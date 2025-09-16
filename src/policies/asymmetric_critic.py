"""
Asymmetric Critic: Privileged critic with onboard-sensor actor for Go2 quadruped locomotion.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import Distribution, DiagGaussianDistribution
from stable_baselines3.common.utils import get_device
from .ppo_baseline import PPOBaseline


class PrivilegedCritic(BaseFeaturesExtractor):
    """
    Privileged critic that has access to privileged information (e.g., ground truth states).
    """
    
    def __init__(self, observation_space, privileged_obs_space, features_dim: int = 128,
                 hidden_dims: List[int] = [512, 256, 128], activation: str = "tanh"):
        super().__init__(observation_space, features_dim)
        
        self.privileged_obs_space = privileged_obs_space
        self.hidden_dims = hidden_dims
        self.activation = getattr(F, activation)
        
        # Onboard sensor encoder
        self.onboard_encoder = nn.Sequential(
            nn.Linear(observation_space.shape[0], hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            self.activation,
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            self.activation
        )
        
        # Privileged information encoder
        self.privileged_encoder = nn.Sequential(
            nn.Linear(privileged_obs_space.shape[0], hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            self.activation,
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            self.activation
        )
        
        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dims[1] * 2, hidden_dims[2]),
            nn.LayerNorm(hidden_dims[2]),
            self.activation,
            nn.Linear(hidden_dims[2], features_dim)
        )
        
    def forward(self, observations: torch.Tensor, privileged_obs: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through privileged critic."""
        # Encode onboard sensors
        onboard_features = self.onboard_encoder(observations)
        
        # Encode privileged information
        if privileged_obs is not None:
            privileged_features = self.privileged_encoder(privileged_obs)
        else:
            # Use zeros if no privileged info available
            batch_size = observations.shape[0]
            device = observations.device
            privileged_features = torch.zeros(batch_size, self.hidden_dims[1], device=device)
        
        # Fuse features
        fused_features = torch.cat([onboard_features, privileged_features], dim=-1)
        output = self.fusion_network(fused_features)
        
        return output


class OnboardActor(BaseFeaturesExtractor):
    """
    Actor that only has access to onboard sensors (no privileged information).
    """
    
    def __init__(self, observation_space, features_dim: int = 128,
                 hidden_dims: List[int] = [512, 256, 128], activation: str = "tanh"):
        super().__init__(observation_space, features_dim)
        
        self.hidden_dims = hidden_dims
        self.activation = getattr(F, activation)
        
        # Build network
        layers = []
        input_dim = observation_space.shape[0]
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self.activation
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, features_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through onboard actor."""
        return self.network(observations)


class AsymmetricCriticPolicy(BasePolicy):
    """
    Asymmetric policy with privileged critic and onboard actor.
    """
    
    def __init__(self, observation_space, action_space, lr_schedule, 
                 privileged_obs_space=None, net_arch=None, activation_fn=nn.Tanh,
                 squash_output=False, features_extractor_class=None,
                 features_extractor_kwargs=None, normalize_images=True,
                 optimizer_class=torch.optim.Adam, optimizer_kwargs=None,
                 n_critics=2, share_features_extractor=True):
        
        super().__init__(
            observation_space, action_space, lr_schedule, net_arch, activation_fn,
            squash_output, features_extractor_class, features_extractor_kwargs,
            normalize_images, optimizer_class, optimizer_kwargs, n_critics, share_features_extractor
        )
        
        self.privileged_obs_space = privileged_obs_space
        
        # Create actor (onboard sensors only)
        self.actor = OnboardActor(
            observation_space=observation_space,
            features_dim=net_arch['pi'][-1] if net_arch else 128,
            hidden_dims=net_arch['pi'][:-1] if net_arch else [512, 256],
            activation="tanh"
        )
        
        # Create critic (privileged + onboard)
        self.critic = PrivilegedCritic(
            observation_space=observation_space,
            privileged_obs_space=privileged_obs_space or observation_space,
            features_dim=net_arch['qf'][-1] if net_arch else 128,
            hidden_dims=net_arch['qf'][:-1] if net_arch else [512, 256],
            activation="tanh"
        )
        
        # Action distribution
        self.action_dist = DiagGaussianDistribution(action_space.shape[0])
        
        # Value head
        self.value_head = nn.Linear(
            net_arch['qf'][-1] if net_arch else 128, 1
        )
        
        # Action head
        self.action_head = nn.Linear(
            net_arch['pi'][-1] if net_arch else 128, action_space.shape[0] * 2
        )
        
    def forward(self, obs: torch.Tensor, privileged_obs: torch.Tensor = None, 
                deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # Actor forward (onboard sensors only)
        actor_features = self.actor(obs)
        action_logits = self.action_head(actor_features)
        
        # Create action distribution
        mean_actions, log_std = torch.chunk(action_logits, 2, dim=1)
        actions = self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic)
        log_prob = self.action_dist.log_prob(actions)
        
        # Critic forward (privileged + onboard)
        critic_features = self.critic(obs, privileged_obs)
        values = self.value_head(critic_features)
        
        return actions, values, log_prob
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, 
                        privileged_obs: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for training."""
        # Actor forward
        actor_features = self.actor(obs)
        action_logits = self.action_head(actor_features)
        
        # Create action distribution
        mean_actions, log_std = torch.chunk(action_logits, 2, dim=1)
        log_prob = self.action_dist.log_prob(actions)
        entropy = self.action_dist.entropy()
        
        # Critic forward
        critic_features = self.critic(obs, privileged_obs)
        values = self.value_head(critic_features)
        
        return values, log_prob, entropy


class AsymmetricCritic(PPOBaseline):
    """
    Asymmetric Critic implementation with privileged critic and onboard actor.
    """
    
    def __init__(self, config, observation_space, action_space, privileged_obs_space=None, device="auto"):
        self.config = config
        self.observation_space = observation_space
        self.action_space = action_space
        self.privileged_obs_space = privileged_obs_space or observation_space
        self.device = device
        
        # Create custom policy
        self.policy = AsymmetricCriticPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda _: config.policy.learning_rate,
            privileged_obs_space=privileged_obs_space,
            net_arch={
                'pi': config.policy.policy_hidden_dims,
                'qf': config.policy.value_hidden_dims
            },
            activation_fn=nn.Tanh,
            features_extractor_class=None,  # We use custom extractors
            features_extractor_kwargs=None
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.policy.learning_rate,
            eps=1e-5
        )
        
        # Training state
        self.training_data = {
            'observations': [],
            'privileged_obs': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        # Privileged information availability
        self.privileged_available = True
    
    def predict(self, observation: np.ndarray, privileged_obs: np.ndarray = None, 
                deterministic: bool = True) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Predict action given observation."""
        self.policy.eval()
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            priv_tensor = torch.FloatTensor(privileged_obs).unsqueeze(0).to(self.device) if privileged_obs is not None else None
            
            action, value, log_prob = self.policy(obs_tensor, priv_tensor, deterministic)
            action = action.cpu().numpy().squeeze()
            value = value.cpu().numpy().squeeze()
        
        # Clip to action space bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        info = {
            'value': value,
            'log_prob': log_prob.cpu().numpy().squeeze() if log_prob is not None else None
        }
        
        return action, info
    
    def predict_without_privileged(self, observation: np.ndarray, 
                                  deterministic: bool = True) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Predict action without privileged information (for deployment)."""
        return self.predict(observation, privileged_obs=None, deterministic=deterministic)
    
    def learn(self, total_timesteps: int, callback=None, **kwargs) -> None:
        """Train the asymmetric critic policy."""
        # This is a simplified training loop - in practice, you'd implement
        # a full PPO training loop with privileged critic updates
        
        print(f"Training Asymmetric Critic for {total_timesteps} timesteps")
        
        # Training loop would go here
        # For now, just a placeholder
        for step in range(total_timesteps):
            if step % 1000 == 0:
                print(f"Training step {step}/{total_timesteps}")
            
            # Actual training implementation would go here
            pass
        
        print("Asymmetric Critic training completed")
    
    def add_training_data(self, obs: np.ndarray, privileged_obs: np.ndarray, 
                         action: np.ndarray, reward: float, value: float, 
                         log_prob: float, done: bool) -> None:
        """Add training data for batch learning."""
        self.training_data['observations'].append(obs)
        self.training_data['privileged_obs'].append(privileged_obs)
        self.training_data['actions'].append(action)
        self.training_data['rewards'].append(reward)
        self.training_data['values'].append(value)
        self.training_data['log_probs'].append(log_prob)
        self.training_data['dones'].append(done)
    
    def clear_training_data(self) -> None:
        """Clear accumulated training data."""
        for key in self.training_data:
            self.training_data[key] = []
    
    def get_training_batch(self) -> Dict[str, np.ndarray]:
        """Get training batch from accumulated data."""
        batch = {}
        for key, values in self.training_data.items():
            if values:
                batch[key] = np.array(values)
            else:
                batch[key] = np.array([])
        return batch
    
    def save(self, path: str) -> None:
        """Save the model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'observation_space': self.observation_space,
            'action_space': self.action_space,
            'privileged_obs_space': self.privileged_obs_space
        }, path)
        print(f"Asymmetric Critic model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load the model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Asymmetric Critic model loaded from {path}")
    
    def get_parameters(self) -> Dict[str, int]:
        """Get model parameters count."""
        total_params = sum(p.numel() for p in self.policy.parameters())
        trainable_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'actor_parameters': sum(p.numel() for p in self.policy.actor.parameters()),
            'critic_parameters': sum(p.numel() for p in self.policy.critic.parameters())
        }
    
    def set_privileged_available(self, available: bool) -> None:
        """Set whether privileged information is available."""
        self.privileged_available = available
    
    def get_privileged_info_usage(self) -> Dict[str, Any]:
        """Get information about privileged information usage."""
        return {
            'privileged_available': self.privileged_available,
            'privileged_obs_dim': self.privileged_obs_space.shape[0] if self.privileged_obs_space else 0,
            'onboard_obs_dim': self.observation_space.shape[0]
        }

