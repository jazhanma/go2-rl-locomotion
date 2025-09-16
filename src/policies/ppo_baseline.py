"""
PPO Baseline Policy with MLP + LSTM architecture for Go2 quadruped locomotion.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import Distribution


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """LSTM-based feature extractor for sequential observations."""
    
    def __init__(self, observation_space, features_dim: int = 128, 
                 lstm_hidden_size: int = 128, num_lstm_layers: int = 1):
        super().__init__(observation_space, features_dim)
        
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        
        # Input projection
        self.input_projection = nn.Linear(observation_space.shape[0], lstm_hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(lstm_hidden_size, features_dim)
        
        # Initialize hidden state
        self.hidden_state = None
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM feature extractor."""
        batch_size = observations.shape[0]
        
        # Project input
        projected = self.input_projection(observations)
        
        # Add sequence dimension if needed
        if len(projected.shape) == 2:
            projected = projected.unsqueeze(1)  # [batch, 1, hidden]
        
        # LSTM forward pass
        lstm_out, self.hidden_state = self.lstm(projected, self.hidden_state)
        
        # Take the last output
        if len(lstm_out.shape) == 3:
            lstm_out = lstm_out[:, -1, :]  # [batch, hidden]
        
        # Project to output features
        features = self.output_projection(lstm_out)
        
        return features
    
    def reset_hidden_state(self, batch_size: int = 1):
        """Reset LSTM hidden state."""
        device = next(self.parameters()).device
        self.hidden_state = (
            torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_size, device=device),
            torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        )


class MLPFeatureExtractor(BaseFeaturesExtractor):
    """MLP-based feature extractor."""
    
    def __init__(self, observation_space, features_dim: int = 128, 
                 hidden_dims: list = [512, 256, 128], activation: str = "tanh"):
        super().__init__(observation_space, features_dim)
        
        self.hidden_dims = hidden_dims
        self.activation = getattr(F, activation)
        
        # Build MLP layers
        layers = []
        input_dim = observation_space.shape[0]
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU() if activation == "relu" else nn.Tanh()
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, features_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        return self.network(observations)


class PPOBaseline:
    """
    PPO Baseline Policy with MLP + LSTM architecture.
    
    This implements a standard PPO policy that can use either:
    1. MLP-only architecture for non-sequential tasks
    2. LSTM architecture for sequential/partially observable tasks
    3. Hybrid MLP + LSTM architecture
    """
    
    def __init__(self, config, observation_space, action_space, device="auto"):
        self.config = config
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        
        # Choose feature extractor based on configuration
        if hasattr(config.policy, 'use_lstm') and config.policy.use_lstm:
            feature_extractor = LSTMFeatureExtractor(
                observation_space=observation_space,
                features_dim=config.policy.policy_hidden_dims[-1],
                lstm_hidden_size=config.policy.lstm_hidden_size,
                num_lstm_layers=1
            )
        else:
            feature_extractor = MLPFeatureExtractor(
                observation_space=observation_space,
                features_dim=config.policy.policy_hidden_dims[-1],
                hidden_dims=config.policy.policy_hidden_dims[:-1],
                activation=config.policy.activation
            )
        
        # Store spaces for later use
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Create a dummy environment for PPO initialization
        from gymnasium import spaces
        dummy_env = type('DummyEnv', (), {
            'observation_space': observation_space,
            'action_space': action_space,
            'reset': lambda: (observation_space.sample(), {}),
            'step': lambda action: (observation_space.sample(), 0, False, False, {})
        })()
        
        # Create PPO model
        self.model = PPO(
            "MlpPolicy" if not hasattr(config.policy, 'use_lstm') or not config.policy.use_lstm else "CnnPolicy",
            dummy_env,
            learning_rate=config.policy.learning_rate,
            n_steps=config.policy.n_steps,
            batch_size=config.policy.batch_size,
            n_epochs=config.policy.n_epochs,
            gamma=config.policy.gamma,
            gae_lambda=config.policy.gae_lambda,
            clip_range=config.policy.clip_range,
            ent_coef=config.policy.ent_coef,
            vf_coef=config.policy.vf_coef,
            max_grad_norm=config.policy.max_grad_norm,
            policy_kwargs={
                "features_extractor_class": type(feature_extractor),
                "features_extractor_kwargs": {
                    "observation_space": observation_space,
                    "features_dim": config.policy.policy_hidden_dims[-1],
                    "hidden_dims": config.policy.policy_hidden_dims[:-1] if not hasattr(config.policy, 'use_lstm') or not config.policy.use_lstm else None,
                    "lstm_hidden_size": config.policy.lstm_hidden_size if hasattr(config.policy, 'use_lstm') and config.policy.use_lstm else None,
                    "activation": config.policy.activation
                }
            },
            device=device,
            verbose=1
        )
        
        # Store feature extractor for manual access if needed
        self.feature_extractor = feature_extractor
        
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Predict action given observation."""
        return self.model.predict(observation, deterministic=deterministic)
    
    def learn(self, total_timesteps: int, callback=None, **kwargs) -> None:
        """Train the policy."""
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            **kwargs
        )
    
    def save(self, path: str) -> None:
        """Save the model."""
        self.model.save(path)
    
    def load(self, path: str) -> None:
        """Load the model."""
        self.model = PPO.load(path, device=self.device)
    
    def reset(self) -> None:
        """Reset internal state (useful for LSTM)."""
        if hasattr(self.feature_extractor, 'reset_hidden_state'):
            self.feature_extractor.reset_hidden_state()
    
    def get_action_distribution(self, observation: np.ndarray) -> Distribution:
        """Get action distribution for analysis."""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.feature_extractor(obs_tensor)
            # This would need to be implemented based on the specific policy architecture
            # For now, return None
            return None
    
    def get_value(self, observation: np.ndarray) -> float:
        """Get value estimate for observation."""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.feature_extractor(obs_tensor)
            # This would need to be implemented based on the specific value function architecture
            # For now, return 0
            return 0.0
    
    def get_parameters(self) -> Dict[str, int]:
        """Get model parameters count."""
        total_params = sum(p.numel() for p in self.model.policy.parameters())
        trainable_params = sum(p.numel() for p in self.model.policy.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }
    
    def set_training_mode(self, training: bool) -> None:
        """Set training mode for the policy."""
        if training:
            self.model.policy.train()
        else:
            self.model.policy.eval()


class PPOBaselineWithCustomReward(PPOBaseline):
    """
    PPO Baseline with custom reward function integration.
    """
    
    def __init__(self, config, observation_space, action_space, reward_function, device="auto"):
        super().__init__(config, observation_space, action_space, device)
        self.reward_function = reward_function
    
    def compute_reward(self, obs: np.ndarray, action: np.ndarray, 
                      next_obs: np.ndarray, done: bool, info: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Compute reward using the custom reward function."""
        total_reward, components = self.reward_function.compute_reward(obs, action, next_obs, done, info)
        
        # Convert components to dictionary for logging
        reward_dict = {
            'forward_speed': components.forward_speed,
            'yaw_tracking': components.yaw_tracking,
            'lateral_drift': components.lateral_drift,
            'tilt_penalty': components.tilt_penalty,
            'height_penalty': components.height_penalty,
            'energy_penalty': components.energy_penalty,
            'action_smoothness': components.action_smoothness,
            'foot_slip': components.foot_slip,
            'early_termination': components.early_termination,
            'total': components.total
        }
        
        return total_reward, reward_dict

