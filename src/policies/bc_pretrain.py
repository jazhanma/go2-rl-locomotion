"""
Behavior Cloning Pretraining + PPO Fine-tune for Go2 quadruped locomotion.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from .ppo_baseline import PPOBaseline
import os
import pickle
from tqdm import tqdm


class BCDataset(Dataset):
    """Dataset for Behavior Cloning training."""
    
    def __init__(self, observations: np.ndarray, actions: np.ndarray):
        """
        Initialize BC dataset.
        
        Args:
            observations: Array of observations (N, obs_dim)
            actions: Array of actions (N, action_dim)
        """
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.FloatTensor(actions)
        
        assert len(self.observations) == len(self.actions), "Observations and actions must have same length"
    
    def __len__(self) -> int:
        return len(self.observations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.observations[idx], self.actions[idx]


class BCNetwork(nn.Module):
    """Neural network for Behavior Cloning."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = [512, 256, 128], 
                 activation: str = "tanh", dropout: float = 0.1):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.activation = getattr(nn, activation.capitalize())()
        self.dropout = dropout
        
        # Build network
        layers = []
        input_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class BCPretrain:
    """
    Behavior Cloning pretraining followed by PPO fine-tuning.
    """
    
    def __init__(self, config, observation_space, action_space, device="auto"):
        self.config = config
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        
        # Initialize BC network
        self.bc_network = BCNetwork(
            obs_dim=observation_space.shape[0],
            action_dim=action_space.shape[0],
            hidden_dims=config.policy.policy_hidden_dims,
            activation=config.policy.activation,
            dropout=0.1
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.bc_network.parameters(),
            lr=config.policy.bc_learning_rate
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training history
        self.training_losses = []
        self.validation_losses = []
        
    def train_bc(self, observations: np.ndarray, actions: np.ndarray, 
                 validation_split: float = 0.2, epochs: int = None) -> Dict[str, List[float]]:
        """
        Train Behavior Cloning network.
        
        Args:
            observations: Expert observations (N, obs_dim)
            actions: Expert actions (N, action_dim)
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs (uses config if None)
            
        Returns:
            Dictionary with training history
        """
        if epochs is None:
            epochs = self.config.policy.bc_epochs
        
        # Split data
        n_train = int(len(observations) * (1 - validation_split))
        train_obs = observations[:n_train]
        train_actions = actions[:n_train]
        val_obs = observations[n_train:]
        val_actions = actions[n_train:]
        
        # Create datasets
        train_dataset = BCDataset(train_obs, train_actions)
        val_dataset = BCDataset(val_obs, val_actions)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Training loop
        for epoch in range(epochs):
            # Training
            self.bc_network.train()
            train_loss = 0.0
            for obs_batch, action_batch in train_loader:
                obs_batch = obs_batch.to(self.device)
                action_batch = action_batch.to(self.device)
                
                self.optimizer.zero_grad()
                predicted_actions = self.bc_network(obs_batch)
                loss = self.criterion(predicted_actions, action_batch)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.bc_network.eval()
            val_loss = 0.0
            with torch.no_grad():
                for obs_batch, action_batch in val_loader:
                    obs_batch = obs_batch.to(self.device)
                    action_batch = action_batch.to(self.device)
                    
                    predicted_actions = self.bc_network(obs_batch)
                    loss = self.criterion(predicted_actions, action_batch)
                    val_loss += loss.item()
            
            # Record losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            self.training_losses.append(avg_train_loss)
            self.validation_losses.append(avg_val_loss)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        
        return {
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses
        }
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Predict action using BC network."""
        self.bc_network.eval()
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action = self.bc_network(obs_tensor)
            action = action.cpu().numpy().squeeze()
        
        # Clip to action space bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        return action, None
    
    def save_bc_model(self, path: str) -> None:
        """Save BC model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.bc_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses
        }, path)
        print(f"BC model saved to {path}")
    
    def load_bc_model(self, path: str) -> None:
        """Load BC model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.bc_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_losses = checkpoint.get('training_losses', [])
        self.validation_losses = checkpoint.get('validation_losses', [])
        print(f"BC model loaded from {path}")
    
    def get_parameters(self) -> Dict[str, int]:
        """Get model parameters count."""
        total_params = sum(p.numel() for p in self.bc_network.parameters())
        trainable_params = sum(p.numel() for p in self.bc_network.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


class BCPretrainPPO(PPOBaseline):
    """
    Complete BC pretraining + PPO fine-tuning pipeline.
    """
    
    def __init__(self, config, observation_space, action_space, device="auto"):
        super().__init__(config, observation_space, action_space, device)
        
        # Initialize BC component
        self.bc_pretrain = BCPretrain(config, observation_space, action_space, device)
        
        # Expert data storage
        self.expert_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }
        
        # Training state
        self.bc_trained = False
        self.ppo_trained = False
    
    def add_expert_data(self, observations: np.ndarray, actions: np.ndarray, 
                       rewards: np.ndarray = None, dones: np.ndarray = None) -> None:
        """Add expert demonstration data."""
        self.expert_data['observations'].append(observations)
        self.expert_data['actions'].append(actions)
        
        if rewards is not None:
            self.expert_data['rewards'].append(rewards)
        if dones is not None:
            self.expert_data['dones'].append(dones)
    
    def load_expert_data(self, data_path: str) -> None:
        """Load expert data from file."""
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.expert_data = data
        print(f"Loaded expert data: {len(self.expert_data['observations'])} trajectories")
    
    def save_expert_data(self, data_path: str) -> None:
        """Save expert data to file."""
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, 'wb') as f:
            pickle.dump(self.expert_data, f)
        print(f"Expert data saved to {data_path}")
    
    def pretrain_bc(self, validation_split: float = 0.2, epochs: int = None) -> Dict[str, List[float]]:
        """Perform BC pretraining."""
        if not self.expert_data['observations']:
            raise ValueError("No expert data available for BC pretraining")
        
        # Concatenate all expert data
        all_obs = np.concatenate(self.expert_data['observations'], axis=0)
        all_actions = np.concatenate(self.expert_data['actions'], axis=0)
        
        print(f"Starting BC pretraining with {len(all_obs)} expert samples")
        
        # Train BC network
        history = self.bc_pretrain.train_bc(all_obs, all_actions, validation_split, epochs)
        
        self.bc_trained = True
        print("BC pretraining completed")
        
        return history
    
    def fine_tune_ppo(self, total_timesteps: int, callback=None, **kwargs) -> None:
        """Fine-tune with PPO after BC pretraining."""
        if not self.bc_trained:
            print("Warning: BC pretraining not completed. Starting PPO from scratch.")
        
        # Initialize PPO policy with BC weights if available
        if self.bc_trained:
            self._transfer_bc_weights()
        
        # Train with PPO
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            **kwargs
        )
        
        self.ppo_trained = True
        print("PPO fine-tuning completed")
    
    def _transfer_bc_weights(self) -> None:
        """Transfer BC network weights to PPO policy."""
        # This is a simplified version - in practice, you'd need to match
        # the exact architecture between BC network and PPO policy
        try:
            # Get BC network state dict
            bc_state_dict = self.bc_pretrain.bc_network.state_dict()
            
            # Get PPO policy state dict
            ppo_state_dict = self.model.policy.state_dict()
            
            # Transfer compatible weights
            transferred_keys = []
            for key, value in bc_state_dict.items():
                if key in ppo_state_dict and value.shape == ppo_state_dict[key].shape:
                    ppo_state_dict[key] = value
                    transferred_keys.append(key)
            
            # Load updated state dict
            self.model.policy.load_state_dict(ppo_state_dict)
            
            print(f"Transferred {len(transferred_keys)} weights from BC to PPO")
            
        except Exception as e:
            print(f"Warning: Could not transfer BC weights to PPO: {e}")
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Predict action using BC (if trained) or PPO."""
        if self.bc_trained and not self.ppo_trained:
            # Use BC network
            return self.bc_pretrain.predict(observation, deterministic)
        else:
            # Use PPO policy
            return super().predict(observation, deterministic)
    
    def save(self, path: str) -> None:
        """Save both BC and PPO models."""
        # Save PPO model
        super().save(path)
        
        # Save BC model
        bc_path = path.replace('.zip', '_bc.pth')
        self.bc_pretrain.save_bc_model(bc_path)
    
    def load(self, path: str) -> None:
        """Load both BC and PPO models."""
        # Load PPO model
        super().load(path)
        
        # Try to load BC model
        bc_path = path.replace('.zip', '_bc.pth')
        if os.path.exists(bc_path):
            self.bc_pretrain.load_bc_model(bc_path)
            self.bc_trained = True
    
    def get_training_status(self) -> Dict[str, bool]:
        """Get training status."""
        return {
            'bc_trained': self.bc_trained,
            'ppo_trained': self.ppo_trained,
            'expert_data_available': len(self.expert_data['observations']) > 0
        }

