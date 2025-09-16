"""
Configuration management for Go2 locomotion framework.
"""
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import os


@dataclass
class PolicyConfig:
    """Configuration for RL policies."""
    # PPO parameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Network architecture
    policy_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    value_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    lstm_hidden_size: int = 128
    activation: str = "tanh"
    
    # Residual RL
    use_residual: bool = False
    safe_controller_gain: float = 0.1
    
    # Behavior Cloning
    use_bc_pretrain: bool = False
    bc_epochs: int = 100
    bc_learning_rate: float = 1e-3
    
    # Asymmetric critic
    use_asymmetric_critic: bool = False
    privileged_obs_dim: int = 0


@dataclass
class RewardConfig:
    """Configuration for reward functions."""
    # Speed rewards
    forward_speed_target: float = 0.4
    forward_speed_weight: float = 1.0
    yaw_tracking_weight: float = 0.5
    
    # Stability penalties
    lateral_drift_weight: float = -0.1
    tilt_penalty_weight: float = -0.2
    height_penalty_weight: float = -0.1
    
    # Action penalties
    energy_penalty_weight: float = -0.01
    action_smoothness_weight: float = -0.1
    foot_slip_weight: float = -0.05
    
    # Termination
    early_termination_weight: float = -1.0


@dataclass
class EnvConfig:
    """Configuration for simulation environment."""
    # Simulation
    sim_backend: str = "pybullet"  # pybullet, mujoco, isaac_gym
    timestep: float = 0.01
    control_freq: int = 50
    max_episode_steps: int = 1000
    
    # Robot parameters
    robot_urdf: str = "assets/go2.urdf"
    initial_height: float = 0.3
    initial_pose: List[float] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0])
    
    # Observation space
    obs_noise_std: float = 0.01
    action_noise_std: float = 0.01
    
    # Action space
    action_scale: float = 1.0
    action_clip: float = 1.0


@dataclass
class VizConfig:
    """Configuration for visualization."""
    # Training visualization
    plot_freq: int = 100
    save_plots: bool = True
    plot_backend: str = "matplotlib"  # matplotlib, plotly
    
    # 3D visualization
    enable_3d_viz: bool = True
    viz_freq: int = 10
    save_videos: bool = True
    video_fps: int = 30
    
    # Rollout visualization
    rollout_plot_metrics: List[str] = field(default_factory=lambda: [
        "reward", "forward_speed", "lateral_drift", "tilt_angle", "height"
    ])


@dataclass
class LogConfig:
    """Configuration for logging."""
    # Logging directories
    log_dir: str = "logs"
    model_dir: str = "models"
    plot_dir: str = "plots"
    video_dir: str = "videos"
    
    # TensorBoard
    use_tensorboard: bool = True
    tensorboard_log_dir: str = "logs/tensorboard"
    
    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = "go2-locomotion"
    wandb_entity: str = ""
    
    # Checkpointing
    save_freq: int = 10000
    keep_checkpoints: int = 5


@dataclass
class Config:
    """Main configuration class."""
    # Sub-configurations
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    log: LogConfig = field(default_factory=LogConfig)
    
    # Training parameters
    total_timesteps: int = 1000000
    eval_freq: int = 10000
    n_eval_episodes: int = 10
    seed: int = 42
    
    # Experiment name
    experiment_name: str = "go2_experiment"
    
    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Create sub-configurations
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                if isinstance(value, dict):
                    sub_config = getattr(config, key)
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_config, sub_key):
                            setattr(sub_config, sub_key, sub_value)
                else:
                    setattr(config, key, value)
        
        return config
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in updates.items():
            if hasattr(self, key):
                if isinstance(value, dict) and hasattr(getattr(self, key), '__dict__'):
                    sub_config = getattr(self, key)
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_config, sub_key):
                            setattr(sub_config, sub_key, sub_value)
                else:
                    setattr(self, key, value)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def create_experiment_config(experiment_name: str, **kwargs) -> Config:
    """Create configuration for a specific experiment."""
    config = get_default_config()
    config.experiment_name = experiment_name
    
    # Update with provided parameters
    config.update_from_dict(kwargs)
    
    return config

