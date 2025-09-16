"""
Tests for policy implementations.
"""
import pytest
import numpy as np
import gymnasium as gym
from unittest.mock import Mock

# Add src to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config import get_default_config
from policies.ppo_baseline import PPOBaseline
from policies.residual_rl import ResidualRL
from policies.bc_pretrain import BCPretrainPPO
from policies.asymmetric_critic import AsymmetricCritic


@pytest.fixture
def config():
    """Create test configuration."""
    config = get_default_config()
    config.total_timesteps = 1000  # Short for testing
    return config


@pytest.fixture
def observation_space():
    """Create test observation space."""
    return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(41,), dtype=np.float32)


@pytest.fixture
def action_space():
    """Create test action space."""
    return gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)


def test_ppo_baseline_creation(config, observation_space, action_space):
    """Test PPO baseline policy creation."""
    policy = PPOBaseline(config, observation_space, action_space)
    
    assert policy is not None
    assert hasattr(policy, 'predict')
    assert hasattr(policy, 'learn')
    assert hasattr(policy, 'save')
    assert hasattr(policy, 'load')


def test_ppo_baseline_prediction(config, observation_space, action_space):
    """Test PPO baseline prediction."""
    policy = PPOBaseline(config, observation_space, action_space)
    
    # Test prediction
    obs = observation_space.sample()
    action, info = policy.predict(obs)
    
    assert action is not None
    assert len(action) == action_space.shape[0]
    assert action_space.contains(action)


def test_residual_rl_creation(config, observation_space, action_space):
    """Test Residual RL policy creation."""
    policy = ResidualRL(config, observation_space, action_space)
    
    assert policy is not None
    assert hasattr(policy, 'predict')
    assert hasattr(policy, 'learn')
    assert hasattr(policy, 'safe_controller')


def test_residual_rl_prediction(config, observation_space, action_space):
    """Test Residual RL prediction."""
    policy = ResidualRL(config, observation_space, action_space)
    
    # Test prediction
    obs = observation_space.sample()
    action, info = policy.predict(obs)
    
    assert action is not None
    assert len(action) == action_space.shape[0]
    assert action_space.contains(action)
    assert 'safe_action' in info
    assert 'rl_delta' in info


def test_bc_pretrain_creation(config, observation_space, action_space):
    """Test BC pretrain policy creation."""
    policy = BCPretrainPPO(config, observation_space, action_space)
    
    assert policy is not None
    assert hasattr(policy, 'predict')
    assert hasattr(policy, 'pretrain_bc')
    assert hasattr(policy, 'fine_tune_ppo')


def test_asymmetric_critic_creation(config, observation_space, action_space):
    """Test Asymmetric Critic policy creation."""
    privileged_obs_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(51,), dtype=np.float32
    )
    
    policy = AsymmetricCritic(config, observation_space, action_space, privileged_obs_space)
    
    assert policy is not None
    assert hasattr(policy, 'predict')
    assert hasattr(policy, 'predict_without_privileged')
    assert hasattr(policy, 'learn')


def test_asymmetric_critic_prediction(config, observation_space, action_space):
    """Test Asymmetric Critic prediction."""
    privileged_obs_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(51,), dtype=np.float32
    )
    
    policy = AsymmetricCritic(config, observation_space, action_space, privileged_obs_space)
    
    # Test prediction with privileged info
    obs = observation_space.sample()
    privileged_obs = privileged_obs_space.sample()
    
    action, info = policy.predict(obs, privileged_obs)
    
    assert action is not None
    assert len(action) == action_space.shape[0]
    assert action_space.contains(action)
    
    # Test prediction without privileged info
    action, info = policy.predict_without_privileged(obs)
    
    assert action is not None
    assert len(action) == action_space.shape[0]
    assert action_space.contains(action)


def test_policy_parameters(config, observation_space, action_space):
    """Test policy parameter counts."""
    policy = PPOBaseline(config, observation_space, action_space)
    
    params = policy.get_parameters()
    
    assert 'total_parameters' in params
    assert 'trainable_parameters' in params
    assert params['total_parameters'] > 0
    assert params['trainable_parameters'] > 0


def test_policy_save_load(config, observation_space, action_space, tmp_path):
    """Test policy save and load."""
    policy = PPOBaseline(config, observation_space, action_space)
    
    # Save policy
    save_path = tmp_path / "test_policy.pth"
    policy.save(str(save_path))
    
    assert save_path.exists()
    
    # Load policy
    new_policy = PPOBaseline(config, observation_space, action_space)
    new_policy.load(str(save_path))
    
    # Test that loaded policy works
    obs = observation_space.sample()
    action, info = new_policy.predict(obs)
    
    assert action is not None
    assert len(action) == action_space.shape[0]


if __name__ == "__main__":
    pytest.main([__file__])

