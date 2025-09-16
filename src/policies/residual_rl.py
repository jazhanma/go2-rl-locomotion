"""
Residual RL Policy: RL policy adds deltas to a safe controller.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, Optional
from .ppo_baseline import PPOBaseline


class SafeController:
    """
    Safe controller that provides baseline actions for the robot.
    This could be a simple PD controller, trajectory following controller, etc.
    """
    
    def __init__(self, config, action_space):
        self.config = config
        self.action_space = action_space
        self.action_dim = action_space.shape[0]
        
        # Simple PD controller parameters
        self.kp = 0.5  # Position gain
        self.kd = 0.1  # Velocity gain
        self.target_positions = np.zeros(self.action_dim)  # Target joint positions
        
    def compute_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> np.ndarray:
        """
        Compute safe baseline action.
        
        Args:
            observation: Current observation
            info: Additional info dict
            
        Returns:
            Safe baseline action
        """
        # Extract joint positions and velocities from observation
        # This is a simplified version - in practice, you'd need to know
        # the exact structure of your observation space
        if len(observation) >= 24:  # Assuming 12 joint positions + 12 joint velocities
            joint_positions = observation[:12]
            joint_velocities = observation[12:24]
        else:
            # Fallback: return zero action
            return np.zeros(self.action_dim)
        
        # Simple PD controller
        position_error = self.target_positions - joint_positions
        velocity_error = -joint_velocities  # Target velocity is 0
        
        action = self.kp * position_error + self.kd * velocity_error
        
        # Clip to action space bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        return action
    
    def update_target(self, target_positions: np.ndarray) -> None:
        """Update target joint positions."""
        self.target_positions = target_positions.copy()
    
    def reset(self) -> None:
        """Reset controller state."""
        self.target_positions = np.zeros(self.action_dim)


class ResidualRL(PPOBaseline):
    """
    Residual RL Policy: RL policy learns to add deltas to a safe controller.
    
    The total action is: action = safe_controller_action + rl_delta
    where rl_delta is learned by the RL policy.
    """
    
    def __init__(self, config, observation_space, action_space, device="auto"):
        super().__init__(config, observation_space, action_space, device)
        
        # Initialize safe controller
        self.safe_controller = SafeController(config, action_space)
        
        # Scale factor for RL deltas
        self.delta_scale = config.policy.safe_controller_gain
        
        # Action space bounds for deltas
        self.delta_bounds = action_space.high - action_space.low
        
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Predict action given observation."""
        # Get safe baseline action
        safe_action = self.safe_controller.compute_action(observation)
        
        # Get RL delta
        rl_action, info = self.model.predict(observation, deterministic=deterministic)
        
        # Scale and add delta to safe action
        delta = rl_action * self.delta_scale
        total_action = safe_action + delta
        
        # Clip to action space bounds
        total_action = np.clip(total_action, self.action_space.low, self.action_space.high)
        
        # Store additional info
        if info is None:
            info = {}
        info.update({
            'safe_action': safe_action,
            'rl_delta': delta,
            'total_action': total_action
        })
        
        return total_action, info
    
    def learn(self, total_timesteps: int, callback=None, **kwargs) -> None:
        """Train the policy."""
        # The RL policy learns to predict deltas, not absolute actions
        # We need to modify the environment to provide delta targets
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            **kwargs
        )
    
    def update_safe_controller(self, target_positions: np.ndarray) -> None:
        """Update the safe controller's target positions."""
        self.safe_controller.update_target(target_positions)
    
    def get_safe_action(self, observation: np.ndarray) -> np.ndarray:
        """Get only the safe controller action (without RL delta)."""
        return self.safe_controller.compute_action(observation)
    
    def get_rl_delta(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Get only the RL delta (without safe controller)."""
        rl_action, _ = self.model.predict(observation, deterministic=deterministic)
        return rl_action * self.delta_scale
    
    def reset(self) -> None:
        """Reset both RL policy and safe controller."""
        super().reset()
        self.safe_controller.reset()


class ResidualRLWithAdaptiveGain(ResidualRL):
    """
    Residual RL with adaptive gain based on performance.
    """
    
    def __init__(self, config, observation_space, action_space, device="auto"):
        super().__init__(config, observation_space, action_space, device)
        
        # Adaptive gain parameters
        self.min_gain = 0.01
        self.max_gain = 1.0
        self.gain_decay = 0.995
        self.gain_increase_threshold = 0.8  # Performance threshold for increasing gain
        self.gain_decrease_threshold = 0.3  # Performance threshold for decreasing gain
        
        # Performance tracking
        self.recent_rewards = []
        self.reward_window = 100
        
    def update_gain(self, reward: float) -> None:
        """Update the delta scale based on recent performance."""
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.reward_window:
            self.recent_rewards.pop(0)
        
        if len(self.recent_rewards) >= 10:  # Need some history
            avg_reward = np.mean(self.recent_rewards)
            
            if avg_reward > self.gain_increase_threshold:
                # Good performance: increase RL influence
                self.delta_scale = min(self.max_gain, self.delta_scale * 1.01)
            elif avg_reward < self.gain_decrease_threshold:
                # Poor performance: decrease RL influence
                self.delta_scale = max(self.min_gain, self.delta_scale * self.gain_decay)
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Predict action with adaptive gain."""
        action, info = super().predict(observation, deterministic)
        
        # Add gain information to info
        if info is None:
            info = {}
        info['delta_scale'] = self.delta_scale
        
        return action, info
    
    def learn(self, total_timesteps: int, callback=None, **kwargs) -> None:
        """Train with adaptive gain updates."""
        # Custom callback to update gain during training
        if callback is None:
            callback = self._adaptive_gain_callback()
        else:
            # Combine with existing callback
            callback = self._combine_callbacks(callback, self._adaptive_gain_callback())
        
        super().learn(total_timesteps, callback=callback, **kwargs)
    
    def _adaptive_gain_callback(self):
        """Create callback for adaptive gain updates."""
        class AdaptiveGainCallback:
            def __init__(self, residual_rl):
                self.residual_rl = residual_rl
                self.last_reward = 0.0
            
            def __call__(self, locals_, globals_):
                # Update gain based on recent reward
                if 'rewards' in locals_ and len(locals_['rewards']) > 0:
                    recent_reward = locals_['rewards'][-1]
                    self.residual_rl.update_gain(recent_reward)
                    self.last_reward = recent_reward
                return True
        
        return AdaptiveGainCallback(self)
    
    def _combine_callbacks(self, callback1, callback2):
        """Combine two callbacks."""
        class CombinedCallback:
            def __init__(self, cb1, cb2):
                self.cb1 = cb1
                self.cb2 = cb2
            
            def __call__(self, locals_, globals_):
                result1 = self.cb1(locals_, globals_) if self.cb1 else True
                result2 = self.cb2(locals_, globals_) if self.cb2 else True
                return result1 and result2
        
        return CombinedCallback(callback1, callback2)

