"""
Reward Function Rules (RFR) for Go2 quadruped locomotion.
"""
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class RewardComponents:
    """Container for individual reward components."""
    forward_speed: float = 0.0
    yaw_tracking: float = 0.0
    lateral_drift: float = 0.0
    tilt_penalty: float = 0.0
    height_penalty: float = 0.0
    energy_penalty: float = 0.0
    action_smoothness: float = 0.0
    foot_slip: float = 0.0
    early_termination: float = 0.0
    total: float = 0.0


class RewardFunctionRules:
    """
    Comprehensive reward function for quadruped locomotion.
    
    Implements all the required reward components:
    - Forward speed reward (≥0.4 m/s target)
    - Yaw tracking reward
    - Lateral drift penalty
    - Tilt & height stability penalties
    - Energy/torque penalty
    - Action smoothness penalty
    - Foot slip penalty
    - Early termination penalty
    """
    
    def __init__(self, config):
        self.config = config
        self.prev_actions = None
        self.prev_velocities = None
        
    def compute_reward(self, obs: np.ndarray, action: np.ndarray, 
                      next_obs: np.ndarray, done: bool, info: Dict[str, Any]) -> Tuple[float, RewardComponents]:
        """
        Compute total reward and individual components.
        
        Args:
            obs: Current observation
            action: Action taken
            next_obs: Next observation
            done: Whether episode is done
            info: Additional info dict
            
        Returns:
            Tuple of (total_reward, reward_components)
        """
        components = RewardComponents()
        
        # Extract relevant information from observations and info
        robot_state = self._extract_robot_state(obs, next_obs, info)
        
        # 1. Forward speed reward (≥0.4 m/s target)
        components.forward_speed = self._compute_forward_speed_reward(robot_state)
        
        # 2. Yaw tracking reward
        components.yaw_tracking = self._compute_yaw_tracking_reward(robot_state)
        
        # 3. Lateral drift penalty
        components.lateral_drift = self._compute_lateral_drift_penalty(robot_state)
        
        # 4. Tilt & height stability penalties
        components.tilt_penalty = self._compute_tilt_penalty(robot_state)
        components.height_penalty = self._compute_height_penalty(robot_state)
        
        # 5. Energy/torque penalty
        components.energy_penalty = self._compute_energy_penalty(action, robot_state)
        
        # 6. Action smoothness penalty
        components.action_smoothness = self._compute_action_smoothness_penalty(action)
        
        # 7. Foot slip penalty
        components.foot_slip = self._compute_foot_slip_penalty(robot_state)
        
        # 8. Early termination penalty
        components.early_termination = self._compute_early_termination_penalty(done, info)
        
        # Compute total reward
        components.total = (
            self.config.reward.forward_speed_weight * components.forward_speed +
            self.config.reward.yaw_tracking_weight * components.yaw_tracking +
            self.config.reward.lateral_drift_weight * components.lateral_drift +
            self.config.reward.tilt_penalty_weight * components.tilt_penalty +
            self.config.reward.height_penalty_weight * components.height_penalty +
            self.config.reward.energy_penalty_weight * components.energy_penalty +
            self.config.reward.action_smoothness_weight * components.action_smoothness +
            self.config.reward.foot_slip_weight * components.foot_slip +
            self.config.reward.early_termination_weight * components.early_termination
        )
        
        # Update internal state
        self.prev_actions = action.copy()
        if 'robot_velocities' in info:
            self.prev_velocities = info['robot_velocities'].copy()
        
        return components.total, components
    
    def _extract_robot_state(self, obs: np.ndarray, next_obs: np.ndarray, 
                           info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract robot state information from observations and info."""
        # This is a simplified extraction - in practice, you'd need to know
        # the exact structure of your observation space
        state = {}
        
        # Assume observation contains: [position(3), orientation(4), linear_vel(3), angular_vel(3), ...]
        if len(obs) >= 13:
            state['position'] = obs[:3]
            state['orientation'] = obs[3:7]  # quaternion
            state['linear_velocity'] = obs[7:10]
            state['angular_velocity'] = obs[10:13]
        
        # Extract from info if available
        state.update({
            'target_velocity': info.get('target_velocity', np.array([0.4, 0.0, 0.0])),
            'target_yaw': info.get('target_yaw', 0.0),
            'foot_contacts': info.get('foot_contacts', np.array([True, True, True, True])),
            'foot_positions': info.get('foot_positions', np.zeros((4, 3))),
            'foot_velocities': info.get('foot_velocities', np.zeros((4, 3))),
            'joint_torques': info.get('joint_torques', np.zeros(12)),
            'joint_positions': info.get('joint_positions', np.zeros(12)),
            'joint_velocities': info.get('joint_velocities', np.zeros(12)),
        })
        
        return state
    
    def _compute_forward_speed_reward(self, state: Dict[str, Any]) -> float:
        """Compute forward speed reward (≥0.4 m/s target)."""
        forward_velocity = state['linear_velocity'][0]  # x-component
        target_speed = self.config.reward.forward_speed_target
        
        # Exponential reward that peaks at target speed
        speed_ratio = forward_velocity / target_speed
        if speed_ratio >= 1.0:
            # Full reward when at or above target speed
            return 1.0
        else:
            # Exponential reward that encourages reaching target speed
            return np.exp(-2 * (1 - speed_ratio) ** 2)
    
    def _compute_yaw_tracking_reward(self, state: Dict[str, Any]) -> float:
        """Compute yaw tracking reward."""
        # Extract yaw from quaternion
        orientation = state['orientation']
        if len(orientation) >= 4:
            # Convert quaternion to yaw angle
            w, x, y, z = orientation
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        else:
            yaw = 0.0
        
        target_yaw = state['target_yaw']
        yaw_error = np.abs(yaw - target_yaw)
        
        # Normalize to [0, 1] range
        yaw_error = np.minimum(yaw_error, np.pi) / np.pi
        
        # Reward decreases with yaw error
        return np.exp(-5 * yaw_error)
    
    def _compute_lateral_drift_penalty(self, state: Dict[str, Any]) -> float:
        """Compute lateral drift penalty."""
        lateral_velocity = state['linear_velocity'][1]  # y-component
        target_lateral = 0.0  # No lateral movement desired
        
        # Penalty proportional to lateral velocity magnitude
        lateral_error = np.abs(lateral_velocity - target_lateral)
        return -lateral_error
    
    def _compute_tilt_penalty(self, state: Dict[str, Any]) -> float:
        """Compute tilt penalty (roll and pitch)."""
        orientation = state['orientation']
        if len(orientation) >= 4:
            # Convert quaternion to roll and pitch
            w, x, y, z = orientation
            
            # Roll (rotation around x-axis)
            roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
            
            # Pitch (rotation around y-axis)
            pitch = np.arcsin(2 * (w * y - z * x))
            
            # Penalty for excessive tilt
            tilt_magnitude = np.sqrt(roll**2 + pitch**2)
            return -tilt_magnitude
        else:
            return 0.0
    
    def _compute_height_penalty(self, state: Dict[str, Any]) -> float:
        """Compute height stability penalty."""
        position = state['position']
        target_height = 0.3  # Target height in meters
        
        if len(position) >= 3:
            height_error = np.abs(position[2] - target_height)
            return -height_error
        else:
            return 0.0
    
    def _compute_energy_penalty(self, action: np.ndarray, state: Dict[str, Any]) -> float:
        """Compute energy/torque penalty."""
        joint_torques = state['joint_torques']
        
        if len(joint_torques) > 0:
            # Penalty proportional to squared torques (energy consumption)
            energy = np.sum(joint_torques ** 2)
            return -energy
        else:
            # Fallback: use action magnitude as proxy for energy
            return -np.sum(action ** 2)
    
    def _compute_action_smoothness_penalty(self, action: np.ndarray) -> float:
        """Compute action smoothness penalty."""
        if self.prev_actions is not None:
            action_diff = action - self.prev_actions
            smoothness_penalty = np.sum(action_diff ** 2)
            return -smoothness_penalty
        else:
            return 0.0
    
    def _compute_foot_slip_penalty(self, state: Dict[str, Any]) -> float:
        """Compute foot slip penalty."""
        foot_contacts = state['foot_contacts']
        foot_velocities = state['foot_velocities']
        
        if len(foot_contacts) > 0 and len(foot_velocities) > 0:
            slip_penalty = 0.0
            for i, (contact, velocity) in enumerate(zip(foot_contacts, foot_velocities)):
                if contact:  # If foot is in contact with ground
                    # Penalize high velocity when foot should be stationary
                    slip_magnitude = np.linalg.norm(velocity)
                    slip_penalty += slip_magnitude
            
            return -slip_penalty
        else:
            return 0.0
    
    def _compute_early_termination_penalty(self, done: bool, info: Dict[str, Any]) -> float:
        """Compute early termination penalty."""
        if done and info.get('timeout', False):
            # Episode ended due to timeout (good)
            return 0.0
        elif done:
            # Episode ended early (bad)
            return -1.0
        else:
            return 0.0
    
    def get_reward_breakdown(self) -> Dict[str, float]:
        """Get the last computed reward breakdown."""
        # This would store the last reward components for analysis
        return getattr(self, '_last_reward_components', {})
    
    def reset(self) -> None:
        """Reset internal state."""
        self.prev_actions = None
        self.prev_velocities = None

