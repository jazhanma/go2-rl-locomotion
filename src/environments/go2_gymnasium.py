"""
Go2 Gymnasium Environment for RL training.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
from typing import Dict, Any, Tuple, Optional

class Go2GymnasiumEnv(gym.Env):
    """
    Go2 Quadruped Environment using PyBullet with proper Gymnasium interface.
    """
    
    def __init__(self, config=None, render_mode: str = None):
        super().__init__()
        
        self.config = config or self._get_default_config()
        self.render_mode = render_mode
        
        # Simulation parameters
        self.timestep = 0.01
        self.control_freq = 50
        self.max_episode_steps = 1000
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )
        
        # Observation space: [position(3), orientation(4), linear_vel(3), angular_vel(3), 
        #                     joint_pos(12), joint_vel(12), foot_contacts(4)]
        obs_dim = 3 + 4 + 3 + 3 + 12 + 12 + 4  # 41 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Simulation state
        self.physics_client = None
        self.robot_id = None
        self.plane_id = None
        self.step_count = 0
        
        # Reward tracking
        self.episode_reward = 0.0
        self.prev_position = np.array([0.0, 0.0, 0.0])
        
    def _get_default_config(self):
        """Get default configuration."""
        class Config:
            def __init__(self):
                self.env = type('EnvConfig', (), {
                    'timestep': 0.01,
                    'control_freq': 50,
                    'max_episode_steps': 1000,
                    'robot_urdf': 'r2d2.urdf',
                    'initial_height': 0.5,
                    'initial_pose': [0, 0, 0, 0, 0, 0]
                })()
        return Config()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Clean up previous connection
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except:
                pass
        
        # Initialize PyBullet
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        try:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            
            # Load ground plane
            self.plane_id = p.loadURDF("plane.urdf")
            
            # Load robot
            start_pos = [0, 0, self.config.env.initial_height]
            start_orientation = p.getQuaternionFromEuler(self.config.env.initial_pose[3:])
            self.robot_id = p.loadURDF(self.config.env.robot_urdf, start_pos, start_orientation)
            
            # Reset tracking variables
            self.step_count = 0
            self.episode_reward = 0.0
            self.prev_position = np.array(start_pos)
            
            # Get initial observation
            obs = self._get_observation()
            info = self._get_info()
            
            return obs, info
            
        except Exception as e:
            print(f"Error in reset: {e}")
            return np.zeros(self.observation_space.shape[0]), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step the environment."""
        if self.physics_client is None:
            return np.zeros(self.observation_space.shape[0]), 0.0, True, False, {}
        
        self.step_count += 1
        
        try:
            # Apply action (simple force application for now)
            force = [action[0] * 10, action[1] * 10, 0]  # Use first two actions
            p.applyExternalForce(self.robot_id, -1, force, [0, 0, 0], p.WORLD_FRAME)
            
            # Step simulation
            p.stepSimulation()
            
            # Get observation
            obs = self._get_observation()
            
            # Calculate reward
            reward = self._calculate_reward(obs, action)
            self.episode_reward += reward
            
            # Check termination
            terminated = self._is_terminated(obs)
            truncated = self.step_count >= self.max_episode_steps
            
            # Get info
            info = self._get_info()
            
            return obs, reward, terminated, truncated, info
            
        except Exception as e:
            print(f"Error in step: {e}")
            return np.zeros(self.observation_space.shape[0]), 0.0, True, False, {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        try:
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            vel, ang_vel = p.getBaseVelocity(self.robot_id)
            
            # Get joint states (simplified for R2D2)
            joint_states = p.getJointStates(self.robot_id, range(12))
            joint_positions = [state[0] for state in joint_states]
            joint_velocities = [state[1] for state in joint_states]
            
            # Simulate foot contacts (simplified)
            foot_contacts = [1.0, 1.0, 1.0, 1.0]  # All feet in contact
            
            # Combine observation
            obs = np.concatenate([
                pos,                    # 3: position
                orn,                    # 4: orientation (quaternion)
                vel,                    # 3: linear velocity
                ang_vel,                # 3: angular velocity
                joint_positions,        # 12: joint positions
                joint_velocities,       # 12: joint velocities
                foot_contacts           # 4: foot contacts
            ]).astype(np.float32)
            
            return obs
        except:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
    
    def _calculate_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Calculate reward based on observation and action."""
        try:
            # Extract observation components
            position = obs[:3]
            velocity = obs[7:10]  # linear velocity
            
            # Reward components
            forward_reward = velocity[0] * 10  # Forward movement
            stability_reward = -abs(velocity[1]) * 2  # Lateral stability
            height_reward = -abs(position[2] - 0.5) * 5  # Height maintenance
            energy_penalty = -np.sum(action**2) * 0.01  # Energy efficiency
            
            # Fall penalty
            fall_penalty = -100 if position[2] < 0.3 else 0
            
            total_reward = forward_reward + stability_reward + height_reward + energy_penalty + fall_penalty
            
            return total_reward
        except:
            return 0.0
    
    def _is_terminated(self, obs: np.ndarray) -> bool:
        """Check if episode should be terminated."""
        try:
            position = obs[:3]
            # Terminate if robot falls
            return position[2] < 0.3
        except:
            return True
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information."""
        return {
            'episode_reward': self.episode_reward,
            'step_count': self.step_count,
            'position': self.prev_position.tolist() if hasattr(self, 'prev_position') else [0, 0, 0]
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            time.sleep(1/240)  # 240 Hz rendering
        return None
    
    def close(self):
        """Close the environment."""
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except:
                pass




