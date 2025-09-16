#!/usr/bin/env python3
"""
Domain randomization for robust Go2 training.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional

class RandomizedGo2Env(gym.Env):
    """Go2 environment with domain randomization."""
    
    def __init__(self, config=None, render_mode: str = None):
        super().__init__()
        
        self.config = config or self._get_default_config()
        self.render_mode = render_mode
        
        # Action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        obs_dim = 3 + 4 + 3 + 3 + 12 + 12 + 4  # 41 dimensions
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Domain randomization parameters
        self.randomize_gravity = True
        self.randomize_mass = True
        self.randomize_friction = True
        self.randomize_restitution = True
        self.randomize_initial_pose = True
        
        # Randomization ranges
        self.gravity_range = (-12.0, -8.0)  # Gravity variation
        self.mass_range = (0.5, 2.0)        # Mass variation
        self.friction_range = (0.1, 1.5)    # Friction variation
        self.restitution_range = (0.0, 0.8) # Restitution variation
        self.initial_height_range = (0.3, 0.7)  # Initial height variation
        
        # Simulation state
        self.physics_client = None
        self.robot_id = None
        self.plane_id = None
        self.step_count = 0
        self.max_episode_steps = 1000
        
        # Reward tracking
        self.episode_reward = 0.0
        
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
    
    def _randomize_environment(self):
        """Randomize environment parameters."""
        if self.randomize_gravity:
            gravity = np.random.uniform(*self.gravity_range)
            p.setGravity(0, 0, gravity)
        
        if self.randomize_mass and self.robot_id is not None:
            mass = np.random.uniform(*self.mass_range)
            p.changeDynamics(self.robot_id, -1, mass=mass)
        
        if self.randomize_friction and self.plane_id is not None:
            friction = np.random.uniform(*self.friction_range)
            p.changeDynamics(self.plane_id, -1, lateralFriction=friction)
        
        if self.randomize_restitution and self.plane_id is not None:
            restitution = np.random.uniform(*self.restitution_range)
            p.changeDynamics(self.plane_id, -1, restitution=restitution)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment with randomization."""
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
            
            # Set base gravity (will be randomized)
            p.setGravity(0, 0, -9.81)
            
            # Load ground plane
            self.plane_id = p.loadURDF("plane.urdf")
            
            # Randomize initial pose
            if self.randomize_initial_pose:
                height = np.random.uniform(*self.initial_height_range)
                yaw = np.random.uniform(-0.5, 0.5)  # Small yaw variation
                start_pos = [0, 0, height]
                start_orientation = p.getQuaternionFromEuler([0, 0, yaw])
            else:
                start_pos = [0, 0, self.config.env.initial_height]
                start_orientation = p.getQuaternionFromEuler(self.config.env.initial_pose[3:])
            
            # Load robot
            self.robot_id = p.loadURDF(self.config.env.robot_urdf, start_pos, start_orientation)
            
            # Apply domain randomization
            self._randomize_environment()
            
            # Reset tracking variables
            self.step_count = 0
            self.episode_reward = 0.0
            
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
            # Apply action
            force = [action[0] * 10, action[1] * 10, 0]
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
            
            # Get joint states
            joint_states = p.getJointStates(self.robot_id, range(12))
            joint_positions = [state[0] for state in joint_states]
            joint_velocities = [state[1] for state in joint_states]
            
            # Simulate foot contacts
            foot_contacts = [1.0, 1.0, 1.0, 1.0]
            
            # Combine observation
            obs = np.concatenate([
                pos,                    # 3: position
                orn,                    # 4: orientation
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
        """Calculate reward."""
        try:
            position = obs[:3]
            velocity = obs[7:10]
            
            # Reward components
            forward_reward = velocity[0] * 10
            stability_reward = -abs(velocity[1]) * 2
            height_reward = -abs(position[2] - 0.5) * 5
            energy_penalty = -np.sum(action**2) * 0.01
            fall_penalty = -100 if position[2] < 0.3 else 0
            
            total_reward = forward_reward + stability_reward + height_reward + energy_penalty + fall_penalty
            
            return total_reward
        except:
            return 0.0
    
    def _is_terminated(self, obs: np.ndarray) -> bool:
        """Check termination."""
        try:
            position = obs[:3]
            return position[2] < 0.3
        except:
            return True
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info."""
        return {
            'episode_reward': self.episode_reward,
            'step_count': self.step_count
        }
    
    def close(self):
        """Close environment."""
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except:
                pass

def train_with_domain_randomization():
    """Train with domain randomization."""
    print("Go2 Domain Randomization Training")
    print("=" * 40)
    
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    
    # Create randomized environment
    env = RandomizedGo2Env(render_mode=None)
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log="./logs/tensorboard_randomized/"
    )
    
    print("Training with domain randomization...")
    print("Randomized parameters:")
    print(f"  - Gravity: {env.gravity_range}")
    print(f"  - Mass: {env.mass_range}")
    print(f"  - Friction: {env.friction_range}")
    print(f"  - Restitution: {env.restitution_range}")
    print(f"  - Initial Height: {env.initial_height_range}")
    
    try:
        # Train for longer with randomization
        model.learn(total_timesteps=50000, progress_bar=True)
        
        # Save model
        model.save("go2_randomized_final")
        print("Randomized model saved as 'go2_randomized_final'")
        
        # Test robustness
        test_robustness(model, env)
        
    except KeyboardInterrupt:
        print("Training interrupted")
        model.save("go2_randomized_interrupted")
    
    finally:
        env.close()

def test_robustness(model, env, n_episodes=10):
    """Test model robustness."""
    print(f"\nTesting robustness for {n_episodes} episodes...")
    
    rewards = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        
        for step in range(env.max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    print(f"Robustness test - Mean: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Min: {np.min(rewards):.2f}, Max: {np.max(rewards):.2f}")

if __name__ == "__main__":
    train_with_domain_randomization()




