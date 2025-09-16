#!/usr/bin/env python3
"""
Robust Go2 training example with proper error handling.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pybullet as p
import pybullet_data
import time
import matplotlib.pyplot as plt
from collections import deque

class RobustGo2Env:
    """Robust Go2 environment for training."""
    
    def __init__(self, render=False):
        self.render = render
        self.physics_client = None
        self.robot_id = None
        self.plane_id = None
        self.step_count = 0
        self.max_steps = 500  # Shorter episodes
        
        # Action and observation spaces
        self.action_space = np.array([-1.0, 1.0])
        self.observation_space = np.array([-np.inf, np.inf])
        
        # Reward tracking
        self.episode_reward = 0.0
        self.prev_position = np.array([0.0, 0.0, 0.0])
        
    def reset(self):
        """Reset the environment."""
        # Clean up previous connection
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except:
                pass
        
        # Initialize PyBullet
        if self.render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        try:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            
            # Load ground plane
            self.plane_id = p.loadURDF("plane.urdf")
            
            # Load robot (using R2D2 as placeholder)
            start_pos = [0, 0, 0.5]
            start_orientation = p.getQuaternionFromEuler([0, 0, 0])
            self.robot_id = p.loadURDF("r2d2.urdf", start_pos, start_orientation)
            
            # Reset tracking variables
            self.step_count = 0
            self.episode_reward = 0.0
            self.prev_position = np.array(start_pos)
            
            # Get initial observation
            obs = self._get_observation()
            return obs, {}
            
        except Exception as e:
            print(f"Error in reset: {e}")
            # Return dummy observation
            return np.zeros(9), {}
    
    def step(self, action):
        """Step the environment."""
        if self.physics_client is None:
            return np.zeros(9), 0, True, False, {}
        
        self.step_count += 1
        
        try:
            # Apply action (simple force application)
            force = [action * 5, 0, 0]  # Reduced force
            p.applyExternalForce(self.robot_id, -1, force, [0, 0, 0], p.WORLD_FRAME)
            
            # Step simulation
            p.stepSimulation()
            
            # Get observation
            obs = self._get_observation()
            
            # Calculate reward
            reward = self._calculate_reward(obs)
            self.episode_reward += reward
            
            # Check if done
            done = self.step_count >= self.max_steps
            
            # Get info
            info = {
                'episode_reward': self.episode_reward,
                'step_count': self.step_count
            }
            
            return obs, reward, done, False, info
            
        except Exception as e:
            print(f"Error in step: {e}")
            return np.zeros(9), 0, True, False, {}
    
    def _get_observation(self):
        """Get current observation."""
        try:
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            vel, ang_vel = p.getBaseVelocity(self.robot_id)
            
            # Simple observation: position and velocity
            obs = np.array([
                pos[0], pos[1], pos[2],  # position
                vel[0], vel[1], vel[2],  # linear velocity
                ang_vel[0], ang_vel[1], ang_vel[2]  # angular velocity
            ])
            
            return obs
        except:
            return np.zeros(9)
    
    def _calculate_reward(self, obs):
        """Calculate reward based on observation."""
        try:
            # Simple reward: encourage forward movement
            current_pos = obs[:3]
            forward_velocity = obs[3]  # x-velocity
            
            # Reward for forward movement
            reward = forward_velocity * 5
            
            # Penalty for falling
            if current_pos[2] < 0.3:
                reward -= 50
            
            # Small penalty for energy usage
            reward -= 0.01
            
            return reward
        except:
            return 0.0
    
    def close(self):
        """Close the environment."""
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except:
                pass

class SimplePolicy:
    """Simple random policy for demonstration."""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def predict(self, obs, deterministic=True):
        """Predict action."""
        # Simple random action
        action = np.random.uniform(-1, 1)
        return action, {}

def train_simple_policy():
    """Train a simple policy."""
    print("Robust Go2 Training Demo")
    print("=" * 40)
    
    # Create environment
    env = RobustGo2Env(render=True)
    policy = SimplePolicy(env.action_space)
    
    # Training parameters
    n_episodes = 5  # Fewer episodes for demo
    episode_rewards = []
    
    print(f"Training for {n_episodes} episodes...")
    
    try:
        for episode in range(n_episodes):
            print(f"Starting episode {episode + 1}...")
            obs, _ = env.reset()
            episode_reward = 0.0
            
            for step in range(env.max_steps):
                # Get action from policy
                action, _ = policy.predict(obs)
                
                # Step environment
                obs, reward, done, _, info = env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
            
            # Small delay for visualization
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        env.close()
    
    # Plot results
    if episode_rewards:
        plt.figure(figsize=(10, 6))
        plt.plot(episode_rewards)
        plt.title('Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.grid(True)
        plt.savefig('training_progress.png')
        print("Training plot saved as 'training_progress.png'")
        
        print(f"Average reward: {np.mean(episode_rewards):.2f}")
        print("Training completed!")
    else:
        print("No episodes completed successfully.")

if __name__ == "__main__":
    train_simple_policy()





