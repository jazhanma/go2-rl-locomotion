#!/usr/bin/env python3
"""
Working Go2 training example with simplified RL setup.
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

class SimpleGo2Env:
    """Simplified Go2 environment for training."""
    
    def __init__(self, render=False):
        self.render = render
        self.physics_client = None
        self.robot_id = None
        self.step_count = 0
        self.max_steps = 1000
        
        # Action and observation spaces
        self.action_space = np.array([-1.0, 1.0])  # Simple action range
        self.observation_space = np.array([-np.inf, np.inf])  # Observation range
        
        # Reward tracking
        self.episode_reward = 0.0
        self.prev_position = np.array([0.0, 0.0, 0.0])
        
    def reset(self):
        """Reset the environment."""
        if self.physics_client is not None:
            p.disconnect()
        
        # Initialize PyBullet
        if self.render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
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
    
    def step(self, action):
        """Step the environment."""
        self.step_count += 1
        
        # Apply action (simple force application)
        force = [action * 10, 0, 0]  # Apply force in x-direction
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
    
    def _get_observation(self):
        """Get current observation."""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        vel, ang_vel = p.getBaseVelocity(self.robot_id)
        
        # Simple observation: position and velocity
        obs = np.array([
            pos[0], pos[1], pos[2],  # position
            vel[0], vel[1], vel[2],  # linear velocity
            ang_vel[0], ang_vel[1], ang_vel[2]  # angular velocity
        ])
        
        return obs
    
    def _calculate_reward(self, obs):
        """Calculate reward based on observation."""
        # Simple reward: encourage forward movement
        current_pos = obs[:3]
        forward_velocity = obs[3]  # x-velocity
        
        # Reward for forward movement
        reward = forward_velocity * 10
        
        # Penalty for falling
        if current_pos[2] < 0.3:
            reward -= 100
        
        # Small penalty for energy usage
        reward -= 0.01
        
        return reward
    
    def close(self):
        """Close the environment."""
        if self.physics_client is not None:
            p.disconnect()

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
    print("Simple Go2 Training Demo")
    print("=" * 40)
    
    # Create environment
    env = SimpleGo2Env(render=True)
    policy = SimplePolicy(env.action_space)
    
    # Training parameters
    n_episodes = 10
    episode_rewards = []
    
    print(f"Training for {n_episodes} episodes...")
    
    try:
        for episode in range(n_episodes):
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
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        env.close()
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.grid(True)
    plt.savefig('training_progress.png')
    plt.show()
    
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print("Training completed!")

if __name__ == "__main__":
    train_simple_policy()




