#!/usr/bin/env python3
"""
Next steps demonstration - enhanced Go2 training.
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

class EnhancedGo2Env:
    """Enhanced Go2 environment with better rewards and visualization."""
    
    def __init__(self, render=False):
        self.render = render
        self.physics_client = None
        self.robot_id = None
        self.plane_id = None
        self.step_count = 0
        self.max_steps = 1000
        
        # Action and observation spaces
        self.action_space = np.array([-1.0, 1.0])
        self.observation_space = np.array([-np.inf, np.inf])
        
        # Enhanced tracking
        self.episode_reward = 0.0
        self.prev_position = np.array([0.0, 0.0, 0.0])
        self.reward_history = []
        
    def reset(self):
        """Reset the environment."""
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except:
                pass
        
        if self.render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        try:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            
            # Load ground plane
            self.plane_id = p.loadURDF("plane.urdf")
            
            # Load robot
            start_pos = [0, 0, 0.5]
            start_orientation = p.getQuaternionFromEuler([0, 0, 0])
            self.robot_id = p.loadURDF("r2d2.urdf", start_pos, start_orientation)
            
            # Reset tracking
            self.step_count = 0
            self.episode_reward = 0.0
            self.prev_position = np.array(start_pos)
            self.reward_history = []
            
            return self._get_observation(), {}
            
        except Exception as e:
            print(f"Error in reset: {e}")
            return np.zeros(9), {}
    
    def step(self, action):
        """Step the environment."""
        if self.physics_client is None:
            return np.zeros(9), 0, True, False, {}
        
        self.step_count += 1
        
        try:
            # Enhanced action application
            force = [action * 8, 0, 0]
            p.applyExternalForce(self.robot_id, -1, force, [0, 0, 0], p.WORLD_FRAME)
            
            p.stepSimulation()
            
            obs = self._get_observation()
            reward = self._calculate_enhanced_reward(obs)
            self.episode_reward += reward
            self.reward_history.append(reward)
            
            done = self.step_count >= self.max_steps
            
            info = {
                'episode_reward': self.episode_reward,
                'step_count': self.step_count,
                'position': obs[:3],
                'velocity': obs[3:6]
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
            
            obs = np.array([
                pos[0], pos[1], pos[2],  # position
                vel[0], vel[1], vel[2],  # linear velocity
                ang_vel[0], ang_vel[1], ang_vel[2]  # angular velocity
            ])
            
            return obs
        except:
            return np.zeros(9)
    
    def _calculate_enhanced_reward(self, obs):
        """Enhanced reward function."""
        try:
            current_pos = obs[:3]
            velocity = obs[3:6]
            
            # Multiple reward components
            forward_reward = velocity[0] * 10  # Forward movement
            stability_reward = -abs(velocity[1]) * 2  # Lateral stability
            height_reward = -abs(current_pos[2] - 0.5) * 5  # Height maintenance
            energy_penalty = -0.01  # Energy efficiency
            
            # Fall penalty
            fall_penalty = -100 if current_pos[2] < 0.3 else 0
            
            total_reward = forward_reward + stability_reward + height_reward + energy_penalty + fall_penalty
            
            return total_reward
        except:
            return 0.0
    
    def close(self):
        """Close the environment."""
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except:
                pass

class ImprovedPolicy:
    """Improved policy with some intelligence."""
    
    def __init__(self, action_space):
        self.action_space = action_space
        self.action_history = deque(maxlen=10)
    
    def predict(self, obs, deterministic=True):
        """Predict action with some strategy."""
        # Simple strategy: if moving too fast, slow down
        velocity = obs[3] if len(obs) > 3 else 0
        
        if velocity > 2.0:
            action = -0.5  # Slow down
        elif velocity < -1.0:
            action = 0.5   # Speed up
        else:
            action = np.random.uniform(-0.8, 0.8)  # Random action
        
        self.action_history.append(action)
        return action, {}

def demonstrate_next_steps():
    """Demonstrate next steps capabilities."""
    print("Go2 Next Steps Demonstration")
    print("=" * 40)
    print("Features:")
    print("- Enhanced reward function")
    print("- Improved policy strategy")
    print("- Better visualization")
    print("- Performance tracking")
    print()
    
    # Create environment
    env = EnhancedGo2Env(render=True)
    policy = ImprovedPolicy(env.action_space)
    
    # Training parameters
    n_episodes = 3
    episode_rewards = []
    episode_lengths = []
    
    print(f"Running {n_episodes} episodes with enhanced features...")
    
    try:
        for episode in range(n_episodes):
            print(f"\nEpisode {episode + 1}:")
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            for step in range(env.max_steps):
                action, _ = policy.predict(obs)
                obs, reward, done, _, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if step % 100 == 0:
                    print(f"  Step {step}: Reward = {reward:.2f}, Position = {info['position']}")
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            print(f"  Episode {episode + 1} completed: Reward = {episode_reward:.2f}, Length = {episode_length}")
            
            time.sleep(1)  # Pause between episodes
    
    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user")
    
    finally:
        env.close()
    
    # Analysis
    if episode_rewards:
        print(f"\nResults Summary:")
        print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
        print(f"Best Episode: {np.max(episode_rewards):.2f}")
        print(f"Worst Episode: {np.min(episode_rewards):.2f}")
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(episode_rewards, 'b-o')
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True)
        
        ax2.plot(episode_lengths, 'r-o')
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('enhanced_training_results.png')
        print("Enhanced training results saved as 'enhanced_training_results.png'")
        
        print("\nNext Steps Available:")
        print("1. Implement real RL algorithms (PPO, SAC, TD3)")
        print("2. Add Go2 robot URDF model")
        print("3. Create curriculum learning")
        print("4. Add domain randomization")
        print("5. Implement sim-to-real transfer")
        print("6. Build evaluation benchmarks")
        print("7. Add advanced visualization tools")
    
    print("\nDemonstration completed!")

if __name__ == "__main__":
    demonstrate_next_steps()
