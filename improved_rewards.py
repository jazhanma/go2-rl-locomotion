#!/usr/bin/env python3
"""
Improved reward functions for better Go2 learning.
"""
import numpy as np

class ImprovedRewardFunction:
    """Enhanced reward function for better learning."""
    
    def __init__(self):
        self.prev_position = None
        self.prev_velocity = None
        self.step_count = 0
        
    def compute_reward(self, obs, action, next_obs, done, info):
        """Compute improved reward."""
        self.step_count += 1
        
        # Extract state information
        position = obs[:3]
        velocity = obs[7:10]  # linear velocity
        angular_velocity = obs[10:13]
        
        # Initialize previous values
        if self.prev_position is None:
            self.prev_position = position.copy()
            self.prev_velocity = velocity.copy()
            return 0.0
        
        # Calculate deltas
        position_delta = position - self.prev_position
        velocity_delta = velocity - self.prev_velocity
        
        # 1. Forward Progress Reward (most important)
        forward_progress = position_delta[0]  # x-direction movement
        forward_reward = np.tanh(forward_progress * 10) * 10  # Bounded reward
        
        # 2. Speed Maintenance Reward
        target_speed = 0.5  # m/s
        speed_error = abs(np.linalg.norm(velocity) - target_speed)
        speed_reward = np.exp(-speed_error * 5) * 5
        
        # 3. Stability Reward
        lateral_velocity = abs(velocity[1])  # y-direction
        angular_velocity_magnitude = np.linalg.norm(angular_velocity)
        stability_reward = -lateral_velocity * 2 - angular_velocity_magnitude * 0.5
        
        # 4. Height Maintenance
        target_height = 0.5
        height_error = abs(position[2] - target_height)
        height_reward = -height_error * 10
        
        # 5. Energy Efficiency
        action_magnitude = np.linalg.norm(action)
        energy_penalty = -action_magnitude * 0.1
        
        # 6. Smoothness Reward
        action_change = np.linalg.norm(action - getattr(self, 'prev_action', np.zeros_like(action)))
        smoothness_reward = -action_change * 0.5
        
        # 7. Survival Bonus
        survival_bonus = 1.0 if not done else 0.0
        
        # 8. Progress Bonus (encourage longer episodes)
        progress_bonus = 0.01 * self.step_count
        
        # 9. Fall Penalty
        fall_penalty = -100 if position[2] < 0.3 else 0
        
        # Combine all rewards
        total_reward = (
            forward_reward +
            speed_reward +
            stability_reward +
            height_reward +
            energy_penalty +
            smoothness_reward +
            survival_bonus +
            progress_bonus +
            fall_penalty
        )
        
        # Update previous values
        self.prev_position = position.copy()
        self.prev_velocity = velocity.copy()
        self.prev_action = action.copy()
        
        return total_reward

def test_reward_function():
    """Test the improved reward function."""
    print("Testing Improved Reward Function")
    print("=" * 40)
    
    reward_fn = ImprovedRewardFunction()
    
    # Simulate some episodes
    for episode in range(3):
        print(f"\nEpisode {episode + 1}:")
        total_reward = 0.0
        
        # Reset
        reward_fn.prev_position = None
        reward_fn.prev_velocity = None
        reward_fn.step_count = 0
        
        for step in range(10):
            # Simulate observation and action
            obs = np.random.randn(41)
            action = np.random.uniform(-1, 1, 12)
            next_obs = obs + np.random.randn(41) * 0.1
            done = step == 9
            
            reward = reward_fn.compute_reward(obs, action, next_obs, done, {})
            total_reward += reward
            
            print(f"  Step {step + 1}: Reward = {reward:.3f}")
        
        print(f"  Total Episode Reward: {total_reward:.3f}")

if __name__ == "__main__":
    test_reward_function()
