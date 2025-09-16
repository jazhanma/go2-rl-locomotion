#!/usr/bin/env python3
"""
Research-focused Go2 RL training with proper PPO integration.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from environments.go2_gymnasium import Go2GymnasiumEnv

def create_go2_env(render=False):
    """Create Go2 environment with proper Gymnasium interface."""
    env = Go2GymnasiumEnv(render_mode="human" if render else None)
    env = Monitor(env)  # Add monitoring
    return env

def train_ppo_policy(total_timesteps=100000):
    """Train PPO policy on Go2 environment."""
    print("Go2 Research Training - PPO")
    print("=" * 40)
    
    # Create training environment
    print("Creating training environment...")
    train_env = create_go2_env(render=False)
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = create_go2_env(render=True)
    
    # Create PPO model
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./logs/tensorboard/"
    )
    
    # Setup callbacks
    print("Setting up callbacks...")
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=100, verbose=1)
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./models/',
        log_path='./logs/eval/',
        eval_freq=10000,
        callback_on_new_best=stop_callback,
        verbose=1
    )
    
    # Training parameters
    print(f"Starting training for {total_timesteps} timesteps...")
    
    try:
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        print("Training completed!")
        
        # Save the final model
        model.save("go2_ppo_final")
        print("Final model saved as 'go2_ppo_final'")
        
        # Test the trained model
        print("Testing trained model...")
        test_model(model, eval_env, n_episodes=5)
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
        model.save("go2_ppo_interrupted")
        print("Model saved as 'go2_ppo_interrupted'")
    
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    
    finally:
        # Cleanup
        train_env.close()
        eval_env.close()
    
    print("Research training completed!")

def test_model(model, env, n_episodes=5):
    """Test the trained model."""
    print(f"Testing model for {n_episodes} episodes...")
    
    episode_rewards = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        step = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, 'b-o')
    plt.title('Test Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig('test_results.png')
    print(f"Test results saved as 'test_results.png'")
    print(f"Average test reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")

def run_quick_demo():
    """Run a quick demonstration."""
    print("Go2 Quick Demo")
    print("=" * 20)
    
    # Create environment
    env = create_go2_env(render=True)
    
    # Create a simple random policy
    obs, _ = env.reset()
    episode_reward = 0.0
    
    print("Running random policy for 100 steps...")
    
    try:
        for step in range(100):
            # Random action
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            print(f"Step {step + 1}: Reward = {reward:.2f}, Total = {episode_reward:.2f}")
            
            if terminated or truncated:
                print("Episode ended early")
                break
            
            # Small delay for visualization
            import time
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("Demo interrupted by user")
    
    finally:
        env.close()
    
    print(f"Demo completed! Total reward: {episode_reward:.2f}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Go2 Research Training')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test', 'demo'],
                       help='Mode: train, test, or demo')
    parser.add_argument('--model', type=str, default='go2_ppo_final',
                       help='Model path for testing')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes for testing')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Number of training timesteps')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_ppo_policy(args.timesteps)
    elif args.mode == 'test':
        # Load model and test
        model = PPO.load(args.model)
        env = create_go2_env(render=True)
        test_model(model, env, args.episodes)
        env.close()
    elif args.mode == 'demo':
        run_quick_demo()

if __name__ == "__main__":
    main()

