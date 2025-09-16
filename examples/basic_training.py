#!/usr/bin/env python3
"""
Basic training example for Go2 quadruped locomotion.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config import get_default_config
from environments.go2_pybullet import Go2PyBulletEnv
from rewards.reward_functions import RewardFunctionRules
from policies.ppo_baseline import PPOBaselineWithCustomReward
from visualization.training_viz import TrainingVisualizer
from visualization.viz_3d import Viz3D
from utils.logging import Logger


def main():
    """Basic training example."""
    print("Go2 Basic Training Example")
    print("=" * 40)
    
    # Create configuration
    config = get_default_config()
    config.experiment_name = "basic_training_example"
    config.total_timesteps = 100000  # Shorter for example
    config.eval_freq = 5000
    config.n_eval_episodes = 5
    
    # Create environment
    print("Creating environment...")
    env = Go2PyBulletEnv(config, render_mode="human")
    
    # Create reward function
    print("Creating reward function...")
    reward_function = RewardFunctionRules(config)
    
    # Create logger
    print("Setting up logging...")
    logger = Logger(config)
    
    # Create policy
    print("Creating PPO policy...")
    policy = PPOBaselineWithCustomReward(config, env.observation_space, env.action_space, reward_function)
    
    # Create visualizers
    print("Setting up visualization...")
    training_viz = TrainingVisualizer(config, backend="matplotlib")
    viz_3d = Viz3D(config, enable_gui=True)
    
    # Start training visualization
    training_viz.start_live_plotting()
    
    try:
        # Training loop
        print("Starting training...")
        episode = 0
        total_steps = 0
        
        while total_steps < config.total_timesteps:
            episode += 1
            
            # Reset environment
            obs, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
            episode_metrics = {}
            
            # Episode loop
            done = False
            while not done and episode_length < config.env.max_episode_steps:
                # Get action from policy
                action, action_info = policy.predict(obs)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Update episode metrics
                episode_reward += reward
                episode_length += 1
                total_steps += 1
                
                # Update 3D visualization
                if episode % config.viz.viz_freq == 0:
                    position = obs[:3]
                    orientation = obs[3:7]
                    joint_positions = obs[13:25]
                    
                    viz_3d.update_robot_state(position, orientation, joint_positions)
                
                # Check termination
                done = terminated or truncated
                obs = next_obs
                
                # Log step metrics
                if 'reward_info' in info:
                    for key, value in info['reward_info'].items():
                        if key not in episode_metrics:
                            episode_metrics[key] = []
                        episode_metrics[key].append(value)
            
            # Log episode
            avg_metrics = {key: sum(values)/len(values) for key, values in episode_metrics.items()}
            logger.log_episode(episode_reward, episode_length, avg_metrics)
            
            # Update training visualization
            training_viz.add_episode_data(episode, episode_reward, episode_length, avg_metrics)
            
            # Print progress
            if episode % 10 == 0:
                print(f"Episode {episode}: Reward = {episode_reward:.2f}, Length = {episode_length}, "
                      f"Total Steps = {total_steps}")
        
        print("Training completed!")
        
        # Save final model
        model_path = logger.save_model(policy)
        print(f"Final model saved to {model_path}")
        
        # Save plots
        training_viz.save_plots(logger.plot_dir)
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # Cleanup
        training_viz.close()
        viz_3d.close()
        logger.close()
        env.close()
    
    print("Example completed!")


if __name__ == "__main__":
    main()

