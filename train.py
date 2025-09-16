#!/usr/bin/env python3
"""
Main training script for Go2 quadruped locomotion framework.
"""
import argparse
import os
import sys
import time
import numpy as np
from typing import Dict, Any, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.config import Config, get_default_config, create_experiment_config
from utils.logging import Logger
from environments.go2_pybullet import Go2PyBulletEnv
from rewards.reward_functions import RewardFunctionRules
from policies.ppo_baseline import PPOBaseline, PPOBaselineWithCustomReward
from policies.residual_rl import ResidualRL
from policies.bc_pretrain import BCPretrainPPO
from policies.asymmetric_critic import AsymmetricCritic
from visualization.training_viz import TrainingVisualizer
from visualization.rollout_viz import RolloutVisualizer
from visualization.viz_3d import Viz3D


def create_policy(config: Config, observation_space, action_space, reward_function=None):
    """Create policy based on configuration."""
    policy_type = getattr(config, 'policy_type', 'ppo_baseline')
    
    if policy_type == 'ppo_baseline':
        if reward_function:
            return PPOBaselineWithCustomReward(config, observation_space, action_space, reward_function)
        else:
            return PPOBaseline(config, observation_space, action_space)
    
    elif policy_type == 'residual_rl':
        return ResidualRL(config, observation_space, action_space)
    
    elif policy_type == 'bc_pretrain':
        return BCPretrainPPO(config, observation_space, action_space)
    
    elif policy_type == 'asymmetric_critic':
        # Create privileged observation space (simplified)
        privileged_obs_space = type(observation_space)(
            low=-np.inf, high=np.inf, shape=(observation_space.shape[0] + 10,)
        )
        return AsymmetricCritic(config, observation_space, action_space, privileged_obs_space)
    
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def train_policy(config: Config, policy, env, logger: Logger, 
                training_viz: TrainingVisualizer, viz_3d: Viz3D = None):
    """Train the policy."""
    print(f"Starting training with {config.total_timesteps} timesteps...")
    
    # Training loop
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
            if hasattr(policy, 'predict_without_privileged'):
                # Asymmetric critic without privileged info
                action, action_info = policy.predict_without_privileged(obs)
            else:
                action, action_info = policy.predict(obs)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Update episode metrics
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            # Update 3D visualization
            if viz_3d and episode % config.viz.viz_freq == 0:
                # Extract robot state from observation
                position = obs[:3]
                orientation = obs[3:7]
                joint_positions = obs[13:25]  # Assuming joint positions are at indices 13-25
                
                viz_3d.update_robot_state(position, orientation, joint_positions)
                
                # Render frame
                frame = viz_3d.render_frame()
                if frame is not None and logger.recording:
                    logger.add_video_frame(frame)
            
            # Check termination
            done = terminated or truncated
            
            # Update observation
            obs = next_obs
            
            # Log step metrics
            if 'reward_info' in info:
                for key, value in info['reward_info'].items():
                    if key not in episode_metrics:
                        episode_metrics[key] = []
                    episode_metrics[key].append(value)
        
        # Log episode
        avg_metrics = {key: np.mean(values) for key, values in episode_metrics.items()}
        logger.log_episode(episode_reward, episode_length, avg_metrics)
        
        # Update training visualization
        training_viz.add_episode_data(episode, episode_reward, episode_length, avg_metrics)
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}: Reward = {episode_reward:.2f}, Length = {episode_length}, "
                  f"Total Steps = {total_steps}")
        
        # Save model periodically
        if episode % config.log.save_freq == 0:
            model_path = logger.save_model(policy)
            print(f"Model saved to {model_path}")
        
        # Evaluate policy periodically
        if episode % config.eval_freq == 0:
            evaluate_policy(policy, env, logger, n_episodes=config.n_eval_episodes)
    
    print("Training completed!")
    return policy


def evaluate_policy(policy, env, logger: Logger, n_episodes: int = 10) -> Dict[str, float]:
    """Evaluate policy performance."""
    print(f"Evaluating policy for {n_episodes} episodes...")
    
    eval_rewards = []
    eval_lengths = []
    eval_metrics = {}
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_metrics = {}
        
        done = False
        while not done and episode_length < env.config.env.max_episode_steps:
            if hasattr(policy, 'predict_without_privileged'):
                action, _ = policy.predict_without_privileged(obs)
            else:
                action, _ = policy.predict(obs)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if 'reward_info' in info:
                for key, value in info['reward_info'].items():
                    if key not in episode_metrics:
                        episode_metrics[key] = []
                    episode_metrics[key].append(value)
            
            done = terminated or truncated
            obs = next_obs
        
        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)
        
        # Aggregate metrics
        for key, values in episode_metrics.items():
            if key not in eval_metrics:
                eval_metrics[key] = []
            eval_metrics[key].extend(values)
    
    # Compute evaluation statistics
    eval_stats = {
        'eval_mean_reward': np.mean(eval_rewards),
        'eval_std_reward': np.std(eval_rewards),
        'eval_mean_length': np.mean(eval_lengths),
        'eval_std_length': np.std(eval_lengths)
    }
    
    # Add metric statistics
    for key, values in eval_metrics.items():
        eval_stats[f'eval_mean_{key}'] = np.mean(values)
        eval_stats[f'eval_std_{key}'] = np.std(values)
    
    # Log evaluation results
    logger.log_scalars(eval_stats)
    
    print(f"Evaluation Results:")
    print(f"  Mean Reward: {eval_stats['eval_mean_reward']:.2f} ± {eval_stats['eval_std_reward']:.2f}")
    print(f"  Mean Length: {eval_stats['eval_mean_length']:.2f} ± {eval_stats['eval_std_length']:.2f}")
    
    return eval_stats


def run_rollout_analysis(policy, env, logger: Logger, n_episodes: int = 50) -> Dict[str, Any]:
    """Run rollout analysis with visualization."""
    print(f"Running rollout analysis for {n_episodes} episodes...")
    
    rollout_data = {
        'episodes': [],
        'rewards': [],
        'lengths': [],
        'metrics': {}
    }
    
    # Collect rollout data
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_metrics = {}
        
        done = False
        while not done and episode_length < env.config.env.max_episode_steps:
            if hasattr(policy, 'predict_without_privileged'):
                action, _ = policy.predict_without_privileged(obs)
            else:
                action, _ = policy.predict(obs)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if 'reward_info' in info:
                for key, value in info['reward_info'].items():
                    if key not in episode_metrics:
                        episode_metrics[key] = []
                    episode_metrics[key].append(value)
            
            done = terminated or truncated
            obs = next_obs
        
        # Store episode data
        rollout_data['episodes'].append(episode)
        rollout_data['rewards'].append(episode_reward)
        rollout_data['lengths'].append(episode_length)
        
        # Store metrics
        for key, values in episode_metrics.items():
            if key not in rollout_data['metrics']:
                rollout_data['metrics'][key] = []
            rollout_data['metrics'][key].append(np.mean(values))
    
    # Analyze rollout data
    rollout_viz = RolloutVisualizer(env.config)
    analysis = rollout_viz.analyze_rollout(rollout_data)
    
    # Create plots
    plot_paths = rollout_viz.create_rollout_plots(rollout_data, analysis, logger.plot_dir)
    
    # Save analysis report
    report_path = rollout_viz.save_analysis_report(analysis, logger.plot_dir)
    
    # Print summary
    rollout_viz.print_summary(analysis)
    
    return analysis


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Go2 quadruped locomotion policies')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--policy', type=str, default='ppo_baseline', 
                       choices=['ppo_baseline', 'residual_rl', 'bc_pretrain', 'asymmetric_critic'],
                       help='Policy type to train')
    parser.add_argument('--timesteps', type=int, default=1000000, help='Total training timesteps')
    parser.add_argument('--experiment', type=str, default='go2_experiment', help='Experiment name')
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('--record', action='store_true', help='Record videos')
    parser.add_argument('--eval', action='store_true', help='Run evaluation after training')
    parser.add_argument('--rollout', action='store_true', help='Run rollout analysis after training')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = Config.load(args.config)
    else:
        config = get_default_config()
    
    # Update config with command line arguments
    config.policy_type = args.policy
    config.total_timesteps = args.timesteps
    config.experiment_name = args.experiment
    
    # Create environment
    print("Creating environment...")
    env = Go2PyBulletEnv(config, render_mode="human" if args.render else None)
    
    # Create reward function
    print("Creating reward function...")
    reward_function = RewardFunctionRules(config)
    
    # Create logger
    print("Setting up logging...")
    logger = Logger(config, args.experiment)
    
    # Create policy
    print(f"Creating {args.policy} policy...")
    policy = create_policy(config, env.observation_space, env.action_space, reward_function)
    
    # Create visualizers
    print("Setting up visualization...")
    training_viz = TrainingVisualizer(config, backend=config.viz.plot_backend)
    
    viz_3d = None
    if config.viz.enable_3d_viz and args.render:
        viz_3d = Viz3D(config, enable_gui=args.render)
        if args.record:
            logger.start_video_recording()
    
    # Start training visualization
    training_viz.start_live_plotting()
    
    try:
        # Train policy
        print("Starting training...")
        trained_policy = train_policy(config, policy, env, logger, training_viz, viz_3d)
        
        # Save final model
        final_model_path = logger.save_model(trained_policy)
        print(f"Final model saved to {final_model_path}")
        
        # Run evaluation
        if args.eval:
            print("Running evaluation...")
            eval_stats = evaluate_policy(trained_policy, env, logger, config.n_eval_episodes)
        
        # Run rollout analysis
        if args.rollout:
            print("Running rollout analysis...")
            analysis = run_rollout_analysis(trained_policy, env, logger, n_episodes=50)
        
        # Save plots
        training_viz.save_plots(logger.plot_dir)
        
        # Stop video recording
        if args.record and logger.recording:
            video_path = logger.stop_video_recording()
            if video_path:
                print(f"Training video saved to {video_path}")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        # Cleanup
        print("Cleaning up...")
        training_viz.close()
        if viz_3d:
            viz_3d.close()
        logger.close()
        env.close()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()

