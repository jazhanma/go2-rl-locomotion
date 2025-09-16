#!/usr/bin/env python3
"""
Advanced RL algorithms for Go2 research.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from environments.go2_gymnasium import Go2GymnasiumEnv

def create_go2_env(render=False):
    """Create Go2 environment."""
    env = Go2GymnasiumEnv(render_mode="human" if render else None)
    env = Monitor(env)
    return env

def compare_algorithms():
    """Compare different RL algorithms."""
    print("Go2 Algorithm Comparison")
    print("=" * 40)
    
    algorithms = {
        'PPO': PPO,
        'SAC': SAC,
        'TD3': TD3,
        'DDPG': DDPG
    }
    
    results = {}
    
    for name, algorithm_class in algorithms.items():
        print(f"\nTraining {name}...")
        
        # Create environments
        train_env = create_go2_env(render=False)
        eval_env = create_go2_env(render=False)
        
        # Create model
        if name in ['SAC', 'TD3', 'DDPG']:
            model = algorithm_class(
                "MlpPolicy",
                train_env,
                learning_rate=3e-4,
                verbose=1,
                tensorboard_log=f"./logs/tensorboard_{name.lower()}/"
            )
        else:  # PPO
            model = algorithm_class(
                "MlpPolicy",
                train_env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1,
                tensorboard_log=f"./logs/tensorboard_{name.lower()}/"
            )
        
        # Train for shorter time for comparison
        try:
            model.learn(total_timesteps=20000, progress_bar=True)
            
            # Test the model
            test_rewards = []
            for _ in range(5):
                obs, _ = eval_env.reset()
                episode_reward = 0.0
                for _ in range(1000):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    episode_reward += reward
                    if terminated or truncated:
                        break
                test_rewards.append(episode_reward)
            
            results[name] = {
                'mean_reward': np.mean(test_rewards),
                'std_reward': np.std(test_rewards),
                'rewards': test_rewards
            }
            
            print(f"{name} - Mean Reward: {results[name]['mean_reward']:.2f} ± {results[name]['std_reward']:.2f}")
            
        except Exception as e:
            print(f"Error training {name}: {e}")
            results[name] = {'mean_reward': 0, 'std_reward': 0, 'rewards': [0]}
        
        finally:
            train_env.close()
            eval_env.close()
    
    # Plot comparison
    plot_algorithm_comparison(results)
    
    return results

def plot_algorithm_comparison(results):
    """Plot algorithm comparison results."""
    names = list(results.keys())
    means = [results[name]['mean_reward'] for name in names]
    stds = [results[name]['std_reward'] for name in names]
    
    plt.figure(figsize=(12, 8))
    
    # Bar plot
    plt.subplot(2, 1, 1)
    bars = plt.bar(names, means, yerr=stds, capsize=5, alpha=0.7)
    plt.title('Algorithm Comparison - Mean Rewards')
    plt.ylabel('Mean Episode Reward')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 10,
                f'{mean:.1f}±{std:.1f}', ha='center', va='bottom')
    
    # Individual episode rewards
    plt.subplot(2, 1, 2)
    for name in names:
        rewards = results[name]['rewards']
        plt.plot([name] * len(rewards), rewards, 'o', alpha=0.7, label=name)
    plt.title('Individual Episode Rewards')
    plt.ylabel('Episode Reward')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nAlgorithm comparison saved as 'algorithm_comparison.png'")

def train_curriculum_learning():
    """Implement curriculum learning."""
    print("Go2 Curriculum Learning")
    print("=" * 30)
    
    # Define curriculum stages
    stages = [
        {'max_steps': 200, 'reward_scale': 1.0, 'name': 'Stage 1: Basic Movement'},
        {'max_steps': 500, 'reward_scale': 2.0, 'name': 'Stage 2: Extended Movement'},
        {'max_steps': 1000, 'reward_scale': 3.0, 'name': 'Stage 3: Full Episodes'}
    ]
    
    model = None
    
    for stage_idx, stage in enumerate(stages):
        print(f"\n{stage['name']}")
        print(f"Max steps: {stage['max_steps']}, Reward scale: {stage['reward_scale']}")
        
        # Create environment with stage-specific parameters
        env = create_go2_env(render=False)
        env.max_episode_steps = stage['max_steps']
        
        if model is None:
            # Create new model for first stage
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                verbose=1,
                tensorboard_log=f"./logs/tensorboard_curriculum/"
            )
        else:
            # Set environment for existing model
            model.set_env(env)
        
        # Train on this stage
        try:
            model.learn(total_timesteps=10000, progress_bar=True)
            
            # Test performance
            test_rewards = []
            for _ in range(3):
                obs, _ = env.reset()
                episode_reward = 0.0
                for _ in range(stage['max_steps']):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    if terminated or truncated:
                        break
                test_rewards.append(episode_reward)
            
            print(f"Stage {stage_idx + 1} - Mean Reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
            
        except Exception as e:
            print(f"Error in stage {stage_idx + 1}: {e}")
        
        finally:
            env.close()
    
    # Save final curriculum model
    if model:
        model.save("go2_curriculum_final")
        print("Curriculum learning model saved as 'go2_curriculum_final'")

def sac_deep_dive():
    """Deep dive into SAC algorithm with extensive analysis."""
    print("SAC Deep Dive Analysis")
    print("=" * 40)
    
    # SAC hyperparameter configurations to test
    sac_configs = {
        'SAC_Default': {
            'learning_rate': 3e-4,
            'buffer_size': 100000,
            'learning_starts': 1000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': 1,
            'gradient_steps': 1,
            'ent_coef': 'auto',
            'target_update_interval': 1,
            'target_entropy': 'auto'
        },
        'SAC_HighLR': {
            'learning_rate': 1e-3,
            'buffer_size': 100000,
            'learning_starts': 1000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': 1,
            'gradient_steps': 1,
            'ent_coef': 'auto',
            'target_update_interval': 1,
            'target_entropy': 'auto'
        },
        'SAC_LowLR': {
            'learning_rate': 1e-4,
            'buffer_size': 100000,
            'learning_starts': 1000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': 1,
            'gradient_steps': 1,
            'ent_coef': 'auto',
            'target_update_interval': 1,
            'target_entropy': 'auto'
        },
        'SAC_HighTau': {
            'learning_rate': 3e-4,
            'buffer_size': 100000,
            'learning_starts': 1000,
            'batch_size': 256,
            'tau': 0.01,
            'gamma': 0.99,
            'train_freq': 1,
            'gradient_steps': 1,
            'ent_coef': 'auto',
            'target_update_interval': 1,
            'target_entropy': 'auto'
        },
        'SAC_LowTau': {
            'learning_rate': 3e-4,
            'buffer_size': 100000,
            'learning_starts': 1000,
            'batch_size': 256,
            'tau': 0.001,
            'gamma': 0.99,
            'train_freq': 1,
            'gradient_steps': 1,
            'ent_coef': 'auto',
            'target_update_interval': 1,
            'target_entropy': 'auto'
        }
    }
    
    results = {}
    training_curves = {}
    
    for config_name, config in sac_configs.items():
        print(f"\nTraining {config_name}...")
        print(f"Learning Rate: {config['learning_rate']}, Tau: {config['tau']}")
        
        # Create environments
        train_env = create_go2_env(render=False)
        eval_env = create_go2_env(render=False)
        
        # Create SAC model with specific config
        model = SAC(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log=f"./logs/tensorboard_sac_deep_dive/",
            **config
        )
        
        # Training with evaluation tracking
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"./models/sac_deep_dive_{config_name}/",
            log_path=f"./logs/sac_deep_dive_{config_name}/",
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        try:
            # Train for longer to see full learning curve
            model.learn(
                total_timesteps=50000,
                callback=eval_callback,
                progress_bar=True
            )
            
            # Test the model
            test_rewards = []
            for _ in range(10):  # More test episodes for better statistics
                obs, _ = eval_env.reset()
                episode_reward = 0.0
                for _ in range(1000):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    episode_reward += reward
                    if terminated or truncated:
                        break
                test_rewards.append(episode_reward)
            
            results[config_name] = {
                'mean_reward': np.mean(test_rewards),
                'std_reward': np.std(test_rewards),
                'rewards': test_rewards,
                'config': config
            }
            
            print(f"{config_name} - Mean Reward: {results[config_name]['mean_reward']:.2f} ± {results[config_name]['std_reward']:.2f}")
            
            # Save model
            model.save(f"go2_sac_deep_dive_{config_name}")
            
        except Exception as e:
            print(f"Error training {config_name}: {e}")
            results[config_name] = {'mean_reward': 0, 'std_reward': 0, 'rewards': [0], 'config': config}
        
        finally:
            train_env.close()
            eval_env.close()
    
    # Plot detailed comparison
    plot_sac_deep_dive(results)
    
    # Print detailed analysis
    print_sac_analysis(results)
    
    return results

def plot_sac_deep_dive(results):
    """Plot detailed SAC analysis."""
    configs = list(results.keys())
    means = [results[config]['mean_reward'] for config in configs]
    stds = [results[config]['std_reward'] for config in configs]
    
    plt.figure(figsize=(15, 10))
    
    # Bar plot comparison
    plt.subplot(2, 2, 1)
    bars = plt.bar(configs, means, yerr=stds, capsize=5, alpha=0.7)
    plt.title('SAC Hyperparameter Comparison')
    plt.ylabel('Mean Episode Reward')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 5,
                f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Individual episode rewards
    plt.subplot(2, 2, 2)
    for config in configs:
        rewards = results[config]['rewards']
        plt.plot([config] * len(rewards), rewards, 'o', alpha=0.7, label=config)
    plt.title('Individual Episode Rewards')
    plt.ylabel('Episode Reward')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Learning rate vs performance
    plt.subplot(2, 2, 3)
    lr_values = [results[config]['config']['learning_rate'] for config in configs]
    plt.scatter(lr_values, means, s=100, alpha=0.7)
    plt.xlabel('Learning Rate')
    plt.ylabel('Mean Episode Reward')
    plt.title('Learning Rate vs Performance')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    # Tau vs performance
    plt.subplot(2, 2, 4)
    tau_values = [results[config]['config']['tau'] for config in configs]
    plt.scatter(tau_values, means, s=100, alpha=0.7)
    plt.xlabel('Tau (Target Update Rate)')
    plt.ylabel('Mean Episode Reward')
    plt.title('Tau vs Performance')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sac_deep_dive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nSAC deep dive analysis saved as 'sac_deep_dive_analysis.png'")

def print_sac_analysis(results):
    """Print detailed SAC analysis."""
    print("\n" + "="*60)
    print("SAC DEEP DIVE ANALYSIS")
    print("="*60)
    
    # Sort by performance
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_reward'], reverse=True)
    
    print("\n🏆 PERFORMANCE RANKING:")
    for i, (config_name, result) in enumerate(sorted_results, 1):
        print(f"{i}. {config_name}: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
    
    print("\n🔬 HYPERPARAMETER ANALYSIS:")
    best_config = sorted_results[0][0]
    best_result = sorted_results[0][1]
    
    print(f"\nBest Configuration: {best_config}")
    print(f"Performance: {best_result['mean_reward']:.2f} ± {best_result['std_reward']:.2f}")
    
    # Analyze learning rate impact
    lr_analysis = {}
    for config_name, result in results.items():
        lr = result['config']['learning_rate']
        if lr not in lr_analysis:
            lr_analysis[lr] = []
        lr_analysis[lr].append(result['mean_reward'])
    
    print(f"\n📊 LEARNING RATE IMPACT:")
    for lr, rewards in lr_analysis.items():
        mean_perf = np.mean(rewards)
        print(f"LR {lr}: {mean_perf:.2f} (from {len(rewards)} configs)")
    
    # Analyze tau impact
    tau_analysis = {}
    for config_name, result in results.items():
        tau = result['config']['tau']
        if tau not in tau_analysis:
            tau_analysis[tau] = []
        tau_analysis[tau].append(result['mean_reward'])
    
    print(f"\n📊 TAU IMPACT:")
    for tau, rewards in tau_analysis.items():
        mean_perf = np.mean(rewards)
        print(f"Tau {tau}: {mean_perf:.2f} (from {len(rewards)} configs)")
    
    print(f"\n✅ RECOMMENDATIONS:")
    print(f"1. Best overall: {best_config}")
    print(f"2. Learning rate: {best_result['config']['learning_rate']}")
    print(f"3. Tau: {best_result['config']['tau']}")
    print(f"4. Use this config for production training!")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Go2 Research')
    parser.add_argument('--mode', type=str, default='compare',
                       choices=['compare', 'curriculum', 'sac_deep_dive'],
                       help='Mode: compare algorithms, curriculum learning, or SAC deep dive')
    
    args = parser.parse_args()
    
    if args.mode == 'compare':
        compare_algorithms()
    elif args.mode == 'curriculum':
        train_curriculum_learning()
    elif args.mode == 'sac_deep_dive':
        sac_deep_dive()

if __name__ == "__main__":
    main()

