#!/usr/bin/env python3
"""
Visualization demo for Go2 quadruped locomotion.
"""
import sys
import os
import numpy as np
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config import get_default_config
from environments.go2_pybullet import Go2PyBulletEnv
from visualization.training_viz import TrainingVisualizer
from visualization.rollout_viz import RolloutVisualizer
from visualization.viz_3d import Viz3D


def generate_demo_data(config, n_episodes=100):
    """Generate demo data for visualization."""
    print("Generating demo data...")
    
    # Create environment
    env = Go2PyBulletEnv(config, render_mode=None)
    
    rollout_data = {
        'episodes': [],
        'rewards': [],
        'lengths': [],
        'metrics': {
            'forward_speed': [],
            'lateral_drift': [],
            'tilt_angle': [],
            'height': [],
            'energy_consumption': [],
            'action_smoothness': []
        }
    }
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_metrics = {}
        
        done = False
        while not done and episode_length < config.env.max_episode_steps:
            # Random action for demo
            action = env.action_space.sample()
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
            if key in rollout_data['metrics']:
                rollout_data['metrics'][key].append(np.mean(values))
        
        if episode % 10 == 0:
            print(f"Generated episode {episode}/{n_episodes}")
    
    env.close()
    return rollout_data


def demo_training_visualization(config):
    """Demo training visualization."""
    print("\n" + "="*50)
    print("TRAINING VISUALIZATION DEMO")
    print("="*50)
    
    # Create training visualizer
    training_viz = TrainingVisualizer(config, backend="matplotlib")
    
    # Generate demo data
    print("Generating demo training data...")
    for episode in range(100):
        # Simulate training progress
        reward = np.random.normal(0.5, 0.3)
        length = np.random.randint(200, 800)
        
        metrics = {
            'forward_speed': np.random.uniform(0.2, 0.6),
            'lateral_drift': np.random.uniform(-0.1, 0.1),
            'tilt_angle': np.random.uniform(0, 0.3),
            'height': np.random.uniform(0.25, 0.35),
            'energy_consumption': np.random.uniform(0.1, 0.8),
            'action_smoothness': np.random.uniform(0.05, 0.2)
        }
        
        training_viz.add_episode_data(episode, reward, length, metrics)
        
        if episode % 20 == 0:
            print(f"Added episode {episode} data")
    
    # Save plots
    training_viz.save_plots("demo_plots")
    print("Training visualization plots saved to demo_plots/")
    
    training_viz.close()


def demo_rollout_visualization(config):
    """Demo rollout visualization."""
    print("\n" + "="*50)
    print("ROLLOUT VISUALIZATION DEMO")
    print("="*50)
    
    # Generate demo data
    rollout_data = generate_demo_data(config, n_episodes=50)
    
    # Create rollout visualizer
    rollout_viz = RolloutVisualizer(config)
    
    # Analyze rollout data
    print("Analyzing rollout data...")
    analysis = rollout_viz.analyze_rollout(rollout_data)
    
    # Create plots
    print("Creating rollout plots...")
    plot_paths = rollout_viz.create_rollout_plots(rollout_data, analysis, "demo_plots")
    
    # Save analysis report
    report_path = rollout_viz.save_analysis_report(analysis, "demo_plots")
    
    # Print summary
    rollout_viz.print_summary(analysis)
    
    print(f"Rollout analysis plots saved to demo_plots/")
    print(f"Analysis report saved to {report_path}")


def demo_3d_visualization(config):
    """Demo 3D visualization."""
    print("\n" + "="*50)
    print("3D VISUALIZATION DEMO")
    print("="*50)
    
    # Create 3D visualizer
    viz_3d = Viz3D(config, enable_gui=True)
    
    print("3D visualization started. You can:")
    print("- Use mouse to rotate camera")
    print("- Press 'w'/'s' to zoom in/out")
    print("- Press 'a'/'d' to rotate left/right")
    print("- Press 'q'/'e' to tilt up/down")
    print("- Press 'r' to reset camera")
    print("- Press Ctrl+C to exit")
    
    # Start video recording
    video_path = viz_3d.start_recording("demo_3d.mp4")
    print(f"Started recording to {video_path}")
    
    try:
        # Simulate robot movement
        for step in range(1000):
            # Generate random robot state
            position = np.array([
                np.sin(step * 0.01) * 0.5,
                np.cos(step * 0.01) * 0.5,
                0.3 + np.sin(step * 0.02) * 0.1
            ])
            
            orientation = np.array([0, 0, 0, 1])  # Identity quaternion
            
            joint_positions = np.array([
                np.sin(step * 0.1 + i) * 0.5 for i in range(12)
            ])
            
            # Update robot state
            viz_3d.update_robot_state(position, orientation, joint_positions)
            
            # Render frame
            frame = viz_3d.render_frame()
            
            # Add some debug elements
            if step % 100 == 0:
                viz_3d.add_debug_text(f"Step: {step}", [0, 0, 1])
            
            time.sleep(0.01)  # 100 FPS
            
    except KeyboardInterrupt:
        print("\nStopping 3D visualization...")
    
    # Stop recording
    final_video_path = viz_3d.stop_recording()
    if final_video_path:
        print(f"Video saved to {final_video_path}")
    
    viz_3d.close()


def main():
    """Main demo function."""
    print("Go2 Visualization Demo")
    print("=" * 50)
    
    # Create configuration
    config = get_default_config()
    config.experiment_name = "visualization_demo"
    
    # Create demo plots directory
    os.makedirs("demo_plots", exist_ok=True)
    
    # Run demos
    demo_training_visualization(config)
    demo_rollout_visualization(config)
    demo_3d_visualization(config)
    
    print("\n" + "="*50)
    print("DEMO COMPLETED!")
    print("="*50)
    print("Check the 'demo_plots' directory for generated visualizations.")
    print("Check the current directory for the 3D visualization video.")


if __name__ == "__main__":
    main()

