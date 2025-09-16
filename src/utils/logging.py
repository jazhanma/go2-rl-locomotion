"""
Comprehensive logging system for Go2 locomotion framework.
"""
import os
import time
import json
import numpy as np
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
import imageio
from torch.utils.tensorboard import SummaryWriter
import wandb


class Logger:
    """Centralized logging system for training and evaluation."""
    
    def __init__(self, config, experiment_name: str = None):
        self.config = config
        self.experiment_name = experiment_name or config.experiment_name
        
        # Create directories
        self.log_dir = os.path.join(config.log.log_dir, self.experiment_name)
        self.model_dir = os.path.join(config.log.model_dir, self.experiment_name)
        self.plot_dir = os.path.join(config.log.plot_dir, self.experiment_name)
        self.video_dir = os.path.join(config.log.video_dir, self.experiment_name)
        
        for dir_path in [self.log_dir, self.model_dir, self.plot_dir, self.video_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize loggers
        self.tensorboard_writer = None
        if config.log.use_tensorboard:
            tb_dir = os.path.join(config.log.tensorboard_log_dir, self.experiment_name)
            self.tensorboard_writer = SummaryWriter(tb_dir)
        
        if config.log.use_wandb:
            wandb.init(
                project=config.log.wandb_project,
                entity=config.log.wandb_entity,
                name=self.experiment_name,
                config=config.__dict__
            )
        
        # Internal state
        self.step = 0
        self.episode = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.metrics_history = defaultdict(list)
        self.episode_metrics = defaultdict(list)
        
        # Rolling averages
        self.rolling_window = 100
        self.rolling_rewards = deque(maxlen=self.rolling_window)
        self.rolling_lengths = deque(maxlen=self.rolling_window)
        
        # Video recording
        self.video_writer = None
        self.video_frames = []
        self.recording = False
    
    def log_scalar(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a scalar value."""
        step = step or self.step
        self.metrics_history[key].append((step, value))
        
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar(key, value, step)
        
        if self.config.log.use_wandb:
            wandb.log({key: value}, step=step)
    
    def log_scalars(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple scalar values."""
        step = step or self.step
        for key, value in metrics.items():
            self.log_scalar(key, value, step)
    
    def log_episode(self, episode_reward: float, episode_length: int, 
                   episode_metrics: Dict[str, float] = None) -> None:
        """Log episode results."""
        self.episode += 1
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.rolling_rewards.append(episode_reward)
        self.rolling_lengths.append(episode_length)
        
        # Log episode metrics
        episode_data = {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'mean_reward_100': np.mean(self.rolling_rewards),
            'mean_length_100': np.mean(self.rolling_lengths)
        }
        
        if episode_metrics:
            episode_data.update(episode_metrics)
            for key, value in episode_metrics.items():
                self.episode_metrics[key].append(value)
        
        self.log_scalars(episode_data, self.episode)
        
        # Print progress
        if self.episode % 10 == 0:
            print(f"Episode {self.episode}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Length={episode_length}, "
                  f"Mean Reward (100)={np.mean(self.rolling_rewards):.2f}")
    
    def log_histogram(self, key: str, values: np.ndarray, step: Optional[int] = None) -> None:
        """Log a histogram."""
        step = step or self.step
        if self.tensorboard_writer:
            self.tensorboard_writer.add_histogram(key, values, step)
        
        if self.config.log.use_wandb:
            wandb.log({key: wandb.Histogram(values)}, step=step)
    
    def log_image(self, key: str, image: np.ndarray, step: Optional[int] = None) -> None:
        """Log an image."""
        step = step or self.step
        if self.tensorboard_writer:
            self.tensorboard_writer.add_image(key, image, step)
        
        if self.config.log.use_wandb:
            wandb.log({key: wandb.Image(image)}, step=step)
    
    def log_video(self, key: str, frames: List[np.ndarray], step: Optional[int] = None) -> None:
        """Log a video."""
        step = step or self.step
        if self.config.log.use_wandb:
            wandb.log({key: wandb.Video(frames, fps=self.config.viz.video_fps)}, step=step)
    
    def start_video_recording(self) -> None:
        """Start recording video frames."""
        self.recording = True
        self.video_frames = []
    
    def add_video_frame(self, frame: np.ndarray) -> None:
        """Add a frame to the current video recording."""
        if self.recording:
            self.video_frames.append(frame.copy())
    
    def stop_video_recording(self, filename: str = None) -> str:
        """Stop recording and save video."""
        if not self.recording:
            return None
        
        self.recording = False
        
        if filename is None:
            filename = f"episode_{self.episode}_{int(time.time())}.mp4"
        
        video_path = os.path.join(self.video_dir, filename)
        
        if self.video_frames:
            # Save as MP4
            imageio.mimsave(video_path, self.video_frames, fps=self.config.viz.video_fps)
            
            # Also save as GIF for GitHub
            gif_path = video_path.replace('.mp4', '.gif')
            imageio.mimsave(gif_path, self.video_frames, fps=self.config.viz.video_fps)
            
            print(f"Video saved: {video_path}")
            return video_path
        
        return None
    
    def save_plots(self) -> None:
        """Save training plots."""
        if not self.config.viz.save_plots:
            return
        
        # Training progress plot
        self._plot_training_progress()
        
        # Episode metrics plot
        self._plot_episode_metrics()
        
        # Reward components plot
        self._plot_reward_components()
    
    def _plot_training_progress(self) -> None:
        """Plot training progress."""
        if not self.episode_rewards:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Episode rewards
        episodes = range(1, len(self.episode_rewards) + 1)
        ax1.plot(episodes, self.episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
        
        if len(self.episode_rewards) >= self.rolling_window:
            rolling_episodes = episodes[self.rolling_window-1:]
            rolling_rewards = [np.mean(self.episode_rewards[i-self.rolling_window:i]) 
                             for i in range(self.rolling_window, len(self.episode_rewards) + 1)]
            ax1.plot(rolling_episodes, rolling_rewards, color='red', linewidth=2, 
                    label=f'Mean Reward ({self.rolling_window})')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Progress - Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Episode lengths
        ax2.plot(episodes, self.episode_lengths, alpha=0.3, color='green', label='Episode Length')
        
        if len(self.episode_lengths) >= self.rolling_window:
            rolling_lengths = [np.mean(self.episode_lengths[i-self.rolling_window:i]) 
                             for i in range(self.rolling_window, len(self.episode_lengths) + 1)]
            ax2.plot(rolling_episodes, rolling_lengths, color='orange', linewidth=2, 
                    label=f'Mean Length ({self.rolling_window})')
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Length')
        ax2.set_title('Training Progress - Episode Lengths')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.plot_dir, 'training_progress.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_episode_metrics(self) -> None:
        """Plot episode-specific metrics."""
        if not self.episode_metrics:
            return
        
        n_metrics = len(self.episode_metrics)
        if n_metrics == 0:
            return
        
        fig, axes = plt.subplots((n_metrics + 1) // 2, 2, figsize=(15, 5 * ((n_metrics + 1) // 2)))
        if n_metrics == 1:
            axes = [axes]
        elif n_metrics == 2:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (metric_name, values) in enumerate(self.episode_metrics.items()):
            if i >= len(axes):
                break
            
            episodes = range(1, len(values) + 1)
            axes[i].plot(episodes, values, alpha=0.7, color=f'C{i}')
            axes[i].set_xlabel('Episode')
            axes[i].set_ylabel(metric_name.replace('_', ' ').title())
            axes[i].set_title(f'Episode {metric_name.replace("_", " ").title()}')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(self.episode_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_path = os.path.join(self.plot_dir, 'episode_metrics.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_reward_components(self) -> None:
        """Plot reward components over time."""
        # This would be implemented based on specific reward logging
        pass
    
    def save_model(self, model, filename: str = None) -> str:
        """Save model checkpoint."""
        if filename is None:
            filename = f"model_step_{self.step}.pth"
        
        model_path = os.path.join(self.model_dir, filename)
        model.save(model_path)
        print(f"Model saved: {model_path}")
        return model_path
    
    def load_model(self, model, filename: str) -> None:
        """Load model checkpoint."""
        model_path = os.path.join(self.model_dir, filename)
        model.load(model_path)
        print(f"Model loaded: {model_path}")
    
    def close(self) -> None:
        """Close all loggers."""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.config.log.use_wandb:
            wandb.finish()
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.episode_rewards:
            return {}
        
        return {
            'total_episodes': len(self.episode_rewards),
            'total_steps': self.step,
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths),
            'std_length': np.std(self.episode_lengths),
            'best_reward': np.max(self.episode_rewards),
            'worst_reward': np.min(self.episode_rewards),
            'recent_mean_reward': np.mean(self.rolling_rewards) if self.rolling_rewards else 0,
            'recent_mean_length': np.mean(self.rolling_lengths) if self.rolling_lengths else 0,
        }

