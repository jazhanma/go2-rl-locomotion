"""
Training Visualizer: Live graphs of reward, speed, stability metrics during training.
"""
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Any
import time
import threading
from collections import deque
import os


class TrainingVisualizer:
    """
    Real-time training visualization with live graphs.
    """
    
    def __init__(self, config, backend: str = "matplotlib"):
        self.config = config
        self.backend = backend
        self.plot_freq = config.viz.plot_freq
        
        # Data storage
        self.episode_data = {
            'episodes': [],
            'rewards': [],
            'lengths': [],
            'forward_speeds': [],
            'lateral_drifts': [],
            'tilt_angles': [],
            'heights': [],
            'energy_consumptions': [],
            'action_smoothness': []
        }
        
        # Rolling averages
        self.rolling_window = 100
        self.rolling_rewards = deque(maxlen=self.rolling_window)
        self.rolling_lengths = deque(maxlen=self.rolling_window)
        self.rolling_speeds = deque(maxlen=self.rolling_window)
        
        # Plotting state
        self.fig = None
        self.axes = None
        self.lines = {}
        self.plot_thread = None
        self.running = False
        
        # Initialize plots
        if self.backend == "matplotlib":
            self._init_matplotlib()
        elif self.backend == "plotly":
            self._init_plotly()
    
    def _init_matplotlib(self):
        """Initialize matplotlib plots."""
        plt.ion()  # Interactive mode
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.suptitle('Go2 Training Progress', fontsize=16)
        
        # Configure subplots
        self.axes[0, 0].set_title('Episode Rewards')
        self.axes[0, 0].set_xlabel('Episode')
        self.axes[0, 0].set_ylabel('Reward')
        self.axes[0, 0].grid(True, alpha=0.3)
        
        self.axes[0, 1].set_title('Episode Lengths')
        self.axes[0, 1].set_xlabel('Episode')
        self.axes[0, 1].set_ylabel('Length')
        self.axes[0, 1].grid(True, alpha=0.3)
        
        self.axes[0, 2].set_title('Forward Speed')
        self.axes[0, 2].set_xlabel('Episode')
        self.axes[0, 2].set_ylabel('Speed (m/s)')
        self.axes[0, 2].grid(True, alpha=0.3)
        
        self.axes[1, 0].set_title('Lateral Drift')
        self.axes[1, 0].set_xlabel('Episode')
        self.axes[1, 0].set_ylabel('Drift (m/s)')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        self.axes[1, 1].set_title('Tilt Angle')
        self.axes[1, 1].set_xlabel('Episode')
        self.axes[1, 1].set_ylabel('Angle (rad)')
        self.axes[1, 1].grid(True, alpha=0.3)
        
        self.axes[1, 2].set_title('Height')
        self.axes[1, 2].set_xlabel('Episode')
        self.axes[1, 2].set_ylabel('Height (m)')
        self.axes[1, 2].grid(True, alpha=0.3)
        
        # Initialize line objects
        self.lines['rewards'] = self.axes[0, 0].plot([], [], 'b-', alpha=0.7, label='Episode Reward')[0]
        self.lines['rewards_rolling'] = self.axes[0, 0].plot([], [], 'r-', linewidth=2, label=f'Rolling Avg ({self.rolling_window})')[0]
        
        self.lines['lengths'] = self.axes[0, 1].plot([], [], 'g-', alpha=0.7, label='Episode Length')[0]
        self.lines['lengths_rolling'] = self.axes[0, 1].plot([], [], 'orange', linewidth=2, label=f'Rolling Avg ({self.rolling_window})')[0]
        
        self.lines['speeds'] = self.axes[0, 2].plot([], [], 'purple', alpha=0.7, label='Forward Speed')[0]
        self.lines['speeds_rolling'] = self.axes[0, 2].plot([], [], 'red', linewidth=2, label=f'Rolling Avg ({self.rolling_window})')[0]
        
        self.lines['lateral_drifts'] = self.axes[1, 0].plot([], [], 'brown', alpha=0.7, label='Lateral Drift')[0]
        self.lines['tilt_angles'] = self.axes[1, 1].plot([], [], 'pink', alpha=0.7, label='Tilt Angle')[0]
        self.lines['heights'] = self.axes[1, 2].plot([], [], 'cyan', alpha=0.7, label='Height')[0]
        
        # Add legends
        for ax in self.axes.flat:
            ax.legend()
        
        plt.tight_layout()
        plt.show(block=False)
    
    def _init_plotly(self):
        """Initialize Plotly plots."""
        self.fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Episode Rewards', 'Episode Lengths', 'Forward Speed',
                          'Lateral Drift', 'Tilt Angle', 'Height'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add traces
        self.fig.add_trace(go.Scatter(x=[], y=[], name='Episode Reward', line=dict(color='blue')), row=1, col=1)
        self.fig.add_trace(go.Scatter(x=[], y=[], name=f'Rolling Avg ({self.rolling_window})', line=dict(color='red', width=2)), row=1, col=1)
        
        self.fig.add_trace(go.Scatter(x=[], y=[], name='Episode Length', line=dict(color='green')), row=1, col=2)
        self.fig.add_trace(go.Scatter(x=[], y=[], name=f'Rolling Avg ({self.rolling_window})', line=dict(color='orange', width=2)), row=1, col=2)
        
        self.fig.add_trace(go.Scatter(x=[], y=[], name='Forward Speed', line=dict(color='purple')), row=1, col=3)
        self.fig.add_trace(go.Scatter(x=[], y=[], name=f'Rolling Avg ({self.rolling_window})', line=dict(color='red', width=2)), row=1, col=3)
        
        self.fig.add_trace(go.Scatter(x=[], y=[], name='Lateral Drift', line=dict(color='brown')), row=2, col=1)
        self.fig.add_trace(go.Scatter(x=[], y=[], name='Tilt Angle', line=dict(color='pink')), row=2, col=2)
        self.fig.add_trace(go.Scatter(x=[], y=[], name='Height', line=dict(color='cyan')), row=2, col=3)
        
        # Update layout
        self.fig.update_layout(
            title='Go2 Training Progress',
            showlegend=True,
            height=800,
            width=1200
        )
        
        self.fig.show()
    
    def add_episode_data(self, episode: int, reward: float, length: int, 
                        metrics: Dict[str, float]) -> None:
        """Add episode data for visualization."""
        self.episode_data['episodes'].append(episode)
        self.episode_data['rewards'].append(reward)
        self.episode_data['lengths'].append(length)
        
        # Add metric data
        self.episode_data['forward_speeds'].append(metrics.get('forward_speed', 0.0))
        self.episode_data['lateral_drifts'].append(metrics.get('lateral_drift', 0.0))
        self.episode_data['tilt_angles'].append(metrics.get('tilt_angle', 0.0))
        self.episode_data['heights'].append(metrics.get('height', 0.0))
        self.episode_data['energy_consumptions'].append(metrics.get('energy_consumption', 0.0))
        self.episode_data['action_smoothness'].append(metrics.get('action_smoothness', 0.0))
        
        # Update rolling averages
        self.rolling_rewards.append(reward)
        self.rolling_lengths.append(length)
        self.rolling_speeds.append(metrics.get('forward_speed', 0.0))
        
        # Update plots if needed
        if episode % self.plot_freq == 0:
            self._update_plots()
    
    def _update_plots(self) -> None:
        """Update the plots with new data."""
        if self.backend == "matplotlib":
            self._update_matplotlib()
        elif self.backend == "plotly":
            self._update_plotly()
    
    def _update_matplotlib(self) -> None:
        """Update matplotlib plots."""
        if not self.episode_data['episodes']:
            return
        
        episodes = self.episode_data['episodes']
        
        # Update reward plot
        self.lines['rewards'].set_data(episodes, self.episode_data['rewards'])
        if len(episodes) >= self.rolling_window:
            rolling_episodes = episodes[self.rolling_window-1:]
            rolling_rewards = [np.mean(self.episode_data['rewards'][i-self.rolling_window:i]) 
                             for i in range(self.rolling_window, len(episodes) + 1)]
            self.lines['rewards_rolling'].set_data(rolling_episodes, rolling_rewards)
        
        # Update length plot
        self.lines['lengths'].set_data(episodes, self.episode_data['lengths'])
        if len(episodes) >= self.rolling_window:
            rolling_lengths = [np.mean(self.episode_data['lengths'][i-self.rolling_window:i]) 
                             for i in range(self.rolling_window, len(episodes) + 1)]
            self.lines['lengths_rolling'].set_data(rolling_episodes, rolling_lengths)
        
        # Update speed plot
        self.lines['speeds'].set_data(episodes, self.episode_data['forward_speeds'])
        if len(episodes) >= self.rolling_window:
            rolling_speeds = [np.mean(self.episode_data['forward_speeds'][i-self.rolling_window:i]) 
                            for i in range(self.rolling_window, len(episodes) + 1)]
            self.lines['speeds_rolling'].set_data(rolling_episodes, rolling_speeds)
        
        # Update other plots
        self.lines['lateral_drifts'].set_data(episodes, self.episode_data['lateral_drifts'])
        self.lines['tilt_angles'].set_data(episodes, self.episode_data['tilt_angles'])
        self.lines['heights'].set_data(episodes, self.episode_data['heights'])
        
        # Auto-scale axes
        for ax in self.axes.flat:
            ax.relim()
            ax.autoscale_view()
        
        # Refresh plot
        plt.draw()
        plt.pause(0.01)
    
    def _update_plotly(self) -> None:
        """Update Plotly plots."""
        if not self.episode_data['episodes']:
            return
        
        episodes = self.episode_data['episodes']
        
        # Update traces
        self.fig.data[0].x = episodes
        self.fig.data[0].y = self.episode_data['rewards']
        
        if len(episodes) >= self.rolling_window:
            rolling_episodes = episodes[self.rolling_window-1:]
            rolling_rewards = [np.mean(self.episode_data['rewards'][i-self.rolling_window:i]) 
                             for i in range(self.rolling_window, len(episodes) + 1)]
            self.fig.data[1].x = rolling_episodes
            self.fig.data[1].y = rolling_rewards
        
        self.fig.data[2].x = episodes
        self.fig.data[2].y = self.episode_data['lengths']
        
        if len(episodes) >= self.rolling_window:
            rolling_lengths = [np.mean(self.episode_data['lengths'][i-self.rolling_window:i]) 
                             for i in range(self.rolling_window, len(episodes) + 1)]
            self.fig.data[3].x = rolling_episodes
            self.fig.data[3].y = rolling_lengths
        
        self.fig.data[4].x = episodes
        self.fig.data[4].y = self.episode_data['forward_speeds']
        
        if len(episodes) >= self.rolling_window:
            rolling_speeds = [np.mean(self.episode_data['forward_speeds'][i-self.rolling_window:i]) 
                            for i in range(self.rolling_window, len(episodes) + 1)]
            self.fig.data[5].x = rolling_episodes
            self.fig.data[5].y = rolling_speeds
        
        self.fig.data[6].x = episodes
        self.fig.data[6].y = self.episode_data['lateral_drifts']
        
        self.fig.data[7].x = episodes
        self.fig.data[7].y = self.episode_data['tilt_angles']
        
        self.fig.data[8].x = episodes
        self.fig.data[8].y = self.episode_data['heights']
        
        # Update layout
        self.fig.update_layout(
            title=f'Go2 Training Progress - Episode {episodes[-1]}'
        )
        
        # Refresh plot
        self.fig.show()
    
    def start_live_plotting(self) -> None:
        """Start live plotting in a separate thread."""
        if self.running:
            return
        
        self.running = True
        self.plot_thread = threading.Thread(target=self._live_plot_loop)
        self.plot_thread.daemon = True
        self.plot_thread.start()
    
    def stop_live_plotting(self) -> None:
        """Stop live plotting."""
        self.running = False
        if self.plot_thread:
            self.plot_thread.join()
    
    def _live_plot_loop(self) -> None:
        """Live plotting loop."""
        while self.running:
            if self.episode_data['episodes']:
                self._update_plots()
            time.sleep(1.0)  # Update every second
    
    def save_plots(self, save_dir: str) -> None:
        """Save current plots to files."""
        os.makedirs(save_dir, exist_ok=True)
        
        if self.backend == "matplotlib":
            # Save matplotlib figure
            plot_path = os.path.join(save_dir, 'training_progress.png')
            self.fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Training plots saved to {plot_path}")
        
        elif self.backend == "plotly":
            # Save Plotly figure
            plot_path = os.path.join(save_dir, 'training_progress.html')
            self.fig.write_html(plot_path)
            print(f"Training plots saved to {plot_path}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of training data."""
        if not self.episode_data['episodes']:
            return {}
        
        return {
            'total_episodes': len(self.episode_data['episodes']),
            'mean_reward': np.mean(self.episode_data['rewards']),
            'std_reward': np.std(self.episode_data['rewards']),
            'best_reward': np.max(self.episode_data['rewards']),
            'worst_reward': np.min(self.episode_data['rewards']),
            'mean_length': np.mean(self.episode_data['lengths']),
            'std_length': np.std(self.episode_data['lengths']),
            'mean_speed': np.mean(self.episode_data['forward_speeds']),
            'std_speed': np.std(self.episode_data['forward_speeds']),
            'recent_mean_reward': np.mean(list(self.rolling_rewards)) if self.rolling_rewards else 0,
            'recent_mean_length': np.mean(list(self.rolling_lengths)) if self.rolling_lengths else 0,
            'recent_mean_speed': np.mean(list(self.rolling_speeds)) if self.rolling_speeds else 0,
        }
    
    def close(self) -> None:
        """Close the visualizer."""
        self.stop_live_plotting()
        if self.backend == "matplotlib" and self.fig:
            plt.close(self.fig)

