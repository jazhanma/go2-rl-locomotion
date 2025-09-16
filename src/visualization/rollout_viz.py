"""
Rollout Visualizer: Plots of episode rewards/lengths with per-metric pass/fail indicators.
"""
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import os
import json
from datetime import datetime


class RolloutVisualizer:
    """
    Visualization for rollout analysis with pass/fail indicators for each metric.
    """
    
    def __init__(self, config, backend: str = "matplotlib"):
        self.config = config
        self.backend = backend
        
        # Metric thresholds for pass/fail analysis
        self.thresholds = {
            'forward_speed': 0.4,  # m/s
            'lateral_drift': 0.1,  # m/s (max allowed)
            'tilt_angle': 0.2,     # rad (max allowed)
            'height_stability': 0.05,  # m (max deviation)
            'energy_efficiency': 0.5,  # normalized energy consumption
            'action_smoothness': 0.1,  # max action change
            'episode_length': 500,  # minimum steps
            'episode_reward': 0.0   # minimum reward
        }
        
        # Color scheme
        self.colors = {
            'pass': '#2E8B57',      # Sea Green
            'fail': '#DC143C',      # Crimson
            'warning': '#FF8C00',   # Dark Orange
            'neutral': '#4682B4'    # Steel Blue
        }
    
    def analyze_rollout(self, rollout_data: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Analyze rollout data and determine pass/fail for each metric.
        
        Args:
            rollout_data: Dictionary containing rollout data with keys:
                - 'episodes': List of episode numbers
                - 'rewards': List of episode rewards
                - 'lengths': List of episode lengths
                - 'metrics': Dictionary of metric lists
                
        Returns:
            Analysis results with pass/fail indicators
        """
        analysis = {
            'episode_count': len(rollout_data['episodes']),
            'metrics_analysis': {},
            'overall_pass_rate': 0.0,
            'summary': {}
        }
        
        # Analyze each metric
        for metric_name, threshold in self.thresholds.items():
            if metric_name in rollout_data['metrics']:
                values = rollout_data['metrics'][metric_name]
                analysis['metrics_analysis'][metric_name] = self._analyze_metric(
                    metric_name, values, threshold
                )
        
        # Analyze episode rewards and lengths
        if 'rewards' in rollout_data:
            analysis['metrics_analysis']['episode_reward'] = self._analyze_metric(
                'episode_reward', rollout_data['rewards'], self.thresholds['episode_reward']
            )
        
        if 'lengths' in rollout_data:
            analysis['metrics_analysis']['episode_length'] = self._analyze_metric(
                'episode_length', rollout_data['lengths'], self.thresholds['episode_length']
            )
        
        # Calculate overall pass rate
        pass_counts = [result['pass_count'] for result in analysis['metrics_analysis'].values()]
        total_counts = [result['total_count'] for result in analysis['metrics_analysis'].values()]
        
        if total_counts:
            analysis['overall_pass_rate'] = sum(pass_counts) / sum(total_counts)
        
        # Generate summary
        analysis['summary'] = self._generate_summary(analysis['metrics_analysis'])
        
        return analysis
    
    def _analyze_metric(self, metric_name: str, values: List[float], threshold: float) -> Dict[str, Any]:
        """Analyze a single metric and determine pass/fail."""
        if not values:
            return {
                'pass_count': 0,
                'total_count': 0,
                'pass_rate': 0.0,
                'mean_value': 0.0,
                'std_value': 0.0,
                'min_value': 0.0,
                'max_value': 0.0,
                'status': 'no_data'
            }
        
        values = np.array(values)
        
        # Determine pass/fail based on metric type
        if metric_name in ['forward_speed', 'episode_length', 'episode_reward']:
            # Higher is better
            passes = values >= threshold
        else:
            # Lower is better (for penalties, drift, etc.)
            passes = values <= threshold
        
        pass_count = np.sum(passes)
        total_count = len(values)
        pass_rate = pass_count / total_count if total_count > 0 else 0.0
        
        # Determine status
        if pass_rate >= 0.9:
            status = 'excellent'
        elif pass_rate >= 0.7:
            status = 'good'
        elif pass_rate >= 0.5:
            status = 'warning'
        else:
            status = 'fail'
        
        return {
            'pass_count': int(pass_count),
            'total_count': total_count,
            'pass_rate': pass_rate,
            'mean_value': float(np.mean(values)),
            'std_value': float(np.std(values)),
            'min_value': float(np.min(values)),
            'max_value': float(np.max(values)),
            'status': status,
            'threshold': threshold,
            'values': values.tolist(),
            'passes': passes.tolist()
        }
    
    def _generate_summary(self, metrics_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of analysis results."""
        summary = {
            'excellent_metrics': [],
            'good_metrics': [],
            'warning_metrics': [],
            'failed_metrics': [],
            'recommendations': []
        }
        
        for metric_name, analysis in metrics_analysis.items():
            status = analysis['status']
            
            if status == 'excellent':
                summary['excellent_metrics'].append(metric_name)
            elif status == 'good':
                summary['good_metrics'].append(metric_name)
            elif status == 'warning':
                summary['warning_metrics'].append(metric_name)
            else:
                summary['failed_metrics'].append(metric_name)
        
        # Generate recommendations
        if summary['failed_metrics']:
            summary['recommendations'].append(
                f"Focus on improving: {', '.join(summary['failed_metrics'])}"
            )
        
        if summary['warning_metrics']:
            summary['recommendations'].append(
                f"Monitor closely: {', '.join(summary['warning_metrics'])}"
            )
        
        if summary['excellent_metrics']:
            summary['recommendations'].append(
                f"Strong performance in: {', '.join(summary['excellent_metrics'])}"
            )
        
        return summary
    
    def create_rollout_plots(self, rollout_data: Dict[str, List[Any]], 
                           analysis: Dict[str, Any], save_dir: str = None) -> Dict[str, str]:
        """Create comprehensive rollout visualization plots."""
        if self.backend == "matplotlib":
            return self._create_matplotlib_plots(rollout_data, analysis, save_dir)
        elif self.backend == "plotly":
            return self._create_plotly_plots(rollout_data, analysis, save_dir)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _create_matplotlib_plots(self, rollout_data: Dict[str, List[Any]], 
                               analysis: Dict[str, Any], save_dir: str = None) -> Dict[str, str]:
        """Create matplotlib plots."""
        plot_paths = {}
        
        # Main rollout overview
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Go2 Rollout Analysis', fontsize=16, fontweight='bold')
        
        episodes = rollout_data['episodes']
        
        # Plot 1: Episode Rewards with Pass/Fail
        ax1 = axes[0, 0]
        rewards = rollout_data['rewards']
        reward_analysis = analysis['metrics_analysis'].get('episode_reward', {})
        
        if 'passes' in reward_analysis:
            passes = np.array(reward_analysis['passes'])
            ax1.scatter(episodes, rewards, c=[self.colors['pass'] if p else self.colors['fail'] 
                                            for p in passes], alpha=0.7, s=20)
        else:
            ax1.plot(episodes, rewards, 'o-', alpha=0.7, color=self.colors['neutral'])
        
        ax1.axhline(y=self.thresholds['episode_reward'], color='red', linestyle='--', 
                   label=f'Threshold: {self.thresholds["episode_reward"]}')
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Episode Lengths with Pass/Fail
        ax2 = axes[0, 1]
        lengths = rollout_data['lengths']
        length_analysis = analysis['metrics_analysis'].get('episode_length', {})
        
        if 'passes' in length_analysis:
            passes = np.array(length_analysis['passes'])
            ax2.scatter(episodes, lengths, c=[self.colors['pass'] if p else self.colors['fail'] 
                                            for p in passes], alpha=0.7, s=20)
        else:
            ax2.plot(episodes, lengths, 'o-', alpha=0.7, color=self.colors['neutral'])
        
        ax2.axhline(y=self.thresholds['episode_length'], color='red', linestyle='--',
                   label=f'Threshold: {self.thresholds["episode_length"]}')
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Length (steps)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Forward Speed
        ax3 = axes[0, 2]
        if 'forward_speed' in rollout_data['metrics']:
            speeds = rollout_data['metrics']['forward_speed']
            speed_analysis = analysis['metrics_analysis'].get('forward_speed', {})
            
            if 'passes' in speed_analysis:
                passes = np.array(speed_analysis['passes'])
                ax3.scatter(episodes, speeds, c=[self.colors['pass'] if p else self.colors['fail'] 
                                               for p in passes], alpha=0.7, s=20)
            else:
                ax3.plot(episodes, speeds, 'o-', alpha=0.7, color=self.colors['neutral'])
            
            ax3.axhline(y=self.thresholds['forward_speed'], color='red', linestyle='--',
                       label=f'Target: {self.thresholds["forward_speed"]} m/s')
            ax3.set_title('Forward Speed')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Speed (m/s)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Stability Metrics
        ax4 = axes[1, 0]
        stability_metrics = ['lateral_drift', 'tilt_angle', 'height_stability']
        for i, metric in enumerate(stability_metrics):
            if metric in rollout_data['metrics']:
                values = rollout_data['metrics'][metric]
                ax4.plot(episodes, values, 'o-', alpha=0.7, label=metric.replace('_', ' ').title())
        
        ax4.set_title('Stability Metrics')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Value')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Pass/Fail Summary
        ax5 = axes[1, 1]
        metric_names = list(analysis['metrics_analysis'].keys())
        pass_rates = [analysis['metrics_analysis'][m]['pass_rate'] for m in metric_names]
        
        colors = [self.colors['pass'] if pr >= 0.7 else self.colors['warning'] if pr >= 0.5 else self.colors['fail'] 
                 for pr in pass_rates]
        
        bars = ax5.bar(range(len(metric_names)), pass_rates, color=colors, alpha=0.7)
        ax5.set_title('Pass Rate by Metric')
        ax5.set_xlabel('Metric')
        ax5.set_ylabel('Pass Rate')
        ax5.set_xticks(range(len(metric_names)))
        ax5.set_xticklabels([m.replace('_', ' ').title() for m in metric_names], rotation=45)
        ax5.set_ylim(0, 1)
        ax5.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, rate in zip(bars, pass_rates):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.2f}', ha='center', va='bottom')
        
        # Plot 6: Overall Performance
        ax6 = axes[1, 2]
        overall_pass_rate = analysis['overall_pass_rate']
        
        # Create pie chart
        pass_count = int(overall_pass_rate * 100)
        fail_count = 100 - pass_count
        
        ax6.pie([pass_count, fail_count], labels=['Pass', 'Fail'], 
               colors=[self.colors['pass'], self.colors['fail']], autopct='%1.1f%%')
        ax6.set_title(f'Overall Performance\n({overall_pass_rate:.1%} Pass Rate)')
        
        plt.tight_layout()
        
        # Save plot
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plot_path = os.path.join(save_dir, 'rollout_analysis.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plot_paths['overview'] = plot_path
            print(f"Rollout analysis plot saved to {plot_path}")
        
        plt.show()
        
        return plot_paths
    
    def _create_plotly_plots(self, rollout_data: Dict[str, List[Any]], 
                           analysis: Dict[str, Any], save_dir: str = None) -> Dict[str, str]:
        """Create Plotly plots."""
        plot_paths = {}
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Episode Rewards', 'Episode Lengths', 'Forward Speed',
                          'Stability Metrics', 'Pass Rate Summary', 'Overall Performance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"type": "pie"}]]
        )
        
        episodes = rollout_data['episodes']
        
        # Plot 1: Episode Rewards
        rewards = rollout_data['rewards']
        reward_analysis = analysis['metrics_analysis'].get('episode_reward', {})
        
        if 'passes' in reward_analysis:
            passes = np.array(reward_analysis['passes'])
            colors = [self.colors['pass'] if p else self.colors['fail'] for p in passes]
            fig.add_trace(go.Scatter(x=episodes, y=rewards, mode='markers', 
                                   marker=dict(color=colors, size=8), name='Rewards'), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=episodes, y=rewards, mode='lines+markers', 
                                   name='Rewards'), row=1, col=1)
        
        fig.add_hline(y=self.thresholds['episode_reward'], line_dash="dash", 
                     line_color="red", row=1, col=1)
        
        # Plot 2: Episode Lengths
        lengths = rollout_data['lengths']
        length_analysis = analysis['metrics_analysis'].get('episode_length', {})
        
        if 'passes' in length_analysis:
            passes = np.array(length_analysis['passes'])
            colors = [self.colors['pass'] if p else self.colors['fail'] for p in passes]
            fig.add_trace(go.Scatter(x=episodes, y=lengths, mode='markers', 
                                   marker=dict(color=colors, size=8), name='Lengths'), row=1, col=2)
        else:
            fig.add_trace(go.Scatter(x=episodes, y=lengths, mode='lines+markers', 
                                   name='Lengths'), row=1, col=2)
        
        fig.add_hline(y=self.thresholds['episode_length'], line_dash="dash", 
                     line_color="red", row=1, col=2)
        
        # Plot 3: Forward Speed
        if 'forward_speed' in rollout_data['metrics']:
            speeds = rollout_data['metrics']['forward_speed']
            speed_analysis = analysis['metrics_analysis'].get('forward_speed', {})
            
            if 'passes' in speed_analysis:
                passes = np.array(speed_analysis['passes'])
                colors = [self.colors['pass'] if p else self.colors['fail'] for p in passes]
                fig.add_trace(go.Scatter(x=episodes, y=speeds, mode='markers', 
                                       marker=dict(color=colors, size=8), name='Speed'), row=1, col=3)
            else:
                fig.add_trace(go.Scatter(x=episodes, y=speeds, mode='lines+markers', 
                                       name='Speed'), row=1, col=3)
            
            fig.add_hline(y=self.thresholds['forward_speed'], line_dash="dash", 
                         line_color="red", row=1, col=3)
        
        # Plot 4: Stability Metrics
        stability_metrics = ['lateral_drift', 'tilt_angle', 'height_stability']
        for metric in stability_metrics:
            if metric in rollout_data['metrics']:
                values = rollout_data['metrics'][metric]
                fig.add_trace(go.Scatter(x=episodes, y=values, mode='lines+markers', 
                                       name=metric.replace('_', ' ').title()), row=2, col=1)
        
        # Plot 5: Pass Rate Summary
        metric_names = list(analysis['metrics_analysis'].keys())
        pass_rates = [analysis['metrics_analysis'][m]['pass_rate'] for m in metric_names]
        
        colors = [self.colors['pass'] if pr >= 0.7 else self.colors['warning'] if pr >= 0.5 else self.colors['fail'] 
                 for pr in pass_rates]
        
        fig.add_trace(go.Bar(x=[m.replace('_', ' ').title() for m in metric_names], 
                            y=pass_rates, marker_color=colors, name='Pass Rate'), row=2, col=2)
        
        # Plot 6: Overall Performance Pie Chart
        overall_pass_rate = analysis['overall_pass_rate']
        pass_count = int(overall_pass_rate * 100)
        fail_count = 100 - pass_count
        
        fig.add_trace(go.Pie(labels=['Pass', 'Fail'], values=[pass_count, fail_count],
                            marker_colors=[self.colors['pass'], self.colors['fail']]), row=2, col=3)
        
        # Update layout
        fig.update_layout(
            title='Go2 Rollout Analysis',
            showlegend=True,
            height=800,
            width=1200
        )
        
        # Save plot
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plot_path = os.path.join(save_dir, 'rollout_analysis.html')
            fig.write_html(plot_path)
            plot_paths['overview'] = plot_path
            print(f"Rollout analysis plot saved to {plot_path}")
        
        fig.show()
        
        return plot_paths
    
    def save_analysis_report(self, analysis: Dict[str, Any], save_dir: str) -> str:
        """Save analysis report to JSON file."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Add timestamp
        analysis['timestamp'] = datetime.now().isoformat()
        analysis['config'] = self.config.__dict__ if hasattr(self.config, '__dict__') else {}
        
        # Save to JSON
        report_path = os.path.join(save_dir, 'rollout_analysis_report.json')
        with open(report_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"Analysis report saved to {report_path}")
        return report_path
    
    def print_summary(self, analysis: Dict[str, Any]) -> None:
        """Print analysis summary to console."""
        print("\n" + "="*60)
        print("GO2 ROLLOUT ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Total Episodes: {analysis['episode_count']}")
        print(f"Overall Pass Rate: {analysis['overall_pass_rate']:.1%}")
        
        print("\nMETRIC ANALYSIS:")
        print("-" * 40)
        
        for metric_name, result in analysis['metrics_analysis'].items():
            status_emoji = {
                'excellent': '🟢',
                'good': '🟡',
                'warning': '🟠',
                'fail': '🔴',
                'no_data': '⚪'
            }.get(result['status'], '❓')
            
            print(f"{status_emoji} {metric_name.replace('_', ' ').title()}: "
                  f"{result['pass_rate']:.1%} ({result['pass_count']}/{result['total_count']}) "
                  f"[{result['mean_value']:.3f} ± {result['std_value']:.3f}]")
        
        print("\nRECOMMENDATIONS:")
        print("-" * 40)
        for rec in analysis['summary']['recommendations']:
            print(f"• {rec}")
        
        print("="*60)

