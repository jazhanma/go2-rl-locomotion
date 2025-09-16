#!/usr/bin/env python3
"""
Go2 Quadruped Demo Recorder - Fixed Version
===========================================

A professional video recording tool for trained Go2 quadruped RL policies.
This version handles observation space mismatches and provides better error handling.

Usage:
    python examples/demo_recorder_fixed.py --policy_path models/best_model.zip --duration 90
"""

import argparse
import os
import sys
import time
import numpy as np
import cv2
import pybullet as p
import pybullet_data
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config import Config, get_default_config
from environments.go2_pybullet import Go2PyBulletEnv
from rewards.reward_functions import RewardFunctionRules


class FixedDemoRecorder:
    """Fixed demo recorder for Go2 quadruped with better error handling."""
    
    def __init__(self, config: Config, policy_path: str, output_path: str, duration: int = 90):
        self.config = config
        self.policy_path = policy_path
        self.output_path = output_path
        self.duration = duration
        
        # Create output directory
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize PyBullet
        try:
            self.physics_client = p.connect(p.GUI)
        except p.error:
            p.disconnect()
            self.physics_client = p.connect(p.GUI)
        
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(0.01, physicsClientId=self.physics_client)
        
        # Load ground plane
        p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), "plane.urdf"),
            physicsClientId=self.physics_client
        )
        
        # Initialize environment
        self.env = Go2PyBulletEnv(config, render_mode="human")
        
        # Load policy with observation space handling
        self.policy = self._load_policy(policy_path)
        self.observation_mismatch = False
        
        # Video recording
        self.video_writer = None
        self.frame_count = 0
        self.start_time = time.time()
        
    def _load_policy(self, policy_path: str):
        """Load trained policy with observation space handling."""
        if not os.path.exists(policy_path):
            print(f"⚠️  Policy file not found: {policy_path}")
            print("Using random policy for demo")
            return None
        
        try:
            from stable_baselines3 import PPO, SAC, TD3, DDPG
            
            # Determine policy type from filename
            if 'ppo' in policy_path.lower():
                policy = PPO.load(policy_path)
            elif 'sac' in policy_path.lower():
                policy = SAC.load(policy_path)
            elif 'td3' in policy_path.lower():
                policy = TD3.load(policy_path)
            elif 'ddpg' in policy_path.lower():
                policy = DDPG.load(policy_path)
            else:
                policy = PPO.load(policy_path)
            
            # Check observation space compatibility
            if hasattr(policy, 'observation_space'):
                expected_obs_dim = policy.observation_space.shape[0]
                actual_obs_dim = self.env.observation_space.shape[0]
                
                if expected_obs_dim != actual_obs_dim:
                    print(f"⚠️  Observation space mismatch:")
                    print(f"   Policy expects: {expected_obs_dim} dimensions")
                    print(f"   Environment provides: {actual_obs_dim} dimensions")
                    print("   Using observation space adaptation")
                    self.observation_mismatch = True
                else:
                    print(f"✅ Policy loaded successfully: {type(policy).__name__}")
            
            return policy
            
        except Exception as e:
            print(f"⚠️  Could not load policy: {e}")
            print("Using random policy for demo")
            return None
    
    def _adapt_observation(self, obs: np.ndarray) -> np.ndarray:
        """Adapt observation to match policy expectations."""
        if not self.observation_mismatch or self.policy is None:
            return obs
        
        expected_dim = self.policy.observation_space.shape[0]
        actual_dim = len(obs)
        
        if actual_dim > expected_dim:
            # Truncate observation
            return obs[:expected_dim]
        elif actual_dim < expected_dim:
            # Pad observation with zeros
            padding = np.zeros(expected_dim - actual_dim)
            return np.concatenate([obs, padding])
        else:
            return obs
    
    def run_demo(self):
        """Run the demo recording."""
        print("🚀 Starting Go2 Quadruped Demo Recording")
        print(f"📁 Policy: {self.policy_path}")
        print(f"⏱️  Duration: {self.duration} seconds")
        print(f"🎬 Output: {self.output_path}")
        print()
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_path, fourcc, 30, (1920, 1080)
        )
        
        # Reset environment
        obs, info = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        # Main simulation loop
        max_steps = int(self.duration / 0.01)  # Convert to simulation steps
        
        try:
            for step in range(max_steps):
                # Get action from policy
                if self.policy is not None:
                    try:
                        # Adapt observation if needed
                        adapted_obs = self._adapt_observation(obs)
                        action, _ = self.policy.predict(adapted_obs, deterministic=True)
                    except Exception as e:
                        print(f"⚠️  Policy prediction failed: {e}")
                        print("Switching to random actions")
                        action = self.env.action_space.sample()
                        self.policy = None  # Disable policy
                else:
                    # Random action for demo
                    action = self.env.action_space.sample()
                
                # Step environment
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Update metrics
                episode_reward += reward
                episode_length += 1
                
                # Update camera to follow robot
                self._update_camera(obs)
                
                # Get camera image
                camera_image = self._get_camera_image()
                
                # Add overlay
                frame_with_overlay = self._add_overlay(
                    camera_image, step, episode_reward, info, episode_length
                )
                
                # Write frame
                self.video_writer.write(frame_with_overlay)
                self.frame_count += 1
                
                # Check termination
                if terminated or truncated:
                    obs, info = self.env.reset()
                    episode_reward = 0.0
                    episode_length = 0
                
                obs = next_obs
                
                # Step simulation
                p.stepSimulation(self.physics_client)
                
                # Control frame rate
                time.sleep(0.01)
                
                # Print progress
                if step % 1000 == 0:
                    elapsed = time.time() - self.start_time
                    progress = (step / max_steps) * 100
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    print(f"📊 Progress: {progress:.1f}% | Step: {step} | Reward: {episode_reward:.2f} | FPS: {fps:.1f}")
        
        except KeyboardInterrupt:
            print("\n⏹️  Demo interrupted by user")
        
        finally:
            # Stop recording
            if self.video_writer:
                self.video_writer.release()
            self.cleanup()
            print(f"✅ Recording saved to {self.output_path}")
    
    def _update_camera(self, obs: np.ndarray):
        """Update camera to follow robot."""
        robot_position = obs[:3]
        
        # Camera follows behind and above robot
        camera_distance = 2.5
        camera_height = 1.0
        camera_offset = 1.5
        
        camera_x = robot_position[0] - camera_offset
        camera_y = robot_position[1]
        camera_z = robot_position[2] + camera_height
        
        camera_pos = [camera_x, camera_y, camera_z]
        camera_target = robot_position.tolist()
        
        # Set camera in PyBullet
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=0,
            cameraPitch=-30,
            cameraTargetPosition=camera_target,
            physicsClientId=self.physics_client
        )
    
    def _get_camera_image(self) -> np.ndarray:
        """Get camera image from PyBullet."""
        width, height = 1920, 1080
        
        # Get camera parameters
        camera_info = p.getDebugVisualizerCamera(self.physics_client)
        camera_pos = camera_info[2]
        camera_target = camera_info[3]
        
        # Calculate view matrix with proper parameters
        distance = np.linalg.norm(np.array(camera_pos) - np.array(camera_target))
        
        # Use simple camera positioning
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=list(camera_target),
            distance=distance,
            yaw=0,
            pitch=-30,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self.physics_client
        )
        
        # Calculate projection matrix
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=width/height, nearVal=0.1, farVal=100.0,
            physicsClientId=self.physics_client
        )
        
        # Get camera image
        _, _, rgb_array, _, _ = p.getCameraImage(
            width, height, view_matrix, projection_matrix,
            physicsClientId=self.physics_client
        )
        
        # Convert to BGR for OpenCV
        rgb_array = np.array(rgb_array)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        
        return bgr_array
    
    def _add_overlay(self, frame: np.ndarray, step: int, reward: float, 
                    info: Dict[str, Any], episode_length: int) -> np.ndarray:
        """Add performance overlay to frame."""
        # Create overlay
        overlay = frame.copy()
        
        # Performance metrics
        forward_speed = info.get('robot_velocities', [0, 0, 0])[0] if 'robot_velocities' in info else 0.0
        energy = np.sum(np.square(info.get('joint_torques', [0] * 12))) if 'joint_torques' in info else 0.0
        
        # Calculate FPS
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        # Text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_color = (255, 255, 255)  # White
        bg_color = (0, 0, 0)  # Black
        
        # Background rectangle
        cv2.rectangle(overlay, (10, 10), (450, 200), bg_color, -1)
        cv2.rectangle(overlay, (10, 10), (450, 200), (255, 255, 255), 2)
        
        # Text lines
        policy_type = type(self.policy).__name__ if self.policy else "Random"
        if self.observation_mismatch:
            policy_type += " (Adapted)"
        
        lines = [
            f"Step: {step:4d}",
            f"Reward: {reward:6.2f}",
            f"Speed: {forward_speed:5.2f} m/s",
            f"Energy: {energy:6.2f}",
            f"Length: {episode_length:4d}",
            f"Policy: {policy_type}",
            f"FPS: {fps:5.1f}"
        ]
        
        y_pos = 40
        for line in lines:
            cv2.putText(overlay, line, (20, y_pos), font, font_scale, text_color, font_thickness)
            y_pos += 25
        
        # Performance indicator
        performance = min(1.0, max(0.0, reward / 100.0))
        bar_width = int(performance * 200)
        bar_x = 1920 - 220
        bar_y = 1080 - 40
        
        # Color based on performance
        if performance > 0.8:
            color = (0, 255, 0)  # Green
        elif performance > 0.6:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), color, -1)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + 200, bar_y + 20), (255, 255, 255), 2)
        
        return overlay
    
    def cleanup(self):
        """Cleanup resources."""
        p.disconnect(self.physics_client)


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description='Go2 Quadruped Demo Recorder (Fixed)')
    parser.add_argument('--policy_path', type=str, default='models/best_model.zip',
                       help='Path to trained policy model')
    parser.add_argument('--duration', type=int, default=90,
                       help='Demo duration in seconds')
    parser.add_argument('--output', type=str, default='videos/go2_demo_fixed.mp4',
                       help='Output video file path')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = Config.load(args.config)
    else:
        config = get_default_config()
    
    # Create demo recorder
    recorder = FixedDemoRecorder(
        config=config,
        policy_path=args.policy_path,
        output_path=args.output,
        duration=args.duration
    )
    
    # Run demo
    recorder.run_demo()


if __name__ == "__main__":
    main()
