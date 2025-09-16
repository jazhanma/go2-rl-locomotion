#!/usr/bin/env python3
"""
Go2 Quadruped Demo Recorder
===========================

A professional video recording tool for trained Go2 quadruped RL policies.
Creates cinematic, YouTube-ready demo videos with real-time overlays and
smooth camera tracking.

Features:
- Loads trained RL policies (PPO, SAC, TD3, DDPG, etc.)
- Smooth third-person camera tracking with orbit controls
- Real-time performance overlays
- High-quality MP4 recording (1080p, 30fps)
- CLI interface for easy customization
- Modular design for easy modification

Usage:
    python demo_recorder.py --policy_path models/best_model.zip --duration 90 --output go2_demo.mp4
"""

import argparse
import os
import sys
import time
import numpy as np
import cv2
import pybullet as p
import pybullet_data
from typing import Dict, Any, Tuple, Optional, List
import threading
import queue
from dataclasses import dataclass
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.config import Config, get_default_config
from environments.go2_pybullet import Go2PyBulletEnv
from rewards.reward_functions import RewardFunctionRules


@dataclass
class CameraState:
    """Camera state for smooth tracking."""
    distance: float = 2.5
    yaw: float = 0.0
    pitch: float = -30.0
    target_position: np.ndarray = None
    orbit_speed: float = 0.5
    zoom_speed: float = 0.1
    pitch_speed: float = 1.0
    yaw_speed: float = 1.0


@dataclass
class OverlayData:
    """Real-time overlay information."""
    episode_step: int = 0
    total_reward: float = 0.0
    forward_speed: float = 0.0
    energy_consumption: float = 0.0
    episode_length: int = 0
    policy_type: str = "Unknown"
    fps: float = 0.0


class CameraController:
    """Advanced camera controller with smooth tracking and cinematic movements."""
    
    def __init__(self, physics_client: int, config: Config):
        self.physics_client = physics_client
        self.config = config
        self.camera = CameraState()
        self.camera.target_position = np.array([0, 0, 0.3])
        
        # Camera modes
        self.mode = "follow"  # "follow", "orbit", "fixed"
        self.orbit_angle = 0.0
        self.orbit_radius = 2.5
        
        # Smooth interpolation
        self.target_camera = CameraState()
        self.interpolation_speed = 0.1
        
        # Cinematic presets
        self.presets = {
            "close_follow": CameraState(distance=1.8, pitch=-25, yaw=0),
            "wide_shot": CameraState(distance=4.0, pitch=-45, yaw=0),
            "side_view": CameraState(distance=2.0, pitch=-15, yaw=90),
            "top_down": CameraState(distance=3.0, pitch=-80, yaw=0),
            "cinematic": CameraState(distance=2.5, pitch=-35, yaw=0)
        }
        self.current_preset = "cinematic"
        
    def update_target(self, robot_position: np.ndarray, robot_orientation: np.ndarray):
        """Update camera target to follow robot."""
        self.camera.target_position = robot_position.copy()
        
        # Smooth interpolation to target
        self.camera.distance += (self.target_camera.distance - self.camera.distance) * self.interpolation_speed
        self.camera.pitch += (self.target_camera.pitch - self.camera.pitch) * self.interpolation_speed
        self.camera.yaw += (self.target_camera.yaw - self.camera.yaw) * self.interpolation_speed
        
        # Orbit mode
        if self.mode == "orbit":
            self.orbit_angle += self.camera.orbit_speed * 0.1
            self.camera.yaw = self.orbit_angle
            
    def set_camera_position(self):
        """Set camera position in PyBullet."""
        if self.mode == "follow":
            # Follow behind and above robot
            robot_pos = self.camera.target_position
            distance = self.camera.distance
            yaw_rad = np.radians(self.camera.yaw)
            pitch_rad = np.radians(self.camera.pitch)
            
            # Calculate camera position
            camera_x = robot_pos[0] - distance * np.cos(pitch_rad) * np.cos(yaw_rad)
            camera_y = robot_pos[1] - distance * np.cos(pitch_rad) * np.sin(yaw_rad)
            camera_z = robot_pos[2] + distance * np.sin(pitch_rad)
            
            camera_pos = [camera_x, camera_y, camera_z]
            camera_target = robot_pos.tolist()
            
        elif self.mode == "orbit":
            # Orbit around robot
            robot_pos = self.camera.target_position
            angle = np.radians(self.orbit_angle)
            
            camera_x = robot_pos[0] + self.orbit_radius * np.cos(angle)
            camera_y = robot_pos[1] + self.orbit_radius * np.sin(angle)
            camera_z = robot_pos[2] + 1.0
            
            camera_pos = [camera_x, camera_y, camera_z]
            camera_target = robot_pos.tolist()
            
        else:  # fixed
            # Fixed camera position
            camera_pos = [0, -3, 2]
            camera_target = [0, 0, 0.3]
        
        # Set camera in PyBullet
        p.resetDebugVisualizerCamera(
            cameraDistance=np.linalg.norm(np.array(camera_pos) - np.array(camera_target)),
            cameraYaw=np.degrees(np.arctan2(camera_y - robot_pos[1], camera_x - robot_pos[0])),
            cameraPitch=np.degrees(np.arcsin((camera_z - robot_pos[2]) / np.linalg.norm(np.array(camera_pos) - np.array(camera_target)))),
            cameraTargetPosition=camera_target,
            physicsClientId=self.physics_client
        )
    
    def handle_keyboard_input(self, key: str):
        """Handle keyboard input for camera control."""
        if key == 'w':
            self.target_camera.distance = max(1.0, self.target_camera.distance - self.camera.zoom_speed)
        elif key == 's':
            self.target_camera.distance = min(10.0, self.target_camera.distance + self.camera.zoom_speed)
        elif key == 'a':
            self.target_camera.yaw -= self.camera.yaw_speed
        elif key == 'd':
            self.target_camera.yaw += self.camera.yaw_speed
        elif key == 'q':
            self.target_camera.pitch = max(-89, self.target_camera.pitch - self.camera.pitch_speed)
        elif key == 'e':
            self.target_camera.pitch = min(89, self.target_camera.pitch + self.camera.pitch_speed)
        elif key == 'r':
            self.reset_camera()
        elif key == '1':
            self.set_preset("close_follow")
        elif key == '2':
            self.set_preset("wide_shot")
        elif key == '3':
            self.set_preset("side_view")
        elif key == '4':
            self.set_preset("top_down")
        elif key == '5':
            self.set_preset("cinematic")
        elif key == 'm':
            self.cycle_mode()
    
    def set_preset(self, preset_name: str):
        """Set camera to a predefined preset."""
        if preset_name in self.presets:
            self.current_preset = preset_name
            preset = self.presets[preset_name]
            self.target_camera.distance = preset.distance
            self.target_camera.pitch = preset.pitch
            self.target_camera.yaw = preset.yaw
    
    def cycle_mode(self):
        """Cycle through camera modes."""
        modes = ["follow", "orbit", "fixed"]
        current_idx = modes.index(self.mode)
        self.mode = modes[(current_idx + 1) % len(modes)]
        print(f"Camera mode: {self.mode}")
    
    def reset_camera(self):
        """Reset camera to default position."""
        self.target_camera = CameraState()
        self.orbit_angle = 0.0


class OverlayRenderer:
    """Real-time overlay renderer for performance metrics."""
    
    def __init__(self, config: Config):
        self.config = config
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2
        self.text_color = (255, 255, 255)  # White
        self.bg_color = (0, 0, 0)  # Black
        self.alpha = 0.7
        
        # Overlay positions
        self.text_start_x = 20
        self.text_start_y = 40
        self.line_height = 35
        
    def render_overlay(self, frame: np.ndarray, data: OverlayData) -> np.ndarray:
        """Render overlay on frame."""
        height, width = frame.shape[:2]
        
        # Create overlay background
        overlay = frame.copy()
        
        # Performance metrics
        metrics = [
            f"Step: {data.episode_step:4d}",
            f"Reward: {data.total_reward:6.2f}",
            f"Speed: {data.forward_speed:5.2f} m/s",
            f"Energy: {data.energy_consumption:6.2f}",
            f"Length: {data.episode_length:4d}",
            f"Policy: {data.policy_type}",
            f"FPS: {data.fps:5.1f}"
        ]
        
        # Draw background rectangle
        rect_height = len(metrics) * self.line_height + 20
        cv2.rectangle(overlay, (10, 10), (400, rect_height), self.bg_color, -1)
        cv2.rectangle(overlay, (10, 10), (400, rect_height), (255, 255, 255), 2)
        
        # Draw text
        for i, metric in enumerate(metrics):
            y_pos = self.text_start_y + i * self.line_height
            cv2.putText(overlay, metric, (self.text_start_x, y_pos), 
                       self.font, self.font_scale, self.text_color, self.font_thickness)
        
        # Draw performance indicator
        self._draw_performance_indicator(overlay, data, width, height)
        
        # Blend overlay
        result = cv2.addWeighted(frame, 1 - self.alpha, overlay, self.alpha, 0)
        return result
    
    def _draw_performance_indicator(self, frame: np.ndarray, data: OverlayData, width: int, height: int):
        """Draw performance indicator bar."""
        # Performance based on reward
        performance = min(1.0, max(0.0, data.total_reward / 100.0))  # Normalize to 0-1
        
        # Color based on performance
        if performance > 0.8:
            color = (0, 255, 0)  # Green
        elif performance > 0.6:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        # Draw performance bar
        bar_width = int(performance * 200)
        bar_x = width - 220
        bar_y = height - 40
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 200, bar_y + 20), (255, 255, 255), 2)
        
        # Performance text
        cv2.putText(frame, f"Performance: {performance:.1%}", 
                   (bar_x, bar_y - 10), self.font, 0.5, (255, 255, 255), 1)


class VideoRecorder:
    """High-quality video recorder for PyBullet simulation."""
    
    def __init__(self, output_path: str, width: int = 1920, height: int = 1080, fps: int = 30):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Frame queue for threading
        self.frame_queue = queue.Queue(maxsize=100)
        self.recording = False
        self.recording_thread = None
        
    def start_recording(self):
        """Start video recording."""
        self.recording = True
        self.recording_thread = threading.Thread(target=self._recording_worker)
        self.recording_thread.start()
        print(f"🎬 Started recording to {self.output_path}")
    
    def stop_recording(self):
        """Stop video recording."""
        self.recording = False
        if self.recording_thread:
            self.recording_thread.join()
        self.video_writer.release()
        print(f"✅ Recording saved to {self.output_path}")
    
    def add_frame(self, frame: np.ndarray):
        """Add frame to recording queue."""
        if self.recording and not self.frame_queue.full():
            # Resize frame to target resolution
            resized_frame = cv2.resize(frame, (self.width, self.height))
            self.frame_queue.put(resized_frame)
    
    def _recording_worker(self):
        """Worker thread for video recording."""
        while self.recording or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=0.1)
                self.video_writer.write(frame)
                self.frame_queue.task_done()
            except queue.Empty:
                continue


class Go2DemoRecorder:
    """Main demo recorder class."""
    
    def __init__(self, config: Config, policy_path: str, output_path: str, duration: int = 90):
        self.config = config
        self.policy_path = policy_path
        self.output_path = output_path
        self.duration = duration
        
        # Initialize PyBullet
        self.physics_client = p.connect(p.GUI)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(0.01, physicsClientId=self.physics_client)
        
        # Load ground plane
        p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), "plane.urdf"),
            physicsClientId=self.physics_client
        )
        
        # Initialize components
        self.env = Go2PyBulletEnv(config, render_mode="human")
        self.camera_controller = CameraController(self.physics_client, config)
        self.overlay_renderer = OverlayRenderer(config)
        self.video_recorder = VideoRecorder(output_path)
        
        # Load policy
        self.policy = self._load_policy(policy_path)
        
        # Performance tracking
        self.overlay_data = OverlayData()
        self.start_time = time.time()
        self.frame_count = 0
        
    def _load_policy(self, policy_path: str):
        """Load trained policy."""
        try:
            # Try to load with stable-baselines3
            from stable_baselines3 import PPO, SAC, TD3, DDPG
            
            # Determine policy type from filename or try different loaders
            if 'ppo' in policy_path.lower():
                return PPO.load(policy_path)
            elif 'sac' in policy_path.lower():
                return SAC.load(policy_path)
            elif 'td3' in policy_path.lower():
                return TD3.load(policy_path)
            elif 'ddpg' in policy_path.lower():
                return DDPG.load(policy_path)
            else:
                # Default to PPO
                return PPO.load(policy_path)
        except Exception as e:
            print(f"⚠️  Could not load policy from {policy_path}: {e}")
            print("Using random policy for demo")
            return None
    
    def run_demo(self):
        """Run the full demo recording."""
        print("🚀 Starting Go2 Quadruped Demo Recording")
        print(f"📁 Policy: {self.policy_path}")
        print(f"⏱️  Duration: {self.duration} seconds")
        print(f"🎬 Output: {self.output_path}")
        print()
        print("🎮 Controls:")
        print("  W/S: Zoom in/out")
        print("  A/D: Rotate left/right")
        print("  Q/E: Pitch up/down")
        print("  R: Reset camera")
        print("  1-5: Camera presets")
        print("  M: Cycle camera modes")
        print("  ESC: Exit")
        print()
        
        # Start recording
        self.video_recorder.start_recording()
        
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
                    action, _ = self.policy.predict(obs, deterministic=True)
                else:
                    # Random action for demo
                    action = self.env.action_space.sample()
                
                # Step environment
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Update metrics
                episode_reward += reward
                episode_length += 1
                
                # Update camera
                robot_position = obs[:3]
                robot_orientation = obs[3:7]
                self.camera_controller.update_target(robot_position, robot_orientation)
                self.camera_controller.set_camera_position()
                
                # Get camera image
                camera_image = self._get_camera_image()
                
                # Update overlay data
                self._update_overlay_data(step, episode_reward, info, episode_length)
                
                # Render overlay
                frame_with_overlay = self.overlay_renderer.render_overlay(camera_image, self.overlay_data)
                
                # Add frame to recording
                self.video_recorder.add_frame(frame_with_overlay)
                
                # Check termination
                if terminated or truncated:
                    obs, info = self.env.reset()
                    episode_reward = 0.0
                    episode_length = 0
                
                obs = next_obs
                
                # Handle keyboard input
                keys = p.getKeyboardEvents(self.physics_client)
                for key, state in keys.items():
                    if state & p.KEY_WAS_TRIGGERED:
                        self.camera_controller.handle_keyboard_input(chr(key))
                
                # Step simulation
                p.stepSimulation(self.physics_client)
                
                # Control frame rate
                time.sleep(0.01)
                
                # Print progress
                if step % 1000 == 0:
                    elapsed = time.time() - self.start_time
                    progress = (step / max_steps) * 100
                    print(f"📊 Progress: {progress:.1f}% | Step: {step} | Reward: {episode_reward:.2f} | FPS: {self.overlay_data.fps:.1f}")
        
        except KeyboardInterrupt:
            print("\n⏹️  Demo interrupted by user")
        
        finally:
            # Stop recording
            self.video_recorder.stop_recording()
            self.cleanup()
    
    def _get_camera_image(self) -> np.ndarray:
        """Get camera image from PyBullet."""
        # Get camera parameters
        camera_pos = p.getDebugVisualizerCamera(self.physics_client)[2]
        camera_target = p.getDebugVisualizerCamera(self.physics_client)[3]
        
        # Render camera image
        width, height = 1920, 1080
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_target,
            distance=np.linalg.norm(np.array(camera_pos) - np.array(camera_target)),
            yaw=p.getDebugVisualizerCamera(self.physics_client)[4],
            pitch=p.getDebugVisualizerCamera(self.physics_client)[5],
            roll=0,
            upAxisIndex=2,
            physicsClientId=self.physics_client
        )
        
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=width/height, nearVal=0.1, farVal=100.0,
            physicsClientId=self.physics_client
        )
        
        _, _, rgb_array, _, _ = p.getCameraImage(
            width, height, view_matrix, projection_matrix,
            physicsClientId=self.physics_client
        )
        
        # Convert to BGR for OpenCV
        rgb_array = np.array(rgb_array)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        
        return bgr_array
    
    def _update_overlay_data(self, step: int, reward: float, info: Dict[str, Any], episode_length: int):
        """Update overlay data."""
        self.overlay_data.episode_step = step
        self.overlay_data.total_reward = reward
        self.overlay_data.episode_length = episode_length
        
        # Extract metrics from info
        if 'robot_velocities' in info:
            self.overlay_data.forward_speed = info['robot_velocities'][0]  # x-velocity
        
        if 'joint_torques' in info:
            self.overlay_data.energy_consumption = np.sum(np.square(info['joint_torques']))
        
        # Calculate FPS
        current_time = time.time()
        if hasattr(self, 'last_time'):
            dt = current_time - self.last_time
            self.overlay_data.fps = 1.0 / dt if dt > 0 else 0
        self.last_time = current_time
        
        # Policy type
        if self.policy is not None:
            self.overlay_data.policy_type = type(self.policy).__name__
        else:
            self.overlay_data.policy_type = "Random"
    
    def cleanup(self):
        """Cleanup resources."""
        p.disconnect(self.physics_client)
        print("🧹 Cleanup completed")


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description='Go2 Quadruped Demo Recorder')
    parser.add_argument('--policy_path', type=str, default='models/best_model.zip',
                       help='Path to trained policy model')
    parser.add_argument('--duration', type=int, default=90,
                       help='Demo duration in seconds')
    parser.add_argument('--output', type=str, default='videos/go2_demo.mp4',
                       help='Output video file path')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--width', type=int, default=1920,
                       help='Video width')
    parser.add_argument('--height', type=int, default=1080,
                       help='Video height')
    parser.add_argument('--fps', type=int, default=30,
                       help='Video FPS')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = Config.load(args.config)
    else:
        config = get_default_config()
    
    # Create demo recorder
    recorder = Go2DemoRecorder(
        config=config,
        policy_path=args.policy_path,
        output_path=args.output,
        duration=args.duration
    )
    
    # Run demo
    recorder.run_demo()


if __name__ == "__main__":
    main()
