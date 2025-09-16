#!/usr/bin/env python3
"""
Working Demo Video Creator for Go2 Quadruped
============================================

Creates a proper working demo video that can be opened in QuickTime Player.
This version focuses on creating valid MP4 files with proper encoding.
"""

import os
import sys
import time
import numpy as np
import cv2
import pybullet as p
import pybullet_data
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.config import Config, get_default_config
from environments.go2_pybullet import Go2PyBulletEnv

def create_working_demo():
    """Create a working demo video."""
    print("🎬 Creating Working Go2 Demo Video")
    print("=" * 50)
    
    # Load configuration
    config = get_default_config()
    
    # Create output directory
    os.makedirs("videos", exist_ok=True)
    
    # Initialize PyBullet
    try:
        physics_client = p.connect(p.GUI)
    except p.error:
        p.disconnect()
        physics_client = p.connect(p.GUI)
    
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
    p.setTimeStep(0.01, physicsClientId=physics_client)
    
    # Load ground plane
    p.loadURDF(
        os.path.join(pybullet_data.getDataPath(), "plane.urdf"),
        physicsClientId=physics_client
    )
    
    # Create environment
    env = Go2PyBulletEnv(config, render_mode="human")
    
    # Video settings
    width, height = 1920, 1080
    fps = 30
    duration = 10  # seconds
    total_frames = duration * fps
    
    # Initialize video writer with proper codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "videos/working_demo.mp4"
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"📹 Recording {duration} seconds at {fps} FPS")
    print(f"🎯 Target frames: {total_frames}")
    print(f"💾 Output: {output_path}")
    print()
    
    # Reset environment
    obs, info = env.reset()
    episode_reward = 0.0
    episode_length = 0
    
    # Recording loop
    frame_count = 0
    start_time = time.time()
    
    try:
        for frame in range(total_frames):
            # Random action
            action = env.action_space.sample()
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            
            # Get camera image
            camera_image = get_camera_image(physics_client, width, height)
            
            # Add overlay
            frame_with_overlay = add_overlay(
                camera_image, frame, episode_reward, info, episode_length, frame_count
            )
            
            # Write frame
            video_writer.write(frame_with_overlay)
            frame_count += 1
            
            # Check termination
            if terminated or truncated:
                obs, info = env.reset()
                episode_reward = 0.0
                episode_length = 0
            
            obs = next_obs
            
            # Step simulation
            p.stepSimulation(physics_client)
            
            # Control frame rate
            time.sleep(1.0 / fps)
            
            # Print progress
            if frame % 30 == 0:  # Every second
                progress = (frame / total_frames) * 100
                elapsed = time.time() - start_time
                actual_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"📊 Progress: {progress:.1f}% | Frame: {frame}/{total_frames} | FPS: {actual_fps:.1f}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Recording interrupted by user")
    
    finally:
        # Stop recording
        video_writer.release()
        env.close()
        p.disconnect(physics_client)
        
        # Check file
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"\n✅ Recording completed!")
            print(f"📁 File: {output_path}")
            print(f"📊 Size: {file_size:,} bytes")
            print(f"🎬 Frames: {frame_count}")
            
            if file_size > 1000:  # More than 1KB
                print("🎉 Video should be playable in QuickTime Player!")
                return True
            else:
                print("⚠️  Video file is too small, may be corrupted")
                return False
        else:
            print("❌ Video file was not created")
            return False

def get_camera_image(physics_client, width, height):
    """Get camera image from PyBullet."""
    # Simple camera setup
    camera_pos = [0, -3, 2]
    camera_target = [0, 0, 0.3]
    
    # Calculate view matrix
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_target,
        distance=3.0,
        yaw=0,
        pitch=-30,
        roll=0,
        upAxisIndex=2,
        physicsClientId=physics_client
    )
    
    # Calculate projection matrix
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=width/height, nearVal=0.1, farVal=100.0,
        physicsClientId=physics_client
    )
    
    # Get camera image
    _, _, rgb_array, _, _ = p.getCameraImage(
        width, height, view_matrix, projection_matrix,
        physicsClientId=physics_client
    )
    
    # Convert to BGR for OpenCV
    rgb_array = np.array(rgb_array)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    
    return bgr_array

def add_overlay(frame, frame_num, reward, info, episode_length, total_frames):
    """Add overlay to frame."""
    # Create overlay
    overlay = frame.copy()
    
    # Performance metrics
    forward_speed = info.get('robot_velocities', [0, 0, 0])[0] if 'robot_velocities' in info else 0.0
    
    # Text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2
    text_color = (255, 255, 255)  # White
    bg_color = (0, 0, 0)  # Black
    
    # Background rectangle
    cv2.rectangle(overlay, (20, 20), (500, 200), bg_color, -1)
    cv2.rectangle(overlay, (20, 20), (500, 200), (255, 255, 255), 2)
    
    # Text lines
    lines = [
        f"Go2 Quadruped Demo",
        f"Frame: {frame_num:4d}/{total_frames}",
        f"Reward: {reward:6.2f}",
        f"Speed: {forward_speed:5.2f} m/s",
        f"Episode: {episode_length:4d}",
        f"Status: Recording..."
    ]
    
    y_pos = 60
    for line in lines:
        cv2.putText(overlay, line, (40, y_pos), font, font_scale, text_color, font_thickness)
        y_pos += 30
    
    # Progress bar
    progress = frame_num / total_frames
    bar_width = int(progress * 400)
    bar_x = 40
    bar_y = 220
    
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (0, 255, 0), -1)
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + 400, bar_y + 20), (255, 255, 255), 2)
    
    return overlay

if __name__ == "__main__":
    success = create_working_demo()
    if success:
        print("\n🎉 Demo video created successfully!")
        print("📱 You can now open it in QuickTime Player or any video player")
    else:
        print("\n❌ Demo video creation failed")
    
    sys.exit(0 if success else 1)

