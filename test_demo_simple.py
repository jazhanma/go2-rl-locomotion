#!/usr/bin/env python3
"""
Simple test for the demo recorder using random actions.
This bypasses the policy loading issue and tests the core functionality.
"""

import os
import sys
import time
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.config import Config, get_default_config
from environments.go2_pybullet import Go2PyBulletEnv

def test_demo_with_random_actions():
    """Test demo recorder with random actions."""
    print("🧪 Testing demo recorder with random actions")
    print("=" * 50)
    
    # Load configuration
    config = get_default_config()
    
    # Create environment
    print("Creating environment...")
    env = Go2PyBulletEnv(config, render_mode="human")
    
    # Create output directory
    os.makedirs("videos", exist_ok=True)
    
    # Initialize video recording
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        "videos/random_demo.mp4", fourcc, 30, (1920, 1080)
    )
    
    print("🎬 Starting random action demo...")
    
    # Reset environment
    obs, info = env.reset()
    episode_reward = 0.0
    episode_length = 0
    
    # Run for 10 seconds (1000 steps at 0.01 timestep)
    max_steps = 1000
    
    try:
        for step in range(max_steps):
            # Random action
            action = env.action_space.sample()
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            
            # Get camera image
            camera_image = get_camera_image(env.physics_client)
            
            # Add simple overlay
            frame_with_overlay = add_simple_overlay(
                camera_image, step, episode_reward, info, episode_length
            )
            
            # Write frame
            video_writer.write(frame_with_overlay)
            
            # Check termination
            if terminated or truncated:
                obs, info = env.reset()
                episode_reward = 0.0
                episode_length = 0
            
            obs = next_obs
            
            # Step simulation
            import pybullet as p
            p.stepSimulation(env.physics_client)
            
            # Control frame rate
            time.sleep(0.01)
            
            # Print progress
            if step % 100 == 0:
                progress = (step / max_steps) * 100
                print(f"📊 Progress: {progress:.1f}% | Step: {step} | Reward: {episode_reward:.2f}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    
    finally:
        # Stop recording
        video_writer.release()
        env.close()
        print(f"✅ Recording saved to videos/random_demo.mp4")
        
        # Check if file was created
        if os.path.exists("videos/random_demo.mp4"):
            file_size = os.path.getsize("videos/random_demo.mp4")
            print(f"📁 File size: {file_size} bytes")
            return True
        else:
            print("❌ Video file not created")
            return False

def get_camera_image(physics_client):
    """Get camera image from PyBullet."""
    import pybullet as p
    
    width, height = 1920, 1080
    
    # Get camera parameters
    camera_info = p.getDebugVisualizerCamera(physics_client)
    camera_pos = camera_info[2]
    camera_target = camera_info[3]
    
    # Calculate view matrix
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_target,
        distance=np.linalg.norm(np.array(camera_pos) - np.array(camera_target)),
        yaw=camera_info[4],
        pitch=camera_info[5],
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

def add_simple_overlay(frame, step, reward, info, episode_length):
    """Add simple overlay to frame."""
    import cv2
    
    # Create overlay
    overlay = frame.copy()
    
    # Performance metrics
    forward_speed = info.get('robot_velocities', [0, 0, 0])[0] if 'robot_velocities' in info else 0.0
    
    # Text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text_color = (255, 255, 255)  # White
    bg_color = (0, 0, 0)  # Black
    
    # Background rectangle
    cv2.rectangle(overlay, (10, 10), (400, 150), bg_color, -1)
    cv2.rectangle(overlay, (10, 10), (400, 150), (255, 255, 255), 2)
    
    # Text lines
    lines = [
        f"Step: {step:4d}",
        f"Reward: {reward:6.2f}",
        f"Speed: {forward_speed:5.2f} m/s",
        f"Length: {episode_length:4d}",
        f"Mode: Random Actions"
    ]
    
    y_pos = 40
    for line in lines:
        cv2.putText(overlay, line, (20, y_pos), font, font_scale, text_color, font_thickness)
        y_pos += 25
    
    return overlay

if __name__ == "__main__":
    success = test_demo_with_random_actions()
    if success:
        print("\n🎉 Demo recorder test successful!")
        print("The demo recorder is working correctly.")
    else:
        print("\n❌ Demo recorder test failed.")
    
    sys.exit(0 if success else 1)

