#!/usr/bin/env python3
"""
Simple Demo Video Creator for Go2 Quadruped
===========================================

Creates a simple, working demo video that can be opened in QuickTime Player.
"""

import os
import sys
import time
import numpy as np
import cv2
import pybullet as p
import pybullet_data

def create_simple_demo():
    """Create a simple working demo video."""
    print("🎬 Creating Simple Go2 Demo Video")
    print("=" * 40)
    
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
    
    # Load a simple robot (R2D2 as placeholder)
    robot_id = p.loadURDF(
        os.path.join(pybullet_data.getDataPath(), "r2d2.urdf"),
        [0, 0, 1],
        physicsClientId=physics_client
    )
    
    # Video settings
    width, height = 1280, 720  # Smaller resolution for reliability
    fps = 30
    duration = 5  # 5 seconds
    total_frames = duration * fps
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "videos/simple_demo.mp4"
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"📹 Recording {duration} seconds at {fps} FPS")
    print(f"🎯 Target frames: {total_frames}")
    print(f"💾 Output: {output_path}")
    print()
    
    # Recording loop
    frame_count = 0
    start_time = time.time()
    
    try:
        for frame in range(total_frames):
            # Simple robot movement
            angle = frame * 0.1
            pos = [0, 0, 1 + 0.1 * np.sin(angle)]
            orn = p.getQuaternionFromEuler([0, 0, angle])
            p.resetBasePositionAndOrientation(robot_id, pos, orn, physicsClientId=physics_client)
            
            # Get camera image
            camera_image = get_camera_image(physics_client, width, height)
            
            # Add simple overlay
            frame_with_overlay = add_simple_overlay(camera_image, frame, total_frames)
            
            # Write frame
            video_writer.write(frame_with_overlay)
            frame_count += 1
            
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
        p.disconnect(physics_client)
        
        # Check file
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"\n✅ Recording completed!")
            print(f"📁 File: {output_path}")
            print(f"📊 Size: {file_size:,} bytes")
            print(f"🎬 Frames: {frame_count}")
            
            if file_size > 10000:  # More than 10KB
                print("🎉 Video should be playable in QuickTime Player!")
                return True
            else:
                print("⚠️  Video file is too small")
                return False
        else:
            print("❌ Video file was not created")
            return False

def get_camera_image(physics_client, width, height):
    """Get camera image from PyBullet."""
    # Simple camera setup
    camera_pos = [0, -3, 2]
    camera_target = [0, 0, 1]
    
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

def add_simple_overlay(frame, frame_num, total_frames):
    """Add simple overlay to frame."""
    # Create overlay
    overlay = frame.copy()
    
    # Text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2
    text_color = (255, 255, 255)  # White
    bg_color = (0, 0, 0)  # Black
    
    # Background rectangle
    cv2.rectangle(overlay, (20, 20), (400, 120), bg_color, -1)
    cv2.rectangle(overlay, (20, 20), (400, 120), (255, 255, 255), 2)
    
    # Text lines
    lines = [
        f"Go2 Quadruped Demo",
        f"Frame: {frame_num:4d}/{total_frames}",
        f"Status: Recording...",
        f"Time: {frame_num/30:.1f}s"
    ]
    
    y_pos = 50
    for line in lines:
        cv2.putText(overlay, line, (40, y_pos), font, font_scale, text_color, font_thickness)
        y_pos += 25
    
    # Progress bar
    if total_frames > 0:
        progress = frame_num / total_frames
        bar_width = int(progress * 300)
        bar_x = 40
        bar_y = 140
        
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + 15), (0, 255, 0), -1)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + 300, bar_y + 15), (255, 255, 255), 2)
    
    return overlay

if __name__ == "__main__":
    success = create_simple_demo()
    if success:
        print("\n🎉 Demo video created successfully!")
        print("📱 You can now open it in QuickTime Player or any video player")
        print("📁 Location: /Users/yujriohanma/Go2/videos/simple_demo.mp4")
    else:
        print("\n❌ Demo video creation failed")
    
    sys.exit(0 if success else 1)

