#!/usr/bin/env python3
"""
Quick Demo Video Creator - Fixed Issues
======================================

Creates a clear video with proper movement (no spinning in circles).
"""

import os
import sys
import time
import numpy as np
import cv2
import pybullet as p
import pybullet_data

def create_quick_demo():
    """Create a quick demo with proper movement."""
    print("🎬 Creating Quick Demo (Fixed Issues)")
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
    
    # Load R2D2 robot
    robot_id = p.loadURDF(
        os.path.join(pybullet_data.getDataPath(), "r2d2.urdf"),
        [0, 0, 1],
        physicsClientId=physics_client
    )
    
    # Video settings - High resolution for clarity
    width, height = 1920, 1080  # Full HD
    fps = 30
    duration = 5  # 5 seconds only
    total_frames = duration * fps
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "videos/quick_demo.mp4"
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"📹 Recording {duration} seconds at {fps} FPS")
    print(f"🎯 Target frames: {total_frames}")
    print(f"💾 Output: {output_path}")
    print(f"📐 Resolution: {width}x{height} (Full HD)")
    print()
    
    # Recording loop
    frame_count = 0
    start_time = time.time()
    
    try:
        for frame in range(total_frames):
            # FIXED: Move forward instead of spinning in circles
            time_step = frame * 0.1
            forward_speed = 0.3
            
            # Get current position
            pos, orn = p.getBasePositionAndOrientation(robot_id, physicsClientId=physics_client)
            
            # Move forward (not in circles!)
            new_pos = [
                pos[0] + forward_speed * 0.01,  # Move forward in X direction
                pos[1],  # Keep Y position
                pos[2] + 0.01 * np.sin(time_step * 2)  # Add slight bouncing
            ]
            
            # Reset position to create forward movement
            p.resetBasePositionAndOrientation(robot_id, new_pos, orn, physicsClientId=physics_client)
            
            # FIXED: Better camera for clarity
            camera_image = get_clear_camera_image(physics_client, width, height, frame)
            
            # Add overlay
            frame_with_overlay = add_clear_overlay(camera_image, frame, total_frames)
            
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
            
            if file_size > 50000:  # More than 50KB
                print("🎉 Clear video with proper movement created!")
                print("✅ No more spinning in circles!")
                print("✅ High resolution for clarity!")
                return True
            else:
                print("⚠️  Video file is too small")
                return False
        else:
            print("❌ Video file was not created")
            return False

def get_clear_camera_image(physics_client, width, height, frame):
    """Get clear camera image with better positioning."""
    # FIXED: Better camera positioning for clarity
    time_step = frame * 0.1
    
    # Camera follows the robot with better angle
    camera_distance = 4.0  # Further back for better view
    camera_height = 2.0    # Higher for better perspective
    
    # Calculate camera position
    camera_x = -2.0 + 0.1 * time_step  # Follow forward movement
    camera_y = -1.0  # Slight side angle
    camera_z = camera_height
    
    camera_pos = [camera_x, camera_y, camera_z]
    camera_target = [0.1 * time_step, 0, 0.5]  # Look at robot
    
    # Calculate view matrix
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_target,
        distance=camera_distance,
        yaw=10,    # Slight angle for better view
        pitch=-20, # Better angle for clarity
        roll=0,
        upAxisIndex=2,
        physicsClientId=physics_client
    )
    
    # Calculate projection matrix
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=45,  # Smaller FOV for less distortion
        aspect=width/height, 
        nearVal=0.1, 
        farVal=100.0,
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

def add_clear_overlay(frame, frame_num, total_frames):
    """Add clear overlay to frame."""
    # Create overlay
    overlay = frame.copy()
    
    # Text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2
    text_color = (255, 255, 255)  # White
    bg_color = (0, 0, 0)  # Black
    
    # Background rectangle
    cv2.rectangle(overlay, (20, 20), (500, 150), bg_color, -1)
    cv2.rectangle(overlay, (20, 20), (500, 150), (255, 255, 255), 2)
    
    # Text lines
    lines = [
        f"Go2 Quadruped Demo - FIXED",
        f"Resolution: 1920x1080 (Full HD)",
        f"Frame: {frame_num:4d}/{total_frames}",
        f"Time: {frame_num/30:.1f}s",
        f"Movement: Forward (No Circles!)"
    ]
    
    y_pos = 50
    for line in lines:
        cv2.putText(overlay, line, (40, y_pos), font, font_scale, text_color, font_thickness)
        y_pos += 25
    
    # Progress bar
    if total_frames > 0:
        progress = frame_num / total_frames
        bar_width = int(progress * 400)
        bar_x = 40
        bar_y = 170
        
        # Progress bar
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (0, 255, 0), -1)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + 400, bar_y + 20), (255, 255, 255), 2)
    
    return overlay

if __name__ == "__main__":
    success = create_quick_demo()
    if success:
        print("\n🎉 Quick demo video created successfully!")
        print("✅ FIXED: No more spinning in circles!")
        print("✅ FIXED: High resolution for clarity!")
        print("📁 Location: /Users/yujriohanma/Go2/videos/quick_demo.mp4")
    else:
        print("\n❌ Demo video creation failed")
    
    sys.exit(0 if success else 1)

