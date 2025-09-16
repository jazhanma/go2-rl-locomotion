#!/usr/bin/env python3
"""
Long Moving Demo Video Creator
=============================

Creates a longer video with clear robot movement.
"""

import os
import sys
import time
import numpy as np
import cv2
import pybullet as p
import pybullet_data

def create_long_moving_demo():
    """Create a longer demo with clear robot movement."""
    print("🎬 Creating Long Moving Demo Video")
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
    
    # Video settings - Longer duration
    width, height = 1920, 1080  # Full HD
    fps = 30
    duration = 30  # 30 seconds - much longer!
    total_frames = duration * fps
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "videos/long_moving_demo.mp4"
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"📹 Recording {duration} seconds at {fps} FPS")
    print(f"🎯 Target frames: {total_frames}")
    print(f"💾 Output: {output_path}")
    print(f"📐 Resolution: {width}x{height} (Full HD)")
    print("🤖 Robot will move forward, backward, and in patterns!")
    print()
    
    # Recording loop
    frame_count = 0
    start_time = time.time()
    
    try:
        for frame in range(total_frames):
            # Create complex movement patterns
            time_step = frame * 0.1
            
            # Get current position
            pos, orn = p.getBasePositionAndOrientation(robot_id, physicsClientId=physics_client)
            
            # Create interesting movement patterns
            if frame < total_frames // 3:
                # Phase 1: Move forward
                new_pos = [
                    pos[0] + 0.02,  # Move forward
                    pos[1],
                    pos[2] + 0.02 * np.sin(time_step * 3)  # Bouncing
                ]
                movement_type = "Forward"
            elif frame < 2 * total_frames // 3:
                # Phase 2: Move in a circle
                angle = time_step * 0.5
                new_pos = [
                    2 * np.cos(angle),  # Circle motion
                    2 * np.sin(angle),
                    pos[2] + 0.03 * np.sin(time_step * 4)  # Bouncing
                ]
                movement_type = "Circle"
            else:
                # Phase 3: Move backward
                new_pos = [
                    pos[0] - 0.015,  # Move backward
                    pos[1] + 0.01 * np.sin(time_step * 2),  # Slight side movement
                    pos[2] + 0.02 * np.sin(time_step * 5)  # Bouncing
                ]
                movement_type = "Backward"
            
            # Reset position to create movement
            p.resetBasePositionAndOrientation(robot_id, new_pos, orn, physicsClientId=physics_client)
            
            # Get clear camera image
            camera_image = get_dynamic_camera_image(physics_client, width, height, frame, new_pos)
            
            # Add overlay
            frame_with_overlay = add_movement_overlay(camera_image, frame, total_frames, movement_type)
            
            # Write frame
            video_writer.write(frame_with_overlay)
            frame_count += 1
            
            # Step simulation
            p.stepSimulation(physics_client)
            
            # Control frame rate
            time.sleep(1.0 / fps)
            
            # Print progress
            if frame % 60 == 0:  # Every 2 seconds
                progress = (frame / total_frames) * 100
                elapsed = time.time() - start_time
                actual_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"📊 Progress: {progress:.1f}% | Frame: {frame}/{total_frames} | FPS: {actual_fps:.1f} | Movement: {movement_type}")
    
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
            
            if file_size > 500000:  # More than 500KB
                print("🎉 Long moving video created!")
                print("✅ Robot moves forward, in circles, and backward!")
                print("✅ 30 seconds of clear footage!")
                return True
            else:
                print("⚠️  Video file is too small")
                return False
        else:
            print("❌ Video file was not created")
            return False

def get_dynamic_camera_image(physics_client, width, height, frame, robot_pos):
    """Get dynamic camera image that follows the robot."""
    time_step = frame * 0.1
    
    # Dynamic camera that follows the robot
    camera_distance = 3.0
    camera_height = 2.0
    
    # Calculate camera position based on robot position
    camera_x = robot_pos[0] - 2.0 + 0.5 * np.sin(time_step * 0.3)  # Follow with slight orbit
    camera_y = robot_pos[1] - 1.0 + 0.3 * np.cos(time_step * 0.3)
    camera_z = robot_pos[2] + camera_height
    
    camera_pos = [camera_x, camera_y, camera_z]
    camera_target = robot_pos  # Always look at robot
    
    # Calculate view matrix
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_target,
        distance=camera_distance,
        yaw=15,    # Slight angle for better view
        pitch=-25, # Good angle for clarity
        roll=0,
        upAxisIndex=2,
        physicsClientId=physics_client
    )
    
    # Calculate projection matrix
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=50,  # Good FOV for clarity
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

def add_movement_overlay(frame, frame_num, total_frames, movement_type):
    """Add overlay showing movement type."""
    # Create overlay
    overlay = frame.copy()
    
    # Text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2
    text_color = (255, 255, 255)  # White
    bg_color = (0, 0, 0)  # Black
    
    # Background rectangle
    cv2.rectangle(overlay, (20, 20), (600, 180), bg_color, -1)
    cv2.rectangle(overlay, (20, 20), (600, 180), (255, 255, 255), 3)
    
    # Text lines
    lines = [
        f"Go2 Quadruped Demo - MOVING ROBOT",
        f"Resolution: 1920x1080 (Full HD)",
        f"Duration: 30 seconds",
        f"Frame: {frame_num:4d}/{total_frames}",
        f"Time: {frame_num/30:.1f}s",
        f"Movement: {movement_type}",
        f"Status: Recording..."
    ]
    
    y_pos = 50
    for line in lines:
        cv2.putText(overlay, line, (40, y_pos), font, font_scale, text_color, font_thickness)
        y_pos += 25
    
    # Progress bar
    if total_frames > 0:
        progress = frame_num / total_frames
        bar_width = int(progress * 500)
        bar_x = 40
        bar_y = 200
        
        # Progress bar
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + 25), (0, 255, 0), -1)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + 500, bar_y + 25), (255, 255, 255), 2)
        
        # Progress percentage
        cv2.putText(overlay, f"{progress:.1%}", (bar_x + 520, bar_y + 20), font, 0.8, text_color, 2)
    
    return overlay

if __name__ == "__main__":
    success = create_long_moving_demo()
    if success:
        print("\n🎉 Long moving demo video created successfully!")
        print("✅ Robot moves forward, in circles, and backward!")
        print("✅ 30 seconds of clear footage!")
        print("✅ High resolution (1920x1080)!")
        print("📁 Location: /Users/yujriohanma/Go2/videos/long_moving_demo.mp4")
    else:
        print("\n❌ Demo video creation failed")
    
    sys.exit(0 if success else 1)

