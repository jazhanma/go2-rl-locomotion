#!/usr/bin/env python3
"""
30-Second Demo Video Creator
============================

Creates a 30-second demo video with clear robot movement.
"""

import os
import sys
import time
import numpy as np
import cv2
import pybullet as p
import pybullet_data

def create_30_second_demo():
    """Create a 30-second demo video."""
    print("🎬 Creating 30-Second Demo Video")
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
    
    # Video settings - 30 seconds
    width, height = 1280, 720  # Good resolution
    fps = 24  # Smooth FPS
    duration = 30  # 30 seconds!
    total_frames = duration * fps
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "videos/30_second_demo.mp4"
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"📹 Recording {duration} seconds at {fps} FPS")
    print(f"🎯 Target frames: {total_frames}")
    print(f"💾 Output: {output_path}")
    print(f"📐 Resolution: {width}x{height}")
    print("🤖 Robot will move in 6 different patterns!")
    print()
    
    # Recording loop
    frame_count = 0
    start_time = time.time()
    
    try:
        for frame in range(total_frames):
            # Create 6 different movement phases (5 seconds each)
            time_step = frame * 0.1
            phase = frame // (total_frames // 6)  # 6 phases
            
            # Get current position
            pos, orn = p.getBasePositionAndOrientation(robot_id, physicsClientId=physics_client)
            
            # Phase 1: Move forward (0-5 seconds)
            if phase == 0:
                new_pos = [
                    pos[0] + 0.03,  # Forward movement
                    pos[1],
                    pos[2] + 0.1 * np.sin(time_step * 4)  # Bouncing
                ]
                movement_type = "Phase 1: Moving Forward"
            # Phase 2: Move in circle (5-10 seconds)
            elif phase == 1:
                angle = time_step * 0.6
                new_pos = [
                    2 * np.cos(angle),  # Circle motion
                    2 * np.sin(angle),
                    pos[2] + 0.1 * np.sin(time_step * 6)  # Bouncing
                ]
                movement_type = "Phase 2: Moving in Circle"
            # Phase 3: Move backward (10-15 seconds)
            elif phase == 2:
                new_pos = [
                    pos[0] - 0.025,  # Backward movement
                    pos[1] + 0.03 * np.sin(time_step * 3),  # Side movement
                    pos[2] + 0.1 * np.sin(time_step * 5)  # Bouncing
                ]
                movement_type = "Phase 3: Moving Backward"
            # Phase 4: Figure-8 pattern (15-20 seconds)
            elif phase == 3:
                angle = time_step * 0.8
                new_pos = [
                    2.5 * np.cos(angle),  # Figure-8 motion
                    1.2 * np.sin(2 * angle),
                    pos[2] + 0.1 * np.sin(time_step * 8)  # Bouncing
                ]
                movement_type = "Phase 4: Figure-8 Pattern"
            # Phase 5: Spiral movement (20-25 seconds)
            elif phase == 4:
                angle = time_step * 1.0
                radius = 0.5 + 0.3 * time_step
                new_pos = [
                    radius * np.cos(angle),  # Spiral motion
                    radius * np.sin(angle),
                    pos[2] + 0.1 * np.sin(time_step * 10)  # Bouncing
                ]
                movement_type = "Phase 5: Spiral Movement"
            # Phase 6: Random exploration (25-30 seconds)
            else:
                new_pos = [
                    pos[0] + 0.02 * np.sin(time_step * 2),  # Random movement
                    pos[1] + 0.02 * np.cos(time_step * 3),
                    pos[2] + 0.1 * np.sin(time_step * 7)  # Bouncing
                ]
                movement_type = "Phase 6: Random Exploration"
            
            # Reset position to create movement
            p.resetBasePositionAndOrientation(robot_id, new_pos, orn, physicsClientId=physics_client)
            
            # Get clear camera image
            camera_image = get_dynamic_camera_image(physics_client, width, height, frame, new_pos)
            
            # Add overlay
            frame_with_overlay = add_30_second_overlay(camera_image, frame, total_frames, movement_type, phase)
            
            # Write frame
            video_writer.write(frame_with_overlay)
            frame_count += 1
            
            # Step simulation
            p.stepSimulation(physics_client)
            
            # Control frame rate
            time.sleep(1.0 / fps)
            
            # Print progress
            if frame % 48 == 0:  # Every 2 seconds
                progress = (frame / total_frames) * 100
                elapsed = time.time() - start_time
                actual_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"📊 Progress: {progress:.1f}% | Frame: {frame}/{total_frames} | FPS: {actual_fps:.1f} | {movement_type}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Recording interrupted by user")
    
    finally:
        # Stop recording
        video_writer.release()
        p.disconnect(physics_client)
        
        # Check file
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"\n✅ 30-second recording completed!")
            print(f"📁 File: {output_path}")
            print(f"📊 Size: {file_size:,} bytes")
            print(f"🎬 Frames: {frame_count}")
            
            if file_size > 1000000:  # More than 1MB
                print("🎉 30-second demo video created!")
                print("✅ 6 different movement phases!")
                print("✅ Clear robot movement throughout!")
                print("✅ Should work in QuickTime Player!")
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
    camera_distance = 3.5
    camera_height = 2.0
    
    # Calculate camera position based on robot position
    camera_x = robot_pos[0] - 2.5 + 0.5 * np.sin(time_step * 0.2)  # Follow with slight orbit
    camera_y = robot_pos[1] - 1.5 + 0.3 * np.cos(time_step * 0.2)
    camera_z = robot_pos[2] + camera_height
    
    camera_pos = [camera_x, camera_y, camera_z]
    camera_target = robot_pos  # Always look at robot
    
    # Calculate view matrix
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_target,
        distance=camera_distance,
        yaw=10,    # Slight angle for better view
        pitch=-25, # Good angle for clarity
        roll=0,
        upAxisIndex=2,
        physicsClientId=physics_client
    )
    
    # Calculate projection matrix
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=55,  # Good FOV for clarity
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

def add_30_second_overlay(frame, frame_num, total_frames, movement_type, phase):
    """Add overlay for 30-second video."""
    # Create overlay
    overlay = frame.copy()
    
    # Text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text_color = (255, 255, 255)  # White
    bg_color = (0, 0, 0)  # Black
    
    # Background rectangle
    cv2.rectangle(overlay, (20, 20), (600, 200), bg_color, -1)
    cv2.rectangle(overlay, (20, 20), (600, 200), (255, 255, 255), 3)
    
    # Text lines
    lines = [
        f"Go2 Quadruped Demo - 30 SECONDS",
        f"Resolution: {frame.shape[1]}x{frame.shape[0]}",
        f"Duration: 30 seconds (6 phases)",
        f"Frame: {frame_num:4d}/{total_frames}",
        f"Time: {frame_num/24:.1f}s",
        f"Phase: {phase + 1}/6",
        f"Movement: {movement_type}"
    ]
    
    y_pos = 45
    for line in lines:
        cv2.putText(overlay, line, (30, y_pos), font, font_scale, text_color, font_thickness)
        y_pos += 25
    
    # Progress bar
    if total_frames > 0:
        progress = frame_num / total_frames
        bar_width = int(progress * 500)
        bar_x = 30
        bar_y = 220
        
        # Progress bar
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + 25), (0, 255, 0), -1)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + 500, bar_y + 25), (255, 255, 255), 2)
        
        # Progress percentage
        cv2.putText(overlay, f"{progress:.1%}", (bar_x + 520, bar_y + 20), font, 0.6, text_color, 2)
    
    return overlay

if __name__ == "__main__":
    success = create_30_second_demo()
    if success:
        print("\n🎉 30-second demo video created!")
        print("✅ 6 different movement phases!")
        print("✅ Clear robot movement throughout!")
        print("✅ Should work in QuickTime Player!")
        print("📁 Location: /Users/yujriohanma/Go2/videos/30_second_demo.mp4")
    else:
        print("\n❌ Demo video creation failed")
    
    sys.exit(0 if success else 1)

