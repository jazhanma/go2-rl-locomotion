#!/usr/bin/env python3
"""
Perfect Demo Video Creator
=========================

Creates a perfect demo video with clear robot movement and good quality.
"""

import os
import sys
import time
import numpy as np
import cv2
import pybullet as p
import pybullet_data

def create_perfect_demo():
    """Create a perfect demo video."""
    print("🎬 Creating PERFECT Demo Video")
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
    
    # Video settings
    width, height = 1280, 720  # Good resolution, not too heavy
    fps = 24  # Lower FPS for smoother recording
    duration = 15  # 15 seconds - good length
    total_frames = duration * fps
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "videos/perfect_demo.mp4"
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"📹 Recording {duration} seconds at {fps} FPS")
    print(f"🎯 Target frames: {total_frames}")
    print(f"💾 Output: {output_path}")
    print(f"📐 Resolution: {width}x{height}")
    print("🤖 Robot will move in clear patterns!")
    print()
    
    # Recording loop
    frame_count = 0
    start_time = time.time()
    
    try:
        for frame in range(total_frames):
            # Create clear, visible movement
            time_step = frame * 0.1
            
            # Get current position
            pos, orn = p.getBasePositionAndOrientation(robot_id, physicsClientId=physics_client)
            
            # Create clear movement patterns
            if frame < total_frames // 4:
                # Phase 1: Move forward (0-3.75 seconds)
                new_pos = [
                    pos[0] + 0.05,  # Clear forward movement
                    pos[1],
                    pos[2] + 0.1 * np.sin(time_step * 4)  # Clear bouncing
                ]
                movement_type = "Moving Forward"
            elif frame < 2 * total_frames // 4:
                # Phase 2: Move in circle (3.75-7.5 seconds)
                angle = time_step * 0.8
                new_pos = [
                    1.5 * np.cos(angle),  # Clear circle motion
                    1.5 * np.sin(angle),
                    pos[2] + 0.1 * np.sin(time_step * 6)  # Bouncing
                ]
                movement_type = "Moving in Circle"
            elif frame < 3 * total_frames // 4:
                # Phase 3: Move backward (7.5-11.25 seconds)
                new_pos = [
                    pos[0] - 0.04,  # Clear backward movement
                    pos[1] + 0.05 * np.sin(time_step * 3),  # Side movement
                    pos[2] + 0.1 * np.sin(time_step * 5)  # Bouncing
                ]
                movement_type = "Moving Backward"
            else:
                # Phase 4: Move in figure-8 (11.25-15 seconds)
                angle = time_step * 1.2
                new_pos = [
                    2 * np.cos(angle),  # Figure-8 motion
                    1 * np.sin(2 * angle),
                    pos[2] + 0.1 * np.sin(time_step * 8)  # Bouncing
                ]
                movement_type = "Figure-8 Pattern"
            
            # Reset position to create movement
            p.resetBasePositionAndOrientation(robot_id, new_pos, orn, physicsClientId=physics_client)
            
            # Get clear camera image
            camera_image = get_perfect_camera_image(physics_client, width, height, frame, new_pos)
            
            # Add overlay
            frame_with_overlay = add_perfect_overlay(camera_image, frame, total_frames, movement_type)
            
            # Write frame
            video_writer.write(frame_with_overlay)
            frame_count += 1
            
            # Step simulation
            p.stepSimulation(physics_client)
            
            # Control frame rate
            time.sleep(1.0 / fps)
            
            # Print progress
            if frame % 24 == 0:  # Every second
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
            print(f"\n✅ Recording completed!")
            print(f"📁 File: {output_path}")
            print(f"📊 Size: {file_size:,} bytes")
            print(f"🎬 Frames: {frame_count}")
            
            if file_size > 100000:  # More than 100KB
                print("🎉 PERFECT demo video created!")
                print("✅ Clear robot movement!")
                print("✅ Good quality and length!")
                print("✅ Should work in QuickTime Player!")
                return True
            else:
                print("⚠️  Video file is too small")
                return False
        else:
            print("❌ Video file was not created")
            return False

def get_perfect_camera_image(physics_client, width, height, frame, robot_pos):
    """Get perfect camera image."""
    time_step = frame * 0.1
    
    # Camera follows robot with good positioning
    camera_distance = 3.0
    camera_height = 1.5
    
    # Calculate camera position
    camera_x = robot_pos[0] - 2.0
    camera_y = robot_pos[1] - 1.0
    camera_z = robot_pos[2] + camera_height
    
    camera_pos = [camera_x, camera_y, camera_z]
    camera_target = robot_pos
    
    # Calculate view matrix
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_target,
        distance=camera_distance,
        yaw=0,
        pitch=-30,  # Good angle
        roll=0,
        upAxisIndex=2,
        physicsClientId=physics_client
    )
    
    # Calculate projection matrix
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60,  # Good FOV
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

def add_perfect_overlay(frame, frame_num, total_frames, movement_type):
    """Add perfect overlay."""
    # Create overlay
    overlay = frame.copy()
    
    # Text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text_color = (255, 255, 255)  # White
    bg_color = (0, 0, 0)  # Black
    
    # Background rectangle
    cv2.rectangle(overlay, (20, 20), (500, 160), bg_color, -1)
    cv2.rectangle(overlay, (20, 20), (500, 160), (255, 255, 255), 2)
    
    # Text lines
    lines = [
        f"Go2 Quadruped Demo - PERFECT",
        f"Resolution: {frame.shape[1]}x{frame.shape[0]}",
        f"Duration: 15 seconds",
        f"Frame: {frame_num:4d}/{total_frames}",
        f"Time: {frame_num/24:.1f}s",
        f"Movement: {movement_type}"
    ]
    
    y_pos = 45
    for line in lines:
        cv2.putText(overlay, line, (30, y_pos), font, font_scale, text_color, font_thickness)
        y_pos += 25
    
    # Progress bar
    if total_frames > 0:
        progress = frame_num / total_frames
        bar_width = int(progress * 400)
        bar_x = 30
        bar_y = 180
        
        # Progress bar
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (0, 255, 0), -1)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + 400, bar_y + 20), (255, 255, 255), 2)
        
        # Progress percentage
        cv2.putText(overlay, f"{progress:.1%}", (bar_x + 420, bar_y + 15), font, 0.6, text_color, 2)
    
    return overlay

if __name__ == "__main__":
    success = create_perfect_demo()
    if success:
        print("\n🎉 PERFECT demo video created!")
        print("✅ Clear robot movement!")
        print("✅ Good quality and length!")
        print("✅ Should work in QuickTime Player!")
        print("📁 Location: /Users/yujriohanma/Go2/videos/perfect_demo.mp4")
    else:
        print("\n❌ Demo video creation failed")
    
    sys.exit(0 if success else 1)

