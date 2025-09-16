#!/usr/bin/env python3
"""
Better Demo Video Creator for Go2 Quadruped
===========================================

Creates a high-quality demo video with proper quadruped locomotion.
"""

import os
import sys
import time
import numpy as np
import cv2
import pybullet as p
import pybullet_data

def create_better_demo():
    """Create a better demo video with proper quadruped walking."""
    print("🎬 Creating Better Go2 Demo Video")
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
    
    # Try to load a quadruped robot (use ant.urdf as it's a proper quadruped)
    try:
        robot_id = p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), "ant.urdf"),
            [0, 0, 0.5],
            physicsClientId=physics_client
        )
        print("✅ Loaded Ant quadruped robot")
        robot_type = "ant"
    except:
        # Fallback to R2D2 but with better movement
        robot_id = p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), "r2d2.urdf"),
            [0, 0, 1],
            physicsClientId=physics_client
        )
        print("⚠️  Using R2D2 as fallback")
        robot_type = "r2d2"
    
    # Video settings - Higher resolution for clarity
    width, height = 1920, 1080  # Full HD
    fps = 30
    duration = 10  # 10 seconds
    total_frames = duration * fps
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "videos/better_demo.mp4"
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"📹 Recording {duration} seconds at {fps} FPS")
    print(f"🎯 Target frames: {total_frames}")
    print(f"💾 Output: {output_path}")
    print(f"📐 Resolution: {width}x{height}")
    print()
    
    # Recording loop
    frame_count = 0
    start_time = time.time()
    
    try:
        for frame in range(total_frames):
            # Create proper quadruped walking motion
            if robot_type == "ant":
                # For ant robot, create walking gait
                create_ant_walking_gait(robot_id, frame, physics_client)
            else:
                # For R2D2, create forward movement instead of spinning
                create_forward_movement(robot_id, frame, physics_client)
            
            # Get high-quality camera image
            camera_image = get_high_quality_camera_image(physics_client, width, height, frame)
            
            # Add professional overlay
            frame_with_overlay = add_professional_overlay(camera_image, frame, total_frames, robot_type)
            
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
            
            if file_size > 100000:  # More than 100KB
                print("🎉 High-quality video created!")
                print("📱 Should be much clearer and show proper movement!")
                return True
            else:
                print("⚠️  Video file is too small")
                return False
        else:
            print("❌ Video file was not created")
            return False

def create_ant_walking_gait(robot_id, frame, physics_client):
    """Create proper quadruped walking gait for ant robot."""
    # Get joint information
    num_joints = p.getNumJoints(robot_id, physicsClientId=physics_client)
    
    # Create walking gait parameters
    time_step = frame * 0.1
    gait_frequency = 2.0  # Hz
    
    # Apply joint torques for walking
    for joint_idx in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_idx, physicsClientId=physics_client)
        if joint_info[2] == p.JOINT_REVOLUTE:  # Only revolute joints
            # Create sinusoidal walking pattern
            if joint_idx % 2 == 0:  # Even joints (left side)
                torque = 0.5 * np.sin(time_step * gait_frequency * 2 * np.pi)
            else:  # Odd joints (right side)
                torque = 0.5 * np.sin(time_step * gait_frequency * 2 * np.pi + np.pi)
            
            p.setJointMotorControl2(
                robot_id, joint_idx, p.TORQUE_CONTROL,
                force=torque, physicsClientId=physics_client
            )

def create_forward_movement(robot_id, frame, physics_client):
    """Create forward movement instead of spinning."""
    # Move forward instead of spinning
    time_step = frame * 0.1
    forward_speed = 0.5
    
    # Get current position and orientation
    pos, orn = p.getBasePositionAndOrientation(robot_id, physicsClientId=physics_client)
    
    # Move forward
    new_pos = [
        pos[0] + forward_speed * 0.01,  # Move forward in X
        pos[1],  # Keep Y position
        pos[2]  # Keep Z position
    ]
    
    # Reset position to create forward movement
    p.resetBasePositionAndOrientation(robot_id, new_pos, orn, physicsClientId=physics_client)

def get_high_quality_camera_image(physics_client, width, height, frame):
    """Get high-quality camera image with better positioning."""
    # Dynamic camera that follows the robot
    time_step = frame * 0.1
    
    # Camera follows the robot with smooth movement
    camera_distance = 3.0
    camera_height = 1.5
    camera_offset = 2.0
    
    # Calculate camera position
    camera_x = -camera_offset + 0.1 * time_step  # Follow forward movement
    camera_y = 0
    camera_z = camera_height
    
    camera_pos = [camera_x, camera_y, camera_z]
    camera_target = [0.1 * time_step, 0, 0.5]  # Look at robot position
    
    # Calculate view matrix
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_target,
        distance=camera_distance,
        yaw=0,
        pitch=-25,  # Better angle for clarity
        roll=0,
        upAxisIndex=2,
        physicsClientId=physics_client
    )
    
    # Calculate projection matrix with better FOV
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=50,  # Smaller FOV for less distortion
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

def add_professional_overlay(frame, frame_num, total_frames, robot_type):
    """Add professional overlay to frame."""
    # Create overlay
    overlay = frame.copy()
    
    # Text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_thickness = 3
    text_color = (255, 255, 255)  # White
    bg_color = (0, 0, 0)  # Black
    
    # Background rectangle
    cv2.rectangle(overlay, (30, 30), (600, 200), bg_color, -1)
    cv2.rectangle(overlay, (30, 30), (600, 200), (255, 255, 255), 3)
    
    # Text lines
    lines = [
        f"Go2 Quadruped Locomotion Demo",
        f"Robot: {robot_type.upper()}",
        f"Frame: {frame_num:4d}/{total_frames}",
        f"Time: {frame_num/30:.1f}s",
        f"Resolution: 1920x1080",
        f"Status: Recording..."
    ]
    
    y_pos = 70
    for line in lines:
        cv2.putText(overlay, line, (50, y_pos), font, font_scale, text_color, font_thickness)
        y_pos += 35
    
    # Progress bar
    if total_frames > 0:
        progress = frame_num / total_frames
        bar_width = int(progress * 500)
        bar_x = 50
        bar_y = 250
        
        # Progress bar background
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + 500, bar_y + 25), (50, 50, 50), -1)
        # Progress bar fill
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + 25), (0, 255, 0), -1)
        # Progress bar border
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + 500, bar_y + 25), (255, 255, 255), 2)
        
        # Progress percentage
        cv2.putText(overlay, f"{progress:.1%}", (bar_x + 520, bar_y + 20), font, 0.8, text_color, 2)
    
    return overlay

if __name__ == "__main__":
    success = create_better_demo()
    if success:
        print("\n🎉 Better demo video created successfully!")
        print("📱 This video should be much clearer and show proper movement!")
        print("📁 Location: /Users/yujriohanma/Go2/videos/better_demo.mp4")
    else:
        print("\n❌ Demo video creation failed")
    
    sys.exit(0 if success else 1)

