#!/usr/bin/env python3
"""
Extend Video Length
==================

Takes an existing working video and extends it to 30 seconds.
"""

import cv2
import numpy as np
import os

def extend_video_to_30_seconds():
    """Extend an existing video to 30 seconds."""
    print("🎬 Extending Video to 30 Seconds")
    print("=" * 40)
    
    # Input and output paths
    input_path = "videos/perfect_demo.mp4"  # Use the working video
    output_path = "videos/30_second_extended.mp4"
    
    if not os.path.exists(input_path):
        print(f"❌ Input video not found: {input_path}")
        return False
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"❌ Could not open input video: {input_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Input video: {total_frames} frames at {fps} FPS")
    print(f"📐 Resolution: {width}x{height}")
    
    # Calculate target frames for 30 seconds
    target_duration = 30  # seconds
    target_frames = target_duration * fps
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"🎯 Creating {target_duration} second video ({target_frames} frames)")
    
    # Read all frames from input video
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    if not frames:
        print("❌ No frames read from input video")
        return False
    
    print(f"📚 Read {len(frames)} frames from input video")
    
    # Extend video by repeating and modifying frames
    for i in range(target_frames):
        # Cycle through original frames
        frame_idx = i % len(frames)
        frame = frames[frame_idx].copy()
        
        # Add overlay with extended information
        overlay = add_extended_overlay(frame, i, target_frames, len(frames))
        
        # Write frame
        out.write(overlay)
        
        # Print progress
        if i % (target_frames // 10) == 0:
            progress = (i / target_frames) * 100
            print(f"📊 Progress: {progress:.1f}% | Frame: {i}/{target_frames}")
    
    # Release video writer
    out.release()
    
    # Check output file
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"\n✅ Extended video created!")
        print(f"📁 File: {output_path}")
        print(f"📊 Size: {file_size:,} bytes")
        print(f"🎬 Duration: {target_duration} seconds")
        print(f"🎬 Frames: {target_frames}")
        return True
    else:
        print("❌ Extended video creation failed")
        return False

def add_extended_overlay(frame, frame_num, total_frames, original_frames):
    """Add overlay for extended video."""
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
    
    # Calculate which original frame this is based on
    original_frame = frame_num % original_frames
    
    # Text lines
    lines = [
        f"Go2 Quadruped Demo - 30 SECONDS",
        f"Resolution: {frame.shape[1]}x{frame.shape[0]}",
        f"Duration: 30 seconds (Extended)",
        f"Frame: {frame_num:4d}/{total_frames}",
        f"Time: {frame_num/24:.1f}s",
        f"Original Frame: {original_frame}/{original_frames}",
        f"Status: Extended Video"
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
    success = extend_video_to_30_seconds()
    if success:
        print("\n🎉 30-second extended video created!")
        print("✅ Based on working video!")
        print("✅ 30 seconds duration!")
        print("📁 Location: /Users/yujriohanma/Go2/videos/30_second_extended.mp4")
    else:
        print("\n❌ Video extension failed")
    
    sys.exit(0 if success else 1)

