#!/usr/bin/env python3
"""
Super Simple Test Video
=======================

Creates a basic test video to verify everything works.
"""

import cv2
import numpy as np
import os

def create_test_video():
    """Create a simple test video."""
    print("🎬 Creating Simple Test Video")
    
    # Create output directory
    os.makedirs("videos", exist_ok=True)
    
    # Video settings
    width, height = 640, 480
    fps = 30
    duration = 5  # 5 seconds
    total_frames = duration * fps
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "videos/test_video.mp4"
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"📹 Creating {duration} second test video")
    print(f"💾 Output: {output_path}")
    
    # Create frames
    for frame in range(total_frames):
        # Create a simple colored frame
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some color and text
        img[:, :] = [100, 150, 200]  # Blue background
        
        # Add text
        cv2.putText(img, f"Test Video Frame {frame}", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f"Time: {frame/30:.1f}s", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add a moving circle
        center_x = int(320 + 200 * np.sin(frame * 0.1))
        center_y = int(240 + 100 * np.cos(frame * 0.1))
        cv2.circle(img, (center_x, center_y), 30, (0, 255, 0), -1)
        
        # Write frame
        video_writer.write(img)
    
    # Release video writer
    video_writer.release()
    
    # Check file
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"✅ Test video created!")
        print(f"📁 File: {output_path}")
        print(f"📊 Size: {file_size:,} bytes")
        print("🎉 This should definitely work in QuickTime Player!")
        return True
    else:
        print("❌ Test video creation failed")
        return False

if __name__ == "__main__":
    success = create_test_video()
    if success:
        print("\n📱 Try opening: /Users/yujriohanma/Go2/videos/test_video.mp4")
    sys.exit(0 if success else 1)

