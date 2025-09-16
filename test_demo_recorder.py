#!/usr/bin/env python3
"""
Test script for the Go2 demo recorder.
This script tests the demo recorder functionality without requiring a trained model.
"""

import os
import sys
import subprocess
import time

def test_demo_recorder():
    """Test the demo recorder functionality."""
    print("🧪 Testing Go2 Demo Recorder")
    print("=" * 50)
    
    # Check if required packages are available
    try:
        import cv2
        import pybullet
        import numpy as np
        print("✅ Required packages available")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install: pip install opencv-python pybullet numpy")
        return False
    
    # Test the simplified demo recorder
    print("\n🎬 Testing simplified demo recorder...")
    
    try:
        # Run the demo recorder for a short duration
        cmd = [
            sys.executable, "examples/demo_recorder.py",
            "--duration", "5",  # Short test duration
            "--output", "videos/test_demo.mp4"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Demo recorder test completed successfully")
            
            # Check if output file was created
            if os.path.exists("videos/test_demo.mp4"):
                file_size = os.path.getsize("videos/test_demo.mp4")
                print(f"✅ Output video created: videos/test_demo.mp4 ({file_size} bytes)")
                return True
            else:
                print("❌ Output video file not found")
                return False
        else:
            print(f"❌ Demo recorder failed with return code: {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Demo recorder test timed out")
        return False
    except Exception as e:
        print(f"❌ Error running demo recorder: {e}")
        return False

def test_full_demo_recorder():
    """Test the full-featured demo recorder."""
    print("\n🎬 Testing full demo recorder...")
    
    try:
        # Run the full demo recorder for a short duration
        cmd = [
            sys.executable, "demo_recorder.py",
            "--duration", "5",  # Short test duration
            "--output", "videos/test_full_demo.mp4"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Full demo recorder test completed successfully")
            
            # Check if output file was created
            if os.path.exists("videos/test_full_demo.mp4"):
                file_size = os.path.getsize("videos/test_full_demo.mp4")
                print(f"✅ Output video created: videos/test_full_demo.mp4 ({file_size} bytes)")
                return True
            else:
                print("❌ Output video file not found")
                return False
        else:
            print(f"❌ Full demo recorder failed with return code: {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Full demo recorder test timed out")
        return False
    except Exception as e:
        print(f"❌ Error running full demo recorder: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Go2 Demo Recorder Test Suite")
    print("=" * 50)
    
    # Create videos directory
    os.makedirs("videos", exist_ok=True)
    
    # Test simplified demo recorder
    simple_test = test_demo_recorder()
    
    # Test full demo recorder
    full_test = test_full_demo_recorder()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    if simple_test:
        print("✅ Simplified demo recorder: PASSED")
    else:
        print("❌ Simplified demo recorder: FAILED")
    
    if full_test:
        print("✅ Full demo recorder: PASSED")
    else:
        print("❌ Full demo recorder: FAILED")
    
    if simple_test or full_test:
        print("\n🎉 At least one demo recorder is working!")
        print("📁 Check the videos/ directory for test outputs")
        print("\n🚀 Ready to create professional demo videos!")
    else:
        print("\n❌ Both demo recorders failed")
        print("Please check the error messages above and fix any issues")
    
    return simple_test or full_test

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
