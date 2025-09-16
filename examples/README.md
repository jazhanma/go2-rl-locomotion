# Go2 Quadruped Examples

This directory contains example scripts demonstrating how to use the Go2 Quadruped Locomotion Framework.

## 📁 Available Examples

### 1. Basic Training (`basic_training.py`)
Simple example of training a PPO policy on the Go2 environment.

```bash
python examples/basic_training.py
```

### 2. Visualization Demo (`visualization_demo.py`)
Demonstrates the visualization capabilities of the framework.

```bash
python examples/visualization_demo.py
```

### 3. Demo Recorder (`demo_recorder.py`)
**Professional video recording tool for trained policies.**

Creates cinematic, YouTube-ready demo videos with:
- Smooth third-person camera tracking
- Real-time performance overlays
- High-quality MP4 recording (1080p, 30fps)
- Adjustable camera controls

#### Usage

```bash
# Basic usage
python examples/demo_recorder.py --policy_path models/best_model.zip --duration 90

# Custom output
python examples/demo_recorder.py --policy_path models/ppo_model.zip --duration 60 --output my_demo.mp4

# With custom config
python examples/demo_recorder.py --policy_path models/sac_model.zip --config configs/sac_config.yaml
```

#### Features

- **Camera Controls**: Automatic following with smooth tracking
- **Real-time Overlays**: Step count, reward, speed, energy consumption, FPS
- **Performance Indicators**: Color-coded performance bars
- **High Quality**: 1080p recording at 30 FPS
- **Modular Design**: Easy to customize overlays and camera behavior

#### Output

Videos are saved to `videos/` directory by default:
- `videos/go2_demo.mp4` - Default output
- `videos/my_demo.mp4` - Custom output

## 🎬 Creating Professional Demo Videos

The demo recorder is designed to create professional, YouTube-ready videos of your trained Go2 policies. It includes:

### Camera System
- **Smooth Tracking**: Camera follows robot with cinematic angles
- **Automatic Positioning**: Optimal third-person view
- **Stable Footage**: Smooth interpolation prevents jarring movements

### Overlay System
- **Performance Metrics**: Real-time display of key metrics
- **Visual Indicators**: Color-coded performance bars
- **Clean Design**: Non-intrusive but informative overlays

### Video Quality
- **High Resolution**: 1920x1080 (1080p)
- **Smooth Frame Rate**: 30 FPS for professional quality
- **Efficient Encoding**: MP4 format with optimized compression

## 🔧 Customization

The demo recorder is highly modular and can be easily customized:

### Camera Behavior
- Modify `_update_camera()` method for different tracking styles
- Adjust camera distance, height, and offset
- Add cinematic camera movements

### Overlay Design
- Edit `_add_overlay()` method for custom information display
- Modify colors, fonts, and positioning
- Add additional metrics or visual elements

### Video Settings
- Change resolution in `_get_camera_image()`
- Adjust frame rate in video writer initialization
- Modify encoding settings for different quality/size trade-offs

## 📊 Example Output

The demo recorder creates videos showing:
- Robot locomotion behavior
- Real-time performance metrics
- Smooth camera tracking
- Professional visual quality

Perfect for:
- Research presentations
- YouTube demonstrations
- Social media content
- Academic papers
- Portfolio showcases

## 🚀 Quick Start

1. Train a policy using the main training script
2. Run the demo recorder with your trained model
3. Get a professional video of your robot in action!

```bash
# Train a policy
python train.py --policy ppo_baseline --timesteps 1000000

# Record a demo
python examples/demo_recorder.py --policy_path models/go2_experiment/best_model.zip
```

## 📝 Notes

- The demo recorder works with any trained policy compatible with stable-baselines3
- Camera tracking is optimized for quadruped locomotion
- Overlay information is updated in real-time
- Videos are saved in MP4 format for maximum compatibility
