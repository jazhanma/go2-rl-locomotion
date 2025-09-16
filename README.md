# Go2 Quadruped Locomotion Framework

A comprehensive reinforcement learning framework for training and visualizing quadruped locomotion policies, specifically designed for the Go2 robot. This framework provides multiple policy architectures, detailed reward functions, and extensive visualization tools for debugging and sharing results.

## 🚀 Features

### Policy Architectures
- **PPO Baseline**: Standard PPO with MLP + LSTM support
- **Residual RL**: RL policy adds deltas to a safe controller
- **Behavior Cloning**: Pretraining with expert demonstrations + PPO fine-tuning
- **Asymmetric Critic**: Privileged critic with onboard-sensor actor

### Reward Function Rules (RFR)
- Forward speed reward (≥0.4 m/s target)
- Yaw tracking reward
- Lateral drift penalty
- Tilt & height stability penalties
- Energy/torque penalty
- Action smoothness penalty
- Foot slip penalty
- Early termination penalty

### Visualization & Logging
- **Training Visualizer**: Live graphs of rewards, speed, stability metrics
- **Rollout Visualizer**: Episode analysis with pass/fail indicators
- **3D Visualizer**: Real-time 3D rendering with camera controls
- **Video Recording**: MP4/GIF generation for GitHub demos
- **Comprehensive Logging**: TensorBoard, file structure, metrics tracking

## 📁 Project Structure

```
Go2/
├── src/
│   ├── policies/          # RL policy implementations
│   ├── environments/      # Simulation environments
│   ├── rewards/          # Reward function definitions
│   ├── visualization/    # Visualization tools
│   └── utils/           # Configuration and logging
├── configs/             # Configuration files
├── logs/               # Training logs
├── models/             # Saved models
├── plots/              # Generated plots
├── videos/             # Recorded videos
├── examples/           # Example scripts
├── tests/              # Unit tests
└── docs/               # Documentation
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA (for GPU acceleration)
- PyBullet, MuJoCo, or Isaac Gym

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd Go2

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Environment Setup
The framework supports multiple simulation backends:

1. **PyBullet** (default, included)
2. **MuJoCo** (requires license)
3. **Isaac Gym** (requires NVIDIA license)

## 🚀 Quick Start

### Basic Training
```bash
# Train PPO baseline
python train.py --policy ppo_baseline --timesteps 1000000 --render

# Train Residual RL
python train.py --policy residual_rl --timesteps 1000000 --render

# Train with custom config
python train.py --config configs/ppo_baseline.yaml --render --record
```

### Advanced Training
```bash
# Train with video recording
python train.py --policy ppo_baseline --render --record --timesteps 1000000

# Run evaluation after training
python train.py --policy ppo_baseline --eval --rollout

# Train with specific experiment name
python train.py --policy ppo_baseline --experiment my_experiment --render
```

## 📊 Policy Architectures

### 1. PPO Baseline
Standard Proximal Policy Optimization with MLP or LSTM architecture.

**Key Features:**
- Configurable network architecture
- LSTM support for sequential tasks
- Custom reward function integration

**Usage:**
```python
from policies.ppo_baseline import PPOBaseline

policy = PPOBaseline(config, observation_space, action_space)
```

### 2. Residual RL
RL policy learns to add deltas to a safe controller, providing safety guarantees.

**Key Features:**
- Safe baseline controller
- Adaptive gain control
- Smooth policy updates

**Usage:**
```python
from policies.residual_rl import ResidualRL

policy = ResidualRL(config, observation_space, action_space)
```

### 3. Behavior Cloning + PPO
Pretrain with expert demonstrations, then fine-tune with PPO.

**Key Features:**
- Expert data loading
- BC pretraining
- Seamless PPO fine-tuning

**Usage:**
```python
from policies.bc_pretrain import BCPretrainPPO

policy = BCPretrainPPO(config, observation_space, action_space)
policy.add_expert_data(observations, actions)
policy.pretrain_bc()
policy.fine_tune_ppo(total_timesteps=1000000)
```

### 4. Asymmetric Critic
Privileged critic with onboard-sensor actor for sim-to-real transfer.

**Key Features:**
- Privileged information access
- Onboard sensor actor
- Sim-to-real transfer ready

**Usage:**
```python
from policies.asymmetric_critic import AsymmetricCritic

policy = AsymmetricCritic(config, observation_space, action_space, privileged_obs_space)
```

## 🎯 Reward Functions

The framework implements comprehensive reward functions for stable locomotion:

### Forward Speed Reward
- **Target**: ≥0.4 m/s forward velocity
- **Purpose**: Encourages forward movement
- **Implementation**: Exponential reward peaking at target speed

### Stability Rewards
- **Yaw Tracking**: Maintains desired heading
- **Lateral Drift**: Penalizes sideways movement
- **Tilt Penalty**: Prevents robot from falling over
- **Height Penalty**: Maintains stable height

### Efficiency Rewards
- **Energy Penalty**: Reduces power consumption
- **Action Smoothness**: Encourages smooth control
- **Foot Slip**: Penalizes foot sliding

### Termination Rewards
- **Early Termination**: Penalizes premature episode end
- **Success Bonus**: Rewards completing episodes

## 📈 Visualization Tools

### Training Visualizer
Real-time monitoring of training progress:

```python
from visualization.training_viz import TrainingVisualizer

viz = TrainingVisualizer(config, backend="matplotlib")
viz.start_live_plotting()
viz.add_episode_data(episode, reward, length, metrics)
```

**Features:**
- Live reward/length plots
- Rolling averages
- Metric breakdowns
- Automatic saving

### Rollout Visualizer
Comprehensive analysis of policy performance:

```python
from visualization.rollout_viz import RolloutVisualizer

viz = RolloutVisualizer(config)
analysis = viz.analyze_rollout(rollout_data)
viz.create_rollout_plots(rollout_data, analysis)
```

**Features:**
- Pass/fail indicators
- Metric thresholds
- Performance recommendations
- Statistical analysis

### 3D Visualizer
Real-time 3D rendering with camera controls:

```python
from visualization.viz_3d import Viz3D

viz = Viz3D(config, enable_gui=True)
viz.update_robot_state(position, orientation, joint_positions)
viz.start_recording("rollout.mp4")
```

**Features:**
- Real-time rendering
- Camera controls (orbit, zoom)
- Trajectory visualization
- Video recording
- Debug overlays

## 🔧 Configuration

### Configuration Files
The framework uses YAML configuration files for easy customization:

```yaml
# configs/ppo_baseline.yaml
experiment_name: "go2_ppo_baseline"
policy_type: "ppo_baseline"

policy:
  learning_rate: 3e-4
  n_steps: 2048
  batch_size: 64
  # ... more parameters

reward:
  forward_speed_target: 0.4
  forward_speed_weight: 1.0
  # ... more parameters
```

### Key Parameters
- **Policy**: Learning rate, network architecture, training parameters
- **Reward**: Target speeds, penalty weights, thresholds
- **Environment**: Simulation settings, robot parameters
- **Visualization**: Plot settings, video recording, 3D rendering
- **Logging**: TensorBoard, file paths, checkpointing

## 📊 Results & Examples

### Training Progress
The framework generates comprehensive training visualizations:

<img width="4470" height="2953" alt="image" src="https://github.com/user-attachments/assets/94e2e5ad-c5ae-4fc6-8416-87794079044b" />



### Rollout Analysis
Detailed analysis of policy performance with pass/fail indicators:

<img width="5370" height="3541" alt="image" src="https://github.com/user-attachments/assets/890936b6-985d-4247-b6d9-fd59cc3d973c" />



### 3D Visualization
Real-time 3D rendering of robot locomotion:

![3D Visualization](videos/rollout.gif)

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

Run specific tests:
```bash
python -m pytest tests/test_policies.py
python -m pytest tests/test_rewards.py
```

## 📚 Documentation

### API Reference
- [Policies](docs/policies.md)
- [Environments](docs/environments.md)
- [Rewards](docs/rewards.md)
- [Visualization](docs/visualization.md)

### Tutorials
- [Getting Started](docs/tutorials/getting_started.md)
- [Custom Policies](docs/tutorials/custom_policies.md)
- [Reward Engineering](docs/tutorials/reward_engineering.md)
- [Visualization Guide](docs/tutorials/visualization.md)

### Examples
- [Basic Training](examples/basic_training.py)
- [Custom Environment](examples/custom_environment.py)
- [Reward Tuning](examples/reward_tuning.py)
- [Visualization](examples/visualization.py)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Run tests
pytest
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) for RL algorithms
- [PyBullet](https://pybullet.org/) for physics simulation
- [Isaac Gym](https://developer.nvidia.com/isaac-gym) for high-performance simulation
- [MuJoCo](https://mujoco.org/) for advanced physics simulation

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Join our Discord community
- Check the documentation

## 🗺️ Roadmap

- [ ] Multi-robot training
- [ ] Curriculum learning
- [ ] Domain randomization
- [ ] Real robot deployment
- [ ] Web-based visualization
- [ ] Cloud training support

---

**Happy Training! 🐕🤖**

