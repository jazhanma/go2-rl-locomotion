"""
Go2 Quadruped Environment using PyBullet for simulation.
"""
import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import time
from typing import Dict, Any, Tuple, Optional, List
import math


class Go2PyBulletEnv(gym.Env):
    """
    Go2 Quadruped Environment using PyBullet physics simulation.
    """
    
    def __init__(self, config, render_mode: str = None):
        super().__init__()
        
        self.config = config
        self.render_mode = render_mode
        
        # Simulation parameters
        self.timestep = config.env.timestep
        self.control_freq = config.env.control_freq
        self.max_episode_steps = config.env.max_episode_steps
        
        # Robot parameters
        self.robot_urdf = config.env.robot_urdf
        self.initial_height = config.env.initial_height
        self.initial_pose = config.env.initial_pose
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )
        
        # Observation space: [position(3), orientation(4), linear_vel(3), angular_vel(3), 
        #                     joint_pos(12), joint_vel(12), foot_contacts(4)]
        obs_dim = 3 + 4 + 3 + 3 + 12 + 12 + 4  # 41 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Simulation state
        self.physics_client = None
        self.robot_id = None
        self.episode_step = 0
        self.episode_reward = 0.0
        
        # Robot state
        self.joint_indices = None
        self.joint_names = None
        self.foot_link_indices = None
        
        # Target tracking
        self.target_velocity = np.array([0.4, 0.0, 0.0])  # Forward velocity target
        self.target_yaw = 0.0
        
        # Previous state for reward computation
        self.prev_position = None
        self.prev_orientation = None
        self.prev_joint_positions = None
        
        # Visualization
        self.camera_distance = 2.0
        self.camera_yaw = 0.0
        self.camera_pitch = -30.0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Initialize PyBullet if not already done
        if self.physics_client is None:
            self._initialize_simulation()
        
        # Reset episode state
        self.episode_step = 0
        self.episode_reward = 0.0
        
        # Reset robot
        self._reset_robot()
        
        # Get initial observation
        observation = self._get_observation()
        
        # Reset previous state
        self.prev_position = observation[:3].copy()
        self.prev_orientation = observation[3:7].copy()
        self.prev_joint_positions = observation[7:19].copy()
        
        info = {
            'target_velocity': self.target_velocity,
            'target_yaw': self.target_yaw,
            'robot_velocities': observation[7:13],  # linear + angular velocities
            'foot_contacts': observation[37:41],  # foot contact states
            'foot_positions': self._get_foot_positions(),
            'foot_velocities': self._get_foot_velocities(),
            'joint_torques': self._get_joint_torques(),
            'joint_positions': observation[7:19],
            'joint_velocities': observation[19:31]
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one simulation step."""
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Apply action
        self._apply_action(action)
        
        # Step simulation
        for _ in range(self.control_freq):
            p.stepSimulation(self.physics_client)
            if self.render_mode == "human":
                time.sleep(self.timestep)
        
        # Get new observation
        observation = self._get_observation()
        
        # Compute reward
        reward, reward_info = self._compute_reward(observation, action)
        
        # Check termination conditions
        terminated, termination_info = self._check_termination(observation)
        
        # Check truncation (time limit)
        truncated = self.episode_step >= self.max_episode_steps
        
        # Update episode state
        self.episode_step += 1
        self.episode_reward += reward
        
        # Update previous state
        self.prev_position = observation[:3].copy()
        self.prev_orientation = observation[3:7].copy()
        self.prev_joint_positions = observation[7:19].copy()
        
        # Prepare info
        info = {
            'target_velocity': self.target_velocity,
            'target_yaw': self.target_yaw,
            'robot_velocities': observation[7:13],
            'foot_contacts': observation[37:41],
            'foot_positions': self._get_foot_positions(),
            'foot_velocities': self._get_foot_velocities(),
            'joint_torques': self._get_joint_torques(),
            'joint_positions': observation[7:19],
            'joint_velocities': observation[19:31],
            'reward_info': reward_info,
            'termination_info': termination_info,
            'episode_step': self.episode_step,
            'episode_reward': self.episode_reward
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "human":
            # PyBullet GUI rendering
            pass
        elif self.render_mode == "rgb_array":
            # Return RGB array for video recording
            return self._get_camera_image()
        return None
    
    def close(self) -> None:
        """Close the environment."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
    
    def _initialize_simulation(self) -> None:
        """Initialize PyBullet simulation."""
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Set simulation parameters
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(self.timestep, physicsClientId=self.physics_client)
        
        # Load ground plane
        p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), "plane.urdf"),
            physicsClientId=self.physics_client
        )
        
        # Load robot URDF
        self._load_robot()
        
        # Set camera
        self._set_camera()
    
    def _load_robot(self) -> None:
        """Load robot URDF and get joint information."""
        # For now, use a simple box as placeholder
        # In practice, you'd load the actual Go2 URDF
        start_pos = [0, 0, self.initial_height]
        start_orientation = p.getQuaternionFromEuler(self.initial_pose[3:6])
        
        # Create a simple quadruped robot (placeholder)
        self.robot_id = p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), "r2d2.urdf"),
            start_pos,
            start_orientation,
            physicsClientId=self.physics_client
        )
        
        # Get joint information
        self.joint_indices = []
        self.joint_names = []
        
        for i in range(p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            if joint_info[2] == p.JOINT_REVOLUTE:  # Only revolute joints
                self.joint_indices.append(i)
                self.joint_names.append(joint_info[1].decode('utf-8'))
        
        # Limit to 12 joints for quadruped
        self.joint_indices = self.joint_indices[:12]
        self.joint_names = self.joint_names[:12]
        
        # Set joint limits
        for joint_idx in self.joint_indices:
            p.changeDynamics(
                self.robot_id, joint_idx,
                jointLowerLimit=-1.57,  # -90 degrees
                jointUpperLimit=1.57,   # 90 degrees
                physicsClientId=self.physics_client
            )
    
    def _reset_robot(self) -> None:
        """Reset robot to initial state."""
        if self.robot_id is not None:
            # Reset position and orientation
            start_pos = [0, 0, self.initial_height]
            start_orientation = p.getQuaternionFromEuler(self.initial_pose[3:6])
            
            p.resetBasePositionAndOrientation(
                self.robot_id, start_pos, start_orientation,
                physicsClientId=self.physics_client
            )
            
            # Reset joint positions
            for joint_idx in self.joint_indices:
                p.resetJointState(
                    self.robot_id, joint_idx, 0.0, 0.0,
                    physicsClientId=self.physics_client
                )
    
    def _apply_action(self, action: np.ndarray) -> None:
        """Apply action to robot joints."""
        if self.robot_id is None:
            return
        
        # Scale action to joint torque range
        max_torque = 10.0  # Maximum joint torque
        torques = action * max_torque
        
        # Apply torques to joints
        for i, joint_idx in enumerate(self.joint_indices):
            if i < len(torques):
                p.setJointMotorControl2(
                    self.robot_id, joint_idx, p.TORQUE_CONTROL,
                    force=torques[i], physicsClientId=self.physics_client
                )
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.robot_id is None:
            return np.zeros(self.observation_space.shape[0])
        
        # Get base position and orientation
        pos, orn = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )
        
        # Get base velocity
        linear_vel, angular_vel = p.getBaseVelocity(
            self.robot_id, physicsClientId=self.physics_client
        )
        
        # Get joint states
        joint_positions = []
        joint_velocities = []
        
        for joint_idx in self.joint_indices:
            joint_state = p.getJointState(
                self.robot_id, joint_idx, physicsClientId=self.physics_client
            )
            joint_positions.append(joint_state[0])
            joint_velocities.append(joint_state[1])
        
        # Get foot contact states (simplified)
        foot_contacts = self._get_foot_contacts()
        
        # Combine into observation
        observation = np.concatenate([
            pos,                    # 3: position
            orn,                    # 4: orientation (quaternion)
            linear_vel,             # 3: linear velocity
            angular_vel,            # 3: angular velocity
            joint_positions,        # 12: joint positions
            joint_velocities,       # 12: joint velocities
            foot_contacts           # 4: foot contacts
        ])
        
        return observation.astype(np.float32)
    
    def _get_foot_contacts(self) -> np.ndarray:
        """Get foot contact states."""
        # Simplified foot contact detection
        # In practice, you'd check actual foot-ground contacts
        return np.array([1.0, 1.0, 1.0, 1.0])  # All feet in contact
    
    def _get_foot_positions(self) -> np.ndarray:
        """Get foot positions in world frame."""
        # Simplified foot positions
        # In practice, you'd compute actual foot positions
        return np.zeros((4, 3))
    
    def _get_foot_velocities(self) -> np.ndarray:
        """Get foot velocities in world frame."""
        # Simplified foot velocities
        # In practice, you'd compute actual foot velocities
        return np.zeros((4, 3))
    
    def _get_joint_torques(self) -> np.ndarray:
        """Get current joint torques."""
        if self.robot_id is None:
            return np.zeros(12)
        
        torques = []
        for joint_idx in self.joint_indices:
            joint_state = p.getJointState(
                self.robot_id, joint_idx, physicsClientId=self.physics_client
            )
            torques.append(joint_state[3])  # Applied torque
        
        return np.array(torques)
    
    def _compute_reward(self, observation: np.ndarray, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Compute reward for current state."""
        # Extract state information
        position = observation[:3]
        orientation = observation[3:7]
        linear_vel = observation[7:10]
        angular_vel = observation[10:13]
        joint_positions = observation[13:25]
        joint_velocities = observation[25:37]
        foot_contacts = observation[37:41]
        
        # Compute reward components
        reward_info = {}
        
        # 1. Forward speed reward
        forward_vel = linear_vel[0]
        target_speed = self.target_velocity[0]
        if forward_vel >= target_speed:
            speed_reward = 1.0
        else:
            speed_reward = np.exp(-2 * (1 - forward_vel / target_speed) ** 2)
        reward_info['forward_speed'] = speed_reward
        
        # 2. Yaw tracking reward
        yaw = self._quat_to_yaw(orientation)
        yaw_error = abs(yaw - self.target_yaw)
        yaw_reward = np.exp(-5 * yaw_error / np.pi)
        reward_info['yaw_tracking'] = yaw_reward
        
        # 3. Lateral drift penalty
        lateral_vel = linear_vel[1]
        lateral_penalty = -abs(lateral_vel)
        reward_info['lateral_drift'] = lateral_penalty
        
        # 4. Tilt penalty
        roll, pitch = self._quat_to_roll_pitch(orientation)
        tilt_penalty = -np.sqrt(roll**2 + pitch**2)
        reward_info['tilt_penalty'] = tilt_penalty
        
        # 5. Height penalty
        height_error = abs(position[2] - self.initial_height)
        height_penalty = -height_error
        reward_info['height_penalty'] = height_penalty
        
        # 6. Energy penalty
        energy_penalty = -np.sum(action**2)
        reward_info['energy_penalty'] = energy_penalty
        
        # 7. Action smoothness penalty
        if self.prev_joint_positions is not None:
            action_diff = action - self.prev_joint_positions
            smoothness_penalty = -np.sum(action_diff**2)
        else:
            smoothness_penalty = 0.0
        reward_info['action_smoothness'] = smoothness_penalty
        
        # 8. Foot slip penalty (simplified)
        foot_slip_penalty = 0.0  # Would need actual foot velocity computation
        reward_info['foot_slip'] = foot_slip_penalty
        
        # Total reward
        total_reward = (
            speed_reward + yaw_reward + lateral_penalty + tilt_penalty + 
            height_penalty + energy_penalty + smoothness_penalty + foot_slip_penalty
        )
        
        return total_reward, reward_info
    
    def _check_termination(self, observation: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """Check if episode should terminate."""
        position = observation[:3]
        orientation = observation[3:7]
        
        termination_info = {}
        terminated = False
        
        # Check if robot fell over
        roll, pitch = self._quat_to_roll_pitch(orientation)
        if abs(roll) > 0.5 or abs(pitch) > 0.5:  # 30 degrees
            terminated = True
            termination_info['reason'] = 'fall'
        
        # Check if robot fell too low
        if position[2] < 0.1:
            terminated = True
            termination_info['reason'] = 'low_height'
        
        # Check if robot moved too far from origin
        if np.linalg.norm(position[:2]) > 10.0:
            terminated = True
            termination_info['reason'] = 'out_of_bounds'
        
        return terminated, termination_info
    
    def _quat_to_yaw(self, quat: np.ndarray) -> float:
        """Convert quaternion to yaw angle."""
        w, x, y, z = quat
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    def _quat_to_roll_pitch(self, quat: np.ndarray) -> Tuple[float, float]:
        """Convert quaternion to roll and pitch angles."""
        w, x, y, z = quat
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        pitch = math.asin(2 * (w * y - z * x))
        return roll, pitch
    
    def _set_camera(self) -> None:
        """Set camera position for rendering."""
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=self.camera_distance,
                cameraYaw=self.camera_yaw,
                cameraPitch=self.camera_pitch,
                cameraTargetPosition=[0, 0, 0.5],
                physicsClientId=self.physics_client
            )
    
    def _get_camera_image(self) -> np.ndarray:
        """Get camera image for video recording."""
        if self.render_mode == "rgb_array":
            # Get camera image from PyBullet
            width, height = 640, 480
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0.5],
                distance=self.camera_distance,
                yaw=self.camera_yaw,
                pitch=self.camera_pitch,
                roll=0,
                upAxisIndex=2,
                physicsClientId=self.physics_client
            )
            projection_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=width/height, nearVal=0.1, farVal=100.0,
                physicsClientId=self.physics_client
            )
            
            _, _, rgb_array, _, _ = p.getCameraImage(
                width, height, view_matrix, projection_matrix,
                physicsClientId=self.physics_client
            )
            
            return rgb_array
        return None
    
    def set_target_velocity(self, velocity: np.ndarray) -> None:
        """Set target velocity for the robot."""
        self.target_velocity = velocity.copy()
    
    def set_target_yaw(self, yaw: float) -> None:
        """Set target yaw angle for the robot."""
        self.target_yaw = yaw

