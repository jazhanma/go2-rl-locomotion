"""
3D Visualizer: Real-time 3D rendering of robot walking with camera controls.
"""
import numpy as np
import pybullet as p
import pybullet_data
import cv2
import imageio
from typing import Dict, List, Optional, Tuple, Any
import threading
import time
import os
from collections import deque


class Viz3D:
    """
    3D visualization for Go2 quadruped robot with real-time rendering and camera controls.
    """
    
    def __init__(self, config, robot_urdf: str = None, enable_gui: bool = True):
        self.config = config
        self.robot_urdf = robot_urdf
        self.enable_gui = enable_gui
        
        # PyBullet setup
        self.physics_client = None
        self.robot_id = None
        self.ground_id = None
        
        # Camera settings
        self.camera_distance = 3.0
        self.camera_yaw = 0.0
        self.camera_pitch = -30.0
        self.camera_target = [0, 0, 0.5]
        
        # Rendering settings
        self.width = 1280
        self.height = 720
        self.fov = 60
        self.near_plane = 0.1
        self.far_plane = 100.0
        
        # Video recording
        self.recording = False
        self.video_frames = []
        self.video_writer = None
        self.recording_thread = None
        
        # Robot state
        self.joint_indices = None
        self.joint_names = None
        self.foot_link_indices = None
        
        # Trajectory visualization
        self.trajectory_points = deque(maxlen=1000)
        self.trajectory_line_id = None
        
        # Initialize visualization
        self._initialize_visualization()
    
    def _initialize_visualization(self):
        """Initialize PyBullet visualization."""
        if self.enable_gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Set simulation parameters
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(self.config.env.timestep, physicsClientId=self.physics_client)
        
        # Load ground plane
        self.ground_id = p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), "plane.urdf"),
            physicsClientId=self.physics_client
        )
        
        # Load robot
        self._load_robot()
        
        # Set up camera
        self._setup_camera()
        
        # Set up trajectory visualization
        self._setup_trajectory_viz()
    
    def _load_robot(self):
        """Load robot URDF and get joint information."""
        if self.robot_urdf and os.path.exists(self.robot_urdf):
            # Load custom robot URDF
            start_pos = [0, 0, 0.3]
            start_orientation = p.getQuaternionFromEuler([0, 0, 0])
            
            self.robot_id = p.loadURDF(
                self.robot_urdf,
                start_pos,
                start_orientation,
                physicsClientId=self.physics_client
            )
        else:
            # Use placeholder robot (R2D2 for now)
            start_pos = [0, 0, 0.3]
            start_orientation = p.getQuaternionFromEuler([0, 0, 0])
            
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
    
    def _setup_camera(self):
        """Set up camera for rendering."""
        if self.enable_gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=self.camera_distance,
                cameraYaw=self.camera_yaw,
                cameraPitch=self.camera_pitch,
                cameraTargetPosition=self.camera_target,
                physicsClientId=self.physics_client
            )
    
    def _setup_trajectory_viz(self):
        """Set up trajectory visualization."""
        # Create line for trajectory
        self.trajectory_line_id = p.addUserDebugLine(
            [0, 0, 0], [0, 0, 0], [1, 0, 0], lineWidth=3,
            physicsClientId=self.physics_client
        )
    
    def update_robot_state(self, position: np.ndarray, orientation: np.ndarray, 
                          joint_positions: np.ndarray) -> None:
        """Update robot state in visualization."""
        if self.robot_id is None:
            return
        
        # Update base position and orientation
        p.resetBasePositionAndOrientation(
            self.robot_id, position, orientation,
            physicsClientId=self.physics_client
        )
        
        # Update joint positions
        for i, joint_idx in enumerate(self.joint_indices):
            if i < len(joint_positions):
                p.resetJointState(
                    self.robot_id, joint_idx, joint_positions[i], 0.0,
                    physicsClientId=self.physics_client
                )
        
        # Update trajectory
        self._update_trajectory(position)
    
    def _update_trajectory(self, position: np.ndarray):
        """Update trajectory visualization."""
        self.trajectory_points.append(position.copy())
        
        if len(self.trajectory_points) > 1:
            # Create line segments for trajectory
            points = list(self.trajectory_points)
            for i in range(len(points) - 1):
                p.addUserDebugLine(
                    points[i], points[i + 1], [1, 0, 0], lineWidth=2,
                    physicsClientId=self.physics_client
                )
    
    def set_camera_position(self, distance: float = None, yaw: float = None, 
                           pitch: float = None, target: List[float] = None) -> None:
        """Set camera position for rendering."""
        if distance is not None:
            self.camera_distance = distance
        if yaw is not None:
            self.camera_yaw = yaw
        if pitch is not None:
            self.camera_pitch = pitch
        if target is not None:
            self.camera_target = target
        
        if self.enable_gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=self.camera_distance,
                cameraYaw=self.camera_yaw,
                cameraPitch=self.camera_pitch,
                cameraTargetPosition=self.camera_target,
                physicsClientId=self.physics_client
            )
    
    def get_camera_image(self) -> np.ndarray:
        """Get camera image for video recording."""
        if self.physics_client is None:
            return None
        
        # Compute view matrix
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self.physics_client
        )
        
        # Compute projection matrix
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.width / self.height,
            nearVal=self.near_plane,
            farVal=self.far_plane,
            physicsClientId=self.physics_client
        )
        
        # Get camera image
        _, _, rgb_array, _, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            physicsClientId=self.physics_client
        )
        
        return rgb_array
    
    def start_recording(self, video_path: str = None) -> str:
        """Start recording video."""
        if self.recording:
            print("Already recording!")
            return None
        
        if video_path is None:
            timestamp = int(time.time())
            video_path = f"rollout_{timestamp}.mp4"
        
        self.recording = True
        self.video_frames = []
        
        # Start recording thread
        self.recording_thread = threading.Thread(
            target=self._recording_loop,
            args=(video_path,)
        )
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        print(f"Started recording to {video_path}")
        return video_path
    
    def stop_recording(self) -> str:
        """Stop recording and save video."""
        if not self.recording:
            print("Not currently recording!")
            return None
        
        self.recording = False
        
        if self.recording_thread:
            self.recording_thread.join()
        
        # Save video
        if self.video_frames:
            video_path = self._save_video()
            print(f"Video saved to {video_path}")
            return video_path
        
        return None
    
    def _recording_loop(self, video_path: str):
        """Recording loop for video capture."""
        while self.recording:
            frame = self.get_camera_image()
            if frame is not None:
                self.video_frames.append(frame.copy())
            time.sleep(1.0 / self.config.viz.video_fps)
    
    def _save_video(self) -> str:
        """Save recorded frames to video file."""
        if not self.video_frames:
            return None
        
        # Save as MP4
        video_path = f"rollout_{int(time.time())}.mp4"
        imageio.mimsave(video_path, self.video_frames, fps=self.config.viz.video_fps)
        
        # Also save as GIF for GitHub
        gif_path = video_path.replace('.mp4', '.gif')
        imageio.mimsave(gif_path, self.video_frames, fps=self.config.viz.video_fps)
        
        return video_path
    
    def render_frame(self) -> np.ndarray:
        """Render a single frame."""
        return self.get_camera_image()
    
    def add_debug_line(self, start: List[float], end: List[float], 
                      color: List[float] = [1, 0, 0], line_width: float = 2) -> int:
        """Add debug line to visualization."""
        return p.addUserDebugLine(
            start, end, color, lineWidth=line_width,
            physicsClientId=self.physics_client
        )
    
    def add_debug_text(self, text: str, position: List[float], 
                      text_color: List[float] = [1, 1, 1], text_size: float = 1.0) -> int:
        """Add debug text to visualization."""
        return p.addUserDebugText(
            text, position, textColorRGB=text_color, textSize=text_size,
            physicsClientId=self.physics_client
        )
    
    def remove_debug_item(self, item_id: int) -> None:
        """Remove debug item from visualization."""
        p.removeUserDebugItem(item_id, physicsClientId=self.physics_client)
    
    def clear_debug_items(self) -> None:
        """Clear all debug items."""
        p.removeAllUserDebugItems(physicsClientId=self.physics_client)
    
    def set_robot_color(self, color: List[float]) -> None:
        """Set robot color."""
        if self.robot_id is None:
            return
        
        # Change robot color
        p.changeVisualShape(
            self.robot_id, -1, rgbaColor=color,
            physicsClientId=self.physics_client
        )
    
    def add_ground_grid(self, grid_size: float = 1.0, grid_color: List[float] = [0.5, 0.5, 0.5]) -> None:
        """Add ground grid for reference."""
        # This would add a grid to the ground plane
        # Implementation depends on specific needs
        pass
    
    def add_lighting(self, light_position: List[float] = [0, 0, 5], 
                    light_color: List[float] = [1, 1, 1]) -> None:
        """Add lighting to the scene."""
        # PyBullet lighting is limited, but we can add debug spheres as light sources
        p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), "sphere_small.urdf"),
            light_position,
            physicsClientId=self.physics_client
        )
    
    def reset_scene(self) -> None:
        """Reset the visualization scene."""
        # Reset robot position
        if self.robot_id is not None:
            p.resetBasePositionAndOrientation(
                self.robot_id, [0, 0, 0.3], [0, 0, 0, 1],
                physicsClientId=self.physics_client
            )
            
            # Reset joint positions
            for joint_idx in self.joint_indices:
                p.resetJointState(
                    self.robot_id, joint_idx, 0.0, 0.0,
                    physicsClientId=self.physics_client
                )
        
        # Clear trajectory
        self.trajectory_points.clear()
        self.clear_debug_items()
        
        # Reset camera
        self._setup_camera()
    
    def get_robot_info(self) -> Dict[str, Any]:
        """Get robot information."""
        if self.robot_id is None:
            return {}
        
        pos, orn = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )
        
        return {
            'position': pos,
            'orientation': orn,
            'joint_count': len(self.joint_indices),
            'joint_names': self.joint_names,
            'trajectory_points': len(self.trajectory_points)
        }
    
    def close(self) -> None:
        """Close the visualization."""
        if self.recording:
            self.stop_recording()
        
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()


class Viz3DWithControls(Viz3D):
    """
    3D Visualizer with interactive camera controls.
    """
    
    def __init__(self, config, robot_urdf: str = None, enable_gui: bool = True):
        super().__init__(config, robot_urdf, enable_gui)
        
        # Camera control state
        self.camera_mode = "orbit"  # "orbit", "follow", "fixed"
        self.follow_target = None
        self.auto_rotate = False
        self.rotation_speed = 1.0
        
        # Mouse control state
        self.mouse_dragging = False
        self.last_mouse_pos = None
    
    def set_camera_mode(self, mode: str) -> None:
        """Set camera mode: 'orbit', 'follow', or 'fixed'."""
        self.camera_mode = mode
    
    def set_follow_target(self, target: List[float]) -> None:
        """Set target to follow with camera."""
        self.follow_target = target
    
    def enable_auto_rotate(self, enable: bool = True, speed: float = 1.0) -> None:
        """Enable/disable automatic camera rotation."""
        self.auto_rotate = enable
        self.rotation_speed = speed
    
    def update_camera(self, dt: float = 0.01) -> None:
        """Update camera position based on current mode."""
        if self.camera_mode == "follow" and self.follow_target is not None:
            # Follow target
            self.camera_target = self.follow_target.copy()
            self._setup_camera()
        
        elif self.camera_mode == "orbit" and self.auto_rotate:
            # Auto-rotate around target
            self.camera_yaw += self.rotation_speed * dt
            self._setup_camera()
    
    def handle_mouse_input(self, mouse_x: float, mouse_y: float, 
                          mouse_buttons: List[bool]) -> None:
        """Handle mouse input for camera control."""
        if mouse_buttons[0]:  # Left mouse button
            if not self.mouse_dragging:
                self.mouse_dragging = True
                self.last_mouse_pos = [mouse_x, mouse_y]
            else:
                # Rotate camera
                dx = mouse_x - self.last_mouse_pos[0]
                dy = mouse_y - self.last_mouse_pos[1]
                
                self.camera_yaw += dx * 0.01
                self.camera_pitch += dy * 0.01
                
                # Clamp pitch
                self.camera_pitch = np.clip(self.camera_pitch, -89, 89)
                
                self._setup_camera()
                self.last_mouse_pos = [mouse_x, mouse_y]
        else:
            self.mouse_dragging = False
    
    def handle_keyboard_input(self, key: str) -> None:
        """Handle keyboard input for camera control."""
        if key == 'w':
            self.camera_distance = max(0.5, self.camera_distance - 0.1)
        elif key == 's':
            self.camera_distance = min(10.0, self.camera_distance + 0.1)
        elif key == 'a':
            self.camera_yaw -= 5.0
        elif key == 'd':
            self.camera_yaw += 5.0
        elif key == 'q':
            self.camera_pitch = max(-89, self.camera_pitch - 5.0)
        elif key == 'e':
            self.camera_pitch = min(89, self.camera_pitch + 5.0)
        elif key == 'r':
            self.reset_camera()
        
        self._setup_camera()
    
    def reset_camera(self) -> None:
        """Reset camera to default position."""
        self.camera_distance = 3.0
        self.camera_yaw = 0.0
        self.camera_pitch = -30.0
        self.camera_target = [0, 0, 0.5]
        self._setup_camera()

