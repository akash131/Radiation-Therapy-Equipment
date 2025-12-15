"""
CyberKnife-Type Robotic Radiosurgery System Module.

This module implements stereotactic radiosurgery systems featuring:
- Robotic arm with sub-millimeter accuracy
- Real-time motion compensation
- Image-guided targeting
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Dict, Callable
import math
import numpy as np

from .base import (
    RadiationSource,
    BeamDeliverySystem,
    BeamParameters,
    Position3D,
    Vector3D,
    DoseDistribution,
    TreatmentTarget,
    TreatmentPlan,
    OrganAtRisk,
    ParticleType,
    BeamModality,
)


class RobotJoint(Enum):
    """Robot arm joints (6-DOF)."""
    J1_BASE_ROTATION = "j1_base"  # Base rotation
    J2_SHOULDER = "j2_shoulder"  # Shoulder
    J3_ELBOW = "j3_elbow"  # Elbow
    J4_WRIST_1 = "j4_wrist1"  # Wrist rotation 1
    J5_WRIST_2 = "j5_wrist2"  # Wrist rotation 2
    J6_WRIST_3 = "j6_wrist3"  # End effector rotation


@dataclass
class JointLimits:
    """Angular limits for robot joints."""
    min_angle: float  # degrees
    max_angle: float  # degrees
    max_velocity: float  # degrees/second
    max_acceleration: float  # degrees/second^2


@dataclass
class RobotConfiguration:
    """Robot arm configuration (joint angles)."""
    j1: float = 0.0
    j2: float = 0.0
    j3: float = 0.0
    j4: float = 0.0
    j5: float = 0.0
    j6: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.j1, self.j2, self.j3, self.j4, self.j5, self.j6])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "RobotConfiguration":
        """Create from numpy array."""
        return cls(
            j1=float(arr[0]),
            j2=float(arr[1]),
            j3=float(arr[2]),
            j4=float(arr[3]),
            j5=float(arr[4]),
            j6=float(arr[5])
        )


@dataclass
class EndEffectorPose:
    """End effector (LINAC) pose in space."""
    position: Position3D  # Position of source
    orientation: Vector3D  # Beam direction vector
    source_to_target_distance: float = 800.0  # mm (SAD)


class RoboticArm:
    """
    6-DOF Robotic Arm for CyberKnife-type system.

    Provides sub-millimeter positioning accuracy for non-isocentric
    radiosurgery delivery. Can position the LINAC at any point on
    a sphere around the target and aim through target center.
    """

    # DH parameters for a typical medical robot (simplified)
    LINK_LENGTHS = [0.0, 650.0, 600.0, 0.0, 0.0, 0.0]  # mm
    LINK_OFFSETS = [400.0, 0.0, 0.0, 650.0, 0.0, 100.0]  # mm

    def __init__(
        self,
        name: str = "Robotic Arm",
        payload_capacity: float = 150.0,  # kg
        reach: float = 2000.0  # mm
    ):
        self.name = name
        self.payload_capacity = payload_capacity
        self.reach = reach

        # Joint limits
        self._joint_limits: Dict[RobotJoint, JointLimits] = {
            RobotJoint.J1_BASE_ROTATION: JointLimits(-185, 185, 100, 200),
            RobotJoint.J2_SHOULDER: JointLimits(-140, 60, 80, 150),
            RobotJoint.J3_ELBOW: JointLimits(-120, 170, 100, 200),
            RobotJoint.J4_WRIST_1: JointLimits(-350, 350, 150, 300),
            RobotJoint.J5_WRIST_2: JointLimits(-130, 130, 150, 300),
            RobotJoint.J6_WRIST_3: JointLimits(-350, 350, 200, 400),
        }

        # Current configuration
        self._configuration = RobotConfiguration()
        self._current_pose = EndEffectorPose(
            position=Position3D(),
            orientation=Vector3D(0, 0, -1)
        )

        # Positioning accuracy
        self._positioning_accuracy = 0.5  # mm (sub-millimeter)
        self._repeatability = 0.2  # mm

        # State
        self._is_calibrated = False
        self._is_moving = False
        self._collision_detected = False
        self._path_planner: Optional[Callable] = None

    def initialize(self) -> bool:
        """Initialize robot arm."""
        return self.calibrate()

    def calibrate(self) -> bool:
        """
        Calibrate robot using end effector tracking.

        Real systems use optical tracking or specialized calibration
        fixtures to ensure sub-millimeter accuracy.
        """
        # Move to home position
        self._configuration = RobotConfiguration()
        self._update_forward_kinematics()
        self._is_calibrated = True
        return True

    def _update_forward_kinematics(self):
        """
        Calculate end effector pose from joint angles.

        Uses simplified DH convention for forward kinematics.
        """
        # Simplified forward kinematics (not full DH)
        config = self._configuration

        # Base rotation
        theta1 = math.radians(config.j1)

        # Calculate approximate end effector position
        # This is simplified; real systems use full transformation matrices
        r = (
            self.LINK_LENGTHS[1] * math.cos(math.radians(config.j2)) +
            self.LINK_LENGTHS[2] * math.cos(math.radians(config.j2 + config.j3))
        )
        z = (
            self.LINK_OFFSETS[0] +
            self.LINK_LENGTHS[1] * math.sin(math.radians(config.j2)) +
            self.LINK_LENGTHS[2] * math.sin(math.radians(config.j2 + config.j3)) +
            self.LINK_OFFSETS[3]
        )

        x = r * math.cos(theta1)
        y = r * math.sin(theta1)

        self._current_pose.position = Position3D(x, y, z)

        # Calculate beam direction from wrist angles
        # Simplified: just use wrist angles for direction
        phi = math.radians(config.j4 + config.j5)
        psi = math.radians(config.j6)

        self._current_pose.orientation = Vector3D(
            dx=math.sin(phi) * math.cos(psi),
            dy=math.sin(phi) * math.sin(psi),
            dz=-math.cos(phi)
        ).normalize()

    def _inverse_kinematics(
        self,
        target_pose: EndEffectorPose
    ) -> Optional[RobotConfiguration]:
        """
        Calculate joint angles for desired end effector pose.

        Uses iterative numerical method for 6-DOF IK.
        """
        # Simplified analytical IK (real systems use numerical methods)
        pos = target_pose.position
        direction = target_pose.orientation

        # Calculate base rotation
        j1 = math.degrees(math.atan2(pos.y, pos.x))

        # Calculate planar distance
        r = math.sqrt(pos.x ** 2 + pos.y ** 2)
        z = pos.z - self.LINK_OFFSETS[0] - self.LINK_OFFSETS[3]

        # Solve 2-link planar arm for j2 and j3
        L1 = self.LINK_LENGTHS[1]
        L2 = self.LINK_LENGTHS[2]
        d = math.sqrt(r ** 2 + z ** 2)

        if d > L1 + L2:
            return None  # Target out of reach

        # Law of cosines for elbow angle
        cos_j3 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
        cos_j3 = max(-1, min(1, cos_j3))  # Clamp for numerical stability
        j3 = math.degrees(math.acos(cos_j3))

        # Shoulder angle
        alpha = math.atan2(z, r)
        beta = math.acos((L1 ** 2 + d ** 2 - L2 ** 2) / (2 * L1 * d))
        j2 = math.degrees(alpha + beta)

        # Wrist angles from desired direction
        # Simplified: align with direction
        j4 = math.degrees(math.atan2(
            direction.dy, direction.dx
        ))
        j5 = math.degrees(math.acos(-direction.dz))
        j6 = 0.0  # Roll

        config = RobotConfiguration(
            j1=j1, j2=j2, j3=j3, j4=j4, j5=j5, j6=j6
        )

        # Check joint limits
        if not self._check_joint_limits(config):
            return None

        return config

    def _check_joint_limits(self, config: RobotConfiguration) -> bool:
        """Check if configuration is within joint limits."""
        angles = [config.j1, config.j2, config.j3, config.j4, config.j5, config.j6]
        joints = list(RobotJoint)

        for joint, angle in zip(joints, angles):
            limits = self._joint_limits[joint]
            if angle < limits.min_angle or angle > limits.max_angle:
                return False

        return True

    def move_to_pose(self, target_pose: EndEffectorPose) -> bool:
        """
        Move robot to target pose.

        Args:
            target_pose: Desired end effector pose
        """
        config = self._inverse_kinematics(target_pose)
        if config is None:
            return False

        return self.move_to_configuration(config)

    def move_to_configuration(self, config: RobotConfiguration) -> bool:
        """Move robot to specified joint configuration."""
        if not self._check_joint_limits(config):
            return False

        # Check for collisions along path
        if not self._check_path_collision(self._configuration, config):
            return False

        self._is_moving = True
        self._configuration = config
        self._update_forward_kinematics()
        self._is_moving = False

        return True

    def _check_path_collision(
        self,
        start: RobotConfiguration,
        end: RobotConfiguration
    ) -> bool:
        """
        Check for collisions along motion path.

        Uses linear interpolation with collision checking.
        """
        num_steps = 20
        start_arr = start.to_array()
        end_arr = end.to_array()

        for i in range(num_steps + 1):
            t = i / num_steps
            interp = start_arr + t * (end_arr - start_arr)
            config = RobotConfiguration.from_array(interp)

            # Check self-collision and environment collision
            if self._detect_collision(config):
                self._collision_detected = True
                return False

        self._collision_detected = False
        return True

    def _detect_collision(self, config: RobotConfiguration) -> bool:
        """
        Detect collisions for given configuration.

        Simplified collision model - real systems use detailed geometry.
        """
        # Check for extreme configurations that might cause collision
        if abs(config.j3) > 160:  # Elbow near limit
            return True
        if abs(config.j2) > 130:  # Shoulder near limit
            return True
        return False

    def aim_at_target(self, source_position: Position3D, target: Position3D) -> bool:
        """
        Position robot to aim at target from given source position.

        Args:
            source_position: Desired source position
            target: Target position to aim at
        """
        # Calculate direction vector
        dx = target.x - source_position.x
        dy = target.y - source_position.y
        dz = target.z - source_position.z
        distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        if distance == 0:
            return False

        direction = Vector3D(
            dx=dx / distance,
            dy=dy / distance,
            dz=dz / distance
        )

        pose = EndEffectorPose(
            position=source_position,
            orientation=direction,
            source_to_target_distance=distance
        )

        return self.move_to_pose(pose)

    def get_positioning_error(self) -> float:
        """Get current positioning error estimate."""
        # In real systems, this comes from tracking measurements
        return np.random.normal(0, self._repeatability)

    def get_current_pose(self) -> EndEffectorPose:
        """Get current end effector pose."""
        return self._current_pose

    def get_current_configuration(self) -> RobotConfiguration:
        """Get current joint configuration."""
        return self._configuration

    def calculate_path_time(self, target: EndEffectorPose) -> float:
        """Calculate time to move to target pose."""
        target_config = self._inverse_kinematics(target)
        if target_config is None:
            return float('inf')

        current = self._configuration.to_array()
        target_arr = target_config.to_array()
        diff = np.abs(target_arr - current)

        # Use slowest joint as limiting factor
        max_time = 0.0
        joints = list(RobotJoint)
        for i, joint in enumerate(joints):
            velocity = self._joint_limits[joint].max_velocity
            time = diff[i] / velocity
            max_time = max(max_time, time)

        return max_time

    def get_status(self) -> Dict:
        """Get robot arm status."""
        return {
            "name": self.name,
            "is_calibrated": self._is_calibrated,
            "is_moving": self._is_moving,
            "collision_detected": self._collision_detected,
            "configuration": {
                "j1": self._configuration.j1,
                "j2": self._configuration.j2,
                "j3": self._configuration.j3,
                "j4": self._configuration.j4,
                "j5": self._configuration.j5,
                "j6": self._configuration.j6,
            },
            "pose": {
                "position": {
                    "x": self._current_pose.position.x,
                    "y": self._current_pose.position.y,
                    "z": self._current_pose.position.z,
                },
                "direction": {
                    "dx": self._current_pose.orientation.dx,
                    "dy": self._current_pose.orientation.dy,
                    "dz": self._current_pose.orientation.dz,
                },
            },
            "positioning_accuracy": self._positioning_accuracy,
        }


class TrackingModality(Enum):
    """Motion tracking modalities."""
    XRAY_STEREO = "xray_stereo"  # Dual X-ray imaging
    FIDUCIAL = "fiducial"  # Implanted fiducials
    SPINE = "spine"  # Spine tracking
    LUNG = "lung"  # Lung tracking (synchrony)
    SURFACE = "surface"  # Surface optical tracking
    SKULL = "skull"  # Skull tracking for cranial SRS


@dataclass
class MotionSample:
    """A single motion measurement sample."""
    timestamp: float  # seconds
    position: Position3D
    correlation_signal: float = 0.0  # External marker signal


class MotionCompensation:
    """
    Real-time motion compensation system.

    Tracks target motion and adjusts beam delivery in real-time.
    Uses correlation between external markers and internal tumor position.
    """

    def __init__(
        self,
        name: str = "Motion Compensation System",
        update_rate: float = 25.0  # Hz
    ):
        self.name = name
        self.update_rate = update_rate
        self._tracking_modality = TrackingModality.XRAY_STEREO

        # Motion history
        self._motion_history: List[MotionSample] = []
        self._correlation_model: Optional[Callable] = None

        # External marker tracking
        self._external_markers: List[Position3D] = []
        self._correlation_coefficient = 0.0

        # Current state
        self._target_position = Position3D()
        self._predicted_position = Position3D()
        self._is_tracking = False
        self._is_compensating = False

        # Compensation parameters
        self._compensation_latency = 0.1  # seconds (system delay)
        self._prediction_horizon = 0.15  # seconds ahead to predict
        self._motion_tolerance = 3.0  # mm, beam gating threshold

        # Breathing model
        self._breathing_period = 4.0  # seconds
        self._breathing_amplitude = Position3D()

    def set_tracking_modality(self, modality: TrackingModality) -> bool:
        """Set the tracking modality."""
        self._tracking_modality = modality
        return True

    def acquire_reference(self, num_samples: int = 100) -> bool:
        """
        Acquire reference position and build motion model.

        Args:
            num_samples: Number of samples for model building
        """
        self._motion_history = []

        # Simulate reference acquisition
        for i in range(num_samples):
            timestamp = i / self.update_rate

            # Simulate breathing motion
            phase = 2 * math.pi * timestamp / self._breathing_period
            position = Position3D(
                x=10.0 * math.sin(phase),  # Lateral motion
                y=5.0 * math.sin(phase),  # Longitudinal motion
                z=15.0 * math.sin(phase)  # Vertical (primary breathing motion)
            )

            sample = MotionSample(
                timestamp=timestamp,
                position=position,
                correlation_signal=math.sin(phase)  # External marker
            )
            self._motion_history.append(sample)

        # Build correlation model
        self._build_correlation_model()

        return True

    def _build_correlation_model(self):
        """Build correlation model between external markers and tumor position."""
        if len(self._motion_history) < 10:
            return

        # Extract data
        external = np.array([s.correlation_signal for s in self._motion_history])
        internal_z = np.array([s.position.z for s in self._motion_history])

        # Calculate correlation coefficient
        if np.std(external) > 0 and np.std(internal_z) > 0:
            self._correlation_coefficient = np.corrcoef(external, internal_z)[0, 1]

        # Estimate breathing amplitude
        positions = np.array([
            [s.position.x, s.position.y, s.position.z]
            for s in self._motion_history
        ])
        self._breathing_amplitude = Position3D(
            x=float(np.ptp(positions[:, 0])) / 2,
            y=float(np.ptp(positions[:, 1])) / 2,
            z=float(np.ptp(positions[:, 2])) / 2
        )

        # Simple linear model for prediction
        def predict_position(external_signal: float) -> Position3D:
            # Scale factor from correlation
            scale = self._breathing_amplitude.z / 2 if self._correlation_coefficient > 0 else 0
            return Position3D(
                x=external_signal * self._breathing_amplitude.x,
                y=external_signal * self._breathing_amplitude.y,
                z=external_signal * scale * self._correlation_coefficient
            )

        self._correlation_model = predict_position

    def start_tracking(self) -> bool:
        """Start real-time motion tracking."""
        if not self._motion_history:
            return False

        self._is_tracking = True
        return True

    def stop_tracking(self) -> bool:
        """Stop motion tracking."""
        self._is_tracking = False
        return True

    def start_compensation(self) -> bool:
        """Start motion compensation (beam tracking)."""
        if not self._is_tracking:
            return False

        self._is_compensating = True
        return True

    def stop_compensation(self) -> bool:
        """Stop motion compensation."""
        self._is_compensating = False
        return True

    def update_position(self, external_signal: float) -> Position3D:
        """
        Update target position based on external signal.

        Args:
            external_signal: Current external marker signal

        Returns:
            Current target position
        """
        if not self._is_tracking:
            return self._target_position

        # Get position from correlation model
        if self._correlation_model:
            self._target_position = self._correlation_model(external_signal)

        # Add to history
        timestamp = len(self._motion_history) / self.update_rate
        sample = MotionSample(
            timestamp=timestamp,
            position=self._target_position,
            correlation_signal=external_signal
        )
        self._motion_history.append(sample)

        return self._target_position

    def predict_position(self, time_ahead: float) -> Position3D:
        """
        Predict target position at future time.

        Uses motion model to predict position accounting for
        system latency and beam delivery time.
        """
        if len(self._motion_history) < 3:
            return self._target_position

        # Get recent samples
        recent = self._motion_history[-10:]
        times = np.array([s.timestamp for s in recent])
        positions = np.array([
            [s.position.x, s.position.y, s.position.z]
            for s in recent
        ])

        # Fit sinusoidal model (for breathing)
        # Simplified: use linear velocity extrapolation
        if len(recent) >= 2:
            dt = times[-1] - times[-2]
            if dt > 0:
                velocity = (positions[-1] - positions[-2]) / dt
                future_pos = positions[-1] + velocity * time_ahead
                self._predicted_position = Position3D(
                    x=float(future_pos[0]),
                    y=float(future_pos[1]),
                    z=float(future_pos[2])
                )
            else:
                self._predicted_position = self._target_position

        return self._predicted_position

    def get_compensation_offset(self) -> Position3D:
        """
        Get the compensation offset for beam aiming.

        Returns offset to apply to beam position to hit moving target.
        """
        if not self._is_compensating:
            return Position3D()

        # Predict where target will be when beam arrives
        total_delay = self._compensation_latency + self._prediction_horizon
        predicted = self.predict_position(total_delay)

        # Return offset from reference (assumed at origin)
        return predicted

    def is_motion_within_tolerance(self) -> bool:
        """Check if current motion is within gating tolerance."""
        displacement = math.sqrt(
            self._target_position.x ** 2 +
            self._target_position.y ** 2 +
            self._target_position.z ** 2
        )
        return displacement <= self._motion_tolerance

    def get_gating_signal(self) -> bool:
        """Get beam gating signal (True = beam can be on)."""
        return self.is_motion_within_tolerance()

    def get_motion_statistics(self) -> Dict:
        """Calculate motion statistics from tracking data."""
        if not self._motion_history:
            return {}

        positions = np.array([
            [s.position.x, s.position.y, s.position.z]
            for s in self._motion_history
        ])

        return {
            "mean_position": positions.mean(axis=0).tolist(),
            "amplitude": {
                "x": float(np.ptp(positions[:, 0])) / 2,
                "y": float(np.ptp(positions[:, 1])) / 2,
                "z": float(np.ptp(positions[:, 2])) / 2,
            },
            "std": {
                "x": float(np.std(positions[:, 0])),
                "y": float(np.std(positions[:, 1])),
                "z": float(np.std(positions[:, 2])),
            },
            "correlation_coefficient": self._correlation_coefficient,
            "num_samples": len(self._motion_history),
        }

    def get_status(self) -> Dict:
        """Get motion compensation status."""
        return {
            "name": self.name,
            "tracking_modality": self._tracking_modality.value,
            "is_tracking": self._is_tracking,
            "is_compensating": self._is_compensating,
            "update_rate": self.update_rate,
            "current_position": {
                "x": self._target_position.x,
                "y": self._target_position.y,
                "z": self._target_position.z,
            },
            "within_tolerance": self.is_motion_within_tolerance(),
            "motion_tolerance": self._motion_tolerance,
        }


@dataclass
class FiducialMarker:
    """Implanted fiducial marker for tracking."""
    marker_id: str
    position: Position3D  # Position in patient coordinates
    visible: bool = True


class ImageGuidedTargeting:
    """
    Image-guided targeting system.

    Uses stereoscopic X-ray imaging to track target position
    by identifying skeletal landmarks or implanted fiducials.
    """

    def __init__(
        self,
        name: str = "Image Guided Targeting",
        imaging_geometry: str = "stereo_45deg"
    ):
        self.name = name
        self.imaging_geometry = imaging_geometry

        # X-ray imaging parameters
        self._kv_source_1 = 120.0  # kV
        self._kv_source_2 = 120.0  # kV
        self._ma_1 = 50.0  # mA
        self._ma_2 = 50.0  # mA
        self._detector_size = (400, 400)  # pixels
        self._pixel_size = 0.388  # mm

        # Fiducials
        self._fiducials: List[FiducialMarker] = []

        # Reference images (DRRs - Digitally Reconstructed Radiographs)
        self._drr_1: Optional[np.ndarray] = None
        self._drr_2: Optional[np.ndarray] = None

        # Current images
        self._current_image_1: Optional[np.ndarray] = None
        self._current_image_2: Optional[np.ndarray] = None

        # Targeting state
        self._target_position = Position3D()
        self._registration_error = 0.0
        self._is_registered = False

        # Imaging schedule
        self._image_interval = 30.0  # seconds between images during treatment

    def add_fiducial(
        self,
        marker_id: str,
        position: Position3D
    ) -> bool:
        """Add fiducial marker to tracking list."""
        fiducial = FiducialMarker(
            marker_id=marker_id,
            position=position
        )
        self._fiducials.append(fiducial)
        return True

    def generate_drrs(self, ct_data: Optional[np.ndarray] = None) -> bool:
        """
        Generate reference DRRs from CT data.

        In real systems, this ray-traces through the CT volume
        to create synthetic X-ray images from treatment position.
        """
        # Simulate DRR generation
        size = self._detector_size
        self._drr_1 = np.random.random(size) * 100 + 1000  # Simulate CT values
        self._drr_2 = np.random.random(size) * 100 + 1000

        # Add fiducial projections
        for fiducial in self._fiducials:
            # Project fiducial to detector plane (simplified)
            u1, v1 = self._project_point(fiducial.position, 1)
            u2, v2 = self._project_point(fiducial.position, 2)

            if 0 <= u1 < size[0] and 0 <= v1 < size[1]:
                self._add_fiducial_to_image(self._drr_1, u1, v1)
            if 0 <= u2 < size[0] and 0 <= v2 < size[1]:
                self._add_fiducial_to_image(self._drr_2, u2, v2)

        return True

    def _project_point(
        self,
        point: Position3D,
        camera: int
    ) -> Tuple[int, int]:
        """Project 3D point to 2D detector coordinates."""
        # Simplified pinhole camera projection
        # Cameras at 45 degrees to patient
        source_distance = 1000.0  # mm
        detector_distance = 1500.0  # mm

        if camera == 1:
            angle = math.radians(-45)
        else:
            angle = math.radians(45)

        # Rotate point
        x_rot = point.x * math.cos(angle) + point.z * math.sin(angle)
        z_rot = -point.x * math.sin(angle) + point.z * math.cos(angle)

        # Project
        scale = detector_distance / (source_distance - z_rot)
        u = int(self._detector_size[0] / 2 + x_rot * scale / self._pixel_size)
        v = int(self._detector_size[1] / 2 + point.y * scale / self._pixel_size)

        return u, v

    def _add_fiducial_to_image(
        self,
        image: np.ndarray,
        u: int,
        v: int,
        size: int = 5
    ):
        """Add fiducial marker appearance to image."""
        for du in range(-size, size + 1):
            for dv in range(-size, size + 1):
                if (du ** 2 + dv ** 2) <= size ** 2:
                    if 0 <= u + du < image.shape[0] and 0 <= v + dv < image.shape[1]:
                        image[u + du, v + dv] = 3000  # High intensity

    def acquire_images(self) -> bool:
        """Acquire stereoscopic X-ray images."""
        size = self._detector_size

        # Simulate image acquisition with noise
        self._current_image_1 = self._drr_1 + np.random.normal(0, 50, size)
        self._current_image_2 = self._drr_2 + np.random.normal(0, 50, size)

        return True

    def register_images(self) -> Tuple[Position3D, float]:
        """
        Register current images to reference DRRs.

        Uses 2D-3D registration to find patient position offset.

        Returns:
            Tuple of (position offset, registration error in mm)
        """
        if self._current_image_1 is None or self._drr_1 is None:
            return (Position3D(), float('inf'))

        # Simplified registration using cross-correlation
        # Real systems use gradient-based optimization

        # Detect fiducials in current images
        detected_1 = self._detect_fiducials(self._current_image_1)
        detected_2 = self._detect_fiducials(self._current_image_2)

        if not detected_1 or not detected_2:
            return (Position3D(), float('inf'))

        # Triangulate 3D positions
        measured_positions = self._triangulate_fiducials(detected_1, detected_2)

        # Calculate offset from reference
        if self._fiducials and measured_positions:
            offset = Position3D(
                x=measured_positions[0].x - self._fiducials[0].position.x,
                y=measured_positions[0].y - self._fiducials[0].position.y,
                z=measured_positions[0].z - self._fiducials[0].position.z
            )

            # Calculate registration error (TRE - Target Registration Error)
            tre = math.sqrt(
                offset.x ** 2 + offset.y ** 2 + offset.z ** 2
            )

            self._target_position = offset
            self._registration_error = tre
            self._is_registered = tre < 2.0  # Accept if < 2mm

            return (offset, tre)

        return (Position3D(), float('inf'))

    def _detect_fiducials(
        self,
        image: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Detect fiducial markers in image."""
        # Simplified: find brightest spots
        threshold = 2500
        positions = []

        # Find local maxima above threshold
        for i in range(5, image.shape[0] - 5):
            for j in range(5, image.shape[1] - 5):
                if image[i, j] > threshold:
                    # Check if local maximum
                    patch = image[i-2:i+3, j-2:j+3]
                    if image[i, j] >= patch.max():
                        positions.append((i, j))

        return positions

    def _triangulate_fiducials(
        self,
        positions_1: List[Tuple[int, int]],
        positions_2: List[Tuple[int, int]]
    ) -> List[Position3D]:
        """
        Triangulate 3D positions from stereo image positions.

        Uses epipolar geometry to reconstruct 3D positions.
        """
        if not positions_1 or not positions_2:
            return []

        result = []

        # For each fiducial (simplified: assume correspondence by order)
        for i in range(min(len(positions_1), len(positions_2))):
            u1, v1 = positions_1[i]
            u2, v2 = positions_2[i]

            # Back-project and find intersection
            # Simplified triangulation
            source_distance = 1000.0
            detector_distance = 1500.0

            # Convert to detector coordinates (mm)
            x1 = (u1 - self._detector_size[0] / 2) * self._pixel_size
            y1 = (v1 - self._detector_size[1] / 2) * self._pixel_size
            x2 = (u2 - self._detector_size[0] / 2) * self._pixel_size
            y2 = (v2 - self._detector_size[1] / 2) * self._pixel_size

            # Triangulate (simplified geometry)
            # Cameras at ±45 degrees
            z = source_distance * (x1 + x2) / (x1 - x2) * 0.5
            x = (x1 + x2) / 2 * (source_distance - z) / detector_distance
            y = (y1 + y2) / 2 * (source_distance - z) / detector_distance

            result.append(Position3D(x=x, y=y, z=z))

        return result

    def calculate_correction(self) -> Position3D:
        """Calculate position correction for beam targeting."""
        if not self._is_registered:
            return Position3D()

        # Return negative offset to correct position
        return Position3D(
            x=-self._target_position.x,
            y=-self._target_position.y,
            z=-self._target_position.z
        )

    def get_status(self) -> Dict:
        """Get targeting system status."""
        return {
            "name": self.name,
            "imaging_geometry": self.imaging_geometry,
            "num_fiducials": len(self._fiducials),
            "is_registered": self._is_registered,
            "registration_error": self._registration_error,
            "target_offset": {
                "x": self._target_position.x,
                "y": self._target_position.y,
                "z": self._target_position.z,
            },
            "kv_settings": {
                "source_1_kv": self._kv_source_1,
                "source_2_kv": self._kv_source_2,
            },
        }


@dataclass
class TreatmentNode:
    """A single node (beam position) in CyberKnife treatment plan."""
    node_id: int
    source_position: Position3D
    target_position: Position3D
    monitor_units: float
    beam_direction: Vector3D
    dwell_time: float = 0.0  # seconds


class CyberKnifeSystem(BeamDeliverySystem):
    """
    Complete CyberKnife-type Robotic Radiosurgery System.

    Integrates:
    - Compact LINAC (6 MV X-band)
    - 6-DOF robotic arm
    - Real-time motion compensation
    - Image-guided targeting
    """

    def __init__(
        self,
        name: str = "CyberKnife Robotic Radiosurgery System",
        beam_energy: float = 6.0  # MV (fixed)
    ):
        super().__init__(name)
        self.beam_energy = beam_energy

        # Initialize components
        self.robotic_arm = RoboticArm()
        self.motion_compensation = MotionCompensation()
        self.image_guided_targeting = ImageGuidedTargeting()

        # LINAC parameters (compact X-band)
        self._linac_dose_rate = 800.0  # MU/min (high for efficiency)
        self._field_diameter = 60.0  # mm (circular collimator)
        self._available_collimators = [5, 7.5, 10, 12.5, 15, 20, 25, 30, 35, 40, 50, 60]  # mm

        # Treatment plan
        self._treatment_nodes: List[TreatmentNode] = []
        self._current_node_index = 0

        # Operating state
        self._is_ready = False
        self._is_treating = False
        self._beam_on = False
        self._current_collimator = 60.0
        self._total_mu_delivered = 0.0

    def initialize(self) -> bool:
        """Initialize the CyberKnife system."""
        success = True
        success &= self.robotic_arm.initialize()
        self._is_ready = success
        return success

    def calibrate(self) -> bool:
        """Calibrate the system."""
        self.robotic_arm.calibrate()
        self._is_calibrated = True
        return True

    def set_position(self, position: Position3D) -> bool:
        """Set robot source position."""
        pose = EndEffectorPose(
            position=position,
            orientation=Vector3D(0, 0, -1)  # Default: aim down
        )
        return self.robotic_arm.move_to_pose(pose)

    def set_collimator(self, diameter: float) -> bool:
        """Set collimator diameter."""
        if diameter not in self._available_collimators:
            # Find nearest available size
            diameter = min(
                self._available_collimators,
                key=lambda x: abs(x - diameter)
            )

        self._current_collimator = diameter
        self._field_diameter = diameter
        return True

    def load_treatment_plan(
        self,
        nodes: List[TreatmentNode],
        target: TreatmentTarget
    ) -> bool:
        """Load treatment plan with beam positions."""
        if not nodes:
            return False

        self._treatment_nodes = nodes
        self._current_node_index = 0

        # Set up image guidance
        self.image_guided_targeting.add_fiducial(
            "Target_Center",
            target.center
        )
        self.image_guided_targeting.generate_drrs()

        return True

    def generate_treatment_nodes(
        self,
        target: TreatmentTarget,
        num_nodes: int = 100,
        min_source_distance: float = 650.0,
        max_source_distance: float = 1000.0
    ) -> List[TreatmentNode]:
        """
        Generate treatment nodes (beam positions) for target.

        Uses quasi-random distribution on sphere around target.
        """
        nodes = []
        golden_ratio = (1 + math.sqrt(5)) / 2

        for i in range(num_nodes):
            # Fibonacci sphere distribution
            theta = 2 * math.pi * i / golden_ratio
            phi = math.acos(1 - 2 * (i + 0.5) / num_nodes)

            # Random distance within range
            distance = min_source_distance + (
                np.random.random() * (max_source_distance - min_source_distance)
            )

            # Source position on sphere around target
            source_pos = Position3D(
                x=target.center.x + distance * math.sin(phi) * math.cos(theta),
                y=target.center.y + distance * math.sin(phi) * math.sin(theta),
                z=target.center.z + distance * math.cos(phi)
            )

            # Beam direction toward target
            dx = target.center.x - source_pos.x
            dy = target.center.y - source_pos.y
            dz = target.center.z - source_pos.z
            dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

            direction = Vector3D(
                dx=dx / dist,
                dy=dy / dist,
                dz=dz / dist
            )

            node = TreatmentNode(
                node_id=i,
                source_position=source_pos,
                target_position=target.center,
                monitor_units=target.prescribed_dose / num_nodes * 100,  # MU
                beam_direction=direction
            )
            nodes.append(node)

        return nodes

    def optimize_node_weights(
        self,
        nodes: List[TreatmentNode],
        target: TreatmentTarget,
        oars: List[OrganAtRisk]
    ) -> List[float]:
        """
        Optimize beam weights for each node.

        Uses simplified inverse planning to achieve target coverage
        while minimizing OAR doses.
        """
        num_nodes = len(nodes)
        weights = np.ones(num_nodes)

        # Iterative optimization (simplified)
        for iteration in range(50):
            # Calculate dose at target center
            total_dose = sum(
                weights[i] * nodes[i].monitor_units * 0.01
                for i in range(num_nodes)
            )

            # Scale to prescription
            if total_dose > 0:
                scale = target.prescribed_dose / total_dose
                weights *= scale

            # Reduce weights for nodes that hit OARs
            for i, node in enumerate(nodes):
                for oar in oars:
                    if oar.contains_point(node.source_position):
                        weights[i] *= 0.5  # Reduce contribution

        # Update node MU values
        for i, node in enumerate(nodes):
            node.monitor_units *= weights[i]

        return weights.tolist()

    def perform_image_guidance(self) -> bool:
        """
        Perform image guidance check.

        Acquires X-ray images and calculates position correction.
        """
        self.image_guided_targeting.acquire_images()
        offset, error = self.image_guided_targeting.register_images()

        if error > 5.0:  # mm
            # Registration failed or large error
            return False

        # Apply correction to all nodes
        for node in self._treatment_nodes:
            node.target_position = Position3D(
                x=node.target_position.x + offset.x,
                y=node.target_position.y + offset.y,
                z=node.target_position.z + offset.z
            )

        return True

    def start_motion_tracking(self) -> bool:
        """Start motion compensation system."""
        self.motion_compensation.acquire_reference()
        self.motion_compensation.start_tracking()
        self.motion_compensation.start_compensation()
        return True

    def deliver_node(self, node: TreatmentNode) -> bool:
        """
        Deliver dose at a single treatment node.

        Includes real-time motion compensation.
        """
        # Move robot to node position
        target_pos = node.target_position

        # Apply motion compensation offset
        if self.motion_compensation._is_compensating:
            compensation = self.motion_compensation.get_compensation_offset()
            target_pos = Position3D(
                x=target_pos.x + compensation.x,
                y=target_pos.y + compensation.y,
                z=target_pos.z + compensation.z
            )

        # Aim robot at target
        if not self.robotic_arm.aim_at_target(node.source_position, target_pos):
            return False

        # Check if motion is within tolerance
        if not self.motion_compensation.get_gating_signal():
            # Wait for motion to be within tolerance
            return False

        # Deliver beam
        self._beam_on = True
        delivery_time = node.monitor_units / self._linac_dose_rate * 60  # seconds
        node.dwell_time = delivery_time

        self._total_mu_delivered += node.monitor_units
        self._beam_on = False

        return True

    def deliver_treatment(self) -> bool:
        """
        Deliver complete treatment.

        Executes all treatment nodes with image guidance
        and motion compensation.
        """
        if not self._treatment_nodes:
            return False

        self._is_treating = True

        # Perform initial image guidance
        if not self.perform_image_guidance():
            self._is_treating = False
            return False

        # Start motion tracking
        self.start_motion_tracking()

        # Deliver each node
        for i, node in enumerate(self._treatment_nodes):
            self._current_node_index = i

            # Periodic image guidance (every 30 nodes or so)
            if i > 0 and i % 30 == 0:
                self.perform_image_guidance()

            # Deliver node
            if not self.deliver_node(node):
                # Retry with updated motion compensation
                if not self.deliver_node(node):
                    continue  # Skip node if still failing

        self._is_treating = False
        self.motion_compensation.stop_compensation()
        self.motion_compensation.stop_tracking()

        return True

    def deliver_dose(
        self,
        beam_params: BeamParameters,
        monitor_units: float
    ) -> DoseDistribution:
        """Deliver specified dose (simplified for single beam)."""
        # Create dose distribution
        dose_dist = DoseDistribution(
            origin=Position3D(-100, -100, -100),
            spacing=(2.0, 2.0, 2.0),
            dimensions=(100, 100, 100)
        )

        pose = self.robotic_arm.get_current_pose()
        source = pose.position
        direction = pose.orientation

        # Calculate dose from pencil beam
        for i in range(dose_dist.dimensions[0]):
            for j in range(dose_dist.dimensions[1]):
                for k in range(dose_dist.dimensions[2]):
                    point = Position3D(
                        x=dose_dist.origin.x + i * dose_dist.spacing[0],
                        y=dose_dist.origin.y + j * dose_dist.spacing[1],
                        z=dose_dist.origin.z + k * dose_dist.spacing[2]
                    )

                    # Distance from source
                    dx = point.x - source.x
                    dy = point.y - source.y
                    dz = point.z - source.z
                    dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

                    if dist == 0:
                        continue

                    # Distance along beam axis
                    depth = dx * direction.dx + dy * direction.dy + dz * direction.dz

                    # Lateral distance from beam axis
                    lateral = math.sqrt(dist ** 2 - depth ** 2)

                    # Field size cutoff
                    if lateral > self._field_diameter / 2:
                        continue

                    # Depth dose (6 MV photons)
                    if depth > 0:
                        pdd = math.exp(-0.05 * depth) * (100 / (depth + 100)) ** 2
                        dose = monitor_units * 0.01 * pdd
                        dose_dist.dose_grid[i, j, k] += dose

        return dose_dist

    def calculate_treatment_time(self) -> float:
        """Calculate total treatment time."""
        if not self._treatment_nodes:
            return 0.0

        total_time = 0.0

        # Beam-on time
        total_mu = sum(n.monitor_units for n in self._treatment_nodes)
        beam_time = total_mu / self._linac_dose_rate * 60  # seconds

        # Robot motion time
        motion_time = 0.0
        for i in range(1, len(self._treatment_nodes)):
            prev = self._treatment_nodes[i - 1]
            curr = self._treatment_nodes[i]

            prev_pose = EndEffectorPose(
                position=prev.source_position,
                orientation=prev.beam_direction
            )
            curr_pose = EndEffectorPose(
                position=curr.source_position,
                orientation=curr.beam_direction
            )

            motion_time += self.robotic_arm.calculate_path_time(curr_pose)

        # Imaging time (assume 30 seconds per image check)
        num_image_checks = len(self._treatment_nodes) // 30 + 1
        imaging_time = num_image_checks * 30.0

        total_time = beam_time + motion_time + imaging_time

        return total_time

    def get_status(self) -> Dict:
        """Get complete system status."""
        return {
            "name": self.name,
            "beam_energy": self.beam_energy,
            "is_ready": self._is_ready,
            "is_calibrated": self._is_calibrated,
            "is_treating": self._is_treating,
            "beam_on": self._beam_on,
            "current_collimator": self._current_collimator,
            "total_mu_delivered": self._total_mu_delivered,
            "current_node": self._current_node_index,
            "total_nodes": len(self._treatment_nodes),
            "robotic_arm": self.robotic_arm.get_status(),
            "motion_compensation": self.motion_compensation.get_status(),
            "image_guidance": self.image_guided_targeting.get_status(),
        }
