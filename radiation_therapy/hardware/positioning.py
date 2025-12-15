"""
Patient Positioning Systems.

Implements treatment couches and immobilization devices:
- Standard and 6-DOF treatment couches
- Immobilization devices (masks, frames, cushions)
- Indexing systems for reproducible setup
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Dict
from abc import ABC, abstractmethod
import math
import numpy as np

from ..base import Position3D, Vector3D


class CouchType(Enum):
    """Types of treatment couches."""
    STANDARD = "standard"  # 4 DOF
    SIX_DOF = "six_dof"  # 6 DOF (adds pitch, roll, yaw)
    ROBOTIC = "robotic"  # Fully robotic
    PROTON = "proton"  # Specialized for particle therapy


@dataclass
class CouchLimits:
    """Motion limits for treatment couch."""
    lateral: Tuple[float, float] = (-250.0, 250.0)  # mm
    longitudinal: Tuple[float, float] = (-500.0, 1500.0)  # mm
    vertical: Tuple[float, float] = (-100.0, 500.0)  # mm
    rotation: Tuple[float, float] = (-185.0, 185.0)  # degrees
    pitch: Tuple[float, float] = (-3.0, 3.0)  # degrees (6DOF only)
    roll: Tuple[float, float] = (-3.0, 3.0)  # degrees (6DOF only)


@dataclass
class CouchPosition:
    """Complete couch position (6 DOF)."""
    lateral: float = 0.0  # mm (+ = left)
    longitudinal: float = 0.0  # mm (+ = toward gantry)
    vertical: float = 0.0  # mm (+ = up)
    rotation: float = 0.0  # degrees (IEC)
    pitch: float = 0.0  # degrees
    roll: float = 0.0  # degrees

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.lateral, self.longitudinal, self.vertical,
            self.rotation, self.pitch, self.roll
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "CouchPosition":
        """Create from array."""
        return cls(
            lateral=arr[0],
            longitudinal=arr[1],
            vertical=arr[2],
            rotation=arr[3] if len(arr) > 3 else 0.0,
            pitch=arr[4] if len(arr) > 4 else 0.0,
            roll=arr[5] if len(arr) > 5 else 0.0
        )


class TreatmentCouch(ABC):
    """Abstract base class for treatment couches."""

    def __init__(
        self,
        name: str,
        couch_type: CouchType,
        max_weight: float = 200.0
    ):
        self.name = name
        self.couch_type = couch_type
        self.max_weight = max_weight
        self.limits = CouchLimits()
        self._current_position = CouchPosition()
        self._is_calibrated = False
        self._indexing_bar = None

    @abstractmethod
    def move_to(self, position: CouchPosition) -> bool:
        """Move couch to specified position."""
        pass

    @abstractmethod
    def get_position(self) -> CouchPosition:
        """Get current couch position."""
        pass

    def move_relative(
        self,
        delta_lat: float = 0.0,
        delta_long: float = 0.0,
        delta_vert: float = 0.0,
        delta_rot: float = 0.0
    ) -> bool:
        """Move couch by relative amounts."""
        new_pos = CouchPosition(
            lateral=self._current_position.lateral + delta_lat,
            longitudinal=self._current_position.longitudinal + delta_long,
            vertical=self._current_position.vertical + delta_vert,
            rotation=self._current_position.rotation + delta_rot,
            pitch=self._current_position.pitch,
            roll=self._current_position.roll
        )
        return self.move_to(new_pos)

    def is_within_limits(self, position: CouchPosition) -> bool:
        """Check if position is within motion limits."""
        return (
            self.limits.lateral[0] <= position.lateral <= self.limits.lateral[1] and
            self.limits.longitudinal[0] <= position.longitudinal <= self.limits.longitudinal[1] and
            self.limits.vertical[0] <= position.vertical <= self.limits.vertical[1] and
            self.limits.rotation[0] <= position.rotation <= self.limits.rotation[1]
        )

    def calculate_isocenter_shift(
        self,
        couch_shift: CouchPosition
    ) -> Position3D:
        """
        Calculate isocenter position change from couch movement.

        Accounts for couch rotation effects.
        """
        # For rotated couch, lateral/longitudinal map differently
        rot_rad = math.radians(self._current_position.rotation)

        iso_x = (
            couch_shift.lateral * math.cos(rot_rad) -
            couch_shift.longitudinal * math.sin(rot_rad)
        )
        iso_y = (
            couch_shift.lateral * math.sin(rot_rad) +
            couch_shift.longitudinal * math.cos(rot_rad)
        )
        iso_z = couch_shift.vertical

        return Position3D(x=-iso_x, y=-iso_y, z=iso_z)


class SixDOFCouch(TreatmentCouch):
    """
    6 Degrees of Freedom treatment couch.

    Provides corrections for:
    - 3 translations (lateral, longitudinal, vertical)
    - 3 rotations (pitch, roll, yaw/rotation)
    """

    def __init__(
        self,
        name: str = "6DOF Couch",
        max_weight: float = 200.0
    ):
        super().__init__(name, CouchType.SIX_DOF, max_weight)

        # 6DOF limits
        self.limits.pitch = (-3.0, 3.0)  # degrees
        self.limits.roll = (-3.0, 3.0)  # degrees

        # Accuracy specifications
        self._translation_accuracy = 0.5  # mm
        self._rotation_accuracy = 0.1  # degrees

        # Correction capabilities
        self._can_correct_pitch = True
        self._can_correct_roll = True
        self._can_correct_yaw = True

    def move_to(self, position: CouchPosition) -> bool:
        """Move couch to 6DOF position."""
        # Check all limits including rotations
        if not self.is_within_limits(position):
            return False

        if not (self.limits.pitch[0] <= position.pitch <= self.limits.pitch[1]):
            return False
        if not (self.limits.roll[0] <= position.roll <= self.limits.roll[1]):
            return False

        self._current_position = position
        return True

    def get_position(self) -> CouchPosition:
        """Get current 6DOF position."""
        return self._current_position

    def apply_6dof_correction(
        self,
        translation: Position3D,
        rotation: Dict[str, float]
    ) -> bool:
        """
        Apply 6DOF correction from image guidance.

        Args:
            translation: Translation correction (mm)
            rotation: Rotation correction {"pitch": deg, "roll": deg, "yaw": deg}
        """
        new_pos = CouchPosition(
            lateral=self._current_position.lateral - translation.x,
            longitudinal=self._current_position.longitudinal - translation.y,
            vertical=self._current_position.vertical + translation.z,
            rotation=self._current_position.rotation + rotation.get("yaw", 0.0),
            pitch=self._current_position.pitch + rotation.get("pitch", 0.0),
            roll=self._current_position.roll + rotation.get("roll", 0.0)
        )
        return self.move_to(new_pos)


class HexaPOD(SixDOFCouch):
    """
    Elekta HexaPOD evo RT couch.

    Stewart platform design with 6 actuators.
    """

    def __init__(self):
        super().__init__("Elekta HexaPOD evo RT", max_weight=200.0)

        # HexaPOD specifications
        self._actuators = 6
        self._platform_type = "stewart"

        # Motion ranges
        self.limits.lateral = (-30.0, 30.0)  # mm (fine positioning)
        self.limits.longitudinal = (-30.0, 30.0)
        self.limits.vertical = (-30.0, 30.0)
        self.limits.rotation = (-3.0, 3.0)  # degrees

        # Additional HexaPOD limits
        self.limits.pitch = (-3.0, 3.0)
        self.limits.roll = (-3.0, 3.0)

        # Note: Coarse positioning via standard couch
        self._coarse_couch = None


class RoboticCouch(TreatmentCouch):
    """
    Fully robotic treatment couch.

    Used in CyberKnife and some proton therapy systems.
    """

    def __init__(
        self,
        name: str = "Robotic Couch",
        max_weight: float = 180.0
    ):
        super().__init__(name, CouchType.ROBOTIC, max_weight)

        # Extended motion ranges
        self.limits.lateral = (-500.0, 500.0)
        self.limits.longitudinal = (-1000.0, 1000.0)
        self.limits.vertical = (-200.0, 600.0)

        # Full rotation
        self.limits.rotation = (-180.0, 180.0)
        self.limits.pitch = (-10.0, 10.0)
        self.limits.roll = (-5.0, 5.0)

        # Speed
        self._max_speed = 50.0  # mm/s
        self._rotation_speed = 5.0  # deg/s

    def move_to(self, position: CouchPosition) -> bool:
        """Move robotic couch to position."""
        if not self.is_within_limits(position):
            return False

        self._current_position = position
        return True

    def get_position(self) -> CouchPosition:
        """Get current position."""
        return self._current_position

    def calculate_motion_time(self, target: CouchPosition) -> float:
        """Calculate time to reach target position."""
        current = self._current_position.to_array()
        target_arr = target.to_array()

        # Translation time
        trans_distance = np.sqrt(np.sum((target_arr[:3] - current[:3]) ** 2))
        trans_time = trans_distance / self._max_speed

        # Rotation time
        rot_diff = np.abs(target_arr[3:] - current[3:])
        rot_time = np.max(rot_diff) / self._rotation_speed

        return max(trans_time, rot_time)


class ImmobilizationDevice(ABC):
    """Abstract base class for immobilization devices."""

    def __init__(
        self,
        name: str,
        body_region: str
    ):
        self.name = name
        self.body_region = body_region
        self._indexed = False
        self._index_values = {}

    @abstractmethod
    def setup(self, patient_id: str) -> bool:
        """Set up device for patient."""
        pass

    def set_index_values(self, values: Dict[str, float]):
        """Set indexing values for reproducible setup."""
        self._index_values = values
        self._indexed = True

    def get_index_values(self) -> Dict[str, float]:
        """Get indexing values."""
        return self._index_values


class HeadAndNeckMask(ImmobilizationDevice):
    """
    Thermoplastic head and neck immobilization mask.

    Types:
    - Open face (for IGRT)
    - Closed face (for SRS)
    - S-frame (with shoulder fixation)
    """

    def __init__(
        self,
        mask_type: str = "open_face",
        with_shoulder: bool = False
    ):
        super().__init__("Head and Neck Mask", "head_neck")

        self.mask_type = mask_type
        self.with_shoulder = with_shoulder

        # Mask specifications
        self._material = "thermoplastic"
        self._thickness = 2.5  # mm
        self._perforation = "2.4mm" if mask_type == "open_face" else "1.6mm"

        # Setup reproducibility
        self._setup_accuracy = 2.0 if mask_type == "open_face" else 1.0  # mm

        # S-frame for shoulder
        self._s_frame = with_shoulder

    def setup(self, patient_id: str) -> bool:
        """Set up mask for patient."""
        return True

    def mold_mask(self, temperature: float = 70.0) -> bool:
        """Mold mask to patient (simulation)."""
        if temperature < 65 or temperature > 75:
            return False
        return True


class BodyFrame(ImmobilizationDevice):
    """
    Stereotactic body frame for SBRT.

    Provides abdominal compression and indexed setup.
    """

    def __init__(
        self,
        manufacturer: str = "Elekta",
        compression: bool = True
    ):
        super().__init__("Stereotactic Body Frame", "thorax_abdomen")

        self.manufacturer = manufacturer
        self.has_compression = compression

        # Frame specifications
        if manufacturer.lower() == "elekta":
            self._type = "BodyFix"
            self._vacuum_bag = True
            self._compression_plate = compression
        else:
            self._type = "Generic"
            self._vacuum_bag = True
            self._compression_plate = compression

        # Accuracy
        self._setup_accuracy = 3.0  # mm

    def setup(self, patient_id: str) -> bool:
        """Set up body frame."""
        return True

    def set_compression(self, pressure: float) -> bool:
        """Set abdominal compression pressure (mbar)."""
        if not self.has_compression:
            return False
        if pressure < 0 or pressure > 100:
            return False
        self._compression_pressure = pressure
        return True


class VacuumCushion(ImmobilizationDevice):
    """
    Vacuum cushion for patient positioning.

    Molds to patient shape when evacuated.
    """

    def __init__(
        self,
        size: str = "large",
        body_region: str = "whole_body"
    ):
        super().__init__("Vacuum Cushion", body_region)

        self.size = size

        # Size specifications
        sizes = {
            "small": (400, 600),
            "medium": (500, 800),
            "large": (600, 1000),
            "extra_large": (700, 1200)
        }
        self._dimensions = sizes.get(size, sizes["large"])

        self._is_evacuated = False
        self._vacuum_level = 0.0  # mbar below atmospheric

    def setup(self, patient_id: str) -> bool:
        """Set up vacuum cushion."""
        return True

    def evacuate(self, target_vacuum: float = -300.0) -> bool:
        """Evacuate cushion to mold to patient."""
        if target_vacuum > 0 or target_vacuum < -500:
            return False

        self._vacuum_level = target_vacuum
        self._is_evacuated = True
        return True

    def release(self) -> bool:
        """Release vacuum."""
        self._vacuum_level = 0.0
        self._is_evacuated = False
        return True


class BellyBoard(ImmobilizationDevice):
    """
    Belly board for prone positioning.

    Used to reduce small bowel dose in pelvic RT.
    """

    def __init__(self):
        super().__init__("Belly Board", "pelvis")

        # Board specifications
        self._aperture_size = (180.0, 150.0)  # mm
        self._board_angle = 15.0  # degrees
        self._has_head_rest = True

    def setup(self, patient_id: str) -> bool:
        """Set up belly board."""
        return True


class BreastBoard(ImmobilizationDevice):
    """
    Breast board for breast cancer treatment.

    Provides arm positioning and torso tilt.
    """

    def __init__(
        self,
        side: str = "left",
        manufacturer: str = "CIVCO"
    ):
        super().__init__("Breast Board", "breast")

        self.side = side
        self.manufacturer = manufacturer

        # Board angles
        self._arm_angle = 90.0  # degrees (arm abduction)
        self._board_angle = 10.0  # degrees (thorax tilt)

        # Adjustments
        self._adjustable_arm_support = True
        self._adjustable_hand_grip = True

    def setup(self, patient_id: str) -> bool:
        """Set up breast board."""
        return True

    def set_arm_angle(self, angle: float) -> bool:
        """Set arm abduction angle."""
        if angle < 60 or angle > 130:
            return False
        self._arm_angle = angle
        return True

    def set_tilt_angle(self, angle: float) -> bool:
        """Set board tilt angle."""
        if angle < 0 or angle > 25:
            return False
        self._board_angle = angle
        return True


@dataclass
class IndexingPoint:
    """A single indexing reference point."""
    name: str
    bar_position: float  # mm or position number
    lateral_offset: float = 0.0  # mm
    vertical_offset: float = 0.0  # mm


class Indexing:
    """
    Indexing system for reproducible patient setup.

    Provides reference points and bar positions for consistent positioning.
    """

    def __init__(
        self,
        manufacturer: str = "CIVCO",
        bar_type: str = "standard"
    ):
        self.manufacturer = manufacturer
        self.bar_type = bar_type

        # Indexing bar specifications
        self._bar_length = 1000.0  # mm
        self._marking_interval = 10.0  # mm
        self._reference_points: List[IndexingPoint] = []

        # Couch interface
        self._couch_slot = True

    def add_reference_point(self, point: IndexingPoint):
        """Add indexing reference point."""
        self._reference_points.append(point)

    def get_setup_position(self) -> Dict[str, float]:
        """Get position for reproducible setup."""
        if not self._reference_points:
            return {}

        primary = self._reference_points[0]
        return {
            "bar_position": primary.bar_position,
            "lateral_offset": primary.lateral_offset,
            "vertical_offset": primary.vertical_offset,
        }

    def verify_setup(
        self,
        measured_position: float,
        tolerance: float = 2.0
    ) -> bool:
        """Verify patient setup against indexed position."""
        if not self._reference_points:
            return False

        expected = self._reference_points[0].bar_position
        return abs(measured_position - expected) <= tolerance
