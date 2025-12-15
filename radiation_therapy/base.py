"""
Base classes for Radiation Therapy Equipment simulation.

This module provides fundamental data structures and abstractions
used across all radiation therapy systems.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Callable
from abc import ABC, abstractmethod
import math
import numpy as np


@dataclass
class Position3D:
    """3D position in millimeters relative to isocenter."""
    x: float = 0.0  # mm, lateral (patient left/right)
    y: float = 0.0  # mm, longitudinal (patient head/feet)
    z: float = 0.0  # mm, vertical (patient anterior/posterior)

    def distance_to(self, other: "Position3D") -> float:
        """Calculate Euclidean distance to another position."""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Position3D":
        """Create Position3D from numpy array."""
        return cls(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))

    def __add__(self, other: "Position3D") -> "Position3D":
        return Position3D(
            x=self.x + other.x,
            y=self.y + other.y,
            z=self.z + other.z
        )

    def __sub__(self, other: "Position3D") -> "Position3D":
        return Position3D(
            x=self.x - other.x,
            y=self.y - other.y,
            z=self.z - other.z
        )


@dataclass
class Vector3D:
    """3D vector for direction and magnitude."""
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 1.0

    @property
    def magnitude(self) -> float:
        """Calculate vector magnitude."""
        return math.sqrt(self.dx ** 2 + self.dy ** 2 + self.dz ** 2)

    def normalize(self) -> "Vector3D":
        """Return normalized unit vector."""
        mag = self.magnitude
        if mag == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.dx / mag, self.dy / mag, self.dz / mag)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.dx, self.dy, self.dz])

    def dot(self, other: "Vector3D") -> float:
        """Calculate dot product with another vector."""
        return self.dx * other.dx + self.dy * other.dy + self.dz * other.dz

    def cross(self, other: "Vector3D") -> "Vector3D":
        """Calculate cross product with another vector."""
        return Vector3D(
            dx=self.dy * other.dz - self.dz * other.dy,
            dy=self.dz * other.dx - self.dx * other.dz,
            dz=self.dx * other.dy - self.dy * other.dx
        )


class ParticleType(Enum):
    """Types of particles used in radiation therapy."""
    PHOTON = "photon"
    ELECTRON = "electron"
    PROTON = "proton"
    CARBON_ION = "carbon_ion"
    HELIUM_ION = "helium_ion"


class BeamModality(Enum):
    """Beam delivery modalities."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    IMRT = "intensity_modulated"
    VMAT = "volumetric_modulated_arc"
    SBRT = "stereotactic_body"
    SRS = "stereotactic_radiosurgery"


@dataclass
class BeamParameters:
    """Parameters defining a radiation beam."""
    energy: float  # MeV for particles, MV for photons
    particle_type: ParticleType = ParticleType.PHOTON
    dose_rate: float = 600.0  # MU/min (monitor units per minute)
    field_size_x: float = 10.0  # cm
    field_size_y: float = 10.0  # cm
    gantry_angle: float = 0.0  # degrees
    collimator_angle: float = 0.0  # degrees
    couch_angle: float = 0.0  # degrees
    source_to_axis_distance: float = 100.0  # cm (SAD)
    modality: BeamModality = BeamModality.STATIC

    def calculate_inverse_square_factor(self, distance: float) -> float:
        """Calculate inverse square law factor for dose at given distance."""
        return (self.source_to_axis_distance / distance) ** 2


@dataclass
class TreatmentTarget:
    """Definition of a treatment target volume."""
    name: str
    center: Position3D
    dimensions: Tuple[float, float, float]  # (width, height, depth) in mm
    prescribed_dose: float  # Gy
    fractions: int = 1
    priority: int = 1
    margin_ctv: float = 0.0  # Clinical Target Volume margin in mm
    margin_ptv: float = 3.0  # Planning Target Volume margin in mm

    @property
    def dose_per_fraction(self) -> float:
        """Calculate dose per fraction."""
        return self.prescribed_dose / self.fractions

    @property
    def volume(self) -> float:
        """Calculate approximate volume in cubic mm."""
        # Assuming ellipsoid shape
        return (4/3) * math.pi * (
            self.dimensions[0] / 2 *
            self.dimensions[1] / 2 *
            self.dimensions[2] / 2
        )

    def contains_point(self, point: Position3D) -> bool:
        """Check if a point is within the target volume (ellipsoid approximation)."""
        dx = (point.x - self.center.x) / (self.dimensions[0] / 2 + self.margin_ptv)
        dy = (point.y - self.center.y) / (self.dimensions[1] / 2 + self.margin_ptv)
        dz = (point.z - self.center.z) / (self.dimensions[2] / 2 + self.margin_ptv)
        return (dx ** 2 + dy ** 2 + dz ** 2) <= 1.0


@dataclass
class OrganAtRisk:
    """Definition of an organ at risk (OAR)."""
    name: str
    center: Position3D
    dimensions: Tuple[float, float, float]  # mm
    max_dose: float  # Gy - maximum allowable dose
    mean_dose_constraint: Optional[float] = None  # Gy
    volume_constraint: Optional[Tuple[float, float]] = None  # (V%, dose Gy)
    priority: int = 2
    serial: bool = True  # Serial organs have dose limits, parallel have mean dose

    def contains_point(self, point: Position3D) -> bool:
        """Check if a point is within the OAR volume."""
        dx = (point.x - self.center.x) / (self.dimensions[0] / 2)
        dy = (point.y - self.center.y) / (self.dimensions[1] / 2)
        dz = (point.z - self.center.z) / (self.dimensions[2] / 2)
        return (dx ** 2 + dy ** 2 + dz ** 2) <= 1.0


@dataclass
class DosePoint:
    """A single dose measurement point."""
    position: Position3D
    dose: float  # Gy
    uncertainty: float = 0.0  # Gy


@dataclass
class DoseDistribution:
    """3D dose distribution over a volume."""
    origin: Position3D
    spacing: Tuple[float, float, float]  # mm spacing in x, y, z
    dimensions: Tuple[int, int, int]  # grid size in x, y, z
    dose_grid: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        if self.dose_grid.size == 0:
            self.dose_grid = np.zeros(self.dimensions)

    def get_dose_at(self, position: Position3D) -> float:
        """Get interpolated dose at a specific position."""
        # Convert position to grid indices
        i = int((position.x - self.origin.x) / self.spacing[0])
        j = int((position.y - self.origin.y) / self.spacing[1])
        k = int((position.z - self.origin.z) / self.spacing[2])

        # Check bounds
        if (0 <= i < self.dimensions[0] and
            0 <= j < self.dimensions[1] and
            0 <= k < self.dimensions[2]):
            return float(self.dose_grid[i, j, k])
        return 0.0

    def set_dose_at(self, position: Position3D, dose: float):
        """Set dose at a specific position."""
        i = int((position.x - self.origin.x) / self.spacing[0])
        j = int((position.y - self.origin.y) / self.spacing[1])
        k = int((position.z - self.origin.z) / self.spacing[2])

        if (0 <= i < self.dimensions[0] and
            0 <= j < self.dimensions[1] and
            0 <= k < self.dimensions[2]):
            self.dose_grid[i, j, k] = dose

    def get_max_dose(self) -> float:
        """Get maximum dose in the distribution."""
        return float(np.max(self.dose_grid))

    def get_mean_dose(self) -> float:
        """Get mean dose in the distribution."""
        return float(np.mean(self.dose_grid))

    def get_dose_volume_histogram(
        self,
        num_bins: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate dose-volume histogram."""
        max_dose = self.get_max_dose()
        bins = np.linspace(0, max_dose, num_bins + 1)
        hist, _ = np.histogram(self.dose_grid.flatten(), bins=bins)
        # Convert to cumulative (percentage of volume receiving at least dose)
        cumulative = np.cumsum(hist[::-1])[::-1]
        total_voxels = np.prod(self.dimensions)
        dvh = cumulative / total_voxels * 100
        return bins[:-1], dvh


class RadiationSource(ABC):
    """Abstract base class for radiation sources."""

    def __init__(
        self,
        name: str,
        max_energy: float,
        particle_type: ParticleType
    ):
        self.name = name
        self.max_energy = max_energy
        self.particle_type = particle_type
        self._is_active = False
        self._current_energy = 0.0
        self._beam_current = 0.0

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the radiation source."""
        pass

    @abstractmethod
    def set_energy(self, energy: float) -> bool:
        """Set the beam energy."""
        pass

    @abstractmethod
    def beam_on(self) -> bool:
        """Turn on the beam."""
        pass

    @abstractmethod
    def beam_off(self) -> bool:
        """Turn off the beam."""
        pass

    @property
    def is_active(self) -> bool:
        """Check if beam is currently active."""
        return self._is_active

    @property
    def current_energy(self) -> float:
        """Get current beam energy."""
        return self._current_energy


class BeamDeliverySystem(ABC):
    """Abstract base class for beam delivery systems."""

    def __init__(self, name: str):
        self.name = name
        self._position = Position3D()
        self._direction = Vector3D()
        self._is_calibrated = False

    @abstractmethod
    def calibrate(self) -> bool:
        """Calibrate the delivery system."""
        pass

    @abstractmethod
    def set_position(self, position: Position3D) -> bool:
        """Set the delivery position."""
        pass

    @abstractmethod
    def deliver_dose(
        self,
        beam_params: BeamParameters,
        monitor_units: float
    ) -> DoseDistribution:
        """Deliver radiation dose."""
        pass


class SafetyInterlock(ABC):
    """Abstract base class for safety interlock systems."""

    def __init__(self, name: str):
        self.name = name
        self._is_engaged = True
        self._fault_conditions: List[str] = []

    @abstractmethod
    def check_conditions(self) -> bool:
        """Check all safety conditions."""
        pass

    @abstractmethod
    def engage(self) -> bool:
        """Engage the interlock (stop beam)."""
        pass

    @abstractmethod
    def release(self) -> bool:
        """Release the interlock (allow beam)."""
        pass

    @property
    def fault_conditions(self) -> List[str]:
        """Get list of current fault conditions."""
        return self._fault_conditions.copy()


@dataclass
class TreatmentPlan:
    """A complete treatment plan."""
    plan_id: str
    patient_id: str
    targets: List[TreatmentTarget]
    organs_at_risk: List[OrganAtRisk]
    beams: List[BeamParameters]
    total_dose: float  # Gy
    total_fractions: int
    dose_distribution: Optional[DoseDistribution] = None
    approved: bool = False
    approval_date: Optional[str] = None

    def calculate_conformity_index(self) -> float:
        """
        Calculate conformity index (CI).
        CI = (Volume covered by prescription dose) / (Target volume)
        Ideal CI = 1.0
        """
        if self.dose_distribution is None:
            return 0.0

        # Simplified calculation
        prescription_dose = self.targets[0].prescribed_dose if self.targets else 0
        covered_voxels = np.sum(
            self.dose_distribution.dose_grid >= prescription_dose * 0.95
        )
        total_target_voxels = sum(
            t.volume / np.prod(self.dose_distribution.spacing)
            for t in self.targets
        )

        if total_target_voxels == 0:
            return 0.0
        return covered_voxels / total_target_voxels

    def calculate_homogeneity_index(self) -> float:
        """
        Calculate homogeneity index (HI).
        HI = (D2% - D98%) / D50%
        Lower HI indicates more homogeneous dose.
        """
        if self.dose_distribution is None:
            return float('inf')

        doses = self.dose_distribution.dose_grid.flatten()
        d2 = np.percentile(doses, 98)  # D2% (near max)
        d98 = np.percentile(doses, 2)  # D98% (near min)
        d50 = np.percentile(doses, 50)  # D50% (median)

        if d50 == 0:
            return float('inf')
        return (d2 - d98) / d50
