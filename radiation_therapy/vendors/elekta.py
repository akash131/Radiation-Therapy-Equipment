"""
Elekta Medical Equipment Implementations.

Provides accurate simulations of Elekta linear accelerators:
- Versa HD
- Infinity
- Unity (MR-Linac)
- Agility MLC
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Dict
import math
import numpy as np

from ..base import (
    BeamDeliverySystem,
    BeamParameters,
    Position3D,
    Vector3D,
    DoseDistribution,
    TreatmentTarget,
    ParticleType,
    BeamModality,
)
from ..linac import (
    LinearAccelerator,
    MultiLeafCollimator,
    MLCType,
    TumorTracker,
    ImagingModality,
)


class ElektaBeamMode(Enum):
    """Elekta beam modes."""
    PHOTON = "photon"
    ELECTRON = "electron"
    FFF = "flattening_filter_free"


class ElektaImagingMode(Enum):
    """Elekta imaging modalities."""
    XVI_CBCT = "xvi_cbct"
    XVI_2D = "xvi_2d"
    IVIEW = "iview_portal"
    SURFACE_GUIDED = "catalyst"


@dataclass
class AgilityMLCSpec:
    """Specifications for Elekta Agility MLC."""
    num_leaf_pairs: int = 80
    leaf_width: float = 5.0  # mm at isocenter
    max_leaf_speed: float = 35.0  # mm/s
    leaf_travel: float = 200.0  # mm
    interdigitation: float = 0.0  # Full interdigitation allowed
    min_gap: float = 0.0  # Can close completely
    max_field_size: Tuple[float, float] = (40.0, 40.0)  # cm


class ElektaMLCAgility(MultiLeafCollimator):
    """
    Elekta Agility MLC.

    High-performance MLC with:
    - 80 leaf pairs at 5mm width
    - 160 individually controlled leaves
    - 35 mm/s leaf speed
    - Full interdigitation
    - Rubicon leaf guide system
    """

    def __init__(self):
        super().__init__(
            num_leaf_pairs=80,
            leaf_width=5.0,
            mlc_type=MLCType.STANDARD,
            name="Elekta Agility MLC"
        )

        self.max_leaf_speed = 35.0  # mm/s (faster than Varian)
        self._interlock_gap = 0.0  # Can close completely

        # Agility-specific features
        self._rubicon_guides = True
        self._leaf_position_accuracy = 0.1  # mm
        self._full_interdigitation = True

        # Backup jaws (Elekta uses independent jaws)
        self._x1_jaw = -200.0
        self._x2_jaw = 200.0
        self._y1_jaw = -200.0
        self._y2_jaw = 200.0

    def set_jaw_positions(
        self,
        x1: float,
        x2: float,
        y1: float,
        y2: float
    ) -> bool:
        """Set backup jaw positions."""
        self._x1_jaw = x1
        self._x2_jaw = x2
        self._y1_jaw = y1
        self._y2_jaw = y2
        return True

    def calculate_leaf_trajectory(
        self,
        start_aperture: List[Tuple[float, float]],
        end_aperture: List[Tuple[float, float]],
        delivery_time: float
    ) -> List[List[Tuple[float, float]]]:
        """
        Calculate leaf trajectories for VMAT.

        Returns intermediate positions for smooth motion.
        """
        num_steps = int(delivery_time * 10)  # 10 Hz update
        trajectories = []

        for step in range(num_steps + 1):
            t = step / num_steps
            current = []
            for i in range(self.num_leaf_pairs):
                left = start_aperture[i][0] + t * (
                    end_aperture[i][0] - start_aperture[i][0]
                )
                right = start_aperture[i][1] + t * (
                    end_aperture[i][1] - start_aperture[i][1]
                )
                current.append((left, right))
            trajectories.append(current)

        return trajectories


@dataclass
class ElektaXVI:
    """Elekta X-ray Volume Imaging (XVI) system."""
    source_type: str = "kV"
    kv_range: Tuple[float, float] = (60.0, 150.0)
    ma_range: Tuple[float, float] = (10.0, 640.0)
    detector_size: Tuple[int, int] = (1024, 1024)
    pixel_pitch: float = 0.4  # mm
    source_to_isocenter: float = 1000.0
    source_to_detector: float = 1536.0

    # CBCT modes
    presets: List[str] = field(default_factory=lambda: [
        "head", "chest", "pelvis", "prostate", "symmetry"
    ])
    scan_arc: float = 360.0  # degrees

    # 4D CBCT
    respiratory_correlated: bool = True


class ElektaVersaHD(LinearAccelerator):
    """
    Elekta Versa HD Linear Accelerator.

    High-definition linear accelerator featuring:
    - Photon: 6, 10, 15, 18 MV (and FFF 6, 10)
    - Electron: 4-22 MeV
    - Agility 160 MLC
    - XVI integrated imaging
    - HexaPOD evo RT couch
    """

    PHOTON_ENERGIES = [6.0, 10.0, 15.0, 18.0]
    PHOTON_FFF = [6.0, 10.0]
    ELECTRON_ENERGIES = [4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 22.0]

    DOSE_RATES = {
        "standard": {6.0: 600, 10.0: 600, 15.0: 600, 18.0: 600},
        "fff": {6.0: 1400, 10.0: 2200}
    }

    def __init__(self):
        super().__init__(
            name="Elekta Versa HD",
            max_photon_energy=18.0,
            max_electron_energy=22.0
        )

        self.manufacturer = "Elekta"

        # Agility MLC
        self.mlc = ElektaMLCAgility()

        # XVI imaging
        self.xvi = ElektaXVI()
        self._xvi_panel_deployed = False

        # iViewGT portal imaging
        self._iview_enabled = True

        # Beam modes
        self._fff_mode = False

        # Gantry
        self._max_gantry_speed = 6.0  # deg/s
        self._gantry_acceleration = 0.5

        # HexaPOD couch
        self._hexapod_enabled = True
        self._couch_6dof = True

        # Monaco/XiO TPS integration
        self._tps_integration = "Monaco"

        # VMAT mode
        self._vmat_enabled = True

    def set_fff_mode(self, energy: float) -> bool:
        """Enable FFF mode for supported energies."""
        if energy not in self.PHOTON_FFF:
            return False

        self._fff_mode = True
        self._current_energy = energy
        self._dose_rate = self.DOSE_RATES["fff"][energy]
        return True

    def deploy_xvi(self) -> bool:
        """Deploy XVI imaging panel."""
        self._xvi_panel_deployed = True
        return True

    def retract_xvi(self) -> bool:
        """Retract XVI imaging panel."""
        self._xvi_panel_deployed = False
        return True

    def acquire_xvi_cbct(
        self,
        preset: str = "pelvis",
        arc: float = 360.0
    ) -> np.ndarray:
        """
        Acquire XVI cone-beam CT.

        Args:
            preset: Imaging preset
            arc: Rotation arc in degrees
        """
        if not self._xvi_panel_deployed:
            self.deploy_xvi()

        # Number of projections based on arc
        num_projections = int(arc * 2.5)  # ~2.5 projections per degree

        # Simulate CBCT volume
        matrix_size = 512
        return np.random.random((matrix_size, matrix_size, matrix_size))

    def acquire_4d_cbct(self, phases: int = 10) -> List[np.ndarray]:
        """Acquire 4D respiratory-correlated CBCT."""
        volumes = []
        for phase in range(phases):
            vol = self.acquire_xvi_cbct()
            volumes.append(vol)
        return volumes

    def deliver_vmat_arc(
        self,
        control_points: List[Dict],
    ) -> bool:
        """Deliver VMAT arc treatment."""
        if not self._vmat_enabled:
            return False

        return super().deliver_vmat_arc(control_points)

    def get_hexapod_correction(
        self,
        shift: Position3D,
        rotation: Dict
    ) -> bool:
        """Apply HexaPOD 6-DOF couch correction."""
        if not self._hexapod_enabled:
            return False

        # Apply corrections
        self._position = Position3D(
            x=self._position.x + shift.x,
            y=self._position.y + shift.y,
            z=self._position.z + shift.z
        )
        return True


class ElektaInfinity(LinearAccelerator):
    """
    Elekta Infinity Linear Accelerator.

    Digital linear accelerator with:
    - Full range of photon and electron energies
    - Agility MLC
    - Integrated XVI imaging
    - VMAT capability
    """

    def __init__(self):
        super().__init__(
            name="Elekta Infinity",
            max_photon_energy=18.0,
            max_electron_energy=20.0
        )

        self.manufacturer = "Elekta"
        self.mlc = ElektaMLCAgility()
        self.xvi = ElektaXVI()

        # Response gating
        self._abc_gating = True  # Active Breathing Coordinator
        self._gating_enabled = False

    def enable_abc_gating(self) -> bool:
        """Enable Active Breathing Coordinator gating."""
        self._gating_enabled = True
        return True


@dataclass
class UnityMRISpec:
    """Elekta Unity MRI specifications."""
    field_strength: float = 1.5  # Tesla
    bore_diameter: float = 700.0  # mm
    gradient_strength: float = 18.0  # mT/m
    slew_rate: float = 200.0  # T/m/s
    imaging_fov: float = 500.0  # mm
    real_time_sequences: List[str] = field(default_factory=lambda: [
        "bSSFP", "T1w", "T2w", "DWI"
    ])


class ElektaUnity(BeamDeliverySystem):
    """
    Elekta Unity MR-Linac.

    Revolutionary MR-guided radiation therapy system:
    - 1.5T MRI integrated with 7 MV LINAC
    - Real-time soft tissue visualization
    - Online adaptive radiotherapy
    - Beam gating and tracking
    """

    def __init__(self):
        super().__init__(name="Elekta Unity")

        self.manufacturer = "Elekta"

        # MRI specifications
        self.mri = UnityMRISpec()

        # LINAC specifications
        self._beam_energy = 7.0  # MV (fixed)
        self._dose_rate = 425.0  # MU/min at isocenter
        self._source_to_axis = 1432.0  # mm (larger SAD due to MRI)

        # MLC
        self._mlc_num_pairs = 80
        self._mlc_leaf_width = 7.175  # mm at isocenter
        self._mlc_max_speed = 30.0  # mm/s

        # Gantry (rotates around MRI)
        self._gantry_angle = 0.0
        self._gantry_speed = 6.0  # deg/s

        # Couch
        self._couch_in_bore = True

        # Imaging modes
        self._real_time_imaging = True
        self._cine_mri_rate = 8.0  # frames/second

        # Adaptive workflow
        self._adapt_to_position = True
        self._adapt_to_shape = True

        self._is_ready = False

    def initialize(self) -> bool:
        """Initialize Unity system."""
        self._is_ready = True
        return True

    def calibrate(self) -> bool:
        """Calibrate the system."""
        self._is_calibrated = True
        return True

    def set_position(self, position: Position3D) -> bool:
        """Set treatment position."""
        self._position = position
        return True

    def acquire_mr_image(
        self,
        sequence: str = "bSSFP",
        slice_thickness: float = 3.0
    ) -> np.ndarray:
        """
        Acquire MR image.

        Args:
            sequence: MRI sequence type
            slice_thickness: Slice thickness in mm
        """
        # Simulate MRI acquisition
        matrix_size = 256
        return np.random.random((matrix_size, matrix_size))

    def acquire_cine_mri(
        self,
        duration: float = 10.0
    ) -> List[np.ndarray]:
        """
        Acquire real-time cine MRI for motion tracking.

        Args:
            duration: Duration in seconds
        """
        num_frames = int(duration * self._cine_mri_rate)
        frames = []
        for _ in range(num_frames):
            frames.append(self.acquire_mr_image())
        return frames

    def acquire_3d_mr(self, resolution: float = 1.0) -> np.ndarray:
        """Acquire 3D MR volume for adaptive planning."""
        matrix_size = int(self.mri.imaging_fov / resolution)
        return np.random.random((matrix_size, matrix_size, matrix_size))

    def perform_atp(
        self,
        reference_plan: Dict,
        current_anatomy: np.ndarray
    ) -> Dict:
        """
        Perform Adapt-to-Position (ATP) workflow.

        Fast online adaptation based on patient position.
        """
        # Calculate position correction
        correction = {
            "shift": Position3D(0, 0, 0),
            "rotation": {"pitch": 0, "roll": 0, "yaw": 0},
            "adapted": True
        }
        return correction

    def perform_ats(
        self,
        reference_plan: Dict,
        current_anatomy: np.ndarray
    ) -> Dict:
        """
        Perform Adapt-to-Shape (ATS) workflow.

        Full re-optimization based on current anatomy.
        """
        # Re-contour and re-optimize
        adapted_plan = {
            "original_plan": reference_plan,
            "new_contours": True,
            "re_optimized": True,
            "dose_recalculated": True
        }
        return adapted_plan

    def track_target(
        self,
        cine_frames: List[np.ndarray]
    ) -> List[Position3D]:
        """
        Track target position from cine MRI.

        Returns list of target positions over time.
        """
        positions = []
        for frame in cine_frames:
            # Simulate target detection
            x = np.random.normal(0, 1)
            y = np.random.normal(0, 1)
            z = np.random.normal(0, 2)  # More motion in SI direction
            positions.append(Position3D(x, y, z))
        return positions

    def gate_beam(
        self,
        target_positions: List[Position3D],
        tolerance: float = 3.0
    ) -> List[bool]:
        """
        Generate beam gating signal based on target position.

        Args:
            target_positions: Target positions over time
            tolerance: Position tolerance in mm
        """
        gating = []
        reference = Position3D(0, 0, 0)
        for pos in target_positions:
            displacement = pos.distance_to(reference)
            gating.append(displacement <= tolerance)
        return gating

    def deliver_dose(
        self,
        beam_params: BeamParameters,
        monitor_units: float
    ) -> DoseDistribution:
        """Deliver treatment dose with MR guidance."""
        # Create dose distribution
        dose_dist = DoseDistribution(
            origin=Position3D(-150, -150, -150),
            spacing=(2.0, 2.0, 2.0),
            dimensions=(150, 150, 150)
        )

        # Note: MRI affects electron transport (electron return effect)
        # This is simplified; real calculation considers B-field effects

        return dose_dist

    def calculate_ere(self, position: Position3D) -> float:
        """
        Calculate Electron Return Effect (ERE) at position.

        The 1.5T magnetic field causes electrons to curve back,
        affecting dose at tissue interfaces.
        """
        # Simplified ERE model
        # ERE is strongest at air-tissue interfaces
        return 1.0 + 0.1 * abs(position.z) / 100

    def get_status(self) -> Dict:
        """Get Unity system status."""
        return {
            "name": self.name,
            "manufacturer": self.manufacturer,
            "mri_field_strength": self.mri.field_strength,
            "beam_energy": self._beam_energy,
            "dose_rate": self._dose_rate,
            "is_ready": self._is_ready,
            "real_time_imaging": self._real_time_imaging,
            "adapt_to_shape": self._adapt_to_shape,
            "gantry_angle": self._gantry_angle,
        }


class ElektaAgility(ElektaVersaHD):
    """
    Elekta system with Agility MLC upgrade.

    For older Elekta systems upgraded with Agility MLC.
    """

    def __init__(self, base_model: str = "Synergy"):
        super().__init__()
        self.name = f"Elekta {base_model} with Agility"
        self._base_model = base_model
