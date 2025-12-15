"""
Accuray Radiation Therapy Equipment Implementations.

Provides simulations of Accuray systems:
- CyberKnife (M6, S7)
- TomoTherapy (HD, H Series)
- Radixact
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
)
from ..cyberknife import (
    CyberKnifeSystem,
    RoboticArm,
    MotionCompensation,
    ImageGuidedTargeting,
    TrackingModality,
    TreatmentNode,
)


class CyberKnifeTrackingMode(Enum):
    """CyberKnife tracking modes."""
    SKULL = "skull"  # 6D Skull tracking
    SPINE = "spine"  # Xsight Spine
    LUNG = "lung"  # Xsight Lung with Synchrony
    FIDUCIAL = "fiducial"  # Fiducial tracking
    SOFT_TISSUE = "soft_tissue"  # Xsight Lung without fiducials


class CyberKnifeCollimator(Enum):
    """CyberKnife collimator types."""
    FIXED_CONE = "fixed_cone"
    IRIS = "iris"
    INCISE_MLC = "incise_mlc"


@dataclass
class CyberKnifeSpec:
    """CyberKnife specifications."""
    model: str
    robot_dof: int = 6
    positioning_accuracy: float = 0.5  # mm
    linac_energy: float = 6.0  # MV (X-band)
    max_dose_rate: float = 1000.0  # MU/min
    treatment_distance: Tuple[float, float] = (650.0, 1000.0)  # mm
    imaging_system: str = "stereo_xray"


class AccurayCyberKnifeM6(CyberKnifeSystem):
    """
    Accuray CyberKnife M6 System.

    Features:
    - 6-axis robotic arm (KUKA)
    - Synchrony respiratory tracking
    - InCise 2 MLC (optional)
    - IRIS variable aperture collimator (optional)
    - Xsight tracking technologies
    """

    SPEC = CyberKnifeSpec(
        model="M6",
        robot_dof=6,
        positioning_accuracy=0.5,
        linac_energy=6.0,
        max_dose_rate=1000.0,
        treatment_distance=(650.0, 1000.0),
        imaging_system="stereo_xray"
    )

    def __init__(self, mlc_option: str = "incise"):
        super().__init__(
            name="Accuray CyberKnife M6",
            beam_energy=6.0
        )

        self.manufacturer = "Accuray"
        self.model = "M6"

        # Collimator options
        self._collimator_type = CyberKnifeCollimator.INCISE_MLC
        if mlc_option == "iris":
            self._collimator_type = CyberKnifeCollimator.IRIS
        elif mlc_option == "fixed":
            self._collimator_type = CyberKnifeCollimator.FIXED_CONE

        # Fixed cones
        self._fixed_cones = [5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0,
                            35.0, 40.0, 50.0, 60.0]  # mm

        # IRIS collimator
        self._iris_range = (5.0, 60.0)  # mm variable aperture
        self._iris_steps = 12

        # InCise 2 MLC
        self._incise_leaves = 26  # pairs
        self._incise_leaf_width = 3.85  # mm at 800mm SAD
        self._incise_max_field = (120.0, 100.2)  # mm

        # Tracking modes
        self._tracking_modes = list(CyberKnifeTrackingMode)
        self._current_tracking = CyberKnifeTrackingMode.SKULL

        # Synchrony respiratory tracking
        self._synchrony_enabled = True
        self._synchrony_model = None
        self._led_markers = 3  # External LED markers

        # X-ray imaging
        self._xray_sources = 2
        self._detector_panels = 2
        self._imaging_angle = 45.0  # degrees

        # Robotic arm specifications
        self._robot_reach = 2000.0  # mm
        self._robot_payload = 150.0  # kg

    def set_collimator_type(self, coll_type: CyberKnifeCollimator) -> bool:
        """Set collimator type."""
        self._collimator_type = coll_type
        return True

    def set_iris_aperture(self, diameter: float) -> bool:
        """Set IRIS variable aperture diameter."""
        if self._collimator_type != CyberKnifeCollimator.IRIS:
            return False

        if not (self._iris_range[0] <= diameter <= self._iris_range[1]):
            return False

        self._current_collimator = diameter
        return True

    def set_incise_aperture(
        self,
        aperture: List[Tuple[float, float]]
    ) -> bool:
        """Set InCise MLC aperture."""
        if self._collimator_type != CyberKnifeCollimator.INCISE_MLC:
            return False

        if len(aperture) != self._incise_leaves:
            return False

        return True

    def set_tracking_mode(self, mode: CyberKnifeTrackingMode) -> bool:
        """Set target tracking mode."""
        if mode not in self._tracking_modes:
            return False

        self._current_tracking = mode

        # Configure motion compensation
        if mode == CyberKnifeTrackingMode.LUNG:
            self.motion_compensation.set_tracking_modality(TrackingModality.LUNG)
        elif mode == CyberKnifeTrackingMode.SPINE:
            self.motion_compensation.set_tracking_modality(TrackingModality.SPINE)
        elif mode == CyberKnifeTrackingMode.FIDUCIAL:
            self.motion_compensation.set_tracking_modality(TrackingModality.FIDUCIAL)
        else:
            self.motion_compensation.set_tracking_modality(TrackingModality.SKULL)

        return True

    def build_synchrony_model(
        self,
        internal_positions: List[Position3D],
        external_signals: List[float]
    ) -> bool:
        """Build Synchrony correlation model."""
        if not self._synchrony_enabled:
            return False

        # Build correlation model between LED markers and tumor
        self._synchrony_model = {
            "internal": internal_positions,
            "external": external_signals,
            "correlation": np.corrcoef(
                [p.z for p in internal_positions],
                external_signals
            )[0, 1]
        }
        return True

    def predict_tumor_position(
        self,
        external_signal: float
    ) -> Position3D:
        """Predict tumor position from external markers."""
        if not self._synchrony_model:
            return Position3D()

        # Use correlation model for prediction
        correlation = self._synchrony_model["correlation"]
        internal = self._synchrony_model["internal"]

        if internal:
            mean_z = np.mean([p.z for p in internal])
            predicted_z = mean_z + external_signal * correlation * 10
            return Position3D(0, 0, predicted_z)

        return Position3D()

    def generate_treatment_plan(
        self,
        target: TreatmentTarget,
        tracking_mode: CyberKnifeTrackingMode,
        num_nodes: int = 120
    ) -> List[TreatmentNode]:
        """Generate CyberKnife treatment plan."""
        self.set_tracking_mode(tracking_mode)
        return self.generate_treatment_nodes(target, num_nodes)

    def get_status(self) -> Dict:
        """Get M6 system status."""
        base_status = super().get_status()
        base_status.update({
            "model": self.model,
            "collimator_type": self._collimator_type.value,
            "tracking_mode": self._current_tracking.value,
            "synchrony_enabled": self._synchrony_enabled,
            "has_synchrony_model": self._synchrony_model is not None,
        })
        return base_status


class AccurayCyberKnifeS7(AccurayCyberKnifeM6):
    """
    Accuray CyberKnife S7 System.

    Latest generation CyberKnife featuring:
    - Enhanced robotic arm
    - Improved imaging
    - Faster treatment times
    - SRS Optimizer planning
    """

    def __init__(self):
        super().__init__(mlc_option="incise")
        self.name = "Accuray CyberKnife S7"
        self.model = "S7"

        # S7 enhancements
        self._enhanced_robot = True
        self._faster_imaging = True
        self._treatment_speed_improvement = 1.5  # 50% faster

        # Improved tracking
        self._tracking_frequency = 30.0  # Hz (faster updates)

        # SRS Optimizer
        self._srs_optimizer = True


class TomoTherapyDeliveryMode(Enum):
    """TomoTherapy delivery modes."""
    HELICAL = "helical"
    DIRECT = "tomo_direct"
    STATIC = "tomo_static"


@dataclass
class TomoTherapySpec:
    """TomoTherapy specifications."""
    model: str
    ring_diameter: float  # mm
    beam_energy: float  # MV
    mlc_leaves: int
    leaf_width: float  # mm at isocenter
    max_field_length: float  # mm
    slice_widths: List[float]  # mm


class AccurayTomoTherapy(BeamDeliverySystem):
    """
    Accuray TomoTherapy System (Hi-Art, HD).

    Helical tomotherapy system:
    - Ring-mounted LINAC
    - Helical delivery with couch translation
    - Binary MLC
    - Integrated MVCT imaging
    """

    SPEC = TomoTherapySpec(
        model="HD",
        ring_diameter=850.0,
        beam_energy=6.0,
        mlc_leaves=64,
        leaf_width=6.25,
        max_field_length=400.0,
        slice_widths=[1.0, 2.5, 5.0]
    )

    def __init__(self, model: str = "HD"):
        super().__init__(name=f"Accuray TomoTherapy {model}")

        self.manufacturer = "Accuray"
        self.model = model

        # Ring gantry
        self._ring_diameter = self.SPEC.ring_diameter
        self._ring_rotation_speed = 60.0  # seconds per rotation
        self._gantry_angle = 0.0

        # LINAC
        self._beam_energy = 6.0  # MV
        self._dose_rate = 850.0  # MU/min

        # Binary MLC (pneumatic)
        self._mlc_leaves = 64
        self._leaf_width = 6.25  # mm at isocenter
        self._leaf_opening_time = 20.0  # ms
        self._binary_mlc = True  # Open/closed only

        # Field widths (slice widths)
        self._slice_widths = [1.0, 2.5, 5.0]  # cm
        self._current_slice_width = 2.5

        # Pitch (couch travel per rotation)
        self._pitch_range = (0.1, 1.0)
        self._current_pitch = 0.287  # Typical

        # Modulation factor
        self._max_modulation_factor = 3.0
        self._current_modulation = 2.0

        # MVCT imaging
        self._mvct_enabled = True
        self._mvct_energy = 3.5  # MV (detuned)
        self._mvct_modes = ["fine", "normal", "coarse"]

        # Delivery modes
        self._delivery_mode = TomoTherapyDeliveryMode.HELICAL

        self._is_ready = False

    def initialize(self) -> bool:
        """Initialize TomoTherapy system."""
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

    def set_slice_width(self, width: float) -> bool:
        """Set treatment slice width (jaw opening)."""
        if width not in self._slice_widths:
            return False
        self._current_slice_width = width
        return True

    def set_pitch(self, pitch: float) -> bool:
        """Set helical pitch."""
        if not (self._pitch_range[0] <= pitch <= self._pitch_range[1]):
            return False
        self._current_pitch = pitch
        return True

    def set_modulation_factor(self, factor: float) -> bool:
        """Set modulation factor (affects treatment time)."""
        if factor > self._max_modulation_factor:
            return False
        self._current_modulation = factor
        return True

    def set_delivery_mode(self, mode: TomoTherapyDeliveryMode) -> bool:
        """Set delivery mode."""
        self._delivery_mode = mode
        return True

    def acquire_mvct(
        self,
        mode: str = "normal",
        start_position: float = -200.0,
        end_position: float = 200.0
    ) -> np.ndarray:
        """
        Acquire MVCT for patient setup.

        Args:
            mode: Imaging mode (fine, normal, coarse)
            start_position: Start couch position
            end_position: End couch position
        """
        length = abs(end_position - start_position)
        num_slices = int(length / 4)  # 4mm slice reconstruction

        return np.random.random((512, 512, num_slices))

    def calculate_sinogram(
        self,
        target: TreatmentTarget,
        oars: List
    ) -> np.ndarray:
        """
        Calculate optimized sinogram for helical delivery.

        The sinogram represents MLC opening patterns for each
        gantry angle and couch position.
        """
        # Number of projections per rotation
        projections_per_rotation = 51

        # Treatment length
        length = target.dimensions[1] + 100  # mm margin
        num_rotations = int(length / (self._current_slice_width * 10 * self._current_pitch))

        total_projections = projections_per_rotation * num_rotations

        # Sinogram: [projection, leaf]
        sinogram = np.random.random((total_projections, self._mlc_leaves))

        return sinogram

    def deliver_helical(
        self,
        sinogram: np.ndarray,
        total_mu: float
    ) -> bool:
        """
        Deliver helical treatment.

        Args:
            sinogram: Optimized MLC opening patterns
            total_mu: Total monitor units to deliver
        """
        if self._delivery_mode != TomoTherapyDeliveryMode.HELICAL:
            return False

        # Calculate delivery time
        num_projections = sinogram.shape[0]
        projections_per_rotation = 51
        num_rotations = num_projections / projections_per_rotation
        delivery_time = num_rotations * self._ring_rotation_speed

        return True

    def deliver_tomo_direct(
        self,
        gantry_angles: List[float],
        mlc_patterns: List[np.ndarray],
        mu_per_angle: List[float]
    ) -> bool:
        """
        Deliver TomoDirect (static gantry angles).

        3DCRT-like delivery with discrete gantry angles.
        """
        if self._delivery_mode != TomoTherapyDeliveryMode.DIRECT:
            return False

        for angle, pattern, mu in zip(gantry_angles, mlc_patterns, mu_per_angle):
            self._gantry_angle = angle
            # Deliver at this angle
            pass

        return True

    def deliver_dose(
        self,
        beam_params: BeamParameters,
        monitor_units: float
    ) -> DoseDistribution:
        """Deliver treatment dose."""
        dose_dist = DoseDistribution(
            origin=Position3D(-200, -200, -200),
            spacing=(2.0, 2.0, 2.0),
            dimensions=(200, 200, 200)
        )

        # TomoTherapy creates highly conformal dose distributions
        # Simplified calculation
        for i in range(dose_dist.dimensions[0]):
            for j in range(dose_dist.dimensions[1]):
                for k in range(dose_dist.dimensions[2]):
                    x = dose_dist.origin.x + i * dose_dist.spacing[0]
                    y = dose_dist.origin.y + j * dose_dist.spacing[1]
                    z = dose_dist.origin.z + k * dose_dist.spacing[2]

                    r = math.sqrt(x ** 2 + y ** 2)

                    # Ring delivery creates cylindrical dose falloff
                    if r < 50:
                        dose = monitor_units * 0.01
                    else:
                        dose = monitor_units * 0.01 * math.exp(-(r - 50) / 30)

                    dose_dist.dose_grid[i, j, k] = dose

        return dose_dist

    def calculate_treatment_time(
        self,
        target_length: float,
        total_mu: float
    ) -> float:
        """
        Calculate treatment time.

        Args:
            target_length: Target length in mm
            total_mu: Total monitor units

        Returns:
            Treatment time in seconds
        """
        # Calculate number of rotations
        slice_width_mm = self._current_slice_width * 10  # Convert cm to mm
        travel_per_rotation = slice_width_mm * self._current_pitch
        num_rotations = target_length / travel_per_rotation

        # Beam-on time
        beam_on_time = total_mu / self._dose_rate * 60  # seconds

        # Total time (beam-on dominates for TomoTherapy)
        total_time = max(beam_on_time, num_rotations * self._ring_rotation_speed)

        return total_time

    def get_status(self) -> Dict:
        """Get TomoTherapy status."""
        return {
            "name": self.name,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "beam_energy": self._beam_energy,
            "slice_width": self._current_slice_width,
            "pitch": self._current_pitch,
            "modulation_factor": self._current_modulation,
            "delivery_mode": self._delivery_mode.value,
            "mlc_leaves": self._mlc_leaves,
            "is_ready": self._is_ready,
        }


class AccurayRadixact(AccurayTomoTherapy):
    """
    Accuray Radixact System.

    Next-generation TomoTherapy featuring:
    - Faster ring rotation (12 seconds)
    - CTrue imaging (kV + MV)
    - Synchrony motion tracking (optional)
    - Enhanced planning (Precision)
    """

    def __init__(self, synchrony: bool = False):
        super().__init__(model="Radixact")
        self.name = "Accuray Radixact"

        # Faster delivery
        self._ring_rotation_speed = 12.0  # seconds (vs 60 for HD)

        # Enhanced imaging
        self._ctrue_imaging = True
        self._kv_imaging = True  # Added kV source

        # Optional Synchrony
        self._synchrony_option = synchrony
        if synchrony:
            self.motion_compensation = MotionCompensation()
            self._synchrony_enabled = True

        # Precision planning
        self._precision_planning = True
        self._interleaving = True  # Interleaved planning

    def acquire_ctrue(self) -> np.ndarray:
        """Acquire CTrue image (combined kV/MV)."""
        return np.random.random((512, 512, 200))

    def enable_synchrony(self) -> bool:
        """Enable Synchrony motion tracking."""
        if not self._synchrony_option:
            return False

        self._synchrony_enabled = True
        return True

    def deliver_with_synchrony(
        self,
        sinogram: np.ndarray,
        total_mu: float
    ) -> bool:
        """Deliver with real-time motion tracking."""
        if not self._synchrony_enabled:
            return self.deliver_helical(sinogram, total_mu)

        # Modulate delivery based on motion
        return self.deliver_helical(sinogram, total_mu)

    def get_status(self) -> Dict:
        """Get Radixact status."""
        base_status = super().get_status()
        base_status.update({
            "ctrue_imaging": self._ctrue_imaging,
            "synchrony_option": self._synchrony_option,
            "synchrony_enabled": getattr(self, '_synchrony_enabled', False),
            "precision_planning": self._precision_planning,
        })
        return base_status
