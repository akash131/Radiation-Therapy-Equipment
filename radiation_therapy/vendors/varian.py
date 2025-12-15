"""
Varian Medical Systems Equipment Implementations.

Provides accurate simulations of Varian linear accelerators and components:
- TrueBeam (standard and STx)
- Halcyon
- Clinac (iX, 21EX)
- Edge (radiosurgery platform)
- ProBeam (proton therapy)
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
    ElectronGun,
    AcceleratingWaveguide,
    WaveguideType,
    BendingMagnet,
    TumorTracker,
    ImagingModality,
)


class VarianBeamMode(Enum):
    """Varian-specific beam modes."""
    PHOTON_FLATTENED = "photon_flattened"
    PHOTON_FFF = "photon_fff"  # Flattening Filter Free
    ELECTRON = "electron"
    SRS = "srs"  # Stereotactic Radiosurgery


class VarianImagingMode(Enum):
    """Varian imaging modalities."""
    KV_CBCT = "kv_cbct"
    MV_CBCT = "mv_cbct"
    KV_2D = "kv_2d"
    MV_2D = "mv_2d"
    PORTAL_DOSIMETRY = "portal_dosimetry"
    SURFACE_IMAGING = "surface_imaging"


@dataclass
class VarianMLCSpec:
    """Specifications for Varian MLC models."""
    name: str
    num_leaf_pairs: int
    leaf_width_inner: float  # mm at isocenter (central leaves)
    leaf_width_outer: float  # mm at isocenter (outer leaves)
    max_leaf_speed: float  # mm/s
    max_field_size: Tuple[float, float]  # (x, y) in cm
    interdigitation: bool
    leaf_travel: float  # mm maximum travel
    min_gap: float  # mm minimum gap between opposing leaves


class VarianMLCMillennium(MultiLeafCollimator):
    """
    Varian Millennium MLC (80 and 120 leaf versions).

    The Millennium 120 MLC has:
    - 40 pairs of 5mm leaves (central 20cm)
    - 20 pairs of 10mm leaves (outer 20cm total)
    - Maximum field size: 40 x 40 cm
    """

    SPECS_80 = VarianMLCSpec(
        name="Millennium 80",
        num_leaf_pairs=40,
        leaf_width_inner=10.0,
        leaf_width_outer=10.0,
        max_leaf_speed=25.0,
        max_field_size=(40.0, 40.0),
        interdigitation=True,
        leaf_travel=200.0,
        min_gap=0.5
    )

    SPECS_120 = VarianMLCSpec(
        name="Millennium 120",
        num_leaf_pairs=60,
        leaf_width_inner=5.0,
        leaf_width_outer=10.0,
        max_leaf_speed=25.0,
        max_field_size=(40.0, 40.0),
        interdigitation=True,
        leaf_travel=200.0,
        min_gap=0.5
    )

    def __init__(self, model: str = "120"):
        self.spec = self.SPECS_120 if model == "120" else self.SPECS_80
        super().__init__(
            num_leaf_pairs=self.spec.num_leaf_pairs,
            leaf_width=self.spec.leaf_width_inner,
            mlc_type=MLCType.STANDARD,
            name=self.spec.name
        )
        self.max_leaf_speed = self.spec.max_leaf_speed
        self._interlock_gap = self.spec.min_gap

        # Override leaf widths for mixed-width design
        self._configure_leaf_widths()

    def _configure_leaf_widths(self):
        """Configure variable leaf widths for Millennium design."""
        self._leaf_widths = []
        if self.spec.num_leaf_pairs == 60:
            # 10mm outer + 5mm inner + 5mm inner + 10mm outer
            for i in range(60):
                if i < 10 or i >= 50:
                    self._leaf_widths.append(10.0)
                else:
                    self._leaf_widths.append(5.0)
        else:
            self._leaf_widths = [10.0] * 40

    def get_leaf_width(self, leaf_index: int) -> float:
        """Get width of specific leaf."""
        if 0 <= leaf_index < len(self._leaf_widths):
            return self._leaf_widths[leaf_index]
        return self.leaf_width


class VarianMLCHD120(MultiLeafCollimator):
    """
    Varian HD 120 MLC for TrueBeam and Edge.

    High-definition MLC with:
    - 32 pairs of 2.5mm leaves (central 8cm)
    - 28 pairs of 5mm leaves (outer 14cm each side)
    - Maximum field size: 40 x 22 cm
    - Superior for SRS/SBRT applications
    """

    SPECS = VarianMLCSpec(
        name="HD 120 MLC",
        num_leaf_pairs=60,
        leaf_width_inner=2.5,
        leaf_width_outer=5.0,
        max_leaf_speed=25.0,
        max_field_size=(40.0, 22.0),
        interdigitation=True,
        leaf_travel=200.0,
        min_gap=0.25
    )

    def __init__(self):
        super().__init__(
            num_leaf_pairs=60,
            leaf_width=2.5,
            mlc_type=MLCType.MICRO_MLC,
            name="HD 120 MLC"
        )
        self.max_leaf_speed = self.SPECS.max_leaf_speed
        self._interlock_gap = self.SPECS.min_gap
        self._configure_leaf_widths()

    def _configure_leaf_widths(self):
        """Configure HD 120 leaf widths."""
        self._leaf_widths = []
        for i in range(60):
            if 14 <= i < 46:  # Central 32 pairs
                self._leaf_widths.append(2.5)
            else:  # Outer 28 pairs (14 each side)
                self._leaf_widths.append(5.0)

    def get_effective_resolution(self) -> float:
        """Get effective resolution at isocenter."""
        return 2.5  # mm


@dataclass
class VarianOBI:
    """Varian On-Board Imager (OBI) specifications."""
    kv_source_power: float = 80.0  # kW
    kv_tube_voltage_range: Tuple[float, float] = (40.0, 150.0)  # kV
    kv_tube_current_range: Tuple[float, float] = (10.0, 800.0)  # mA
    detector_size: Tuple[int, int] = (2048, 1536)  # pixels
    pixel_pitch: float = 0.194  # mm
    source_to_detector: float = 1500.0  # mm
    source_to_isocenter: float = 1000.0  # mm

    # CBCT parameters
    cbct_rotation_range: float = 360.0  # degrees
    cbct_projections: int = 900
    cbct_reconstruction_matrix: int = 512

    def calculate_fov(self) -> float:
        """Calculate field of view diameter."""
        detector_width = self.detector_size[0] * self.pixel_pitch
        magnification = self.source_to_detector / self.source_to_isocenter
        return detector_width / magnification


@dataclass
class VarianExactrac:
    """Varian ExacTrac patient positioning system."""
    x_ray_sources: int = 2
    detector_panels: int = 2
    imaging_geometry: str = "oblique_45deg"
    positioning_accuracy: float = 0.5  # mm
    rotation_accuracy: float = 0.5  # degrees
    tracking_modes: List[str] = field(default_factory=lambda: [
        "skeletal", "fiducial", "soft_tissue"
    ])


class VarianTrueBeam(LinearAccelerator):
    """
    Varian TrueBeam Linear Accelerator.

    Features:
    - Photon energies: 6, 10, 15, 18 MV (flattened) and 6, 10 FFF
    - Electron energies: 6, 9, 12, 16, 20 MeV
    - Maximum dose rate: 2400 MU/min (FFF mode)
    - HD 120 MLC standard
    - Integrated kV/MV imaging
    - RapidArc VMAT capability
    """

    PHOTON_ENERGIES_FLAT = [6.0, 10.0, 15.0, 18.0]
    PHOTON_ENERGIES_FFF = [6.0, 10.0]
    ELECTRON_ENERGIES = [6.0, 9.0, 12.0, 16.0, 20.0]

    DOSE_RATES_FLAT = {
        6.0: 600, 10.0: 600, 15.0: 600, 18.0: 600
    }
    DOSE_RATES_FFF = {
        6.0: 1400, 10.0: 2400
    }

    def __init__(self, model: str = "standard"):
        """
        Initialize TrueBeam.

        Args:
            model: "standard" or "STx" (radiosurgery configuration)
        """
        super().__init__(
            name=f"Varian TrueBeam {model.upper()}",
            max_photon_energy=18.0 if model == "standard" else 10.0,
            max_electron_energy=20.0
        )

        self.model = model
        self.manufacturer = "Varian Medical Systems"

        # Replace MLC with HD 120
        self.mlc = VarianMLCHD120()

        # Imaging systems
        self.obi = VarianOBI()
        self._kv_imager_deployed = False
        self._mv_imager_deployed = False

        # Beam modes
        self._beam_mode = VarianBeamMode.PHOTON_FLATTENED
        self._fff_mode = False

        # Gantry specifications
        self._max_gantry_speed = 6.0  # degrees/second for VMAT
        self._gantry_acceleration = 0.6  # degrees/second^2

        # Couch specifications
        self._couch_6dof = True
        self._couch_max_weight = 200.0  # kg

        # RapidArc parameters
        self._rapidarc_enabled = True
        self._variable_dose_rate = True
        self._variable_gantry_speed = True

        # Developer mode features
        self._developer_mode = False
        self._machine_log_enabled = True

    def set_beam_mode(self, mode: VarianBeamMode) -> bool:
        """Set beam delivery mode."""
        self._beam_mode = mode
        if mode == VarianBeamMode.PHOTON_FFF:
            self._fff_mode = True
        else:
            self._fff_mode = False
        return True

    def set_energy_fff(self, energy: float) -> bool:
        """Set FFF beam energy."""
        if energy not in self.PHOTON_ENERGIES_FFF:
            return False

        self._beam_mode = VarianBeamMode.PHOTON_FFF
        self._fff_mode = True
        self._current_energy = energy
        self._dose_rate = self.DOSE_RATES_FFF[energy]
        return True

    def get_max_dose_rate(self) -> float:
        """Get maximum dose rate for current mode."""
        if self._fff_mode:
            return self.DOSE_RATES_FFF.get(self._current_energy, 1400)
        return self.DOSE_RATES_FLAT.get(self._current_energy, 600)

    def deploy_kv_imager(self) -> bool:
        """Deploy kV imaging panel."""
        self._kv_imager_deployed = True
        return True

    def retract_kv_imager(self) -> bool:
        """Retract kV imaging panel."""
        self._kv_imager_deployed = False
        return True

    def acquire_cbct(
        self,
        preset: str = "pelvis"
    ) -> np.ndarray:
        """
        Acquire cone-beam CT.

        Args:
            preset: Imaging preset (head, thorax, pelvis, spotlight)
        """
        if not self._kv_imager_deployed:
            self.deploy_kv_imager()

        # Simulate CBCT acquisition
        matrix_size = self.obi.cbct_reconstruction_matrix
        cbct_volume = np.random.random((matrix_size, matrix_size, matrix_size))

        return cbct_volume

    def deliver_rapidarc(
        self,
        control_points: List[Dict],
        clockwise: bool = True
    ) -> bool:
        """
        Deliver RapidArc (VMAT) treatment.

        Args:
            control_points: List of control points with gantry, MLC, MU
            clockwise: Rotation direction
        """
        if not self._rapidarc_enabled:
            return False

        return self.deliver_vmat_arc(control_points)

    def get_machine_parameters(self) -> Dict:
        """Get detailed machine parameters."""
        return {
            "manufacturer": self.manufacturer,
            "model": f"TrueBeam {self.model.upper()}",
            "photon_energies_flat": self.PHOTON_ENERGIES_FLAT,
            "photon_energies_fff": self.PHOTON_ENERGIES_FFF,
            "electron_energies": self.ELECTRON_ENERGIES,
            "mlc_type": "HD 120",
            "max_field_size": self.mlc.SPECS.max_field_size,
            "max_dose_rate_fff": 2400,
            "max_gantry_speed": self._max_gantry_speed,
            "obi_installed": True,
            "rapidarc_enabled": self._rapidarc_enabled,
        }


class VarianHalcyon(LinearAccelerator):
    """
    Varian Halcyon Linear Accelerator.

    Features:
    - Single photon energy: 6 MV FFF
    - Maximum dose rate: 800 MU/min
    - Dual-layer stacked MLC (SX2)
    - Fully enclosed bore design
    - Integrated kV CBCT
    - Simplified workflow
    """

    def __init__(self):
        super().__init__(
            name="Varian Halcyon",
            max_photon_energy=6.0,
            max_electron_energy=0.0  # No electron mode
        )

        self.manufacturer = "Varian Medical Systems"

        # Halcyon uses dual-layer MLC
        self._mlc_upper = self._create_halcyon_mlc("upper")
        self._mlc_lower = self._create_halcyon_mlc("lower")

        # Single energy, FFF only
        self._current_energy = 6.0
        self._fff_mode = True
        self._dose_rate = 800.0

        # Bore design parameters
        self._bore_diameter = 1000.0  # mm
        self._source_to_isocenter = 1000.0  # mm

        # Gantry speed (faster than TrueBeam)
        self._max_gantry_speed = 4.0  # RPM
        self._gantry_rotation_time = 15.0  # seconds for 360°

        # Imaging (integrated MV and kV)
        self._imaging_ring = True
        self._kv_imaging_enabled = True

        # Couch
        self._couch_range_lateral = 230.0  # mm
        self._couch_range_longitudinal = 1300.0  # mm
        self._couch_range_vertical = 420.0  # mm

    def _create_halcyon_mlc(self, layer: str) -> MultiLeafCollimator:
        """Create Halcyon SX2 MLC layer."""
        mlc = MultiLeafCollimator(
            num_leaf_pairs=29,
            leaf_width=10.0,  # mm at isocenter
            mlc_type=MLCType.STANDARD,
            name=f"Halcyon SX2 {layer}"
        )
        mlc.max_leaf_speed = 50.0  # mm/s (very fast)
        return mlc

    def set_field_shape(
        self,
        aperture: List[Tuple[float, float]]
    ) -> bool:
        """
        Set dual-layer MLC aperture.

        Both layers work together for improved modulation.
        """
        # Upper layer defines coarse shape
        self._mlc_upper.set_field_shape(aperture)

        # Lower layer provides fine modulation
        # (offset by half leaf width)
        offset_aperture = [
            (a[0] + 2.5, a[1] + 2.5) for a in aperture
        ]
        self._mlc_lower.set_field_shape(offset_aperture)

        return True

    def calculate_transmission(self, x: float, y: float) -> float:
        """Calculate combined transmission through dual-layer MLC."""
        t_upper = self._mlc_upper.calculate_transmission(x, y)
        t_lower = self._mlc_lower.calculate_transmission(x, y - 5.0)  # Offset
        return t_upper * t_lower

    def acquire_iterative_cbct(self) -> np.ndarray:
        """Acquire iterative CBCT (iCBCT) with reduced dose."""
        # Halcyon uses iterative reconstruction
        matrix_size = 512
        return np.random.random((matrix_size, matrix_size, matrix_size))

    def get_bore_clearance(self, patient_size: float) -> float:
        """Calculate bore clearance for patient."""
        return self._bore_diameter / 2 - patient_size / 2


class VarianClinac(LinearAccelerator):
    """
    Varian Clinac Linear Accelerator (iX, 21EX, etc.).

    Legacy platform still widely used:
    - Photon energies: 6, 15/18 MV (model dependent)
    - Electron energies: 6-20 MeV
    - Millennium MLC
    """

    def __init__(self, model: str = "iX"):
        energy_configs = {
            "iX": {"photon": [6.0, 18.0], "electron": [6, 9, 12, 16, 20]},
            "21EX": {"photon": [6.0, 15.0], "electron": [6, 9, 12, 16, 20]},
            "2100C": {"photon": [6.0, 18.0], "electron": [6, 9, 12, 15, 18]},
        }

        config = energy_configs.get(model, energy_configs["iX"])

        super().__init__(
            name=f"Varian Clinac {model}",
            max_photon_energy=max(config["photon"]),
            max_electron_energy=max(config["electron"])
        )

        self.model = model
        self.manufacturer = "Varian Medical Systems"
        self._photon_energies = config["photon"]
        self._electron_energies = config["electron"]

        # Millennium 120 MLC
        self.mlc = VarianMLCMillennium("120")

        # Standard dose rates
        self._dose_rate = 600.0
        self._max_dose_rate = 600.0

        # Portal imaging
        self._portal_imager = True
        self._portal_imager_type = "aS1000"


class VarianEdge(VarianTrueBeam):
    """
    Varian Edge Radiosurgery System.

    Specialized for SRS/SBRT:
    - TrueBeam platform with radiosurgery enhancements
    - 6 FFF and 10 FFF beams
    - HD 120 MLC standard
    - PerfectPitch 6-DOF couch
    - Calypso real-time tracking (optional)
    """

    def __init__(self):
        super().__init__(model="STx")
        self.name = "Varian Edge"

        # Radiosurgery-specific features
        self._perfectpitch_couch = True
        self._calypso_enabled = False
        self._optical_surface_monitoring = True

        # ExacTrac integration
        self._exactrac = VarianExactrac()
        self._exactrac_enabled = True

        # Specialized SRS cone set
        self._srs_cones = [4.0, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]  # mm

        # Gating capabilities
        self._respiratory_gating = True
        self._rpm_system = True

    def enable_calypso(self) -> bool:
        """Enable Calypso electromagnetic tracking."""
        self._calypso_enabled = True
        return True

    def set_srs_cone(self, diameter: float) -> bool:
        """Set SRS cone collimator."""
        if diameter not in self._srs_cones:
            return False

        # Disable MLC when using cones
        self._srs_cone_active = True
        self._current_cone = diameter
        return True

    def acquire_exactrac_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """Acquire ExacTrac stereoscopic images."""
        if not self._exactrac_enabled:
            return None, None

        # Simulate stereo image acquisition
        img_size = (1024, 1024)
        img1 = np.random.random(img_size)
        img2 = np.random.random(img_size)
        return img1, img2

    def calculate_6dof_correction(
        self,
        target_position: Position3D,
        measured_position: Position3D
    ) -> Dict:
        """Calculate 6-DOF couch correction."""
        translation = Position3D(
            x=target_position.x - measured_position.x,
            y=target_position.y - measured_position.y,
            z=target_position.z - measured_position.z
        )

        # Rotation corrections (simplified)
        rotation = {
            "pitch": 0.0,
            "roll": 0.0,
            "yaw": 0.0
        }

        return {
            "translation": translation,
            "rotation": rotation,
            "within_tolerance": translation.distance_to(Position3D()) < 2.0
        }


class VarianProBeam(BeamDeliverySystem):
    """
    Varian ProBeam Proton Therapy System.

    Features:
    - Superconducting cyclotron (250 MeV)
    - Pencil beam scanning
    - 360° rotating gantry
    - Cone-beam CT integration
    """

    def __init__(self, gantry_rooms: int = 1):
        super().__init__(name="Varian ProBeam")

        self.manufacturer = "Varian Medical Systems"
        self.gantry_rooms = gantry_rooms

        # Cyclotron specifications
        self._cyclotron_energy = 250.0  # MeV
        self._cyclotron_type = "superconducting"
        self._beam_current_range = (1.0, 300.0)  # nA

        # Energy selection system
        self._energy_range = (70.0, 250.0)  # MeV
        self._energy_layers = 94  # Discrete energy layers
        self._energy_switch_time = 0.5  # seconds between layers

        # Scanning system
        self._scan_field_size = (300.0, 400.0)  # mm (x, y)
        self._spot_size_range = (3.0, 10.0)  # mm sigma in air
        self._max_scan_speed = 20.0  # m/s

        # Gantry
        self._gantry_rotation_range = (0.0, 360.0)
        self._gantry_speed = 1.0  # RPM
        self._gantry_weight = 200.0  # tons

        # Imaging
        self._cbct_integrated = True
        self._orthogonal_xray = True

        # Beam delivery
        self._delivery_mode = "PBS"  # Pencil Beam Scanning
        self._is_ready = False

    def initialize(self) -> bool:
        """Initialize ProBeam system."""
        self._is_ready = True
        self._is_calibrated = False
        return True

    def calibrate(self) -> bool:
        """Calibrate the system."""
        self._is_calibrated = True
        return True

    def set_position(self, position: Position3D) -> bool:
        """Set treatment position."""
        self._position = position
        return True

    def set_energy(self, energy: float) -> bool:
        """Set beam energy."""
        if not (self._energy_range[0] <= energy <= self._energy_range[1]):
            return False

        self._current_energy = energy
        return True

    def calculate_range(self, energy: float) -> float:
        """Calculate range in water."""
        # R (mm) = 0.022 * E^1.77 * 10
        return 0.022 * energy ** 1.77 * 10

    def deliver_spot(
        self,
        x: float,
        y: float,
        energy: float,
        mu: float
    ) -> bool:
        """Deliver single spot."""
        self.set_energy(energy)
        return True

    def deliver_layer(
        self,
        spots: List[Tuple[float, float, float]],  # (x, y, MU)
        energy: float
    ) -> bool:
        """Deliver energy layer."""
        self.set_energy(energy)
        for x, y, mu in spots:
            self.deliver_spot(x, y, energy, mu)
        return True

    def deliver_dose(
        self,
        beam_params: BeamParameters,
        monitor_units: float
    ) -> DoseDistribution:
        """Deliver treatment dose."""
        # Create dose distribution
        dose_dist = DoseDistribution(
            origin=Position3D(-150, -150, 0),
            spacing=(2.0, 2.0, 1.0),
            dimensions=(150, 150, 300)
        )

        # Calculate Bragg peak
        range_mm = self.calculate_range(beam_params.energy)

        for k in range(dose_dist.dimensions[2]):
            depth = k * dose_dist.spacing[2]
            # Simplified Bragg curve
            if depth < range_mm:
                dose_factor = 1 + 3 * (depth / range_mm) ** 2
            else:
                dose_factor = math.exp(-((depth - range_mm) ** 2) / 50)

            for i in range(dose_dist.dimensions[0]):
                for j in range(dose_dist.dimensions[1]):
                    x = dose_dist.origin.x + i * dose_dist.spacing[0]
                    y = dose_dist.origin.y + j * dose_dist.spacing[1]
                    r = math.sqrt(x ** 2 + y ** 2)
                    lateral = math.exp(-r ** 2 / 100)
                    dose_dist.dose_grid[i, j, k] = (
                        monitor_units * 0.01 * dose_factor * lateral
                    )

        return dose_dist

    def get_status(self) -> Dict:
        """Get system status."""
        return {
            "name": self.name,
            "manufacturer": self.manufacturer,
            "cyclotron_energy": self._cyclotron_energy,
            "current_energy": getattr(self, '_current_energy', 0),
            "energy_range": self._energy_range,
            "is_ready": self._is_ready,
            "is_calibrated": self._is_calibrated,
            "gantry_rooms": self.gantry_rooms,
        }
