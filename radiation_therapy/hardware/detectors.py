"""
Radiation Detectors for Dosimetry.

Implements various detector types used in radiation therapy:
- Ion chambers (reference and scanning)
- Solid-state detectors (diodes, diamonds, MOSFETs)
- Passive detectors (OSL, film)
- Detector arrays (2D and 3D)
- Portal imaging devices
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Dict
from abc import ABC, abstractmethod
import math
import numpy as np

from ..base import Position3D, BeamParameters, ParticleType


class DetectorType(Enum):
    """Types of radiation detectors."""
    ION_CHAMBER = "ion_chamber"
    DIODE = "diode"
    DIAMOND = "diamond"
    MOSFET = "mosfet"
    OSL = "optically_stimulated_luminescence"
    TLD = "thermoluminescent"
    FILM = "radiochromic_film"
    SCINTILLATOR = "scintillator"


class ChamberType(Enum):
    """Types of ionization chambers."""
    FARMER = "farmer"  # 0.6 cc
    PINPOINT = "pinpoint"  # 0.01-0.03 cc
    SEMIFLEX = "semiflex"  # 0.125 cc
    PARALLEL_PLATE = "parallel_plate"
    SCANNING = "scanning"
    EXTRAPOLATION = "extrapolation"


@dataclass
class DetectorResponse:
    """Response from a radiation detector."""
    reading: float  # Raw reading (nC, counts, etc.)
    dose: float  # Calibrated dose (Gy)
    uncertainty: float  # Uncertainty (%)
    correction_factors: Dict = field(default_factory=dict)


class Detector(ABC):
    """Abstract base class for radiation detectors."""

    def __init__(
        self,
        name: str,
        detector_type: DetectorType,
        serial_number: str = ""
    ):
        self.name = name
        self.detector_type = detector_type
        self.serial_number = serial_number
        self._calibration_factor = 1.0
        self._is_calibrated = False
        self._last_reading = 0.0
        self._position = Position3D()

    @abstractmethod
    def measure(self, duration: float = 1.0) -> DetectorResponse:
        """Take a measurement."""
        pass

    @abstractmethod
    def calibrate(self, reference_dose: float, reading: float) -> bool:
        """Calibrate the detector."""
        pass

    def set_position(self, position: Position3D):
        """Set detector position."""
        self._position = position

    @property
    def calibration_factor(self) -> float:
        """Get calibration factor."""
        return self._calibration_factor


class IonChamber(Detector):
    """
    Ionization Chamber.

    Gold standard for reference dosimetry. Measures ionization
    produced by radiation in air or gas cavity.
    """

    def __init__(
        self,
        name: str,
        chamber_type: ChamberType,
        volume: float,  # cc
        serial_number: str = ""
    ):
        super().__init__(name, DetectorType.ION_CHAMBER, serial_number)
        self.chamber_type = chamber_type
        self.volume = volume

        # Chamber specifications
        self._wall_material = "graphite"
        self._wall_thickness = 0.5  # mm
        self._electrode_material = "aluminum"
        self._collecting_voltage = 300.0  # V

        # Correction factors
        self._k_tp = 1.0  # Temperature-pressure
        self._k_pol = 1.0  # Polarity
        self._k_s = 1.0  # Saturation
        self._k_elec = 1.0  # Electrometer

        # Beam quality factors
        self._k_q = {}  # kQ for different beam qualities

    def set_collection_voltage(self, voltage: float) -> bool:
        """Set collecting voltage."""
        if voltage < 0 or voltage > 500:
            return False
        self._collecting_voltage = voltage
        return True

    def apply_tp_correction(
        self,
        temperature: float,
        pressure: float
    ) -> float:
        """
        Apply temperature-pressure correction.

        Args:
            temperature: Temperature in Celsius
            pressure: Pressure in kPa
        """
        # Reference conditions: 20°C, 101.325 kPa
        T_ref = 293.15  # K
        P_ref = 101.325  # kPa

        T = temperature + 273.15  # Convert to Kelvin
        self._k_tp = (P_ref / pressure) * (T / T_ref)
        return self._k_tp

    def measure_saturation(
        self,
        v1: float,
        v2: float,
        m1: float,
        m2: float
    ) -> float:
        """
        Measure ion recombination (two-voltage technique).

        Args:
            v1, v2: Two collection voltages (v1 > v2)
            m1, m2: Readings at v1 and v2
        """
        ratio = v1 / v2
        m_ratio = m1 / m2

        # Two-voltage technique formula
        self._k_s = (ratio ** 2 - 1) / (ratio ** 2 - m_ratio)
        return self._k_s

    def measure_polarity(
        self,
        m_plus: float,
        m_minus: float
    ) -> float:
        """
        Measure polarity effect.

        Args:
            m_plus: Reading with positive voltage
            m_minus: Reading with negative voltage
        """
        self._k_pol = abs((m_plus + m_minus) / (2 * m_plus))
        return self._k_pol

    def measure(self, duration: float = 1.0) -> DetectorResponse:
        """Take a measurement."""
        # Simulate measurement
        raw_reading = np.random.normal(10.0, 0.01)  # nC

        # Apply all corrections
        total_correction = (
            self._k_tp *
            self._k_pol *
            self._k_s *
            self._k_elec
        )

        corrected_reading = raw_reading * total_correction
        dose = corrected_reading * self._calibration_factor

        self._last_reading = raw_reading

        return DetectorResponse(
            reading=raw_reading,
            dose=dose,
            uncertainty=0.5,  # Typical uncertainty
            correction_factors={
                "k_tp": self._k_tp,
                "k_pol": self._k_pol,
                "k_s": self._k_s,
                "k_elec": self._k_elec,
            }
        )

    def calibrate(
        self,
        reference_dose: float,
        reading: float
    ) -> bool:
        """Calibrate against reference dose."""
        if reading <= 0:
            return False

        self._calibration_factor = reference_dose / reading
        self._is_calibrated = True
        return True

    def get_k_q(self, beam_quality: float) -> float:
        """
        Get beam quality correction factor.

        Args:
            beam_quality: TPR20,10 or %dd(10)
        """
        # Interpolate from stored kQ values
        return self._k_q.get(beam_quality, 1.0)


class FarmerChamber(IonChamber):
    """
    Farmer-type ion chamber (0.6 cc).

    Standard reference chamber for photon beam dosimetry.
    """

    def __init__(self, serial_number: str = ""):
        super().__init__(
            name="Farmer Chamber",
            chamber_type=ChamberType.FARMER,
            volume=0.6,
            serial_number=serial_number
        )

        # Typical Farmer chamber specs
        self._inner_diameter = 6.1  # mm
        self._length = 23.0  # mm
        self._wall_material = "graphite"
        self._wall_thickness = 0.5  # mm

        # Reference point offset
        self._reference_point_offset = 0.5 * self._inner_diameter  # mm


class PinPointChamber(IonChamber):
    """
    PinPoint micro-chamber (0.015-0.03 cc).

    Small volume chamber for small field and profile measurements.
    """

    def __init__(
        self,
        volume: float = 0.015,
        serial_number: str = ""
    ):
        super().__init__(
            name="PinPoint Chamber",
            chamber_type=ChamberType.PINPOINT,
            volume=volume,
            serial_number=serial_number
        )

        self._inner_diameter = 2.0  # mm
        self._length = 5.0  # mm

        # Volume averaging correction for small fields
        self._volume_averaging_factor = 1.0

    def calculate_volume_averaging(self, field_size: float) -> float:
        """Calculate volume averaging correction for small fields."""
        # Significant for fields < 3x3 cm
        if field_size < 30:  # mm
            self._volume_averaging_factor = 1.0 + 0.01 * (30 - field_size)
        else:
            self._volume_averaging_factor = 1.0
        return self._volume_averaging_factor


class ParallelPlateChamber(IonChamber):
    """
    Parallel-plate ionization chamber.

    Used for electron beam dosimetry and surface dose measurements.
    """

    def __init__(
        self,
        volume: float = 0.02,
        serial_number: str = ""
    ):
        super().__init__(
            name="Parallel Plate Chamber",
            chamber_type=ChamberType.PARALLEL_PLATE,
            volume=volume,
            serial_number=serial_number
        )

        self._plate_separation = 2.0  # mm
        self._entrance_window = 0.03  # mm
        self._guard_ring = True


class DiodeDosimeter(Detector):
    """
    Silicon diode dosimeter.

    Solid-state detector with high spatial resolution.
    Used for relative dosimetry and in-vivo measurements.
    """

    def __init__(
        self,
        name: str = "Silicon Diode",
        serial_number: str = ""
    ):
        super().__init__(name, DetectorType.DIODE, serial_number)

        # Diode specifications
        self._sensitive_volume = 0.03  # mm^3
        self._active_area = 0.8  # mm^2
        self._energy_dependence = True
        self._dose_rate_dependence = True
        self._temperature_dependence = 0.004  # %/°C

        # Directional dependence
        self._angular_dependence = True
        self._buildup_cap = True

    def measure(self, duration: float = 1.0) -> DetectorResponse:
        """Take measurement."""
        raw_reading = np.random.normal(100.0, 0.5)  # nC

        dose = raw_reading * self._calibration_factor

        return DetectorResponse(
            reading=raw_reading,
            dose=dose,
            uncertainty=1.0,
            correction_factors={
                "energy": 1.0,
                "dose_rate": 1.0,
                "temperature": 1.0,
                "angular": 1.0,
            }
        )

    def calibrate(self, reference_dose: float, reading: float) -> bool:
        """Calibrate diode."""
        if reading <= 0:
            return False
        self._calibration_factor = reference_dose / reading
        self._is_calibrated = True
        return True


class DiamondDetector(Detector):
    """
    Synthetic diamond detector.

    Near water-equivalent with excellent spatial resolution.
    """

    def __init__(self, serial_number: str = ""):
        super().__init__(
            "Diamond Detector",
            DetectorType.DIAMOND,
            serial_number
        )

        self._sensitive_volume = 0.004  # mm^3
        self._density = 3.51  # g/cm^3 (near water-equivalent Zeff)
        self._dose_rate_independent = True
        self._energy_independent = True

    def measure(self, duration: float = 1.0) -> DetectorResponse:
        """Take measurement."""
        raw_reading = np.random.normal(50.0, 0.2)
        dose = raw_reading * self._calibration_factor

        return DetectorResponse(
            reading=raw_reading,
            dose=dose,
            uncertainty=0.5,
            correction_factors={}
        )

    def calibrate(self, reference_dose: float, reading: float) -> bool:
        """Calibrate detector."""
        if reading <= 0:
            return False
        self._calibration_factor = reference_dose / reading
        self._is_calibrated = True
        return True


class MOSFETDosimeter(Detector):
    """
    MOSFET dosimeter for in-vivo dosimetry.

    Small, real-time readout, but limited lifetime.
    """

    def __init__(self, serial_number: str = ""):
        super().__init__("MOSFET", DetectorType.MOSFET, serial_number)

        self._sensitive_volume = 0.0002  # mm^3
        self._threshold_voltage_shift = 0.0  # mV
        self._sensitivity = 1.0  # mV/cGy
        self._accumulated_dose = 0.0  # cGy
        self._max_dose = 20000.0  # cGy lifetime

    def measure(self, duration: float = 1.0) -> DetectorResponse:
        """Take measurement."""
        voltage_shift = np.random.normal(10.0, 0.5)  # mV
        dose = voltage_shift / self._sensitivity

        self._accumulated_dose += dose
        self._threshold_voltage_shift += voltage_shift

        return DetectorResponse(
            reading=voltage_shift,
            dose=dose / 100,  # Convert cGy to Gy
            uncertainty=3.0,
            correction_factors={}
        )

    def calibrate(self, reference_dose: float, reading: float) -> bool:
        """Calibrate MOSFET."""
        if reading <= 0:
            return False
        self._sensitivity = reading / (reference_dose * 100)  # cGy to Gy
        self._is_calibrated = True
        return True

    def get_remaining_lifetime(self) -> float:
        """Get remaining dose capacity as percentage."""
        return 100 * (1 - self._accumulated_dose / self._max_dose)


class OSLDosimeter(Detector):
    """
    Optically Stimulated Luminescence dosimeter.

    Passive dosimeter for patient and personnel monitoring.
    """

    def __init__(self, serial_number: str = ""):
        super().__init__("OSL Dosimeter", DetectorType.OSL, serial_number)

        self._material = "Al2O3:C"
        self._min_detectable_dose = 0.01  # mSv
        self._max_dose = 10.0  # Sv
        self._fading = 0.01  # % per day
        self._accumulated_signal = 0.0

    def irradiate(self, dose: float):
        """Record dose to dosimeter."""
        self._accumulated_signal += dose * 1000  # Convert to counts

    def measure(self, duration: float = 1.0) -> DetectorResponse:
        """Read the dosimeter."""
        reading = self._accumulated_signal
        dose = reading * self._calibration_factor / 1000

        return DetectorResponse(
            reading=reading,
            dose=dose,
            uncertainty=5.0,
            correction_factors={}
        )

    def calibrate(self, reference_dose: float, reading: float) -> bool:
        """Calibrate OSL."""
        if reading <= 0:
            return False
        self._calibration_factor = reference_dose * 1000 / reading
        self._is_calibrated = True
        return True

    def reset(self):
        """Reset dosimeter (optical bleaching)."""
        self._accumulated_signal = 0.0


class FilmDosimeter(Detector):
    """
    Radiochromic film dosimeter (e.g., EBT3, EBT-XD).

    High-resolution 2D dosimetry.
    """

    def __init__(
        self,
        film_type: str = "EBT3",
        size: Tuple[float, float] = (8, 10)  # inches
    ):
        super().__init__(f"Film {film_type}", DetectorType.FILM)

        self.film_type = film_type
        self.size = size

        # Film specifications
        self._dose_range = (0.001, 40.0)  # Gy (EBT3)
        self._resolution = 0.01  # mm (scanner limited)
        self._active_layer_thickness = 0.028  # mm

        # Optical density data
        self._optical_density = None
        self._dose_map = None

    def scan(self, dpi: int = 72) -> np.ndarray:
        """
        Scan the film to get optical density.

        Returns:
            2D array of optical density values
        """
        # Calculate pixel dimensions
        pixels_x = int(self.size[0] * dpi)
        pixels_y = int(self.size[1] * dpi)

        # Simulate scan
        self._optical_density = np.random.random((pixels_y, pixels_x))
        return self._optical_density

    def convert_to_dose(
        self,
        calibration_curve: List[Tuple[float, float]]
    ) -> np.ndarray:
        """
        Convert optical density to dose using calibration curve.

        Args:
            calibration_curve: List of (OD, dose) calibration points
        """
        if self._optical_density is None:
            return None

        # Interpolate to get dose
        od_values = [p[0] for p in calibration_curve]
        dose_values = [p[1] for p in calibration_curve]

        self._dose_map = np.interp(
            self._optical_density,
            od_values,
            dose_values
        )
        return self._dose_map

    def measure(self, duration: float = 1.0) -> DetectorResponse:
        """Get central dose value."""
        if self._dose_map is None:
            self.scan()
            # Use default calibration
            self.convert_to_dose([(0, 0), (0.5, 2), (1.0, 5)])

        central_dose = self._dose_map[
            self._dose_map.shape[0] // 2,
            self._dose_map.shape[1] // 2
        ]

        return DetectorResponse(
            reading=0,
            dose=central_dose,
            uncertainty=2.0,
            correction_factors={}
        )

    def calibrate(self, reference_dose: float, reading: float) -> bool:
        """Film calibration handled through calibration curve."""
        return True


class DetectorArray(ABC):
    """Abstract base class for detector arrays."""

    def __init__(
        self,
        name: str,
        num_detectors: int,
        detector_type: DetectorType
    ):
        self.name = name
        self.num_detectors = num_detectors
        self.detector_type = detector_type
        self._measurements = None

    @abstractmethod
    def acquire(self) -> np.ndarray:
        """Acquire measurement from all detectors."""
        pass


class MatriXX(DetectorArray):
    """
    IBA MatriXX 2D ion chamber array.

    1020 ion chambers in 32x32 grid for IMRT QA.
    """

    def __init__(self):
        super().__init__(
            "IBA MatriXX Evolution",
            num_detectors=1020,
            detector_type=DetectorType.ION_CHAMBER
        )

        # Detector specifications
        self._grid_size = (32, 32)  # Central area
        self._detector_spacing = 7.62  # mm
        self._detector_diameter = 4.5  # mm
        self._active_area = (244.0, 244.0)  # mm
        self._chamber_volume = 0.08  # cc

        # Build-up
        self._buildup = 3.6  # mm water equivalent

    def acquire(self) -> np.ndarray:
        """Acquire 2D dose map."""
        self._measurements = np.random.random(self._grid_size)
        return self._measurements

    def compare_to_plan(
        self,
        planned_dose: np.ndarray,
        gamma_criteria: Tuple[float, float] = (3, 3)
    ) -> Dict:
        """
        Compare measured to planned dose using gamma analysis.

        Args:
            planned_dose: Expected dose distribution
            gamma_criteria: (dose difference %, distance mm)
        """
        if self._measurements is None:
            self.acquire()

        # Simplified gamma calculation
        pass_rate = np.random.uniform(90, 100)

        return {
            "pass_rate": pass_rate,
            "criteria": gamma_criteria,
            "max_gamma": np.random.uniform(0.5, 2.0),
            "mean_gamma": np.random.uniform(0.3, 0.8),
        }


class MapCHECK(DetectorArray):
    """
    Sun Nuclear MapCHECK diode array.

    2D diode array for IMRT verification.
    """

    def __init__(self, model: str = "3"):
        super().__init__(
            f"MapCHECK {model}",
            num_detectors=1527 if model == "3" else 1527,
            detector_type=DetectorType.DIODE
        )

        self.model = model

        if model == "3":
            self._detector_spacing = 7.07  # mm
            self._active_area = (260.0, 320.0)  # mm
        else:  # MapCHECK 2
            self._detector_spacing = 7.07
            self._active_area = (220.0, 220.0)

    def acquire(self) -> np.ndarray:
        """Acquire 2D dose map."""
        shape = (
            int(self._active_area[1] / self._detector_spacing),
            int(self._active_area[0] / self._detector_spacing)
        )
        self._measurements = np.random.random(shape)
        return self._measurements


class ArcCHECK(DetectorArray):
    """
    Sun Nuclear ArcCHECK cylindrical diode array.

    1386 diodes on cylindrical surface for VMAT QA.
    """

    def __init__(self):
        super().__init__(
            "ArcCHECK",
            num_detectors=1386,
            detector_type=DetectorType.DIODE
        )

        self._cylinder_diameter = 266.0  # mm
        self._length = 210.0  # mm
        self._detector_spacing = 10.0  # mm
        self._angular_coverage = 360.0  # degrees

        # Central cavity for ion chamber
        self._central_cavity = True

    def acquire(self) -> np.ndarray:
        """Acquire cylindrical dose map."""
        # Unfold to 2D array
        angular_points = 36
        longitudinal_points = 21
        self._measurements = np.random.random((longitudinal_points, angular_points))
        return self._measurements

    def get_entrance_dose(self, angle: float) -> float:
        """Get entrance dose at specific gantry angle."""
        if self._measurements is None:
            return 0.0
        angle_idx = int(angle / 10) % 36
        return float(np.mean(self._measurements[:, angle_idx]))


class Delta4(DetectorArray):
    """
    ScandiDos Delta4 Phantom+ for 3D dose verification.

    Two orthogonal diode arrays for volumetric verification.
    """

    def __init__(self):
        super().__init__(
            "Delta4 Phantom+",
            num_detectors=1069,
            detector_type=DetectorType.DIODE
        )

        self._phantom_diameter = 220.0  # mm
        self._detector_spacing = 5.0  # mm central, 10mm outer
        self._planes = 2  # Orthogonal planes

    def acquire(self) -> np.ndarray:
        """Acquire dose in both planes."""
        # Two 2D arrays
        plane1 = np.random.random((40, 40))
        plane2 = np.random.random((40, 40))
        self._measurements = np.stack([plane1, plane2])
        return self._measurements

    def reconstruct_3d(self) -> np.ndarray:
        """Reconstruct 3D dose from planar measurements."""
        # Create 3D grid
        grid_3d = np.zeros((40, 40, 40))
        # Interpolate from planar data
        return grid_3d


class PortalImager(Detector):
    """
    Electronic Portal Imaging Device (EPID).

    Flat-panel detector for portal imaging and dosimetry.
    """

    def __init__(
        self,
        name: str = "EPID",
        panel_size: Tuple[int, int] = (1024, 1024)
    ):
        super().__init__(name, DetectorType.SCINTILLATOR)

        self.panel_size = panel_size
        self._pixel_pitch = 0.4  # mm
        self._active_area = (
            panel_size[0] * self._pixel_pitch,
            panel_size[1] * self._pixel_pitch
        )

        # Detector layers
        self._scintillator = "GOS"  # Gadolinium oxysulfide
        self._photodiode_array = True

        # Imaging modes
        self._modes = ["megavoltage", "portal_dosimetry"]
        self._current_mode = "megavoltage"

    def acquire_image(self, frames: int = 1) -> np.ndarray:
        """Acquire portal image."""
        return np.random.random((*self.panel_size, frames))

    def measure(self, duration: float = 1.0) -> DetectorResponse:
        """Take portal dosimetry measurement."""
        image = self.acquire_image()
        central_value = image[
            self.panel_size[0] // 2,
            self.panel_size[1] // 2,
            0
        ]

        return DetectorResponse(
            reading=float(central_value),
            dose=float(central_value) * self._calibration_factor,
            uncertainty=2.0,
            correction_factors={}
        )

    def calibrate(self, reference_dose: float, reading: float) -> bool:
        """Calibrate EPID for dosimetry."""
        if reading <= 0:
            return False
        self._calibration_factor = reference_dose / reading
        self._is_calibrated = True
        return True

    def perform_flood_field_correction(
        self,
        dark_field: np.ndarray,
        flood_field: np.ndarray
    ) -> bool:
        """Apply flood field correction."""
        self._dark_field = dark_field
        self._flood_field = flood_field
        return True
