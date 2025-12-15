"""
Linear Accelerator (LINAC) System Module.

This module implements the core components of a medical linear accelerator:
- Electron gun and accelerating waveguide
- Bending magnets and beam steering
- Multi-leaf collimators
- Real-time imaging for tumor tracking
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Dict, Callable
import math
import numpy as np

from .base import (
    RadiationSource,
    BeamDeliverySystem,
    SafetyInterlock,
    BeamParameters,
    Position3D,
    Vector3D,
    DoseDistribution,
    TreatmentTarget,
    ParticleType,
    BeamModality,
)


class ElectronGunState(Enum):
    """Operating states of the electron gun."""
    OFF = "off"
    STANDBY = "standby"
    WARMUP = "warmup"
    READY = "ready"
    ACTIVE = "active"
    FAULT = "fault"


@dataclass
class ElectronGunParameters:
    """Parameters for electron gun operation."""
    filament_current: float = 0.0  # Amps
    grid_voltage: float = 0.0  # Volts (controls beam current)
    cathode_temperature: float = 0.0  # Kelvin
    vacuum_pressure: float = 1e-7  # Torr
    pulse_width: float = 4.0  # microseconds
    pulse_repetition_rate: float = 360.0  # Hz


class ElectronGun(RadiationSource):
    """
    Electron gun for linear accelerator.

    Generates electrons through thermionic emission from a heated cathode.
    The electron beam current is controlled by the grid voltage.
    """

    # Physical constants
    BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
    ELECTRON_CHARGE = 1.602176634e-19  # Coulombs
    RICHARDSON_CONSTANT = 1.20173e6  # A/(m^2·K^2)

    def __init__(
        self,
        name: str = "Electron Gun",
        max_beam_current: float = 300.0,  # mA
        cathode_material: str = "tungsten"
    ):
        super().__init__(
            name=name,
            max_energy=0.050,  # 50 keV injection energy
            particle_type=ParticleType.ELECTRON
        )
        self.max_beam_current = max_beam_current
        self.cathode_material = cathode_material
        self.state = ElectronGunState.OFF
        self.parameters = ElectronGunParameters()
        self._target_beam_current = 0.0
        self._work_function = self._get_work_function(cathode_material)

    def _get_work_function(self, material: str) -> float:
        """Get work function for cathode material in eV."""
        work_functions = {
            "tungsten": 4.54,
            "thoriated_tungsten": 2.63,
            "lanthanum_hexaboride": 2.66,
            "barium_oxide": 1.0,
        }
        return work_functions.get(material, 4.54)

    def initialize(self) -> bool:
        """Initialize the electron gun system."""
        self.state = ElectronGunState.STANDBY
        self.parameters.vacuum_pressure = 1e-7  # Establish vacuum
        return True

    def warmup(self, target_temperature: float = 2500.0) -> bool:
        """
        Warm up the cathode to operating temperature.

        Args:
            target_temperature: Target cathode temperature in Kelvin
        """
        if self.state not in [ElectronGunState.STANDBY, ElectronGunState.READY]:
            return False

        self.state = ElectronGunState.WARMUP
        # Simulate gradual heating
        self.parameters.cathode_temperature = target_temperature
        self.parameters.filament_current = self._calculate_filament_current(
            target_temperature
        )
        self.state = ElectronGunState.READY
        return True

    def _calculate_filament_current(self, temperature: float) -> float:
        """Calculate required filament current for target temperature."""
        # Simplified model: I ∝ T^2 for resistive heating
        reference_current = 10.0  # Amps at 2500K
        reference_temp = 2500.0
        return reference_current * (temperature / reference_temp) ** 2

    def set_energy(self, energy: float) -> bool:
        """Set injection energy (typically fixed for electron gun)."""
        if energy <= self.max_energy:
            self._current_energy = energy
            return True
        return False

    def set_beam_current(self, current: float) -> bool:
        """
        Set the beam current by adjusting grid voltage.

        Args:
            current: Desired beam current in mA
        """
        if current > self.max_beam_current:
            return False

        self._target_beam_current = current
        # Grid voltage controls beam current (simplified model)
        self.parameters.grid_voltage = self._calculate_grid_voltage(current)
        self._beam_current = current
        return True

    def _calculate_grid_voltage(self, beam_current: float) -> float:
        """Calculate grid voltage needed for desired beam current."""
        # Child-Langmuir law approximation: I ∝ V^(3/2)
        if beam_current <= 0:
            return -200.0  # Cutoff voltage
        reference_voltage = 50.0  # Volts for 100 mA
        reference_current = 100.0  # mA
        return reference_voltage * (beam_current / reference_current) ** (2/3)

    def beam_on(self) -> bool:
        """Activate electron emission."""
        if self.state != ElectronGunState.READY:
            return False

        if self.parameters.vacuum_pressure > 1e-5:
            self.state = ElectronGunState.FAULT
            return False

        self.state = ElectronGunState.ACTIVE
        self._is_active = True
        return True

    def beam_off(self) -> bool:
        """Stop electron emission."""
        self.state = ElectronGunState.READY
        self._is_active = False
        self._beam_current = 0.0
        self.parameters.grid_voltage = -200.0  # Cutoff
        return True

    def get_emission_current(self) -> float:
        """
        Calculate thermionic emission current using Richardson-Dushman equation.

        Returns:
            Emission current density in A/m^2
        """
        T = self.parameters.cathode_temperature
        if T <= 0:
            return 0.0

        # Richardson-Dushman: J = A * T^2 * exp(-W/(k*T))
        exponent = -self._work_function * self.ELECTRON_CHARGE / (
            self.BOLTZMANN_CONSTANT * T
        )
        return self.RICHARDSON_CONSTANT * T ** 2 * math.exp(exponent)

    def get_status(self) -> Dict:
        """Get current status of electron gun."""
        return {
            "state": self.state.value,
            "cathode_temperature": self.parameters.cathode_temperature,
            "filament_current": self.parameters.filament_current,
            "grid_voltage": self.parameters.grid_voltage,
            "beam_current": self._beam_current,
            "vacuum_pressure": self.parameters.vacuum_pressure,
            "is_active": self._is_active,
        }


class WaveguideType(Enum):
    """Types of accelerating waveguide structures."""
    TRAVELING_WAVE = "traveling_wave"
    STANDING_WAVE = "standing_wave"


@dataclass
class WaveguideParameters:
    """Parameters for the accelerating waveguide."""
    frequency: float = 2.998e9  # Hz (S-band, ~3 GHz)
    rf_power: float = 0.0  # MW
    phase_velocity: float = 0.0  # fraction of c
    shunt_impedance: float = 50.0  # MΩ/m
    quality_factor: float = 10000.0
    temperature: float = 40.0  # Celsius
    length: float = 1.5  # meters
    num_cavities: int = 30


class AcceleratingWaveguide:
    """
    Accelerating waveguide for LINAC.

    Uses RF electromagnetic fields to accelerate electrons to therapeutic energies.
    Supports both traveling-wave and standing-wave structures.
    """

    SPEED_OF_LIGHT = 2.998e8  # m/s

    def __init__(
        self,
        waveguide_type: WaveguideType = WaveguideType.STANDING_WAVE,
        max_energy: float = 25.0,  # MeV
        name: str = "Accelerating Waveguide"
    ):
        self.name = name
        self.waveguide_type = waveguide_type
        self.max_energy = max_energy
        self.parameters = WaveguideParameters()
        self._is_powered = False
        self._current_energy = 0.0
        self._gradient = 0.0  # MeV/m

    def initialize(self) -> bool:
        """Initialize the waveguide system."""
        self.parameters.phase_velocity = 1.0  # Velocity of light
        return True

    def set_rf_power(self, power: float) -> bool:
        """
        Set RF power to achieve desired beam energy.

        Args:
            power: RF power in MW
        """
        if power < 0:
            return False

        self.parameters.rf_power = power
        self._calculate_gradient()
        return True

    def _calculate_gradient(self):
        """Calculate accelerating gradient from RF power."""
        # E = sqrt(P * R * L) where R is shunt impedance
        # Simplified model
        P = self.parameters.rf_power * 1e6  # Convert to W
        R = self.parameters.shunt_impedance * 1e6  # Convert to Ω/m
        L = self.parameters.length

        if P > 0:
            # Energy gain in MeV
            self._gradient = math.sqrt(P * R) / 1e6  # MeV/m
            self._current_energy = min(
                self._gradient * L,
                self.max_energy
            )
        else:
            self._gradient = 0.0
            self._current_energy = 0.0

    def set_energy(self, energy: float) -> bool:
        """
        Configure waveguide for desired output energy.

        Args:
            energy: Desired beam energy in MeV
        """
        if energy > self.max_energy:
            return False

        # Calculate required RF power
        # E = sqrt(P * R * L), so P = E^2 / (R * L)
        R = self.parameters.shunt_impedance * 1e6
        L = self.parameters.length
        required_power = (energy * 1e6) ** 2 / (R * L) / 1e6  # MW

        return self.set_rf_power(required_power)

    def power_on(self) -> bool:
        """Turn on RF power to waveguide."""
        if self.parameters.rf_power <= 0:
            return False
        self._is_powered = True
        return True

    def power_off(self) -> bool:
        """Turn off RF power."""
        self._is_powered = False
        return True

    def get_beam_energy(self) -> float:
        """Get current beam energy in MeV."""
        return self._current_energy if self._is_powered else 0.0

    def calculate_transit_time(self) -> float:
        """Calculate electron transit time through waveguide."""
        # Approximate as speed of light for relativistic electrons
        return self.parameters.length / self.SPEED_OF_LIGHT

    def get_status(self) -> Dict:
        """Get current status of waveguide."""
        return {
            "type": self.waveguide_type.value,
            "rf_power": self.parameters.rf_power,
            "frequency": self.parameters.frequency,
            "gradient": self._gradient,
            "beam_energy": self._current_energy,
            "is_powered": self._is_powered,
            "temperature": self.parameters.temperature,
        }


class MagnetType(Enum):
    """Types of magnets in beam transport."""
    BENDING = "bending"  # 270-degree bend
    STEERING = "steering"  # Small corrections
    FOCUSING = "focusing"  # Quadrupole


@dataclass
class MagnetParameters:
    """Parameters for beam magnets."""
    field_strength: float = 0.0  # Tesla
    pole_gap: float = 20.0  # mm
    effective_length: float = 500.0  # mm
    current: float = 0.0  # Amps
    max_current: float = 500.0  # Amps


class BendingMagnet:
    """
    Bending magnet system for LINAC.

    Typically uses a 270-degree achromatic bending system to:
    1. Bend the electron beam toward the patient
    2. Filter out electrons with incorrect energy
    3. Provide energy analysis capability
    """

    ELECTRON_MASS = 0.511  # MeV/c^2
    SPEED_OF_LIGHT = 2.998e8  # m/s

    def __init__(
        self,
        bend_angle: float = 270.0,  # degrees
        bend_radius: float = 100.0,  # mm
        name: str = "Bending Magnet System"
    ):
        self.name = name
        self.bend_angle = bend_angle
        self.bend_radius = bend_radius
        self.parameters = MagnetParameters()
        self._is_energized = False

    def initialize(self) -> bool:
        """Initialize the bending magnet system."""
        return True

    def set_for_energy(self, energy: float) -> bool:
        """
        Configure bending magnet for specific beam energy.

        Args:
            energy: Beam energy in MeV

        The magnetic rigidity Bρ = p/q where p is momentum
        For relativistic electrons: p = sqrt(E^2 - m^2) in natural units
        """
        if energy <= 0:
            return False

        # Calculate momentum (relativistic)
        # E^2 = (pc)^2 + (mc^2)^2
        # p = sqrt(E^2 - m^2) / c in MeV/c
        momentum = math.sqrt(energy ** 2 - self.ELECTRON_MASS ** 2)  # MeV/c

        # Magnetic rigidity: Bρ = p/(qc) in T·m
        # For electrons: Bρ (T·m) = p (MeV/c) / 299.792
        rigidity = momentum / 299.792  # T·m

        # B = rigidity / radius
        radius_m = self.bend_radius / 1000.0  # Convert to meters
        self.parameters.field_strength = rigidity / radius_m

        # Calculate required current (simplified linear model)
        self.parameters.current = (
            self.parameters.field_strength *
            self.parameters.max_current / 2.0  # Assume 2T max field
        )

        return True

    def energize(self) -> bool:
        """Energize the bending magnet."""
        if self.parameters.current > self.parameters.max_current:
            return False
        self._is_energized = True
        return True

    def de_energize(self) -> bool:
        """De-energize the bending magnet."""
        self._is_energized = False
        return True

    def calculate_energy_filter_window(self) -> Tuple[float, float]:
        """
        Calculate the energy acceptance window of the bending system.

        Returns:
            Tuple of (min_energy, max_energy) in MeV that can pass through
        """
        # Typical achromatic system accepts ±3% energy spread
        nominal_energy = self._get_nominal_energy()
        return (nominal_energy * 0.97, nominal_energy * 1.03)

    def _get_nominal_energy(self) -> float:
        """Calculate nominal energy from current field setting."""
        if self.parameters.field_strength <= 0:
            return 0.0

        radius_m = self.bend_radius / 1000.0
        rigidity = self.parameters.field_strength * radius_m
        momentum = rigidity * 299.792  # MeV/c
        energy = math.sqrt(momentum ** 2 + self.ELECTRON_MASS ** 2)
        return energy

    def get_status(self) -> Dict:
        """Get current status of bending magnet."""
        return {
            "field_strength": self.parameters.field_strength,
            "current": self.parameters.current,
            "bend_angle": self.bend_angle,
            "bend_radius": self.bend_radius,
            "is_energized": self._is_energized,
            "nominal_energy": self._get_nominal_energy(),
        }


class BeamSteering:
    """
    Beam steering system using small correction magnets.

    Provides fine adjustment of beam position and angle at isocenter.
    Uses pairs of steering magnets for X and Y corrections.
    """

    def __init__(self, name: str = "Beam Steering System"):
        self.name = name
        self._x_steering = MagnetParameters()
        self._y_steering = MagnetParameters()
        self._x_offset = 0.0  # mm at isocenter
        self._y_offset = 0.0  # mm at isocenter
        self._is_active = False

    def initialize(self) -> bool:
        """Initialize steering system."""
        self._x_steering.max_current = 10.0  # Smaller than bending
        self._y_steering.max_current = 10.0
        return True

    def set_steering(
        self,
        x_offset: float = 0.0,
        y_offset: float = 0.0
    ) -> bool:
        """
        Set beam steering to achieve desired offset at isocenter.

        Args:
            x_offset: Desired X offset in mm
            y_offset: Desired Y offset in mm

        Returns:
            True if steering is within range
        """
        max_offset = 5.0  # mm maximum steering range

        if abs(x_offset) > max_offset or abs(y_offset) > max_offset:
            return False

        self._x_offset = x_offset
        self._y_offset = y_offset

        # Calculate required currents (linear approximation)
        # 1 mm offset requires approximately 0.5 A
        sensitivity = 0.5  # A/mm
        self._x_steering.current = x_offset * sensitivity
        self._y_steering.current = y_offset * sensitivity

        return True

    def enable(self) -> bool:
        """Enable steering corrections."""
        self._is_active = True
        return True

    def disable(self) -> bool:
        """Disable steering corrections."""
        self._is_active = False
        return True

    def get_current_offset(self) -> Tuple[float, float]:
        """Get current beam offset at isocenter."""
        if not self._is_active:
            return (0.0, 0.0)
        return (self._x_offset, self._y_offset)

    def auto_center(self, measured_x: float, measured_y: float) -> bool:
        """
        Automatically center beam based on measurements.

        Args:
            measured_x: Measured X position in mm
            measured_y: Measured Y position in mm
        """
        # Apply negative feedback to center beam
        correction_x = -measured_x
        correction_y = -measured_y
        return self.set_steering(
            self._x_offset + correction_x,
            self._y_offset + correction_y
        )

    def get_status(self) -> Dict:
        """Get steering system status."""
        return {
            "x_offset": self._x_offset,
            "y_offset": self._y_offset,
            "x_current": self._x_steering.current,
            "y_current": self._y_steering.current,
            "is_active": self._is_active,
        }


@dataclass
class LeafPosition:
    """Position of a single MLC leaf."""
    leaf_index: int
    position: float  # mm from center
    speed: float = 0.0  # mm/s
    max_position: float = 200.0  # mm
    min_position: float = -200.0  # mm


class MLCType(Enum):
    """Types of multi-leaf collimator designs."""
    STANDARD = "standard"  # Rectangular leaves
    DOUBLE_FOCUSED = "double_focused"  # Focused in both planes
    MICRO_MLC = "micro_mlc"  # High resolution


class MultiLeafCollimator:
    """
    Multi-Leaf Collimator (MLC) for beam shaping.

    Consists of opposing banks of tungsten leaves that can be
    independently positioned to create complex field shapes.
    Essential for IMRT and VMAT treatments.
    """

    def __init__(
        self,
        num_leaf_pairs: int = 60,
        leaf_width: float = 5.0,  # mm at isocenter
        mlc_type: MLCType = MLCType.STANDARD,
        name: str = "Multi-Leaf Collimator"
    ):
        self.name = name
        self.num_leaf_pairs = num_leaf_pairs
        self.leaf_width = leaf_width
        self.mlc_type = mlc_type
        self.max_leaf_speed = 25.0  # mm/s
        self.max_overtravel = 150.0  # mm past centerline

        # Initialize leaf positions (A bank and B bank)
        self._leaves_a: List[LeafPosition] = [
            LeafPosition(
                leaf_index=i,
                position=-100.0,  # Retracted
                max_position=self.max_overtravel,
                min_position=-200.0
            )
            for i in range(num_leaf_pairs)
        ]
        self._leaves_b: List[LeafPosition] = [
            LeafPosition(
                leaf_index=i,
                position=100.0,  # Retracted
                max_position=200.0,
                min_position=-self.max_overtravel
            )
            for i in range(num_leaf_pairs)
        ]

        self._is_calibrated = False
        self._interlock_gap = 0.5  # mm minimum gap between opposing leaves

    def initialize(self) -> bool:
        """Initialize MLC system."""
        return self.calibrate()

    def calibrate(self) -> bool:
        """Calibrate leaf positions using encoder feedback."""
        # Reset all leaves to known positions
        for leaf in self._leaves_a:
            leaf.position = -100.0
        for leaf in self._leaves_b:
            leaf.position = 100.0
        self._is_calibrated = True
        return True

    def set_leaf_position(
        self,
        bank: str,
        leaf_index: int,
        position: float
    ) -> bool:
        """
        Set position of a single leaf.

        Args:
            bank: "A" or "B"
            leaf_index: Index of leaf (0 to num_leaf_pairs-1)
            position: Position in mm from centerline
        """
        if leaf_index < 0 or leaf_index >= self.num_leaf_pairs:
            return False

        leaves = self._leaves_a if bank.upper() == "A" else self._leaves_b
        leaf = leaves[leaf_index]

        # Check position limits
        if position < leaf.min_position or position > leaf.max_position:
            return False

        # Check interlock gap with opposing leaf
        opposing = (
            self._leaves_b[leaf_index] if bank.upper() == "A"
            else self._leaves_a[leaf_index]
        )
        if bank.upper() == "A":
            if position > opposing.position - self._interlock_gap:
                return False
        else:
            if position < opposing.position + self._interlock_gap:
                return False

        leaf.position = position
        return True

    def set_field_shape(
        self,
        aperture_positions: List[Tuple[float, float]]
    ) -> bool:
        """
        Set MLC to create specified field shape.

        Args:
            aperture_positions: List of (left, right) positions for each leaf pair
                               in mm from centerline
        """
        if len(aperture_positions) != self.num_leaf_pairs:
            return False

        success = True
        for i, (left, right) in enumerate(aperture_positions):
            # A bank defines left edge, B bank defines right edge
            if not self.set_leaf_position("A", i, left):
                success = False
            if not self.set_leaf_position("B", i, right):
                success = False

        return success

    def set_rectangular_field(
        self,
        width: float,
        height: float,
        center_x: float = 0.0,
        center_y: float = 0.0
    ) -> bool:
        """
        Set MLC to create a rectangular field.

        Args:
            width: Field width in mm
            height: Field height in mm
            center_x: X offset of field center in mm
            center_y: Y offset of field center in mm
        """
        half_width = width / 2
        half_height = height / 2

        # Calculate which leaves are in the field
        center_leaf = self.num_leaf_pairs // 2
        leaves_in_field = int(height / self.leaf_width / 2) + 1

        aperture = []
        for i in range(self.num_leaf_pairs):
            leaf_center = (i - center_leaf) * self.leaf_width + center_y
            if abs(leaf_center) <= half_height:
                # Leaf is in field
                aperture.append((
                    center_x - half_width,
                    center_x + half_width
                ))
            else:
                # Leaf is outside field, close it
                aperture.append((-0.25, 0.25))

        return self.set_field_shape(aperture)

    def get_aperture(self) -> List[Tuple[float, float]]:
        """Get current aperture shape."""
        return [
            (self._leaves_a[i].position, self._leaves_b[i].position)
            for i in range(self.num_leaf_pairs)
        ]

    def calculate_transmission(self, x: float, y: float) -> float:
        """
        Calculate beam transmission at a point.

        Args:
            x: X coordinate in mm
            y: Y coordinate in mm

        Returns:
            Transmission factor (0.0 = blocked, 1.0 = open)
        """
        # Find which leaf pair covers this Y position
        center_leaf = self.num_leaf_pairs // 2
        leaf_index = int(y / self.leaf_width + center_leaf)

        if leaf_index < 0 or leaf_index >= self.num_leaf_pairs:
            return 0.0

        left_pos = self._leaves_a[leaf_index].position
        right_pos = self._leaves_b[leaf_index].position

        # Point is open if between leaf edges
        if left_pos < x < right_pos:
            return 1.0
        else:
            # Leaf leakage (typically 1-2% through tungsten)
            return 0.015

    def calculate_leaf_travel_time(
        self,
        target_aperture: List[Tuple[float, float]]
    ) -> float:
        """Calculate time needed to reach target aperture."""
        max_time = 0.0
        current = self.get_aperture()

        for i in range(self.num_leaf_pairs):
            dist_a = abs(target_aperture[i][0] - current[i][0])
            dist_b = abs(target_aperture[i][1] - current[i][1])
            time_a = dist_a / self.max_leaf_speed
            time_b = dist_b / self.max_leaf_speed
            max_time = max(max_time, time_a, time_b)

        return max_time

    def get_status(self) -> Dict:
        """Get MLC status."""
        return {
            "num_leaf_pairs": self.num_leaf_pairs,
            "leaf_width": self.leaf_width,
            "mlc_type": self.mlc_type.value,
            "is_calibrated": self._is_calibrated,
            "aperture": self.get_aperture(),
        }


class ImagingModality(Enum):
    """Imaging modalities for tumor tracking."""
    PORTAL_IMAGING = "portal"  # MV imaging
    CBCT = "cone_beam_ct"  # kV cone-beam CT
    FLUOROSCOPY = "fluoroscopy"  # Real-time kV
    SURFACE_TRACKING = "surface"  # Optical surface monitoring


@dataclass
class ImageParameters:
    """Parameters for imaging acquisition."""
    modality: ImagingModality
    kv: float = 120.0  # kV for imaging (kV modalities)
    ma: float = 200.0  # mA for imaging
    exposure_time: float = 10.0  # ms
    frame_rate: float = 5.0  # fps for fluoroscopy
    field_of_view: float = 400.0  # mm


class TumorTracker:
    """
    Real-time tumor tracking system.

    Combines multiple imaging modalities to track tumor position
    and motion in real-time during treatment delivery.
    """

    def __init__(self, name: str = "Tumor Tracking System"):
        self.name = name
        self._modalities: List[ImagingModality] = []
        self._is_tracking = False
        self._current_position = Position3D()
        self._motion_history: List[Position3D] = []
        self._tracking_threshold = 3.0  # mm, beam hold threshold
        self._reference_position = Position3D()
        self._gating_window = 5.0  # mm
        self._motion_model: Optional[Callable] = None

    def initialize(self, modalities: List[ImagingModality]) -> bool:
        """Initialize tracking with specified modalities."""
        self._modalities = modalities
        return True

    def set_reference_position(self, position: Position3D) -> bool:
        """Set the reference position for tracking."""
        self._reference_position = position
        return True

    def acquire_image(
        self,
        params: ImageParameters
    ) -> np.ndarray:
        """
        Acquire an image for tracking.

        Returns:
            Simulated image array
        """
        # Simulate image acquisition
        image_size = int(params.field_of_view / 1.0)  # 1mm pixels
        return np.random.random((image_size, image_size))

    def detect_tumor_position(self, image: np.ndarray) -> Position3D:
        """
        Detect tumor position from image.

        In a real system, this would use sophisticated image processing
        and pattern matching algorithms.
        """
        # Simulate detection with small random offset
        offset = Position3D(
            x=np.random.normal(0, 0.5),
            y=np.random.normal(0, 0.5),
            z=np.random.normal(0, 0.5)
        )
        return self._reference_position + offset

    def start_tracking(self) -> bool:
        """Start real-time tumor tracking."""
        if not self._modalities:
            return False
        self._is_tracking = True
        self._motion_history = []
        return True

    def stop_tracking(self) -> bool:
        """Stop tumor tracking."""
        self._is_tracking = False
        return True

    def update_position(self) -> Position3D:
        """Update tracked tumor position."""
        if not self._is_tracking:
            return self._current_position

        # Simulate position update from imaging
        params = ImageParameters(
            modality=self._modalities[0] if self._modalities else ImagingModality.CBCT
        )
        image = self.acquire_image(params)
        self._current_position = self.detect_tumor_position(image)
        self._motion_history.append(self._current_position)

        return self._current_position

    def get_displacement(self) -> float:
        """Get current displacement from reference position."""
        return self._current_position.distance_to(self._reference_position)

    def is_within_tolerance(self) -> bool:
        """Check if tumor is within tracking tolerance."""
        return self.get_displacement() <= self._tracking_threshold

    def is_within_gating_window(self) -> bool:
        """Check if tumor is within respiratory gating window."""
        return self.get_displacement() <= self._gating_window

    def predict_position(self, time_ahead: float) -> Position3D:
        """
        Predict tumor position at future time.

        Uses motion model if available, otherwise extrapolates from history.
        """
        if self._motion_model:
            return self._motion_model(time_ahead)

        # Simple linear extrapolation from last two points
        if len(self._motion_history) < 2:
            return self._current_position

        p1 = self._motion_history[-2]
        p2 = self._motion_history[-1]
        velocity = Position3D(
            x=p2.x - p1.x,
            y=p2.y - p1.y,
            z=p2.z - p1.z
        )

        return Position3D(
            x=p2.x + velocity.x * time_ahead,
            y=p2.y + velocity.y * time_ahead,
            z=p2.z + velocity.z * time_ahead
        )

    def set_motion_model(self, model: Callable) -> bool:
        """Set a predictive motion model."""
        self._motion_model = model
        return True

    def get_motion_statistics(self) -> Dict:
        """Calculate motion statistics from tracking history."""
        if not self._motion_history:
            return {}

        positions = np.array([
            [p.x, p.y, p.z] for p in self._motion_history
        ])

        return {
            "mean_position": positions.mean(axis=0).tolist(),
            "std_position": positions.std(axis=0).tolist(),
            "max_displacement": max(
                p.distance_to(self._reference_position)
                for p in self._motion_history
            ),
            "num_samples": len(self._motion_history),
        }

    def get_status(self) -> Dict:
        """Get tracking system status."""
        return {
            "is_tracking": self._is_tracking,
            "modalities": [m.value for m in self._modalities],
            "current_position": {
                "x": self._current_position.x,
                "y": self._current_position.y,
                "z": self._current_position.z,
            },
            "displacement": self.get_displacement(),
            "within_tolerance": self.is_within_tolerance(),
            "tracking_threshold": self._tracking_threshold,
        }


class LinearAccelerator(BeamDeliverySystem):
    """
    Complete Linear Accelerator (LINAC) system.

    Integrates all components for photon and electron beam delivery:
    - Electron gun
    - Accelerating waveguide
    - Bending magnet system
    - Beam steering
    - Multi-leaf collimator
    - Tumor tracking
    """

    def __init__(
        self,
        name: str = "Medical Linear Accelerator",
        max_photon_energy: float = 18.0,  # MV
        max_electron_energy: float = 22.0,  # MeV
    ):
        super().__init__(name)
        self.max_photon_energy = max_photon_energy
        self.max_electron_energy = max_electron_energy

        # Initialize components
        self.electron_gun = ElectronGun()
        self.waveguide = AcceleratingWaveguide(max_energy=25.0)
        self.bending_magnet = BendingMagnet()
        self.beam_steering = BeamSteering()
        self.mlc = MultiLeafCollimator()
        self.tumor_tracker = TumorTracker()

        # Operating state
        self._current_mode = ParticleType.PHOTON
        self._current_energy = 6.0  # Default 6 MV photon
        self._dose_rate = 600.0  # MU/min
        self._gantry_angle = 0.0  # degrees
        self._collimator_angle = 0.0  # degrees
        self._is_ready = False
        self._beam_on = False
        self._monitor_units_delivered = 0.0

        # Safety systems
        self._interlocks_ok = False
        self._door_closed = False
        self._emergency_stop = False

    def initialize(self) -> bool:
        """Initialize all LINAC subsystems."""
        success = True
        success &= self.electron_gun.initialize()
        success &= self.waveguide.initialize()
        success &= self.bending_magnet.initialize()
        success &= self.beam_steering.initialize()
        success &= self.mlc.initialize()
        self._is_ready = success
        return success

    def calibrate(self) -> bool:
        """Calibrate the LINAC system."""
        # Calibrate all subsystems
        self.mlc.calibrate()
        self._is_calibrated = True
        return True

    def set_mode(self, mode: ParticleType) -> bool:
        """
        Set beam mode (photon or electron).

        In photon mode, electrons hit a tungsten target.
        In electron mode, the target is removed and electrons
        pass through a scattering foil.
        """
        if mode not in [ParticleType.PHOTON, ParticleType.ELECTRON]:
            return False

        if self._beam_on:
            return False  # Cannot change mode while beam is on

        self._current_mode = mode
        return True

    def set_energy(self, energy: float) -> bool:
        """Set beam energy."""
        if self._current_mode == ParticleType.PHOTON:
            if energy > self.max_photon_energy:
                return False
        else:
            if energy > self.max_electron_energy:
                return False

        self._current_energy = energy
        self.waveguide.set_energy(energy)
        self.bending_magnet.set_for_energy(energy)
        return True

    def set_dose_rate(self, rate: float) -> bool:
        """Set dose rate in MU/min."""
        if rate < 0 or rate > 2400:  # Typical max
            return False
        self._dose_rate = rate
        return True

    def set_gantry_angle(self, angle: float) -> bool:
        """Set gantry angle in degrees."""
        # Normalize to 0-360
        self._gantry_angle = angle % 360
        return True

    def set_collimator_angle(self, angle: float) -> bool:
        """Set collimator angle in degrees."""
        self._collimator_angle = angle % 360
        return True

    def set_position(self, position: Position3D) -> bool:
        """Set treatment position (couch position)."""
        self._position = position
        return True

    def set_field(
        self,
        width: float,
        height: float,
        center_x: float = 0.0,
        center_y: float = 0.0
    ) -> bool:
        """Set radiation field size and position."""
        return self.mlc.set_rectangular_field(width, height, center_x, center_y)

    def check_interlocks(self) -> bool:
        """Check all safety interlocks."""
        # Simulate interlock checks
        self._door_closed = True  # Would check actual door sensor
        self._interlocks_ok = (
            self._door_closed and
            not self._emergency_stop and
            self._is_ready
        )
        return self._interlocks_ok

    def beam_on(self) -> bool:
        """Turn on radiation beam."""
        if not self.check_interlocks():
            return False

        if not self._is_calibrated:
            return False

        # Power up systems
        self.electron_gun.warmup()
        self.waveguide.power_on()
        self.bending_magnet.energize()
        self.beam_steering.enable()
        self.electron_gun.beam_on()

        self._beam_on = True
        return True

    def beam_off(self) -> bool:
        """Turn off radiation beam."""
        self.electron_gun.beam_off()
        self._beam_on = False
        return True

    def emergency_stop(self) -> bool:
        """Emergency beam stop."""
        self._emergency_stop = True
        self.beam_off()
        self.waveguide.power_off()
        self.bending_magnet.de_energize()
        return True

    def deliver_dose(
        self,
        beam_params: BeamParameters,
        monitor_units: float
    ) -> DoseDistribution:
        """
        Deliver specified dose to target.

        Args:
            beam_params: Beam parameters for delivery
            monitor_units: Number of monitor units to deliver

        Returns:
            Calculated dose distribution
        """
        # Configure LINAC from beam parameters
        self.set_mode(beam_params.particle_type)
        self.set_energy(beam_params.energy)
        self.set_dose_rate(beam_params.dose_rate)
        self.set_gantry_angle(beam_params.gantry_angle)
        self.set_collimator_angle(beam_params.collimator_angle)
        self.set_field(
            beam_params.field_size_x * 10,  # Convert cm to mm
            beam_params.field_size_y * 10
        )

        # Create dose distribution
        dose_dist = DoseDistribution(
            origin=Position3D(-200, -200, -200),
            spacing=(2.0, 2.0, 2.0),
            dimensions=(200, 200, 200)
        )

        # Calculate dose (simplified model)
        delivery_time = monitor_units / self._dose_rate  # minutes

        if self.beam_on():
            # Simulate dose delivery
            for i in range(dose_dist.dimensions[0]):
                for j in range(dose_dist.dimensions[1]):
                    for k in range(dose_dist.dimensions[2]):
                        x = dose_dist.origin.x + i * dose_dist.spacing[0]
                        y = dose_dist.origin.y + j * dose_dist.spacing[1]
                        z = dose_dist.origin.z + k * dose_dist.spacing[2]

                        # MLC transmission
                        transmission = self.mlc.calculate_transmission(x, y)

                        # Depth dose (simplified)
                        depth = z + 100  # Assuming isocenter at z=0
                        if depth > 0:
                            # Simple exponential depth dose
                            pdd = math.exp(-0.05 * depth) * transmission
                            dose = monitor_units * 0.01 * pdd  # Convert MU to Gy
                            dose_dist.dose_grid[i, j, k] = dose

            self._monitor_units_delivered += monitor_units
            self.beam_off()

        return dose_dist

    def deliver_imrt_segment(
        self,
        mlc_aperture: List[Tuple[float, float]],
        monitor_units: float,
        gantry_angle: float
    ) -> bool:
        """
        Deliver single IMRT segment.

        Args:
            mlc_aperture: MLC leaf positions for this segment
            monitor_units: MU for this segment
            gantry_angle: Gantry angle for this segment
        """
        self.set_gantry_angle(gantry_angle)
        self.mlc.set_field_shape(mlc_aperture)

        if not self.beam_on():
            return False

        # Simulate delivery time
        delivery_time = monitor_units / self._dose_rate * 60  # seconds
        self._monitor_units_delivered += monitor_units

        self.beam_off()
        return True

    def deliver_vmat_arc(
        self,
        control_points: List[Dict],
    ) -> bool:
        """
        Deliver VMAT arc with continuous gantry rotation.

        Args:
            control_points: List of control points with gantry angle,
                           MLC positions, and cumulative MU
        """
        if len(control_points) < 2:
            return False

        if not self.beam_on():
            return False

        for i in range(1, len(control_points)):
            cp = control_points[i]

            # Move gantry and MLC while beam is on
            self.set_gantry_angle(cp["gantry_angle"])
            self.mlc.set_field_shape(cp["mlc_aperture"])

            # MU for this segment
            mu = cp["cumulative_mu"] - control_points[i-1]["cumulative_mu"]
            self._monitor_units_delivered += mu

        self.beam_off()
        return True

    def get_status(self) -> Dict:
        """Get comprehensive LINAC status."""
        return {
            "name": self.name,
            "mode": self._current_mode.value,
            "energy": self._current_energy,
            "dose_rate": self._dose_rate,
            "gantry_angle": self._gantry_angle,
            "collimator_angle": self._collimator_angle,
            "is_ready": self._is_ready,
            "is_calibrated": self._is_calibrated,
            "beam_on": self._beam_on,
            "interlocks_ok": self._interlocks_ok,
            "monitor_units_delivered": self._monitor_units_delivered,
            "electron_gun": self.electron_gun.get_status(),
            "waveguide": self.waveguide.get_status(),
            "bending_magnet": self.bending_magnet.get_status(),
            "beam_steering": self.beam_steering.get_status(),
            "mlc": self.mlc.get_status(),
            "tumor_tracker": self.tumor_tracker.get_status(),
        }
