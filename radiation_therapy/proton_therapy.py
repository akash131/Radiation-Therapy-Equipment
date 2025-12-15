"""
Proton and Heavy Ion Therapy System Module.

This module implements particle therapy systems including:
- Cyclotrons and Synchrotrons
- Beam transport and gantry systems
- Bragg peak optimization
- Treatment planning systems
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


class AcceleratorType(Enum):
    """Types of particle accelerators."""
    CYCLOTRON = "cyclotron"
    SYNCHROTRON = "synchrotron"
    SYNCHROCYCLOTRON = "synchrocyclotron"
    LINAC = "linear_accelerator"  # For FLASH or compact systems


class DeliveryMode(Enum):
    """Beam delivery modes for particle therapy."""
    PASSIVE_SCATTERING = "passive_scattering"
    UNIFORM_SCANNING = "uniform_scanning"
    PENCIL_BEAM_SCANNING = "pencil_beam_scanning"  # PBS/IMPT
    SPOT_SCANNING = "spot_scanning"


@dataclass
class CyclotronParameters:
    """Operating parameters for cyclotron."""
    magnetic_field: float = 1.5  # Tesla
    dee_voltage: float = 50.0  # kV (accelerating voltage)
    rf_frequency: float = 25.0  # MHz
    extraction_radius: float = 0.8  # meters
    beam_current: float = 0.0  # nA
    max_beam_current: float = 500.0  # nA
    extraction_efficiency: float = 0.75  # 75% typical


class Cyclotron(RadiationSource):
    """
    Cyclotron particle accelerator.

    Uses a constant magnetic field and oscillating electric field
    to accelerate charged particles in a spiral path.
    Produces a fixed-energy beam (typically 230-250 MeV for protons).
    """

    PROTON_MASS = 938.272  # MeV/c^2
    SPEED_OF_LIGHT = 2.998e8  # m/s

    def __init__(
        self,
        max_energy: float = 250.0,  # MeV
        particle_type: ParticleType = ParticleType.PROTON,
        name: str = "Cyclotron"
    ):
        super().__init__(name, max_energy, particle_type)
        self.parameters = CyclotronParameters()
        self._extraction_energy = max_energy  # Fixed for cyclotron
        self._is_operating = False

    def initialize(self) -> bool:
        """Initialize cyclotron systems."""
        # Set magnetic field for desired extraction energy
        self._calculate_magnetic_field()
        self._calculate_rf_frequency()
        return True

    def _calculate_magnetic_field(self):
        """Calculate required magnetic field for extraction energy."""
        # For relativistic particles: r = p / (qB)
        # E^2 = (pc)^2 + (mc^2)^2
        E_total = self._extraction_energy + self.PROTON_MASS
        momentum = math.sqrt(E_total ** 2 - self.PROTON_MASS ** 2)  # MeV/c

        # Convert to SI: p (kg·m/s) = p (MeV/c) * 1e6 * e / c
        momentum_si = momentum * 1e6 * 1.602e-19 / self.SPEED_OF_LIGHT

        # B = p / (qr) where r is extraction radius
        q = 1.602e-19  # Coulombs
        r = self.parameters.extraction_radius
        self.parameters.magnetic_field = momentum_si / (q * r)

    def _calculate_rf_frequency(self):
        """Calculate RF frequency for acceleration."""
        # f = qB / (2πm_relativistic)
        gamma = 1 + self._extraction_energy / self.PROTON_MASS
        m_rel = self.PROTON_MASS * gamma * 1e6 * 1.602e-19 / (self.SPEED_OF_LIGHT ** 2)
        q = 1.602e-19
        B = self.parameters.magnetic_field

        # This is approximate - actual cyclotrons use frequency modulation
        self.parameters.rf_frequency = q * B / (2 * math.pi * m_rel) / 1e6  # MHz

    def set_energy(self, energy: float) -> bool:
        """
        Cyclotrons have fixed extraction energy.
        Energy degradation is done downstream.
        """
        if energy > self.max_energy:
            return False
        # Energy will be degraded in beam transport
        self._current_energy = energy
        return True

    def set_beam_current(self, current: float) -> bool:
        """Set beam current in nA."""
        if current > self.parameters.max_beam_current:
            return False
        self.parameters.beam_current = current
        return True

    def beam_on(self) -> bool:
        """Start beam extraction."""
        if not self._is_operating:
            return False
        self._is_active = True
        return True

    def beam_off(self) -> bool:
        """Stop beam extraction."""
        self._is_active = False
        return True

    def start_operation(self) -> bool:
        """Start cyclotron operation (field and RF)."""
        self._is_operating = True
        return True

    def stop_operation(self) -> bool:
        """Stop cyclotron operation."""
        self.beam_off()
        self._is_operating = False
        return True

    def calculate_beam_power(self) -> float:
        """Calculate beam power in watts."""
        # P = I * V where V is extraction energy
        current_amps = self.parameters.beam_current * 1e-9
        voltage = self._extraction_energy * 1e6  # Convert MeV to eV
        return current_amps * voltage

    def get_status(self) -> Dict:
        """Get cyclotron status."""
        return {
            "name": self.name,
            "is_operating": self._is_operating,
            "is_active": self._is_active,
            "extraction_energy": self._extraction_energy,
            "magnetic_field": self.parameters.magnetic_field,
            "rf_frequency": self.parameters.rf_frequency,
            "beam_current": self.parameters.beam_current,
            "beam_power": self.calculate_beam_power(),
        }


@dataclass
class SynchrotronParameters:
    """Operating parameters for synchrotron."""
    circumference: float = 80.0  # meters
    magnetic_field_range: Tuple[float, float] = (0.1, 1.8)  # Tesla
    rf_frequency_range: Tuple[float, float] = (1.0, 4.0)  # MHz
    injection_energy: float = 7.0  # MeV (from injector LINAC)
    spill_duration: float = 5.0  # seconds
    repetition_rate: float = 0.5  # Hz (cycles per second)
    beam_intensity: float = 0.0  # particles per spill


class Synchrotron(RadiationSource):
    """
    Synchrotron particle accelerator.

    Uses time-varying magnetic and electric fields to accelerate
    particles in a fixed circular path. Can produce variable-energy
    beams, essential for carbon ion therapy and energy painting.
    """

    PROTON_MASS = 938.272  # MeV/c^2
    CARBON_MASS = 11177.93  # MeV/c^2 (12C)
    SPEED_OF_LIGHT = 2.998e8  # m/s

    def __init__(
        self,
        max_energy: float = 430.0,  # MeV/u for carbon
        particle_type: ParticleType = ParticleType.CARBON_ION,
        name: str = "Synchrotron"
    ):
        super().__init__(name, max_energy, particle_type)
        self.parameters = SynchrotronParameters()
        self._target_energy = max_energy
        self._is_accelerating = False
        self._cycle_phase = "idle"  # idle, injection, acceleration, extraction

    @property
    def particle_mass(self) -> float:
        """Get mass of accelerated particle."""
        masses = {
            ParticleType.PROTON: self.PROTON_MASS,
            ParticleType.CARBON_ION: self.CARBON_MASS / 12,  # per nucleon
            ParticleType.HELIUM_ION: 3727.38 / 4,  # per nucleon
        }
        return masses.get(self.particle_type, self.PROTON_MASS)

    def initialize(self) -> bool:
        """Initialize synchrotron systems."""
        return True

    def set_energy(self, energy: float) -> bool:
        """
        Set target extraction energy.

        Synchrotrons can produce variable energy beams by adjusting
        the extraction timing during the acceleration cycle.
        """
        if energy > self.max_energy:
            return False
        self._target_energy = energy
        self._current_energy = energy
        return True

    def _calculate_magnetic_field(self, energy: float) -> float:
        """Calculate required magnetic field for given energy."""
        # Relativistic momentum
        mass = self.particle_mass
        if self.particle_type == ParticleType.CARBON_ION:
            mass = self.CARBON_MASS  # Total mass for C12
            charge = 6  # Carbon has 6 protons
        else:
            charge = 1

        E_total = energy + mass
        momentum = math.sqrt(E_total ** 2 - mass ** 2)  # MeV/c

        # Magnetic rigidity: Bρ = p / (Zq) in T·m
        # ρ = circumference / (2π) for average
        rho = self.parameters.circumference / (2 * math.pi)
        rigidity = momentum / 299.792  # Convert MeV/c to T·m

        return rigidity / rho / charge

    def inject_beam(self) -> bool:
        """Inject beam from injector system."""
        self._cycle_phase = "injection"
        self._current_energy = self.parameters.injection_energy
        return True

    def accelerate(self) -> bool:
        """Accelerate beam to target energy."""
        if self._cycle_phase != "injection":
            return False

        self._cycle_phase = "acceleration"
        self._is_accelerating = True

        # Simulate acceleration
        self._current_energy = self._target_energy
        self._is_accelerating = False
        self._cycle_phase = "extraction"

        return True

    def beam_on(self) -> bool:
        """Start beam extraction (slow extraction)."""
        if self._cycle_phase != "extraction":
            return False
        self._is_active = True
        return True

    def beam_off(self) -> bool:
        """Stop beam extraction."""
        self._is_active = False
        self._cycle_phase = "idle"
        return True

    def run_cycle(self) -> bool:
        """Run complete acceleration cycle."""
        self.inject_beam()
        self.accelerate()
        return True

    def set_spill_duration(self, duration: float) -> bool:
        """Set beam spill duration for slow extraction."""
        if duration < 0.1 or duration > 30.0:
            return False
        self.parameters.spill_duration = duration
        return True

    def calculate_range_in_water(self) -> float:
        """
        Calculate beam range in water for current energy.

        Uses power-law approximation: R = α * E^p
        """
        E = self._current_energy
        if self.particle_type == ParticleType.PROTON:
            # R (cm) ≈ 0.0022 * E^1.77 for protons
            return 0.0022 * E ** 1.77
        elif self.particle_type == ParticleType.CARBON_ION:
            # Carbon has ~3x the range of protons at same energy/u
            return 0.0022 * E ** 1.77 * 0.33
        return 0.0

    def get_status(self) -> Dict:
        """Get synchrotron status."""
        return {
            "name": self.name,
            "particle_type": self.particle_type.value,
            "cycle_phase": self._cycle_phase,
            "current_energy": self._current_energy,
            "target_energy": self._target_energy,
            "magnetic_field": self._calculate_magnetic_field(self._current_energy),
            "is_accelerating": self._is_accelerating,
            "is_active": self._is_active,
            "spill_duration": self.parameters.spill_duration,
            "range_in_water": self.calculate_range_in_water(),
        }


@dataclass
class BeamlineElement:
    """A single element in the beam transport line."""
    name: str
    element_type: str  # "dipole", "quadrupole", "degrader", "scanner", etc.
    position: float  # meters along beamline
    length: float  # meters
    strength: float = 0.0  # Field strength or setting
    aperture: float = 100.0  # mm


class BeamTransport:
    """
    Beam transport system from accelerator to treatment room.

    Includes:
    - Dipole magnets for beam steering
    - Quadrupole magnets for focusing
    - Energy degrader for energy selection (cyclotron systems)
    - Beam monitors and diagnostics
    """

    def __init__(self, name: str = "Beam Transport System"):
        self.name = name
        self._elements: List[BeamlineElement] = []
        self._beam_energy = 0.0
        self._energy_spread = 0.0
        self._beam_position = Position3D()
        self._beam_size = (5.0, 5.0)  # mm (sigma x, sigma y)
        self._transmission = 1.0
        self._is_configured = False

    def add_element(self, element: BeamlineElement):
        """Add element to beamline."""
        self._elements.append(element)
        self._elements.sort(key=lambda e: e.position)

    def configure_for_energy(self, energy: float, source_energy: float) -> bool:
        """
        Configure beam transport for target energy.

        Args:
            energy: Desired treatment energy in MeV
            source_energy: Accelerator extraction energy in MeV
        """
        self._beam_energy = energy

        if energy > source_energy:
            return False

        # Calculate degrader setting if needed
        degrader = self._find_element("degrader")
        if degrader and energy < source_energy:
            # Energy loss in degrader
            range_reduction = self._calculate_degrader_thickness(
                source_energy, energy
            )
            degrader.strength = range_reduction
            # Energy spread increases with degradation
            self._energy_spread = (source_energy - energy) * 0.01

        # Configure focusing magnets
        self._configure_optics(energy)

        self._is_configured = True
        return True

    def _find_element(self, element_type: str) -> Optional[BeamlineElement]:
        """Find element by type."""
        for element in self._elements:
            if element.element_type == element_type:
                return element
        return None

    def _calculate_degrader_thickness(
        self,
        initial_energy: float,
        final_energy: float
    ) -> float:
        """Calculate degrader thickness for energy reduction."""
        # Range difference in water equivalent
        range_initial = 0.0022 * initial_energy ** 1.77  # cm
        range_final = 0.0022 * final_energy ** 1.77  # cm
        return (range_initial - range_final) * 10  # mm

    def _configure_optics(self, energy: float):
        """Configure focusing magnets for given energy."""
        # Adjust quadrupole strengths for beam rigidity
        for element in self._elements:
            if element.element_type == "quadrupole":
                # Magnetic rigidity scales with momentum
                momentum = math.sqrt((energy + 938.272) ** 2 - 938.272 ** 2)
                element.strength = momentum / 1000.0  # Normalized

    def transport_beam(self) -> Tuple[Position3D, Tuple[float, float]]:
        """
        Transport beam through beamline.

        Returns:
            Tuple of (beam position, beam size)
        """
        if not self._is_configured:
            return (self._beam_position, self._beam_size)

        # Simulate beam transport through elements
        current_pos = Position3D()
        current_size = list(self._beam_size)

        for element in self._elements:
            if element.element_type == "dipole":
                # Steering
                current_pos.x += element.strength * 0.1  # mm/T
            elif element.element_type == "quadrupole":
                # Focusing/defocusing
                if element.strength > 0:
                    current_size[0] *= 0.9
                    current_size[1] *= 1.1
                else:
                    current_size[0] *= 1.1
                    current_size[1] *= 0.9
            elif element.element_type == "degrader":
                # Scattering increases beam size
                current_size[0] *= 1.5
                current_size[1] *= 1.5
                self._transmission *= 0.8  # Some loss in degrader

        self._beam_position = current_pos
        self._beam_size = tuple(current_size)
        return (current_pos, self._beam_size)

    def get_transmission(self) -> float:
        """Get beam transmission through beamline."""
        return self._transmission

    def get_status(self) -> Dict:
        """Get beam transport status."""
        return {
            "name": self.name,
            "is_configured": self._is_configured,
            "beam_energy": self._beam_energy,
            "energy_spread": self._energy_spread,
            "beam_position": {
                "x": self._beam_position.x,
                "y": self._beam_position.y,
            },
            "beam_size": self._beam_size,
            "transmission": self._transmission,
            "num_elements": len(self._elements),
        }


class GantryType(Enum):
    """Types of treatment gantries."""
    ROTATING = "rotating"  # 360-degree rotation
    FIXED = "fixed"  # Fixed beamline
    COMPACT = "compact"  # Reduced size rotating


@dataclass
class GantryParameters:
    """Parameters for gantry system."""
    rotation_range: Tuple[float, float] = (0.0, 360.0)  # degrees
    rotation_speed: float = 1.0  # degrees/second
    source_to_isocenter: float = 3000.0  # mm
    isocenter_accuracy: float = 0.5  # mm
    weight: float = 200.0  # tons (for rotating gantries)


class GantrySystem:
    """
    Gantry system for particle beam delivery.

    Particle therapy gantries are massive structures (100-200 tons)
    that rotate the beam around the patient for multi-angle treatment.
    """

    def __init__(
        self,
        gantry_type: GantryType = GantryType.ROTATING,
        name: str = "Gantry System"
    ):
        self.name = name
        self.gantry_type = gantry_type
        self.parameters = GantryParameters()
        self._current_angle = 0.0
        self._target_angle = 0.0
        self._is_rotating = False
        self._is_calibrated = False

        # Scanning magnets for pencil beam scanning
        self._scan_x_position = 0.0  # mm at isocenter
        self._scan_y_position = 0.0  # mm at isocenter
        self._scan_x_range = 200.0  # mm
        self._scan_y_range = 200.0  # mm
        self._scan_speed = 20.0  # m/s

    def initialize(self) -> bool:
        """Initialize gantry system."""
        return self.calibrate()

    def calibrate(self) -> bool:
        """Calibrate gantry position encoders."""
        self._is_calibrated = True
        return True

    def set_angle(self, angle: float) -> bool:
        """
        Set gantry to specified angle.

        Args:
            angle: Target angle in degrees
        """
        min_angle, max_angle = self.parameters.rotation_range
        if self.gantry_type == GantryType.ROTATING:
            # Full 360-degree rotation
            angle = angle % 360
        else:
            if angle < min_angle or angle > max_angle:
                return False

        self._target_angle = angle
        return True

    def rotate_to_angle(self) -> float:
        """
        Rotate gantry to target angle.

        Returns:
            Time in seconds to complete rotation
        """
        if self.gantry_type == GantryType.FIXED:
            return 0.0

        angle_diff = abs(self._target_angle - self._current_angle)
        # Take shortest path around circle
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        rotation_time = angle_diff / self.parameters.rotation_speed
        self._current_angle = self._target_angle
        return rotation_time

    def set_scan_position(self, x: float, y: float) -> bool:
        """
        Set scanning magnet position for pencil beam.

        Args:
            x: X position at isocenter in mm
            y: Y position at isocenter in mm
        """
        if abs(x) > self._scan_x_range or abs(y) > self._scan_y_range:
            return False

        self._scan_x_position = x
        self._scan_y_position = y
        return True

    def get_beam_direction(self) -> Vector3D:
        """Get beam direction vector at current gantry angle."""
        angle_rad = math.radians(self._current_angle)
        return Vector3D(
            dx=math.sin(angle_rad),
            dy=0.0,
            dz=-math.cos(angle_rad)
        )

    def calculate_spot_position(self) -> Position3D:
        """Calculate beam spot position at isocenter plane."""
        return Position3D(
            x=self._scan_x_position,
            y=self._scan_y_position,
            z=0.0
        )

    def calculate_scan_time(
        self,
        spot_positions: List[Tuple[float, float]]
    ) -> float:
        """
        Calculate time to scan through spot positions.

        Args:
            spot_positions: List of (x, y) positions in mm
        """
        if len(spot_positions) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(1, len(spot_positions)):
            dx = spot_positions[i][0] - spot_positions[i-1][0]
            dy = spot_positions[i][1] - spot_positions[i-1][1]
            total_distance += math.sqrt(dx ** 2 + dy ** 2)

        return total_distance / (self._scan_speed * 1000)  # Convert m/s to mm/s

    def get_status(self) -> Dict:
        """Get gantry status."""
        return {
            "name": self.name,
            "type": self.gantry_type.value,
            "current_angle": self._current_angle,
            "target_angle": self._target_angle,
            "is_rotating": self._is_rotating,
            "is_calibrated": self._is_calibrated,
            "scan_position": {
                "x": self._scan_x_position,
                "y": self._scan_y_position,
            },
            "source_to_isocenter": self.parameters.source_to_isocenter,
        }


@dataclass
class BraggPeakParameters:
    """Parameters for Bragg peak calculation."""
    energy: float  # MeV
    range: float  # mm in water
    sigma_range: float  # mm range straggling
    lateral_sigma: float  # mm lateral scatter
    peak_to_plateau_ratio: float = 4.0


class BraggPeakOptimizer:
    """
    Bragg peak optimization for treatment planning.

    Calculates and optimizes the depth-dose distribution
    characteristic of charged particle beams.
    """

    WATER_DENSITY = 1.0  # g/cm^3

    def __init__(self, particle_type: ParticleType = ParticleType.PROTON):
        self.particle_type = particle_type
        self._peaks: List[BraggPeakParameters] = []

    def calculate_range(self, energy: float) -> float:
        """
        Calculate range in water for given energy.

        Uses Bragg-Kleeman rule: R = α * E^p
        """
        if self.particle_type == ParticleType.PROTON:
            # R (mm) = 0.022 * E^1.77
            return 0.022 * energy ** 1.77 * 10  # Convert to mm
        elif self.particle_type == ParticleType.CARBON_ION:
            # Carbon: similar but scaled by A/Z^2
            return 0.022 * energy ** 1.77 * 10 * (12 / 36)
        return 0.0

    def calculate_energy_for_range(self, range_mm: float) -> float:
        """Calculate required energy for given range."""
        if self.particle_type == ParticleType.PROTON:
            return (range_mm / 10 / 0.022) ** (1 / 1.77)
        return 0.0

    def calculate_bragg_peak(
        self,
        energy: float,
        depths: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Bragg peak depth-dose curve.

        Uses analytical Bragg curve model.

        Args:
            energy: Beam energy in MeV
            depths: Array of depths in mm

        Returns:
            Normalized dose array
        """
        R = self.calculate_range(energy)  # Range in mm

        # Range straggling (approximately 1% of range)
        sigma = 0.01 * R

        # Simplified Bragg peak model
        dose = np.zeros_like(depths)

        for i, z in enumerate(depths):
            if z < R:
                # Before peak: slowly rising
                # Using simplified analytical formula
                z_rel = z / R
                dose[i] = (1 + 3 * z_rel ** 2) * np.exp(-z_rel ** 10)
            else:
                # After peak: rapid falloff (Gaussian)
                dose[i] = np.exp(-((z - R) ** 2) / (2 * sigma ** 2))

        # Normalize to peak = 1
        if dose.max() > 0:
            dose = dose / dose.max()

        return dose

    def calculate_sobp(
        self,
        min_depth: float,
        max_depth: float,
        depths: np.ndarray,
        num_peaks: int = 20
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Calculate Spread-Out Bragg Peak (SOBP).

        Superposition of weighted pristine Bragg peaks to
        create uniform dose in target region.

        Args:
            min_depth: Proximal target depth in mm
            max_depth: Distal target depth in mm
            depths: Array of depths for calculation
            num_peaks: Number of peaks to use

        Returns:
            Tuple of (dose array, list of weights)
        """
        # Calculate energy range
        max_energy = self.calculate_energy_for_range(max_depth)
        min_energy = self.calculate_energy_for_range(min_depth)

        # Create energy layers
        energies = np.linspace(min_energy, max_energy, num_peaks)

        # Calculate individual peaks
        peaks = []
        for E in energies:
            peak = self.calculate_bragg_peak(E, depths)
            peaks.append(peak)

        # Optimize weights for uniform dose in target
        # Simple approach: weight peaks by depth contribution
        weights = self._optimize_sobp_weights(
            peaks, depths, min_depth, max_depth
        )

        # Sum weighted peaks
        sobp = np.zeros_like(depths)
        for peak, weight in zip(peaks, weights):
            sobp += weight * peak

        # Normalize
        target_mask = (depths >= min_depth) & (depths <= max_depth)
        if sobp[target_mask].max() > 0:
            sobp = sobp / sobp[target_mask].max()

        return sobp, weights.tolist()

    def _optimize_sobp_weights(
        self,
        peaks: List[np.ndarray],
        depths: np.ndarray,
        min_depth: float,
        max_depth: float
    ) -> np.ndarray:
        """
        Optimize peak weights for uniform SOBP.

        Uses simple inverse weighting based on peak contribution
        in target region.
        """
        num_peaks = len(peaks)
        target_mask = (depths >= min_depth) & (depths <= max_depth)

        # Simple weighting: distal peaks get higher weight
        # This is a simplification; real systems use iterative optimization
        weights = np.zeros(num_peaks)
        for i, peak in enumerate(peaks):
            # Weight based on contribution to target uniformity
            target_contribution = peak[target_mask].mean()
            weights[i] = 1.0 / (target_contribution + 0.1)

        # Normalize weights
        weights = weights / weights.sum()

        return weights

    def calculate_lateral_profile(
        self,
        radial_distances: np.ndarray,
        depth: float,
        energy: float,
        spot_size: float = 5.0
    ) -> np.ndarray:
        """
        Calculate lateral dose profile at given depth.

        Accounts for multiple Coulomb scattering.

        Args:
            radial_distances: Array of radial distances in mm
            depth: Depth in mm
            energy: Beam energy in MeV
            spot_size: Initial spot sigma in mm
        """
        # Lateral sigma grows with depth due to scattering
        # σ ≈ σ₀ + 0.023 * depth
        sigma = spot_size + 0.023 * depth

        # Gaussian lateral profile
        return np.exp(-radial_distances ** 2 / (2 * sigma ** 2))

    def calculate_rbe(self, let: float) -> float:
        """
        Calculate Relative Biological Effectiveness (RBE).

        RBE increases with LET, especially for carbon ions.

        Args:
            let: Linear Energy Transfer in keV/μm
        """
        if self.particle_type == ParticleType.PROTON:
            # Protons: RBE ≈ 1.1 (relatively constant)
            return 1.1
        elif self.particle_type == ParticleType.CARBON_ION:
            # Carbon: RBE varies from ~1.5 to ~4 depending on LET
            # Simplified model
            if let < 10:
                return 1.5
            elif let < 100:
                return 1.5 + (let - 10) * 0.025
            else:
                return 3.75
        return 1.0

    def get_peak_parameters(self, energy: float) -> BraggPeakParameters:
        """Get Bragg peak parameters for given energy."""
        range_mm = self.calculate_range(energy)
        return BraggPeakParameters(
            energy=energy,
            range=range_mm,
            sigma_range=0.01 * range_mm,
            lateral_sigma=5.0 + 0.023 * range_mm,
            peak_to_plateau_ratio=4.0 if self.particle_type == ParticleType.PROTON else 3.0
        )


@dataclass
class SpotParameters:
    """Parameters for a single pencil beam spot."""
    position: Position3D  # Position at isocenter
    energy: float  # MeV
    monitor_units: float  # Delivered dose
    spot_size: float  # mm sigma


class TreatmentPlanningSystem:
    """
    Treatment Planning System (TPS) for particle therapy.

    Handles:
    - Dose calculation
    - Beam optimization
    - Plan evaluation
    """

    def __init__(
        self,
        particle_type: ParticleType = ParticleType.PROTON,
        name: str = "Treatment Planning System"
    ):
        self.name = name
        self.particle_type = particle_type
        self.bragg_optimizer = BraggPeakOptimizer(particle_type)
        self._current_plan: Optional[TreatmentPlan] = None
        self._spot_map: List[SpotParameters] = []
        self._calculation_grid_size = 2.5  # mm

    def create_plan(
        self,
        plan_id: str,
        patient_id: str,
        targets: List[TreatmentTarget],
        oars: List[OrganAtRisk],
        prescribed_dose: float,
        fractions: int
    ) -> TreatmentPlan:
        """Create a new treatment plan."""
        # Create beam parameters for IMPT
        beams = self._generate_beam_angles(targets)

        plan = TreatmentPlan(
            plan_id=plan_id,
            patient_id=patient_id,
            targets=targets,
            organs_at_risk=oars,
            beams=beams,
            total_dose=prescribed_dose,
            total_fractions=fractions
        )

        self._current_plan = plan
        return plan

    def _generate_beam_angles(
        self,
        targets: List[TreatmentTarget]
    ) -> List[BeamParameters]:
        """Generate optimal beam angles for targets."""
        # Simple approach: 2-3 fields
        angles = [0.0, 90.0, 180.0]  # Anterior, lateral, posterior

        beams = []
        for angle in angles:
            beam = BeamParameters(
                energy=200.0,  # Will be optimized per spot
                particle_type=self.particle_type,
                gantry_angle=angle,
                modality=BeamModality.IMRT,  # PBS/IMPT
            )
            beams.append(beam)

        return beams

    def generate_spot_map(
        self,
        target: TreatmentTarget,
        gantry_angle: float,
        spot_spacing: float = 5.0,
        layer_spacing: float = 5.0
    ) -> List[SpotParameters]:
        """
        Generate spot positions for pencil beam scanning.

        Args:
            target: Treatment target
            gantry_angle: Gantry angle in degrees
            spot_spacing: Lateral spot spacing in mm
            layer_spacing: Energy layer spacing in mm water equivalent
        """
        spots = []

        # Calculate energy layers needed
        center = target.center
        half_depth = target.dimensions[2] / 2
        min_depth = center.z - half_depth + target.margin_ptv
        max_depth = center.z + half_depth + target.margin_ptv

        min_energy = self.bragg_optimizer.calculate_energy_for_range(
            max(min_depth, 10)
        )
        max_energy = self.bragg_optimizer.calculate_energy_for_range(max_depth)

        # Create energy layers
        energies = np.arange(min_energy, max_energy + 1, layer_spacing)

        # Create spot grid for each layer
        half_x = target.dimensions[0] / 2 + target.margin_ptv
        half_y = target.dimensions[1] / 2 + target.margin_ptv

        for energy in energies:
            range_mm = self.bragg_optimizer.calculate_range(energy)

            # Create grid within target projection
            x_positions = np.arange(
                center.x - half_x,
                center.x + half_x + 1,
                spot_spacing
            )
            y_positions = np.arange(
                center.y - half_y,
                center.y + half_y + 1,
                spot_spacing
            )

            for x in x_positions:
                for y in y_positions:
                    # Check if spot is within target (ellipsoid)
                    point = Position3D(x, y, center.z)
                    if target.contains_point(point):
                        spot = SpotParameters(
                            position=Position3D(x, y, 0),  # At isocenter
                            energy=energy,
                            monitor_units=1.0,  # Will be optimized
                            spot_size=5.0
                        )
                        spots.append(spot)

        self._spot_map = spots
        return spots

    def optimize_spot_weights(
        self,
        target: TreatmentTarget,
        oars: List[OrganAtRisk],
        prescribed_dose: float
    ) -> List[float]:
        """
        Optimize spot weights using inverse planning.

        Simplified gradient descent optimization.
        """
        if not self._spot_map:
            return []

        num_spots = len(self._spot_map)
        weights = np.ones(num_spots)

        # Simple optimization: iterate to improve uniformity
        for iteration in range(50):
            # Calculate current dose
            dose_at_target = self._calculate_dose_at_point(
                target.center, weights
            )

            # Scale weights to match prescription
            if dose_at_target > 0:
                scale = prescribed_dose / dose_at_target
                weights *= scale

        # Update spot MU values
        for i, spot in enumerate(self._spot_map):
            spot.monitor_units = weights[i]

        return weights.tolist()

    def _calculate_dose_at_point(
        self,
        point: Position3D,
        weights: np.ndarray
    ) -> float:
        """Calculate dose at a point from all spots."""
        total_dose = 0.0

        for i, spot in enumerate(self._spot_map):
            # Distance from spot axis
            dx = point.x - spot.position.x
            dy = point.y - spot.position.y
            lateral_dist = math.sqrt(dx ** 2 + dy ** 2)

            # Depth from isocenter (simplified)
            depth = point.z

            # Get Bragg peak contribution
            depths = np.array([depth])
            bragg = self.bragg_optimizer.calculate_bragg_peak(
                spot.energy, depths
            )[0]

            # Lateral falloff
            lateral = np.exp(-lateral_dist ** 2 / (2 * spot.spot_size ** 2))

            total_dose += weights[i] * bragg * lateral

        return total_dose

    def calculate_dose_distribution(self) -> DoseDistribution:
        """Calculate 3D dose distribution for current plan."""
        if not self._current_plan:
            return DoseDistribution(
                origin=Position3D(),
                spacing=(2.5, 2.5, 2.5),
                dimensions=(100, 100, 100)
            )

        # Create dose grid
        grid_size = self._calculation_grid_size
        dims = (100, 100, 100)
        origin = Position3D(-125, -125, -125)

        dose_dist = DoseDistribution(
            origin=origin,
            spacing=(grid_size, grid_size, grid_size),
            dimensions=dims
        )

        # Calculate dose at each voxel
        weights = np.array([s.monitor_units for s in self._spot_map])

        for i in range(dims[0]):
            for j in range(dims[1]):
                for k in range(dims[2]):
                    point = Position3D(
                        x=origin.x + i * grid_size,
                        y=origin.y + j * grid_size,
                        z=origin.z + k * grid_size
                    )
                    dose = self._calculate_dose_at_point(point, weights)
                    dose_dist.dose_grid[i, j, k] = dose

        self._current_plan.dose_distribution = dose_dist
        return dose_dist

    def evaluate_plan(self) -> Dict:
        """Evaluate plan quality metrics."""
        if not self._current_plan or not self._current_plan.dose_distribution:
            return {}

        dose_dist = self._current_plan.dose_distribution

        results = {
            "conformity_index": self._current_plan.calculate_conformity_index(),
            "homogeneity_index": self._current_plan.calculate_homogeneity_index(),
            "max_dose": dose_dist.get_max_dose(),
            "mean_dose": dose_dist.get_mean_dose(),
            "target_coverage": {},
            "oar_doses": {},
        }

        # Target coverage
        for target in self._current_plan.targets:
            coverage = self._calculate_target_coverage(target, dose_dist)
            results["target_coverage"][target.name] = coverage

        # OAR doses
        for oar in self._current_plan.organs_at_risk:
            oar_dose = self._calculate_oar_dose(oar, dose_dist)
            results["oar_doses"][oar.name] = oar_dose

        return results

    def _calculate_target_coverage(
        self,
        target: TreatmentTarget,
        dose_dist: DoseDistribution
    ) -> Dict:
        """Calculate target coverage statistics."""
        doses = []

        # Sample points in target
        for _ in range(1000):
            # Random point in ellipsoid
            r = np.random.random() ** (1/3)
            theta = np.random.random() * 2 * math.pi
            phi = np.random.random() * math.pi

            point = Position3D(
                x=target.center.x + r * target.dimensions[0]/2 * math.sin(phi) * math.cos(theta),
                y=target.center.y + r * target.dimensions[1]/2 * math.sin(phi) * math.sin(theta),
                z=target.center.z + r * target.dimensions[2]/2 * math.cos(phi)
            )
            doses.append(dose_dist.get_dose_at(point))

        doses = np.array(doses)
        prescription = target.prescribed_dose

        return {
            "D95": float(np.percentile(doses, 5)),  # Dose to 95% of volume
            "D50": float(np.percentile(doses, 50)),  # Median dose
            "D5": float(np.percentile(doses, 95)),  # Near-max dose
            "V100": float(np.sum(doses >= prescription) / len(doses) * 100),  # Volume at 100%
        }

    def _calculate_oar_dose(
        self,
        oar: OrganAtRisk,
        dose_dist: DoseDistribution
    ) -> Dict:
        """Calculate OAR dose statistics."""
        doses = []

        # Sample points in OAR
        for _ in range(500):
            r = np.random.random() ** (1/3)
            theta = np.random.random() * 2 * math.pi
            phi = np.random.random() * math.pi

            point = Position3D(
                x=oar.center.x + r * oar.dimensions[0]/2 * math.sin(phi) * math.cos(theta),
                y=oar.center.y + r * oar.dimensions[1]/2 * math.sin(phi) * math.sin(theta),
                z=oar.center.z + r * oar.dimensions[2]/2 * math.cos(phi)
            )
            doses.append(dose_dist.get_dose_at(point))

        doses = np.array(doses)

        return {
            "max_dose": float(np.max(doses)),
            "mean_dose": float(np.mean(doses)),
            "constraint_met": float(np.max(doses)) <= oar.max_dose,
        }

    def get_status(self) -> Dict:
        """Get TPS status."""
        return {
            "name": self.name,
            "particle_type": self.particle_type.value,
            "has_plan": self._current_plan is not None,
            "num_spots": len(self._spot_map),
            "grid_size": self._calculation_grid_size,
        }


class ProtonTherapySystem(BeamDeliverySystem):
    """
    Complete Proton Therapy System.

    Integrates:
    - Cyclotron accelerator
    - Beam transport
    - Gantry with scanning system
    - Treatment planning
    """

    def __init__(
        self,
        name: str = "Proton Therapy System",
        max_energy: float = 250.0
    ):
        super().__init__(name)
        self.max_energy = max_energy

        # Initialize components
        self.accelerator = Cyclotron(max_energy=max_energy)
        self.beam_transport = BeamTransport()
        self.gantry = GantrySystem()
        self.tps = TreatmentPlanningSystem(ParticleType.PROTON)
        self.bragg_optimizer = BraggPeakOptimizer(ParticleType.PROTON)

        # Set up beam transport elements
        self._configure_beamline()

        # Operating state
        self._current_energy = 200.0
        self._is_ready = False
        self._beam_on = False
        self._delivery_mode = DeliveryMode.PENCIL_BEAM_SCANNING

    def _configure_beamline(self):
        """Configure beam transport elements."""
        elements = [
            BeamlineElement("Degrader", "degrader", 5.0, 0.5),
            BeamlineElement("ESS", "quadrupole", 10.0, 1.0, strength=1.0),
            BeamlineElement("BM1", "dipole", 20.0, 2.0),
            BeamlineElement("Q1", "quadrupole", 25.0, 0.5, strength=1.0),
            BeamlineElement("Q2", "quadrupole", 27.0, 0.5, strength=-1.0),
            BeamlineElement("BM2", "dipole", 40.0, 2.0),
        ]
        for element in elements:
            self.beam_transport.add_element(element)

    def initialize(self) -> bool:
        """Initialize the proton therapy system."""
        success = True
        success &= self.accelerator.initialize()
        success &= self.gantry.initialize()
        self._is_ready = success
        return success

    def calibrate(self) -> bool:
        """Calibrate the system."""
        self.gantry.calibrate()
        self._is_calibrated = True
        return True

    def set_position(self, position: Position3D) -> bool:
        """Set treatment position."""
        self._position = position
        return True

    def set_energy(self, energy: float) -> bool:
        """Set beam energy for treatment."""
        if energy > self.max_energy:
            return False

        self._current_energy = energy
        self.accelerator.set_energy(energy)
        self.beam_transport.configure_for_energy(
            energy, self.accelerator._extraction_energy
        )
        return True

    def set_gantry_angle(self, angle: float) -> bool:
        """Set gantry angle."""
        self.gantry.set_angle(angle)
        self.gantry.rotate_to_angle()
        return True

    def deliver_spot(self, spot: SpotParameters) -> bool:
        """
        Deliver a single spot.

        Args:
            spot: Spot parameters including position, energy, and MU
        """
        # Set energy
        self.set_energy(spot.energy)

        # Set scan position
        self.gantry.set_scan_position(
            spot.position.x,
            spot.position.y
        )

        # Deliver dose
        if not self._beam_on:
            self.accelerator.beam_on()
            self._beam_on = True

        # Simulate spot delivery
        return True

    def deliver_layer(
        self,
        spots: List[SpotParameters],
        energy: float
    ) -> bool:
        """
        Deliver all spots at a single energy layer.

        Args:
            spots: List of spots at this energy
            energy: Energy for this layer
        """
        self.set_energy(energy)

        for spot in spots:
            self.gantry.set_scan_position(
                spot.position.x,
                spot.position.y
            )

        return True

    def deliver_dose(
        self,
        beam_params: BeamParameters,
        monitor_units: float
    ) -> DoseDistribution:
        """Deliver treatment dose."""
        # Configure system
        self.set_energy(beam_params.energy)
        self.set_gantry_angle(beam_params.gantry_angle)

        # Start beam
        self.accelerator.start_operation()
        self.accelerator.beam_on()
        self._beam_on = True

        # Create dose distribution
        depths = np.linspace(0, 300, 300)
        bragg = self.bragg_optimizer.calculate_bragg_peak(
            beam_params.energy, depths
        )

        dose_dist = DoseDistribution(
            origin=Position3D(-100, -100, 0),
            spacing=(2.0, 2.0, 1.0),
            dimensions=(100, 100, 300)
        )

        # Simple dose calculation
        for k, depth in enumerate(depths):
            for i in range(dose_dist.dimensions[0]):
                for j in range(dose_dist.dimensions[1]):
                    x = dose_dist.origin.x + i * dose_dist.spacing[0]
                    y = dose_dist.origin.y + j * dose_dist.spacing[1]

                    # Lateral distance from beam axis
                    r = math.sqrt(x ** 2 + y ** 2)
                    lateral = math.exp(-r ** 2 / 50)

                    dose_dist.dose_grid[i, j, k] = (
                        monitor_units * 0.01 * bragg[k] * lateral
                    )

        # Stop beam
        self.accelerator.beam_off()
        self._beam_on = False

        return dose_dist

    def get_status(self) -> Dict:
        """Get system status."""
        return {
            "name": self.name,
            "is_ready": self._is_ready,
            "is_calibrated": self._is_calibrated,
            "current_energy": self._current_energy,
            "beam_on": self._beam_on,
            "delivery_mode": self._delivery_mode.value,
            "accelerator": self.accelerator.get_status(),
            "beam_transport": self.beam_transport.get_status(),
            "gantry": self.gantry.get_status(),
        }


class HeavyIonTherapySystem(BeamDeliverySystem):
    """
    Heavy Ion (Carbon) Therapy System.

    Uses synchrotron for variable-energy carbon ion beams.
    Provides higher RBE for radioresistant tumors.
    """

    def __init__(
        self,
        name: str = "Heavy Ion Therapy System",
        max_energy: float = 430.0,  # MeV/u
        particle_type: ParticleType = ParticleType.CARBON_ION
    ):
        super().__init__(name)
        self.max_energy = max_energy
        self.particle_type = particle_type

        # Initialize components
        self.accelerator = Synchrotron(
            max_energy=max_energy,
            particle_type=particle_type
        )
        self.beam_transport = BeamTransport()
        self.gantry = GantrySystem(
            gantry_type=GantryType.ROTATING,
            name="Carbon Ion Gantry"
        )
        # Carbon gantries are massive
        self.gantry.parameters.weight = 600.0  # tons

        self.tps = TreatmentPlanningSystem(particle_type)
        self.bragg_optimizer = BraggPeakOptimizer(particle_type)

        self._current_energy = 300.0
        self._is_ready = False
        self._beam_on = False
        self._delivery_mode = DeliveryMode.PENCIL_BEAM_SCANNING

    def initialize(self) -> bool:
        """Initialize the heavy ion system."""
        success = True
        success &= self.accelerator.initialize()
        success &= self.gantry.initialize()
        self._is_ready = success
        return success

    def calibrate(self) -> bool:
        """Calibrate the system."""
        self.gantry.calibrate()
        self._is_calibrated = True
        return True

    def set_position(self, position: Position3D) -> bool:
        """Set treatment position."""
        self._position = position
        return True

    def set_energy(self, energy: float) -> bool:
        """Set beam energy (per nucleon)."""
        if energy > self.max_energy:
            return False

        self._current_energy = energy
        self.accelerator.set_energy(energy)
        return True

    def calculate_rbe_weighted_dose(
        self,
        physical_dose: DoseDistribution
    ) -> DoseDistribution:
        """
        Calculate RBE-weighted dose distribution.

        Carbon ions have variable RBE that increases in the Bragg peak.
        """
        rbe_dose = DoseDistribution(
            origin=physical_dose.origin,
            spacing=physical_dose.spacing,
            dimensions=physical_dose.dimensions
        )

        for i in range(physical_dose.dimensions[0]):
            for j in range(physical_dose.dimensions[1]):
                for k in range(physical_dose.dimensions[2]):
                    phys_dose = physical_dose.dose_grid[i, j, k]

                    # Estimate LET based on depth (simplified)
                    depth = k * physical_dose.spacing[2]
                    range_mm = self.bragg_optimizer.calculate_range(
                        self._current_energy
                    )

                    # LET increases near end of range
                    relative_depth = depth / range_mm if range_mm > 0 else 0
                    if relative_depth < 0.8:
                        let = 20.0  # keV/μm entrance
                    elif relative_depth < 1.0:
                        let = 20.0 + (relative_depth - 0.8) * 400  # Increases
                    else:
                        let = 100.0  # Peak LET

                    rbe = self.bragg_optimizer.calculate_rbe(let)
                    rbe_dose.dose_grid[i, j, k] = phys_dose * rbe

        return rbe_dose

    def deliver_dose(
        self,
        beam_params: BeamParameters,
        monitor_units: float
    ) -> DoseDistribution:
        """Deliver treatment dose."""
        self.set_energy(beam_params.energy)

        # Run acceleration cycle
        self.accelerator.run_cycle()
        self.accelerator.beam_on()
        self._beam_on = True

        # Calculate physical dose
        depths = np.linspace(0, 200, 200)
        bragg = self.bragg_optimizer.calculate_bragg_peak(
            beam_params.energy, depths
        )

        dose_dist = DoseDistribution(
            origin=Position3D(-100, -100, 0),
            spacing=(2.0, 2.0, 1.0),
            dimensions=(100, 100, 200)
        )

        for k, depth in enumerate(depths):
            for i in range(dose_dist.dimensions[0]):
                for j in range(dose_dist.dimensions[1]):
                    x = dose_dist.origin.x + i * dose_dist.spacing[0]
                    y = dose_dist.origin.y + j * dose_dist.spacing[1]

                    r = math.sqrt(x ** 2 + y ** 2)
                    # Carbon has sharper lateral falloff
                    lateral = math.exp(-r ** 2 / 30)

                    dose_dist.dose_grid[i, j, k] = (
                        monitor_units * 0.01 * bragg[k] * lateral
                    )

        self.accelerator.beam_off()
        self._beam_on = False

        return dose_dist

    def get_status(self) -> Dict:
        """Get system status."""
        return {
            "name": self.name,
            "particle_type": self.particle_type.value,
            "is_ready": self._is_ready,
            "is_calibrated": self._is_calibrated,
            "current_energy": self._current_energy,
            "beam_on": self._beam_on,
            "accelerator": self.accelerator.get_status(),
            "gantry": self.gantry.get_status(),
        }
