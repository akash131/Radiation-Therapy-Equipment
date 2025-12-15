"""
Particle Therapy Vendor Implementations.

Provides simulations of commercial proton and ion therapy systems:
- IBA (Proteus Plus, Proteus One)
- Hitachi (PROBEAT)
- Mevion (S250i HYPERSCAN)
- SHI (PROTON)
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
from ..proton_therapy import (
    Cyclotron,
    Synchrotron,
    BeamTransport,
    GantrySystem,
    GantryType,
    BraggPeakOptimizer,
    DeliveryMode,
)


class IBADeliveryMode(Enum):
    """IBA beam delivery modes."""
    PBS = "pencil_beam_scanning"
    US = "uniform_scanning"
    DS = "double_scattering"


@dataclass
class IBACyclotronSpec:
    """IBA cyclotron specifications."""
    name: str
    extraction_energy: float  # MeV
    beam_current_range: Tuple[float, float]  # nA
    energy_stability: float  # %
    magnetic_field: float  # Tesla
    weight: float  # tons
    rf_frequency: float  # MHz


class IBAProteusPlus(BeamDeliverySystem):
    """
    IBA Proteus Plus Proton Therapy System.

    Multi-room proton therapy system featuring:
    - C230 cyclotron (230 MeV)
    - Up to 5 treatment rooms
    - 360° isocentric gantries
    - Pencil beam scanning
    - Cone-beam CT imaging
    """

    CYCLOTRON_SPEC = IBACyclotronSpec(
        name="C230",
        extraction_energy=230.0,
        beam_current_range=(1.0, 500.0),
        energy_stability=0.1,
        magnetic_field=2.17,
        weight=220.0,
        rf_frequency=106.0
    )

    def __init__(self, num_rooms: int = 3, compact_gantry: bool = False):
        super().__init__(name="IBA Proteus Plus")

        self.manufacturer = "IBA"
        self.num_rooms = num_rooms
        self.compact_gantry = compact_gantry

        # Cyclotron
        self._cyclotron = Cyclotron(
            max_energy=self.CYCLOTRON_SPEC.extraction_energy,
            particle_type=ParticleType.PROTON,
            name=self.CYCLOTRON_SPEC.name
        )

        # Energy Selection System (ESS)
        self._ess_energy_range = (70.0, 230.0)  # MeV
        self._ess_degrader_material = "carbon"
        self._energy_resolution = 0.1  # MeV

        # Gantry configurations
        self._gantry_rooms = []
        for i in range(num_rooms):
            if compact_gantry:
                gantry = GantrySystem(
                    gantry_type=GantryType.COMPACT,
                    name=f"Compact Gantry {i+1}"
                )
                gantry.parameters.weight = 50.0  # Compact is lighter
            else:
                gantry = GantrySystem(
                    gantry_type=GantryType.ROTATING,
                    name=f"Gantry {i+1}"
                )
                gantry.parameters.weight = 200.0
            self._gantry_rooms.append(gantry)

        # Scanning specifications
        self._max_scan_field = (400.0, 300.0)  # mm (x, y)
        self._spot_size_range = (3.0, 8.0)  # mm sigma
        self._max_scan_speed = 20.0  # m/s

        # Delivery modes
        self._delivery_modes = [
            IBADeliveryMode.PBS,
            IBADeliveryMode.US,
            IBADeliveryMode.DS
        ]
        self._current_delivery_mode = IBADeliveryMode.PBS

        # Imaging
        self._cbct_integrated = True
        self._orthogonal_xray = True

        # Adagio adaptive system
        self._adagio_enabled = True

        self._is_ready = False
        self._current_energy = 200.0

    def initialize(self) -> bool:
        """Initialize Proteus Plus system."""
        self._cyclotron.initialize()
        for gantry in self._gantry_rooms:
            gantry.initialize()
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

    def set_energy(self, energy: float) -> bool:
        """Set beam energy via ESS."""
        if not (self._ess_energy_range[0] <= energy <= self._ess_energy_range[1]):
            return False

        self._current_energy = energy
        return True

    def set_delivery_mode(self, mode: IBADeliveryMode) -> bool:
        """Set beam delivery mode."""
        if mode not in self._delivery_modes:
            return False
        self._current_delivery_mode = mode
        return True

    def select_room(self, room_number: int) -> bool:
        """Select treatment room for beam delivery."""
        if room_number < 1 or room_number > self.num_rooms:
            return False
        # In real system, this would switch beam transport
        return True

    def deliver_spot(
        self,
        x: float,
        y: float,
        energy: float,
        charge: float
    ) -> bool:
        """Deliver single PBS spot."""
        self.set_energy(energy)
        return True

    def deliver_layer(
        self,
        spots: List[Tuple[float, float, float]],
        energy: float
    ) -> float:
        """
        Deliver energy layer.

        Returns delivery time in seconds.
        """
        self.set_energy(energy)
        # Calculate delivery time
        total_charge = sum(s[2] for s in spots)
        beam_current = 100.0  # nA typical
        delivery_time = total_charge / beam_current
        return delivery_time

    def deliver_dose(
        self,
        beam_params: BeamParameters,
        monitor_units: float
    ) -> DoseDistribution:
        """Deliver treatment dose."""
        dose_dist = DoseDistribution(
            origin=Position3D(-150, -150, 0),
            spacing=(2.0, 2.0, 1.0),
            dimensions=(150, 150, 300)
        )

        optimizer = BraggPeakOptimizer(ParticleType.PROTON)
        range_mm = optimizer.calculate_range(beam_params.energy)

        depths = np.linspace(0, 300, 300)
        bragg = optimizer.calculate_bragg_peak(beam_params.energy, depths)

        for k in range(dose_dist.dimensions[2]):
            for i in range(dose_dist.dimensions[0]):
                for j in range(dose_dist.dimensions[1]):
                    x = dose_dist.origin.x + i * dose_dist.spacing[0]
                    y = dose_dist.origin.y + j * dose_dist.spacing[1]
                    r = math.sqrt(x ** 2 + y ** 2)
                    lateral = math.exp(-r ** 2 / 100)
                    dose_dist.dose_grid[i, j, k] = (
                        monitor_units * 0.01 * bragg[k] * lateral
                    )

        return dose_dist

    def acquire_cbct(self, room: int = 1) -> np.ndarray:
        """Acquire cone-beam CT in specified room."""
        return np.random.random((512, 512, 512))

    def get_status(self) -> Dict:
        """Get system status."""
        return {
            "name": self.name,
            "manufacturer": self.manufacturer,
            "num_rooms": self.num_rooms,
            "cyclotron": self.CYCLOTRON_SPEC.name,
            "extraction_energy": self.CYCLOTRON_SPEC.extraction_energy,
            "current_energy": self._current_energy,
            "delivery_mode": self._current_delivery_mode.value,
            "is_ready": self._is_ready,
            "is_calibrated": self._is_calibrated,
            "adagio_enabled": self._adagio_enabled,
        }


class IBAProteusOne(BeamDeliverySystem):
    """
    IBA Proteus One Compact Proton Therapy System.

    Single-room compact proton system:
    - S2C2 superconducting synchrocyclotron (230 MeV)
    - Compact gantry
    - Pencil beam scanning
    - Smaller footprint than Proteus Plus
    """

    def __init__(self):
        super().__init__(name="IBA Proteus One")

        self.manufacturer = "IBA"

        # S2C2 Synchrocyclotron
        self._accelerator_type = "synchrocyclotron"
        self._extraction_energy = 230.0  # MeV
        self._superconducting = True
        self._accelerator_weight = 55.0  # tons (much lighter)

        # Energy modulation (in gantry)
        self._energy_range = (70.0, 230.0)
        self._energy_modulation = "in_gantry"

        # Compact gantry
        self._gantry_type = "compact"
        self._gantry_weight = 55.0  # tons
        self._gantry_rotation = (0.0, 360.0)

        # Scanning
        self._max_scan_field = (300.0, 200.0)
        self._spot_size_range = (4.0, 8.0)

        # Imaging
        self._cbct_integrated = True

        self._is_ready = False
        self._current_energy = 200.0

    def initialize(self) -> bool:
        """Initialize Proteus One."""
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

    def set_energy(self, energy: float) -> bool:
        """Set beam energy."""
        if not (self._energy_range[0] <= energy <= self._energy_range[1]):
            return False
        self._current_energy = energy
        return True

    def deliver_dose(
        self,
        beam_params: BeamParameters,
        monitor_units: float
    ) -> DoseDistribution:
        """Deliver treatment dose."""
        self.set_energy(beam_params.energy)

        dose_dist = DoseDistribution(
            origin=Position3D(-100, -100, 0),
            spacing=(2.0, 2.0, 1.0),
            dimensions=(100, 100, 250)
        )

        return dose_dist

    def get_status(self) -> Dict:
        """Get system status."""
        return {
            "name": self.name,
            "manufacturer": self.manufacturer,
            "accelerator_type": self._accelerator_type,
            "extraction_energy": self._extraction_energy,
            "current_energy": self._current_energy,
            "gantry_type": self._gantry_type,
            "is_ready": self._is_ready,
        }


@dataclass
class HitachiSynchrotronSpec:
    """Hitachi synchrotron specifications."""
    circumference: float = 23.0  # meters
    max_proton_energy: float = 235.0  # MeV
    max_carbon_energy: float = 430.0  # MeV/u
    injection_energy: float = 7.0  # MeV
    extraction_method: str = "RF knockout"
    spill_time: float = 2.0  # seconds


class HitachiPROBEAT(BeamDeliverySystem):
    """
    Hitachi PROBEAT Proton/Carbon Ion Therapy System.

    Features:
    - Synchrotron accelerator
    - Proton and carbon ion capability
    - Real-time tumor tracking (RGRT)
    - Respiratory gating
    - Spot scanning and passive scattering
    """

    SYNCHROTRON_SPEC = HitachiSynchrotronSpec()

    def __init__(self, ion_capability: bool = True):
        super().__init__(name="Hitachi PROBEAT")

        self.manufacturer = "Hitachi"
        self.ion_capability = ion_capability

        # Synchrotron
        self._synchrotron = Synchrotron(
            max_energy=self.SYNCHROTRON_SPEC.max_carbon_energy if ion_capability
                       else self.SYNCHROTRON_SPEC.max_proton_energy,
            particle_type=ParticleType.CARBON_ION if ion_capability
                         else ParticleType.PROTON,
            name="Hitachi Synchrotron"
        )

        # Energy range
        if ion_capability:
            self._proton_energy_range = (70.0, 235.0)
            self._carbon_energy_range = (140.0, 430.0)
        else:
            self._proton_energy_range = (70.0, 235.0)
            self._carbon_energy_range = None

        # Current particle
        self._current_particle = ParticleType.PROTON

        # Gantry
        self._gantry_weight = 300.0  # tons (heavy for carbon)
        self._gantry_rotation = (0.0, 360.0)
        self._gantry_speed = 1.0  # RPM

        # Scanning
        self._scanning_method = "spot_scanning"
        self._max_scan_field = (400.0, 300.0)
        self._spot_positioning_accuracy = 0.5  # mm

        # RGRT (Real-time Gated Radiotherapy)
        self._rgrt_enabled = True
        self._respiratory_gating = True
        self._tumor_tracking = True

        # Layer stacking
        self._layer_stacking = True
        self._repainting = True

        self._is_ready = False
        self._current_energy = 200.0

    def initialize(self) -> bool:
        """Initialize PROBEAT system."""
        self._synchrotron.initialize()
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

    def set_particle_type(self, particle: ParticleType) -> bool:
        """Switch between proton and carbon."""
        if particle == ParticleType.CARBON_ION and not self.ion_capability:
            return False

        self._current_particle = particle
        self._synchrotron.particle_type = particle
        return True

    def set_energy(self, energy: float) -> bool:
        """Set beam energy."""
        if self._current_particle == ParticleType.PROTON:
            energy_range = self._proton_energy_range
        else:
            energy_range = self._carbon_energy_range

        if not (energy_range[0] <= energy <= energy_range[1]):
            return False

        self._current_energy = energy
        self._synchrotron.set_energy(energy)
        return True

    def enable_rgrt(self) -> bool:
        """Enable Real-time Gated Radiotherapy."""
        self._rgrt_enabled = True
        return True

    def set_gating_window(self, phase_start: float, phase_end: float) -> bool:
        """Set respiratory gating window (percentage of cycle)."""
        if not self._rgrt_enabled:
            return False
        return True

    def deliver_with_repainting(
        self,
        layer: List[Tuple[float, float, float]],
        energy: float,
        repaints: int = 4
    ) -> bool:
        """
        Deliver layer with repainting for motion mitigation.

        Args:
            layer: Spot positions and weights
            energy: Layer energy
            repaints: Number of repaints
        """
        self.set_energy(energy)

        for repaint in range(repaints):
            # Deliver fraction of dose per repaint
            for x, y, weight in layer:
                # Deliver with weight/repaints
                pass

        return True

    def deliver_dose(
        self,
        beam_params: BeamParameters,
        monitor_units: float
    ) -> DoseDistribution:
        """Deliver treatment dose."""
        self.set_energy(beam_params.energy)

        dose_dist = DoseDistribution(
            origin=Position3D(-150, -150, 0),
            spacing=(2.0, 2.0, 1.0),
            dimensions=(150, 150, 300)
        )

        optimizer = BraggPeakOptimizer(self._current_particle)
        depths = np.linspace(0, 300, 300)
        bragg = optimizer.calculate_bragg_peak(beam_params.energy, depths)

        for k in range(dose_dist.dimensions[2]):
            for i in range(dose_dist.dimensions[0]):
                for j in range(dose_dist.dimensions[1]):
                    x = dose_dist.origin.x + i * dose_dist.spacing[0]
                    y = dose_dist.origin.y + j * dose_dist.spacing[1]
                    r = math.sqrt(x ** 2 + y ** 2)
                    lateral = math.exp(-r ** 2 / 80)  # Carbon sharper
                    dose_dist.dose_grid[i, j, k] = (
                        monitor_units * 0.01 * bragg[k] * lateral
                    )

        return dose_dist

    def get_status(self) -> Dict:
        """Get system status."""
        return {
            "name": self.name,
            "manufacturer": self.manufacturer,
            "current_particle": self._current_particle.value,
            "ion_capability": self.ion_capability,
            "current_energy": self._current_energy,
            "rgrt_enabled": self._rgrt_enabled,
            "is_ready": self._is_ready,
        }


class MevionS250i(BeamDeliverySystem):
    """
    Mevion S250i HYPERSCAN Proton Therapy System.

    Compact single-room proton system:
    - Superconducting synchrocyclotron mounted on gantry
    - HYPERSCAN pencil beam scanning
    - Discrete spot scanning
    - Very compact footprint
    """

    def __init__(self):
        super().__init__(name="Mevion S250i HYPERSCAN")

        self.manufacturer = "Mevion Medical Systems"

        # Synchrocyclotron on gantry
        self._accelerator_type = "synchrocyclotron"
        self._superconducting = True
        self._extraction_energy = 250.0  # MeV
        self._accelerator_on_gantry = True

        # Energy modulation
        self._energy_range = (60.0, 250.0)  # MeV
        self._energy_modulation = "HYPERSCAN"
        self._adaptive_aperture = True  # Dynamic collimation

        # Gantry
        self._gantry_rotation = (0.0, 360.0)
        self._gantry_speed = 3.0  # deg/s
        self._gantry_weight = 25.0  # tons (very light)

        # HYPERSCAN specifications
        self._max_scan_field = (260.0, 260.0)  # mm
        self._spot_size = 6.0  # mm sigma
        self._discrete_energy_layers = 37

        # Adaptive Aperture
        self._aperture_leaves = 26  # pairs
        self._aperture_leaf_width = 10.0  # mm

        # Imaging
        self._cbct_integrated = True
        self._orthogonal_xray = True

        self._is_ready = False
        self._current_energy = 200.0

    def initialize(self) -> bool:
        """Initialize Mevion system."""
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

    def set_energy(self, energy: float) -> bool:
        """Set beam energy."""
        if not (self._energy_range[0] <= energy <= self._energy_range[1]):
            return False

        # HYPERSCAN uses discrete energy selection
        self._current_energy = energy
        return True

    def set_adaptive_aperture(
        self,
        aperture: List[Tuple[float, float]]
    ) -> bool:
        """
        Set Adaptive Aperture for lateral conformality.

        The aperture provides physical collimation to reduce
        lateral penumbra and improve conformality.
        """
        return True

    def deliver_hyperscan_layer(
        self,
        spots: List[Tuple[float, float, float]],
        energy: float,
        aperture: Optional[List[Tuple[float, float]]] = None
    ) -> bool:
        """
        Deliver layer using HYPERSCAN.

        Combines discrete spot scanning with adaptive aperture.
        """
        self.set_energy(energy)

        if aperture:
            self.set_adaptive_aperture(aperture)

        for x, y, weight in spots:
            # Deliver spot
            pass

        return True

    def deliver_dose(
        self,
        beam_params: BeamParameters,
        monitor_units: float
    ) -> DoseDistribution:
        """Deliver treatment dose."""
        self.set_energy(beam_params.energy)

        dose_dist = DoseDistribution(
            origin=Position3D(-130, -130, 0),
            spacing=(2.0, 2.0, 1.0),
            dimensions=(130, 130, 280)
        )

        optimizer = BraggPeakOptimizer(ParticleType.PROTON)
        depths = np.linspace(0, 280, 280)
        bragg = optimizer.calculate_bragg_peak(beam_params.energy, depths)

        for k in range(dose_dist.dimensions[2]):
            for i in range(dose_dist.dimensions[0]):
                for j in range(dose_dist.dimensions[1]):
                    x = dose_dist.origin.x + i * dose_dist.spacing[0]
                    y = dose_dist.origin.y + j * dose_dist.spacing[1]
                    r = math.sqrt(x ** 2 + y ** 2)
                    # Adaptive aperture sharpens penumbra
                    lateral = math.exp(-r ** 2 / 70)
                    dose_dist.dose_grid[i, j, k] = (
                        monitor_units * 0.01 * bragg[k] * lateral
                    )

        return dose_dist

    def get_footprint(self) -> Dict:
        """Get system footprint requirements."""
        return {
            "vault_length": 9.0,  # meters
            "vault_width": 9.0,
            "vault_height": 4.5,
            "shielding_concrete": 2.0,  # meters typical
        }

    def get_status(self) -> Dict:
        """Get system status."""
        return {
            "name": self.name,
            "manufacturer": self.manufacturer,
            "extraction_energy": self._extraction_energy,
            "current_energy": self._current_energy,
            "accelerator_on_gantry": self._accelerator_on_gantry,
            "adaptive_aperture": self._adaptive_aperture,
            "is_ready": self._is_ready,
        }
