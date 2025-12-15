"""
Siemens Healthineers Radiation Therapy Equipment Implementations.

Provides simulations of Siemens linear accelerators:
- Artiste
- Oncor
- Primus (legacy)

Note: Siemens exited the linear accelerator market in 2012,
but many systems remain in clinical use.
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
    ParticleType,
)
from ..linac import (
    LinearAccelerator,
    MultiLeafCollimator,
    MLCType,
)


@dataclass
class Siemens160MLCSpec:
    """Siemens 160 MLC specifications."""
    num_leaf_pairs: int = 80
    leaf_width_inner: float = 5.0  # mm (central 40 pairs)
    leaf_width_outer: float = 10.0  # mm (outer 20 pairs each side)
    max_leaf_speed: float = 20.0  # mm/s
    leaf_travel: float = 150.0  # mm
    min_gap: float = 0.5  # mm


class Siemens160MLC(MultiLeafCollimator):
    """
    Siemens 160 MLC (80 leaf pairs).

    Features:
    - 40 pairs of 5mm leaves (central)
    - 40 pairs of 10mm leaves (20 each side)
    - Rounded leaf ends
    - Tongue-and-groove design
    """

    def __init__(self):
        super().__init__(
            num_leaf_pairs=80,
            leaf_width=5.0,
            mlc_type=MLCType.STANDARD,
            name="Siemens 160 MLC"
        )

        self.max_leaf_speed = 20.0
        self._interlock_gap = 0.5

        # Configure leaf widths
        self._leaf_widths = []
        for i in range(80):
            if 20 <= i < 60:  # Central 40 pairs
                self._leaf_widths.append(5.0)
            else:  # Outer 20 pairs each side
                self._leaf_widths.append(10.0)


class SiemensBeamMode(Enum):
    """Siemens beam modes."""
    PHOTON = "photon"
    ELECTRON = "electron"
    IMRT = "imrt"
    STEP_AND_SHOOT = "step_and_shoot"
    SLIDING_WINDOW = "sliding_window"


@dataclass
class SiemensMVision:
    """Siemens MVision portal imaging system."""
    detector_size: Tuple[int, int] = (1024, 1024)
    pixel_pitch: float = 0.4  # mm
    active_area: Tuple[float, float] = (410.0, 410.0)  # mm
    imaging_modes: List[str] = field(default_factory=lambda: [
        "high_quality", "high_speed", "dosimetry"
    ])


@dataclass
class SiemensKView:
    """Siemens kView kV imaging system."""
    tube_voltage_range: Tuple[float, float] = (40.0, 150.0)  # kV
    tube_current_range: Tuple[float, float] = (10.0, 630.0)  # mA
    detector_size: Tuple[int, int] = (2048, 1536)
    cbct_capable: bool = True


class SiemensArtiste(LinearAccelerator):
    """
    Siemens Artiste Linear Accelerator.

    High-end Siemens LINAC featuring:
    - Photon: 6, 10, 15, 18 MV
    - Electron: 6-21 MeV
    - 160 MLC (optional 82-leaf for SBRT)
    - MVision and kView imaging
    - Adaptive Targeting
    """

    PHOTON_ENERGIES = [6.0, 10.0, 15.0, 18.0]
    ELECTRON_ENERGIES = [6.0, 9.0, 12.0, 15.0, 18.0, 21.0]

    def __init__(self, configuration: str = "standard"):
        super().__init__(
            name=f"Siemens Artiste {configuration}",
            max_photon_energy=18.0,
            max_electron_energy=21.0
        )

        self.manufacturer = "Siemens Healthcare"
        self.configuration = configuration

        # MLC
        self.mlc = Siemens160MLC()

        # Imaging systems
        self.mvision = SiemensMVision()
        self.kview = SiemensKView()
        self._kview_installed = True

        # Adaptive Targeting system
        self._adaptive_targeting = True

        # Gantry
        self._max_gantry_speed = 5.0  # deg/s

        # IMRT modes
        self._step_and_shoot = True
        self._sliding_window = True

        # Virtual wedge
        self._virtual_wedge_angles = [15, 30, 45, 60]

        # Dose rate
        self._max_dose_rate = 500.0  # MU/min standard

    def set_virtual_wedge(self, angle: float, orientation: str) -> bool:
        """
        Set virtual wedge using MLC.

        Args:
            angle: Wedge angle (15, 30, 45, 60)
            orientation: "in", "out", "left", "right"
        """
        if angle not in self._virtual_wedge_angles:
            return False

        # Calculate MLC pattern for wedge
        wedge_gradient = math.tan(math.radians(angle))
        # Implementation would create wedge-shaped fluence
        return True

    def acquire_cbct(self, preset: str = "thorax") -> np.ndarray:
        """Acquire kView CBCT."""
        if not self._kview_installed:
            return None

        matrix_size = 512
        return np.random.random((matrix_size, matrix_size, matrix_size))

    def perform_adaptive_targeting(
        self,
        reference_position: Position3D,
        cbct_volume: np.ndarray
    ) -> Dict:
        """
        Perform Adaptive Targeting registration.

        Uses automatic bone or soft tissue registration.
        """
        correction = {
            "translation": Position3D(0, 0, 0),
            "rotation": {"pitch": 0, "roll": 0, "yaw": 0},
            "registration_type": "rigid",
            "match_quality": 0.95
        }
        return correction

    def get_machine_parameters(self) -> Dict:
        """Get Artiste machine parameters."""
        return {
            "manufacturer": self.manufacturer,
            "model": "Artiste",
            "configuration": self.configuration,
            "photon_energies": self.PHOTON_ENERGIES,
            "electron_energies": self.ELECTRON_ENERGIES,
            "mlc_type": "160 MLC",
            "max_dose_rate": self._max_dose_rate,
            "kview_installed": self._kview_installed,
            "adaptive_targeting": self._adaptive_targeting,
        }


class SiemensOncor(LinearAccelerator):
    """
    Siemens Oncor Linear Accelerator.

    Mid-range Siemens LINAC:
    - Photon: 6, 15 MV (or 6, 10, 18 MV)
    - Electron: 6-18 MeV
    - Optional 160 MLC or 82 MLC
    """

    def __init__(self, energy_config: str = "6_15"):
        if energy_config == "6_15":
            photon_max = 15.0
            self._photon_energies = [6.0, 15.0]
        else:  # 6_10_18
            photon_max = 18.0
            self._photon_energies = [6.0, 10.0, 18.0]

        super().__init__(
            name="Siemens Oncor",
            max_photon_energy=photon_max,
            max_electron_energy=18.0
        )

        self.manufacturer = "Siemens Healthcare"
        self.mlc = Siemens160MLC()

        # Imaging
        self._mvision = True
        self._kview = False  # Optional

        # IMRT
        self._imrt_capable = True


class SiemensPrimus(LinearAccelerator):
    """
    Siemens Primus Linear Accelerator (Legacy).

    Older Siemens platform:
    - Photon: 6, 15 MV typically
    - Electron: 6-18 MeV
    - 58 or 82 leaf MLC options
    """

    def __init__(self, mlc_option: str = "58"):
        super().__init__(
            name="Siemens Primus",
            max_photon_energy=15.0,
            max_electron_energy=18.0
        )

        self.manufacturer = "Siemens Healthcare"

        if mlc_option == "82":
            self.mlc = MultiLeafCollimator(
                num_leaf_pairs=41,
                leaf_width=10.0,
                name="Siemens 82 MLC"
            )
        else:
            self.mlc = MultiLeafCollimator(
                num_leaf_pairs=29,
                leaf_width=10.0,
                name="Siemens 58 MLC"
            )

        # Legacy system limitations
        self._max_dose_rate = 400.0
        self._imrt_capable = True
        self._vmat_capable = False  # No VMAT on Primus


# Keep compatibility with older naming
SiemensOncorIMPRESSION = SiemensOncor
