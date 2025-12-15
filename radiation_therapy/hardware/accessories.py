"""
Treatment Accessories.

Implements beam modifying devices:
- Physical wedges
- Electron applicators
- Block trays and custom blocks
- Bolus materials
- Shielding devices
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Dict
import math
import numpy as np

from ..base import Position3D


class WedgeType(Enum):
    """Types of physical wedges."""
    PHYSICAL = "physical"  # Fixed angle steel wedge
    DYNAMIC = "dynamic"  # MLC-based virtual wedge
    ENHANCED_DYNAMIC = "enhanced_dynamic"  # EDW
    MOTORIZED = "motorized"  # Universal wedge


class WedgeOrientation(Enum):
    """Wedge orientation."""
    IN = "in"  # Heel toward gantry
    OUT = "out"  # Toe toward gantry
    LEFT = "left"
    RIGHT = "right"


@dataclass
class WedgeSpec:
    """Physical wedge specifications."""
    angle: float  # degrees
    material: str  # Steel, lead, etc.
    transmission_factor: float
    wedge_factor: float
    field_size_range: Tuple[float, float]  # (min, max) cm


class PhysicalWedge:
    """
    Physical wedge filter for beam modification.

    Creates tilted isodose distribution for oblique patient surfaces
    or dose compensation.
    """

    STANDARD_WEDGES = {
        15: WedgeSpec(15, "steel", 0.95, 0.73, (4, 30)),
        30: WedgeSpec(30, "steel", 0.90, 0.65, (4, 25)),
        45: WedgeSpec(45, "steel", 0.85, 0.58, (4, 20)),
        60: WedgeSpec(60, "steel", 0.78, 0.52, (4, 15)),
    }

    def __init__(
        self,
        angle: float,
        wedge_type: WedgeType = WedgeType.PHYSICAL
    ):
        self.angle = angle
        self.wedge_type = wedge_type
        self.spec = self.STANDARD_WEDGES.get(int(angle))

        self._orientation = WedgeOrientation.IN
        self._is_inserted = False

        # Physical properties
        if self.spec:
            self._material = self.spec.material
            self._transmission_factor = self.spec.transmission_factor
            self._wedge_factor = self.spec.wedge_factor
        else:
            self._material = "steel"
            self._transmission_factor = 0.85
            self._wedge_factor = 0.6

    def insert(self, orientation: WedgeOrientation = WedgeOrientation.IN) -> bool:
        """Insert wedge with specified orientation."""
        self._orientation = orientation
        self._is_inserted = True
        return True

    def remove(self) -> bool:
        """Remove wedge from beam path."""
        self._is_inserted = False
        return True

    def calculate_transmission(
        self,
        position: Position3D,
        field_center: Position3D
    ) -> float:
        """
        Calculate wedge transmission at position.

        The wedge creates a gradient in transmission across the field.
        """
        if not self._is_inserted:
            return 1.0

        # Calculate position relative to field center
        if self._orientation in [WedgeOrientation.IN, WedgeOrientation.OUT]:
            offset = position.y - field_center.y
        else:
            offset = position.x - field_center.x

        # Flip for OUT/RIGHT orientations
        if self._orientation in [WedgeOrientation.OUT, WedgeOrientation.RIGHT]:
            offset = -offset

        # Transmission gradient
        wedge_slope = math.tan(math.radians(self.angle))
        transmission_gradient = 0.01 * wedge_slope  # Simplified

        base_transmission = self._transmission_factor
        local_transmission = base_transmission * (1 + offset * transmission_gradient)

        return max(0.1, min(1.0, local_transmission))

    def get_wedge_factor(self, field_size: Tuple[float, float]) -> float:
        """Get wedge factor for given field size."""
        # Wedge factor varies with field size
        avg_size = (field_size[0] + field_size[1]) / 2
        base_factor = self._wedge_factor

        # Slight variation with field size
        size_correction = 1.0 + 0.001 * (avg_size - 10)
        return base_factor * size_correction


class ElectronApplicator:
    """
    Electron beam applicator (cone).

    Defines field size and provides electron scatter equilibrium.
    """

    STANDARD_SIZES = [6, 10, 15, 20, 25]  # cm

    def __init__(
        self,
        size: float,  # cm
        shape: str = "square"  # square, rectangular, circular
    ):
        self.size = size
        self.shape = shape

        # Physical specifications
        self._material = "aluminum"
        self._wall_thickness = 3.0  # mm
        self._end_frame_thickness = 6.0  # mm

        # Insert slot for cutouts
        self._has_insert = True
        self._insert_material = "cerrobend"

        # Applicator-specific output factors
        self._output_factors = {
            6: 0.98,
            10: 1.00,  # Reference
            15: 1.01,
            20: 1.02,
            25: 1.02,
        }

        self._is_attached = False

    def attach(self) -> bool:
        """Attach applicator to LINAC head."""
        self._is_attached = True
        return True

    def remove(self) -> bool:
        """Remove applicator."""
        self._is_attached = False
        return True

    def get_output_factor(self) -> float:
        """Get applicator output factor."""
        return self._output_factors.get(int(self.size), 1.0)

    def get_effective_ssd(self, nominal_ssd: float = 100.0) -> float:
        """
        Get effective SSD for electron beams.

        Accounts for virtual source position.
        """
        # Virtual source shift depends on energy and applicator
        virtual_source_shift = 2.0  # cm typical
        return nominal_ssd - virtual_source_shift


@dataclass
class BlockDimensions:
    """Dimensions for cerrobend block."""
    width: float  # cm
    height: float  # cm
    thickness: float  # cm


class BlockTray:
    """
    Block tray for holding custom cerrobend blocks.

    Used for field shaping before MLC era, still used for electrons.
    """

    def __init__(
        self,
        tray_size: Tuple[float, float] = (30.0, 30.0),
        tray_factor: float = 0.97
    ):
        self.tray_size = tray_size  # cm
        self.tray_factor = tray_factor  # Transmission through tray

        self._tray_material = "acrylic"
        self._tray_thickness = 5.0  # mm

        self._blocks: List[Dict] = []
        self._is_inserted = False

    def insert(self) -> bool:
        """Insert block tray."""
        self._is_inserted = True
        return True

    def remove(self) -> bool:
        """Remove block tray."""
        self._is_inserted = False
        return True

    def add_block(
        self,
        shape: np.ndarray,  # 2D array defining shape
        material: str = "cerrobend"
    ):
        """Add custom block to tray."""
        block = {
            "shape": shape,
            "material": material,
            "thickness": 7.5 if material == "cerrobend" else 5.0,  # cm
        }
        self._blocks.append(block)

    def calculate_transmission(
        self,
        position: Position3D
    ) -> float:
        """Calculate total transmission at position."""
        if not self._is_inserted:
            return 1.0

        transmission = self.tray_factor

        for block in self._blocks:
            # Check if position is under block
            # Simplified - actual implementation checks 2D shape
            if self._is_under_block(position, block["shape"]):
                # Cerrobend transmission ~1%
                transmission *= 0.01

        return transmission

    def _is_under_block(
        self,
        position: Position3D,
        shape: np.ndarray
    ) -> bool:
        """Check if position is under the block."""
        # Simplified check
        return False


class BolusType(Enum):
    """Types of bolus material."""
    SUPERFLAB = "superflab"  # Gel-like
    WAX = "paraffin_wax"
    SHEET = "sheet_bolus"
    CUSTOM_3D = "3d_printed"
    WET_GAUZE = "wet_gauze"


class Bolus:
    """
    Bolus material for surface dose enhancement.

    Tissue-equivalent material placed on skin to:
    - Increase surface dose
    - Shift dose buildup toward surface
    - Fill air gaps
    """

    MATERIAL_PROPERTIES = {
        BolusType.SUPERFLAB: {"density": 1.02, "thickness_range": (3, 20)},
        BolusType.WAX: {"density": 0.92, "thickness_range": (5, 30)},
        BolusType.SHEET: {"density": 1.01, "thickness_range": (5, 10)},
        BolusType.CUSTOM_3D: {"density": 1.05, "thickness_range": (2, 50)},
        BolusType.WET_GAUZE: {"density": 1.0, "thickness_range": (1, 5)},
    }

    def __init__(
        self,
        bolus_type: BolusType,
        thickness: float,  # mm
        dimensions: Tuple[float, float] = (200, 200)  # mm
    ):
        self.bolus_type = bolus_type
        self.thickness = thickness
        self.dimensions = dimensions

        props = self.MATERIAL_PROPERTIES[bolus_type]
        self._density = props["density"]
        self._thickness_range = props["thickness_range"]

    def get_water_equivalent_thickness(self) -> float:
        """Get water-equivalent thickness."""
        return self.thickness * self._density

    def get_surface_dose_factor(
        self,
        energy: float,
        original_surface_dose: float
    ) -> float:
        """
        Calculate surface dose enhancement.

        Bolus shifts the depth dose curve, increasing surface dose.
        """
        # Simplified model
        wet = self.get_water_equivalent_thickness()

        # For 6 MV, dmax ~ 15mm
        if energy <= 6:
            dmax = 15.0
        elif energy <= 10:
            dmax = 25.0
        else:
            dmax = 35.0

        # If bolus thickness approaches dmax, surface dose approaches 100%
        shift_factor = min(1.0, wet / dmax)
        new_surface_dose = original_surface_dose + (100 - original_surface_dose) * shift_factor

        return new_surface_dose


class ShieldType(Enum):
    """Types of shielding."""
    EYE_SHIELD = "eye_shield"
    LIP_SHIELD = "lip_shield"
    GONAD_SHIELD = "gonad_shield"
    LEAD_APRON = "lead_apron"
    TESTICULAR_SHIELD = "testicular_shield"


class LeadShield:
    """
    Lead shielding for critical structure protection.

    Surface shields for electron beams or external shields
    for scattered radiation.
    """

    SHIELD_SPECS = {
        ShieldType.EYE_SHIELD: {"thickness": 2.0, "shape": "circular", "size": 25.0},
        ShieldType.LIP_SHIELD: {"thickness": 3.0, "shape": "curved", "size": 40.0},
        ShieldType.GONAD_SHIELD: {"thickness": 1.0, "shape": "triangular", "size": 100.0},
        ShieldType.TESTICULAR_SHIELD: {"thickness": 2.0, "shape": "cup", "size": 80.0},
    }

    def __init__(
        self,
        shield_type: ShieldType,
        internal: bool = False  # Internal (in beam) or external (scatter)
    ):
        self.shield_type = shield_type
        self.internal = internal

        spec = self.SHIELD_SPECS[shield_type]
        self._thickness = spec["thickness"]  # mm lead
        self._shape = spec["shape"]
        self._size = spec["size"]  # mm characteristic dimension

        self._is_placed = False
        self._position = Position3D()

    def place(self, position: Position3D) -> bool:
        """Place shield at specified position."""
        self._position = position
        self._is_placed = True
        return True

    def remove(self) -> bool:
        """Remove shield."""
        self._is_placed = False
        return True

    def get_transmission(self, energy: float) -> float:
        """
        Calculate transmission through lead shield.

        Depends on beam energy and lead thickness.
        """
        # Lead HVL for electrons ~0.5mm, for photons ~5mm at 6MV
        if energy < 25:  # Electron
            hvl = 0.5
        else:  # Photon energy (MV)
            hvl = 5.0

        transmission = 2 ** (-self._thickness / hvl)
        return transmission

    def get_dose_reduction(self, energy: float) -> float:
        """Calculate dose reduction percentage."""
        transmission = self.get_transmission(energy)
        return (1 - transmission) * 100
