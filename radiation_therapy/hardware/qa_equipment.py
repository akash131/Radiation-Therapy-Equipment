"""
Quality Assurance Equipment.

Implements QA devices and phantoms:
- Water and solid phantoms
- Anthropomorphic phantoms
- Daily QA devices
- Test patterns (Winston-Lutz, Picket Fence, etc.)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Dict
from abc import ABC, abstractmethod
import math
import numpy as np

from ..base import Position3D, DoseDistribution


class PhantomMaterial(Enum):
    """Phantom materials."""
    WATER = "water"
    SOLID_WATER = "solid_water"
    PMMA = "pmma"  # Acrylic/Lucite
    POLYSTYRENE = "polystyrene"
    VIRTUAL_WATER = "virtual_water"
    LUNG_EQUIVALENT = "lung"
    BONE_EQUIVALENT = "bone"


@dataclass
class MaterialProperties:
    """Physical properties of phantom materials."""
    name: str
    density: float  # g/cm^3
    electron_density_ratio: float  # relative to water
    effective_z: float
    stopping_power_ratio: float  # for protons


MATERIAL_PROPERTIES = {
    PhantomMaterial.WATER: MaterialProperties(
        "Water", 1.0, 1.0, 7.42, 1.0
    ),
    PhantomMaterial.SOLID_WATER: MaterialProperties(
        "Solid Water", 1.03, 1.003, 7.49, 1.003
    ),
    PhantomMaterial.PMMA: MaterialProperties(
        "PMMA", 1.18, 1.147, 6.56, 1.156
    ),
    PhantomMaterial.POLYSTYRENE: MaterialProperties(
        "Polystyrene", 1.05, 0.979, 5.69, 0.984
    ),
    PhantomMaterial.LUNG_EQUIVALENT: MaterialProperties(
        "Lung", 0.30, 0.293, 7.40, 0.293
    ),
    PhantomMaterial.BONE_EQUIVALENT: MaterialProperties(
        "Bone", 1.85, 1.695, 13.8, 1.52
    ),
}


class Phantom(ABC):
    """Abstract base class for phantoms."""

    def __init__(
        self,
        name: str,
        material: PhantomMaterial,
        dimensions: Tuple[float, float, float]
    ):
        self.name = name
        self.material = material
        self.dimensions = dimensions
        self.properties = MATERIAL_PROPERTIES.get(
            material, MATERIAL_PROPERTIES[PhantomMaterial.WATER]
        )

        self._detector_positions: List[Position3D] = []

    @abstractmethod
    def get_depth_at_position(self, position: Position3D) -> float:
        """Get water-equivalent depth at position."""
        pass

    def add_detector_position(self, position: Position3D):
        """Add detector measurement position."""
        self._detector_positions.append(position)


class WaterPhantom(Phantom):
    """
    Scanning water phantom for beam commissioning.

    Features:
    - 3D scanning capability
    - Automated detector positioning
    - Temperature monitoring
    """

    def __init__(
        self,
        name: str = "Water Phantom",
        dimensions: Tuple[float, float, float] = (500.0, 500.0, 500.0)
    ):
        super().__init__(name, PhantomMaterial.WATER, dimensions)

        # Tank specifications
        self._wall_material = "PMMA"
        self._wall_thickness = 10.0  # mm
        self._water_level = dimensions[2] - 20  # mm below top

        # Scanning system
        self._scanner_type = "3D"
        self._position_accuracy = 0.1  # mm
        self._scan_speed = 10.0  # mm/s

        # Current detector position
        self._detector_position = Position3D(0, 0, 100)

        # Temperature
        self._water_temperature = 20.0  # °C

    def get_depth_at_position(self, position: Position3D) -> float:
        """Get depth in water."""
        # Depth is z-coordinate from water surface
        return self._water_level - position.z

    def move_detector(self, position: Position3D) -> bool:
        """Move detector to specified position."""
        # Check within tank bounds
        half_x = self.dimensions[0] / 2 - self._wall_thickness
        half_y = self.dimensions[1] / 2 - self._wall_thickness

        if abs(position.x) > half_x or abs(position.y) > half_y:
            return False
        if position.z < 0 or position.z > self._water_level:
            return False

        self._detector_position = position
        return True

    def scan_pdd(
        self,
        depths: List[float],
        field_size: Tuple[float, float] = (100.0, 100.0)
    ) -> List[Tuple[float, float]]:
        """
        Scan percentage depth dose.

        Returns list of (depth, relative_dose) pairs.
        """
        results = []
        for depth in depths:
            self.move_detector(Position3D(0, 0, self._water_level - depth))
            # Simulated reading
            dose = 100.0 * math.exp(-0.05 * depth)  # Simplified
            results.append((depth, dose))
        return results

    def scan_profile(
        self,
        axis: str,
        depth: float,
        positions: List[float]
    ) -> List[Tuple[float, float]]:
        """
        Scan beam profile.

        Args:
            axis: "x" or "y"
            depth: Measurement depth in mm
            positions: Off-axis positions in mm
        """
        results = []
        z = self._water_level - depth

        for pos in positions:
            if axis.lower() == "x":
                self.move_detector(Position3D(pos, 0, z))
            else:
                self.move_detector(Position3D(0, pos, z))

            # Simulated profile (Gaussian falloff)
            dose = 100.0 * math.exp(-pos ** 2 / 5000)
            results.append((pos, dose))

        return results

    def set_temperature(self, temperature: float) -> bool:
        """Set water temperature (for corrections)."""
        if temperature < 15 or temperature > 30:
            return False
        self._water_temperature = temperature
        return True


class SolidWaterPhantom(Phantom):
    """
    Solid water phantom for reference dosimetry.

    Slabs of water-equivalent material.
    """

    def __init__(
        self,
        dimensions: Tuple[float, float, float] = (300.0, 300.0, 300.0),
        slab_thickness: float = 10.0
    ):
        super().__init__("Solid Water Phantom", PhantomMaterial.SOLID_WATER, dimensions)

        self.slab_thickness = slab_thickness
        self._num_slabs = int(dimensions[2] / slab_thickness)

        # Cavity for ion chamber
        self._cavity_depth = 100.0  # mm
        self._cavity_size = (10.0, 10.0)  # mm

    def get_depth_at_position(self, position: Position3D) -> float:
        """Get water-equivalent depth."""
        physical_depth = self.dimensions[2] / 2 - position.z
        return physical_depth * self.properties.electron_density_ratio

    def set_measurement_depth(self, depth: float) -> int:
        """
        Calculate number of slabs for specified depth.

        Returns number of slabs to place above detector.
        """
        return int(depth / self.slab_thickness)


class CylindricalPhantom(Phantom):
    """
    Cylindrical phantom for rotational QA.

    Used for VMAT and TomoTherapy verification.
    """

    def __init__(
        self,
        diameter: float = 200.0,
        length: float = 200.0,
        material: PhantomMaterial = PhantomMaterial.SOLID_WATER
    ):
        super().__init__(
            "Cylindrical Phantom",
            material,
            (diameter, diameter, length)
        )

        self.diameter = diameter
        self.length = length

        # Central cavity for ion chamber
        self._central_cavity = True
        self._cavity_diameter = 8.0  # mm

    def get_depth_at_position(self, position: Position3D) -> float:
        """Get depth from surface."""
        r = math.sqrt(position.x ** 2 + position.y ** 2)
        return (self.diameter / 2 - r) * self.properties.electron_density_ratio


class AnthropomorphicPhantom(Phantom):
    """
    Anthropomorphic phantom for end-to-end testing.

    Realistic human anatomy with tissue-equivalent materials.
    """

    def __init__(
        self,
        body_region: str = "thorax",
        manufacturer: str = "CIRS"
    ):
        dimensions = {
            "head": (180.0, 220.0, 220.0),
            "thorax": (350.0, 300.0, 400.0),
            "pelvis": (350.0, 250.0, 300.0),
        }

        super().__init__(
            f"Anthropomorphic {body_region.title()} Phantom",
            PhantomMaterial.SOLID_WATER,
            dimensions.get(body_region, (300.0, 300.0, 300.0))
        )

        self.body_region = body_region
        self.manufacturer = manufacturer

        # Tissue-equivalent inserts
        self._inserts: Dict[str, PhantomMaterial] = {}
        self._setup_inserts()

        # Detector locations
        self._dosimeter_locations: List[str] = []

    def _setup_inserts(self):
        """Set up anatomical inserts."""
        if self.body_region == "thorax":
            self._inserts = {
                "lung_left": PhantomMaterial.LUNG_EQUIVALENT,
                "lung_right": PhantomMaterial.LUNG_EQUIVALENT,
                "spine": PhantomMaterial.BONE_EQUIVALENT,
                "tumor": PhantomMaterial.SOLID_WATER,
            }
            self._dosimeter_locations = ["tumor", "lung_left", "spine"]

        elif self.body_region == "head":
            self._inserts = {
                "brain": PhantomMaterial.SOLID_WATER,
                "skull": PhantomMaterial.BONE_EQUIVALENT,
                "target": PhantomMaterial.SOLID_WATER,
            }
            self._dosimeter_locations = ["target", "brainstem"]

    def get_depth_at_position(self, position: Position3D) -> float:
        """Get effective depth considering heterogeneities."""
        # Simplified - actual implementation would ray-trace
        return abs(position.z) * 1.0

    def measure_at_location(self, location: str) -> float:
        """Measure dose at anatomical location."""
        if location not in self._dosimeter_locations:
            return 0.0
        # Return simulated measurement
        return np.random.normal(2.0, 0.05)


class DailyQADevice(ABC):
    """Abstract base class for daily QA devices."""

    def __init__(self, name: str):
        self.name = name
        self._is_calibrated = False
        self._last_measurement = None

    @abstractmethod
    def measure(self) -> Dict:
        """Perform daily QA measurement."""
        pass

    @abstractmethod
    def analyze(self, measurement: Dict) -> Dict:
        """Analyze measurement results."""
        pass


class QuickCheck(DailyQADevice):
    """
    PTW QuickCheck daily QA device.

    Measures output, flatness, symmetry in single measurement.
    """

    def __init__(self):
        super().__init__("PTW QuickCheck")

        # Detector arrangement
        self._num_chambers = 13
        self._central_chamber = 1
        self._radial_chambers = 12

        # Measurement capabilities
        self._measures_output = True
        self._measures_flatness = True
        self._measures_symmetry = True
        self._measures_energy = True

    def measure(self) -> Dict:
        """Perform QuickCheck measurement."""
        # Simulate measurements
        readings = {
            "central": np.random.normal(1.0, 0.001),
            "radial": [np.random.normal(1.0, 0.002) for _ in range(12)],
        }

        self._last_measurement = readings
        return readings

    def analyze(self, measurement: Dict) -> Dict:
        """Analyze QuickCheck results."""
        central = measurement.get("central", 1.0)
        radial = measurement.get("radial", [1.0] * 12)

        # Calculate parameters
        flatness = (max(radial) - min(radial)) / (max(radial) + min(radial)) * 100
        symmetry_x = abs(radial[0] - radial[6]) / central * 100  # Cross-plane
        symmetry_y = abs(radial[3] - radial[9]) / central * 100  # In-plane

        return {
            "output": central,
            "flatness": flatness,
            "symmetry_x": symmetry_x,
            "symmetry_y": symmetry_y,
            "pass": flatness < 3.0 and symmetry_x < 2.0 and symmetry_y < 2.0,
        }


class DailyQA3(DailyQADevice):
    """
    Sun Nuclear Daily QA3 device.

    Array-based daily QA with trend analysis.
    """

    def __init__(self):
        super().__init__("Sun Nuclear Daily QA3")

        # Detector array
        self._detector_type = "ion_chamber"
        self._array_size = (10, 10)

        # Measurements
        self._baseline = None

    def measure(self) -> Dict:
        """Perform Daily QA3 measurement."""
        readings = np.random.normal(1.0, 0.005, self._array_size)

        self._last_measurement = {
            "array": readings,
            "output": readings[5, 5],
            "timestamp": "now",
        }
        return self._last_measurement

    def analyze(self, measurement: Dict) -> Dict:
        """Analyze Daily QA3 results."""
        array = measurement.get("array", np.ones(self._array_size))
        center = array[5, 5]

        # Calculate beam parameters
        flatness = (array.max() - array.min()) / (array.max() + array.min()) * 100

        return {
            "output": center,
            "flatness": flatness,
            "symmetry_x": 0.5,
            "symmetry_y": 0.5,
            "pass": flatness < 3.0,
        }

    def set_baseline(self):
        """Set current measurement as baseline."""
        self._baseline = self._last_measurement


@dataclass
class QATestResult:
    """Result from a QA test."""
    test_name: str
    passed: bool
    measured_value: float
    tolerance: float
    unit: str
    details: Dict = field(default_factory=dict)


class StarShot:
    """
    Star shot test for isocenter verification.

    Exposes multiple spoke fields to verify gantry, collimator,
    and couch rotation accuracy.
    """

    def __init__(
        self,
        num_spokes: int = 8,
        spoke_width: float = 3.0  # mm
    ):
        self.num_spokes = num_spokes
        self.spoke_width = spoke_width
        self._angles = [i * 360 / num_spokes for i in range(num_spokes)]

        # Results
        self._radius = 0.0
        self._center = Position3D()

    def analyze_film(self, film_image: np.ndarray) -> QATestResult:
        """
        Analyze star shot film.

        Finds the minimum circle encompassing all spoke centers.
        """
        # Simulate analysis
        self._radius = np.random.uniform(0.3, 0.8)  # mm
        self._center = Position3D(
            x=np.random.uniform(-0.2, 0.2),
            y=np.random.uniform(-0.2, 0.2),
            z=0
        )

        return QATestResult(
            test_name="Star Shot",
            passed=self._radius < 1.0,
            measured_value=self._radius,
            tolerance=1.0,
            unit="mm",
            details={
                "center_x": self._center.x,
                "center_y": self._center.y,
                "angles": self._angles,
            }
        )


class WinstonLutz:
    """
    Winston-Lutz test for isocenter accuracy.

    Uses ball bearing and multiple gantry/couch angles.
    """

    def __init__(
        self,
        ball_diameter: float = 5.0,
        test_angles: Optional[List[Tuple[float, float]]] = None
    ):
        self.ball_diameter = ball_diameter

        if test_angles is None:
            # Standard test angles (gantry, couch)
            self.test_angles = [
                (0, 0), (90, 0), (180, 0), (270, 0),
                (0, 45), (0, 90), (0, 270), (0, 315),
            ]
        else:
            self.test_angles = test_angles

        self._results: List[Dict] = []

    def analyze_images(
        self,
        images: List[np.ndarray]
    ) -> QATestResult:
        """
        Analyze Winston-Lutz images.

        Measures offset between radiation field center and ball center.
        """
        offsets = []

        for i, image in enumerate(images):
            # Simulate offset measurement
            offset_x = np.random.uniform(-0.5, 0.5)
            offset_y = np.random.uniform(-0.5, 0.5)
            offset = math.sqrt(offset_x ** 2 + offset_y ** 2)

            self._results.append({
                "gantry": self.test_angles[i][0],
                "couch": self.test_angles[i][1],
                "offset_x": offset_x,
                "offset_y": offset_y,
                "offset_total": offset,
            })
            offsets.append(offset)

        max_offset = max(offsets)
        mean_offset = np.mean(offsets)

        return QATestResult(
            test_name="Winston-Lutz",
            passed=max_offset < 1.0,
            measured_value=max_offset,
            tolerance=1.0,
            unit="mm",
            details={
                "mean_offset": mean_offset,
                "all_offsets": offsets,
                "individual_results": self._results,
            }
        )


class PicketFence:
    """
    Picket fence test for MLC position accuracy.

    Creates pattern of equally-spaced MLC strips.
    """

    def __init__(
        self,
        strip_width: float = 2.0,
        num_strips: int = 7,
        strip_spacing: float = 30.0
    ):
        self.strip_width = strip_width
        self.num_strips = num_strips
        self.strip_spacing = strip_spacing

        self._leaf_positions: List[float] = []
        self._errors: List[float] = []

    def generate_mlc_pattern(self) -> List[Tuple[float, float]]:
        """Generate MLC positions for picket fence."""
        pattern = []

        for i in range(self.num_strips):
            center = -self.strip_spacing * (self.num_strips - 1) / 2 + i * self.strip_spacing
            left = center - self.strip_width / 2
            right = center + self.strip_width / 2
            pattern.append((left, right))

        return pattern

    def analyze_image(
        self,
        image: np.ndarray,
        expected_spacing: float = None
    ) -> QATestResult:
        """
        Analyze picket fence image.

        Measures deviation of each strip from expected position.
        """
        if expected_spacing is None:
            expected_spacing = self.strip_spacing

        # Simulate analysis
        self._leaf_positions = []
        self._errors = []

        for i in range(self.num_strips):
            expected_pos = -expected_spacing * (self.num_strips - 1) / 2 + i * expected_spacing
            measured_pos = expected_pos + np.random.uniform(-0.3, 0.3)
            error = measured_pos - expected_pos

            self._leaf_positions.append(measured_pos)
            self._errors.append(error)

        max_error = max(abs(e) for e in self._errors)

        return QATestResult(
            test_name="Picket Fence",
            passed=max_error < 0.5,
            measured_value=max_error,
            tolerance=0.5,
            unit="mm",
            details={
                "errors": self._errors,
                "positions": self._leaf_positions,
                "mean_error": np.mean(self._errors),
                "std_error": np.std(self._errors),
            }
        )
