"""
Spatially Fractionated Radiotherapy.

Implements spatial dose modulation techniques:
- GRID therapy (high-dose regions interspersed with low-dose)
- LATTICE therapy (3D version of GRID)
- Minibeam therapy (submillimeter beams)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any
import numpy as np

from ..base import Position3D


class SpatialPattern(Enum):
    """Spatial dose patterns."""
    GRID_2D = "grid_2d"
    LATTICE_3D = "lattice_3d"
    MINIBEAM_PLANAR = "minibeam_planar"
    MINIBEAM_CROSS = "minibeam_cross"


@dataclass
class GridParameters:
    """Parameters for GRID/LATTICE therapy."""
    pattern: SpatialPattern
    peak_spacing: float  # cm (center-to-center)
    peak_width: float  # cm
    valley_width: float  # cm
    peak_dose: float  # Gy
    valley_dose: float  # Gy

    @property
    def peak_to_valley_ratio(self) -> float:
        """Calculate peak-to-valley dose ratio (PVDR)."""
        if self.valley_dose > 0:
            return self.peak_dose / self.valley_dose
        return float('inf')

    @property
    def geometric_ratio(self) -> float:
        """Calculate geometric peak/valley ratio."""
        return self.peak_width / (self.peak_width + self.valley_width)


@dataclass
class MinibeamParameters:
    """Parameters for minibeam therapy."""
    beam_width: float  # mm (typically 0.3-0.7mm)
    beam_spacing: float  # mm (center-to-center)
    num_beams: int
    beam_energy: float  # MeV
    modality: str  # "proton", "photon", "electron"

    @property
    def pvdr_at_surface(self) -> float:
        """Estimate PVDR at surface."""
        # Simplified - actual depends on beam profile
        return 10.0 * (self.beam_spacing / self.beam_width)


class GRIDTherapy:
    """
    GRID radiotherapy implementation.

    Uses spatially fractionated dose delivery with
    high-dose peaks and low-dose valleys.
    Primarily used for large tumors.
    """

    def __init__(
        self,
        pattern: SpatialPattern = SpatialPattern.GRID_2D
    ):
        self.pattern = pattern

        # Default GRID parameters
        self._peak_spacing = 1.5  # cm
        self._peak_diameter = 1.0  # cm
        self._block_material = "cerrobend"
        self._block_thickness = 7.5  # cm

        # Collimator/block specifications
        self._grid_openings: List[Tuple[float, float]] = []
        self._field_size = (20.0, 20.0)  # cm

    def set_grid_parameters(
        self,
        peak_spacing: float,
        peak_diameter: float
    ):
        """Set GRID geometry parameters."""
        self._peak_spacing = peak_spacing
        self._peak_diameter = peak_diameter

    def generate_grid_openings(
        self,
        field_size: Tuple[float, float]
    ) -> List[Tuple[float, float]]:
        """Generate positions of GRID openings."""
        self._field_size = field_size
        self._grid_openings = []

        # Generate hexagonal pattern
        num_x = int(field_size[0] / self._peak_spacing) + 1
        num_y = int(field_size[1] / (self._peak_spacing * 0.866)) + 1

        for i in range(num_x):
            for j in range(num_y):
                x = -field_size[0] / 2 + i * self._peak_spacing
                y = -field_size[1] / 2 + j * self._peak_spacing * 0.866

                # Offset every other row
                if j % 2 == 1:
                    x += self._peak_spacing / 2

                # Check within field
                if abs(x) <= field_size[0] / 2 and abs(y) <= field_size[1] / 2:
                    self._grid_openings.append((x, y))

        return self._grid_openings

    def calculate_transmission(
        self,
        x: float,
        y: float
    ) -> float:
        """Calculate transmission factor at point."""
        # Check if under any opening
        for ox, oy in self._grid_openings:
            distance = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
            if distance <= self._peak_diameter / 2:
                return 1.0  # Full transmission through opening

        # Under blocked region
        return 0.05  # Small transmission through block

    def calculate_dose_distribution(
        self,
        open_field_dose: np.ndarray,
        pixel_size: float = 1.0  # mm
    ) -> np.ndarray:
        """Apply GRID pattern to open field dose."""
        rows, cols = open_field_dose.shape
        grid_dose = np.zeros_like(open_field_dose)

        for i in range(rows):
            for j in range(cols):
                x = (j - cols / 2) * pixel_size / 10  # cm
                y = (i - rows / 2) * pixel_size / 10  # cm

                transmission = self.calculate_transmission(x, y)
                grid_dose[i, j] = open_field_dose[i, j] * transmission

        return grid_dose

    def analyze_dose_distribution(
        self,
        dose_grid: np.ndarray
    ) -> Dict[str, float]:
        """Analyze GRID dose distribution."""
        peak_dose = np.max(dose_grid)
        valley_dose = np.percentile(dose_grid[dose_grid > 0], 10)

        return {
            "peak_dose": float(peak_dose),
            "valley_dose": float(valley_dose),
            "pvdr": float(peak_dose / valley_dose) if valley_dose > 0 else float('inf'),
            "mean_dose": float(np.mean(dose_grid)),
            "coverage_fraction": float(np.mean(dose_grid > peak_dose * 0.5)),
        }


class LATTICETherapy(GRIDTherapy):
    """
    LATTICE (3D-GRID) radiotherapy.

    Extension of GRID to 3D with high-dose vertices
    in a lattice pattern throughout the tumor volume.
    """

    def __init__(self):
        super().__init__(SpatialPattern.LATTICE_3D)

        # LATTICE-specific parameters
        self._vertex_spacing = 1.5  # cm
        self._vertex_diameter = 1.0  # cm
        self._num_vertices: int = 0

        # Vertex positions (3D)
        self._vertices: List[Position3D] = []

    def generate_lattice_vertices(
        self,
        target_dimensions: Tuple[float, float, float],
        target_center: Position3D
    ) -> List[Position3D]:
        """Generate 3D lattice vertex positions."""
        self._vertices = []

        dx, dy, dz = target_dimensions
        spacing = self._vertex_spacing

        # Number of vertices in each dimension
        nx = int(dx / spacing) + 1
        ny = int(dy / spacing) + 1
        nz = int(dz / spacing) + 1

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x = target_center.x - dx / 2 + i * spacing
                    y = target_center.y - dy / 2 + j * spacing
                    z = target_center.z - dz / 2 + k * spacing

                    # Offset for body-centered cubic pattern
                    if (i + j + k) % 2 == 1:
                        x += spacing / 2
                        y += spacing / 2
                        z += spacing / 2

                    # Check within target
                    if (abs(x - target_center.x) <= dx / 2 and
                        abs(y - target_center.y) <= dy / 2 and
                        abs(z - target_center.z) <= dz / 2):
                        self._vertices.append(Position3D(x, y, z))

        self._num_vertices = len(self._vertices)
        return self._vertices

    def create_vertex_plan(
        self,
        vertex_dose: float,
        background_dose: float
    ) -> List[Dict]:
        """Create treatment plan for LATTICE vertices."""
        plan_vertices = []

        for vertex in self._vertices:
            plan_vertices.append({
                "position": vertex,
                "dose": vertex_dose,
                "diameter": self._vertex_diameter,
            })

        return plan_vertices

    def calculate_dvh_metrics(
        self,
        vertex_dose: float,
        background_dose: float,
        target_volume: float  # cc
    ) -> Dict[str, float]:
        """Calculate DVH metrics for LATTICE plan."""
        # Volume of vertices
        vertex_volume = (
            self._num_vertices *
            (4 / 3 * np.pi * (self._vertex_diameter / 2) ** 3)
        )

        # Volume fractions
        vertex_fraction = min(1.0, vertex_volume / target_volume)
        background_fraction = 1.0 - vertex_fraction

        # DVH metrics
        return {
            "num_vertices": self._num_vertices,
            "vertex_volume_cc": vertex_volume,
            "vertex_fraction": vertex_fraction,
            "d_mean": vertex_fraction * vertex_dose + background_fraction * background_dose,
            "d_max": vertex_dose,
            "d_min": background_dose,
            "pvdr": vertex_dose / background_dose if background_dose > 0 else float('inf'),
        }


class MinibeamTherapy:
    """
    Minibeam radiotherapy implementation.

    Uses submillimeter beams (0.3-0.7mm) with
    high peak-to-valley dose ratios for
    improved normal tissue sparing.
    """

    def __init__(
        self,
        modality: str = "proton"
    ):
        self.modality = modality

        # Default minibeam parameters
        self._beam_width = 0.5  # mm
        self._beam_spacing = 2.0  # mm (center-to-center)
        self._collimator_type = "multislit"

        # Beam divergence
        self._divergence = 0.1  # degrees

        # Arrays
        self._beam_positions: List[float] = []

    def set_parameters(
        self,
        beam_width: float,
        beam_spacing: float
    ):
        """Set minibeam parameters."""
        self._beam_width = beam_width
        self._beam_spacing = beam_spacing

    def generate_beam_array(
        self,
        field_width: float  # mm
    ) -> List[float]:
        """Generate minibeam positions."""
        self._beam_positions = []

        num_beams = int(field_width / self._beam_spacing) + 1
        start_pos = -field_width / 2

        for i in range(num_beams):
            pos = start_pos + i * self._beam_spacing
            if abs(pos) <= field_width / 2:
                self._beam_positions.append(pos)

        return self._beam_positions

    def calculate_lateral_profile(
        self,
        depth: float,  # mm
        positions: np.ndarray  # mm
    ) -> np.ndarray:
        """Calculate lateral dose profile at depth."""
        profile = np.zeros_like(positions, dtype=float)

        # Beam width increases with depth due to scattering
        scatter_factor = 1.0 + depth * 0.001  # Simplified
        effective_width = self._beam_width * scatter_factor

        for beam_pos in self._beam_positions:
            # Gaussian beam profile
            contribution = np.exp(
                -((positions - beam_pos) ** 2) /
                (2 * (effective_width / 2.355) ** 2)
            )
            profile += contribution

        # Normalize
        profile = profile / np.max(profile) * 100

        return profile

    def calculate_pvdr_vs_depth(
        self,
        depths: np.ndarray
    ) -> np.ndarray:
        """Calculate PVDR as function of depth."""
        pvdr = np.zeros_like(depths, dtype=float)

        for i, depth in enumerate(depths):
            # Generate profile at this depth
            positions = np.linspace(-10, 10, 201)
            profile = self.calculate_lateral_profile(depth, positions)

            # Find peaks and valleys
            peak = np.max(profile)
            valley = np.min(profile[profile > 0])

            pvdr[i] = peak / valley if valley > 0 else peak

        return pvdr

    def estimate_tissue_sparing(
        self,
        pvdr: float,
        tissue_type: str = "normal"
    ) -> float:
        """
        Estimate tissue sparing factor from PVDR.

        Higher PVDR = better normal tissue sparing.
        """
        # Simplified model based on literature
        if tissue_type == "tumor":
            # Tumor receives effective dose from both peaks and valleys
            return 1.0
        else:
            # Normal tissue spared due to valley recovery
            if pvdr > 10:
                return 0.5  # 50% sparing
            elif pvdr > 5:
                return 0.7
            elif pvdr > 2:
                return 0.9
            else:
                return 1.0  # No sparing


class SpatiallyFractionatedPlan:
    """
    Treatment planning for spatially fractionated therapy.

    Combines spatial modulation with conventional
    dose optimization.
    """

    def __init__(
        self,
        technique: str = "GRID"  # "GRID", "LATTICE", "MINIBEAM"
    ):
        self.technique = technique

        self._target_volume: float = 0.0
        self._prescribed_peak_dose: float = 0.0
        self._prescribed_valley_dose: float = 0.0

        # Planning constraints
        self._min_pvdr: float = 5.0
        self._max_valley_dose: float = 10.0  # Gy

    def set_prescription(
        self,
        peak_dose: float,
        valley_dose: float,
        target_volume: float
    ):
        """Set dose prescription."""
        self._prescribed_peak_dose = peak_dose
        self._prescribed_valley_dose = valley_dose
        self._target_volume = target_volume

    def optimize_pattern(
        self,
        target_dimensions: Tuple[float, float, float]
    ) -> GridParameters:
        """Optimize spatial pattern for target."""
        # Determine optimal spacing based on target size
        min_dimension = min(target_dimensions)

        # Smaller targets need finer spacing
        if min_dimension < 3.0:
            peak_spacing = 1.0
            peak_width = 0.5
        elif min_dimension < 6.0:
            peak_spacing = 1.5
            peak_width = 0.8
        else:
            peak_spacing = 2.0
            peak_width = 1.0

        valley_width = peak_spacing - peak_width

        pattern = SpatialPattern.LATTICE_3D if self.technique == "LATTICE" else SpatialPattern.GRID_2D

        return GridParameters(
            pattern=pattern,
            peak_spacing=peak_spacing,
            peak_width=peak_width,
            valley_width=valley_width,
            peak_dose=self._prescribed_peak_dose,
            valley_dose=self._prescribed_valley_dose,
        )

    def calculate_biological_effective_dose(
        self,
        params: GridParameters,
        alpha_beta: float = 10.0
    ) -> Dict[str, float]:
        """Calculate biologically effective doses."""
        # BED for peak regions
        bed_peak = params.peak_dose * (1 + params.peak_dose / alpha_beta)

        # BED for valley regions
        bed_valley = params.valley_dose * (1 + params.valley_dose / alpha_beta)

        # Volume-weighted average
        peak_fraction = params.geometric_ratio
        valley_fraction = 1 - peak_fraction

        bed_average = peak_fraction * bed_peak + valley_fraction * bed_valley

        return {
            "bed_peak": bed_peak,
            "bed_valley": bed_valley,
            "bed_average": bed_average,
            "eqd2_peak": bed_peak / (1 + 2 / alpha_beta),
            "eqd2_valley": bed_valley / (1 + 2 / alpha_beta),
            "eqd2_average": bed_average / (1 + 2 / alpha_beta),
        }

    def evaluate_plan(
        self,
        params: GridParameters
    ) -> Dict[str, Any]:
        """Evaluate plan quality."""
        issues = []
        passed = True

        # Check PVDR
        if params.peak_to_valley_ratio < self._min_pvdr:
            issues.append(
                f"PVDR {params.peak_to_valley_ratio:.1f} < minimum {self._min_pvdr}"
            )
            passed = False

        # Check valley dose
        if params.valley_dose > self._max_valley_dose:
            issues.append(
                f"Valley dose {params.valley_dose:.1f} Gy > maximum {self._max_valley_dose} Gy"
            )
            passed = False

        return {
            "passed": passed,
            "issues": issues,
            "pvdr": params.peak_to_valley_ratio,
            "geometric_ratio": params.geometric_ratio,
            "peak_dose": params.peak_dose,
            "valley_dose": params.valley_dose,
        }
