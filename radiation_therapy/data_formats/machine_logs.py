"""
Machine Log File Parsing and Analysis.

Implements parsers for various machine log formats:
- Varian Trajectory Logs
- Varian Dynalog files
- Elekta log files
- Generic log analysis tools
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any, BinaryIO
from datetime import datetime
import struct
import numpy as np
from abc import ABC, abstractmethod


class LogFileType(Enum):
    """Machine log file types."""
    TRAJECTORY_LOG = "trajectory_log"
    DYNALOG = "dynalog"
    ELEKTA_LOG = "elekta_log"
    ACCURAY_LOG = "accuray_log"


@dataclass
class LogRecord:
    """Single record from machine log."""
    timestamp: float  # seconds from start
    gantry_angle: float
    collimator_angle: float
    couch_angle: float
    couch_lat: float
    couch_lng: float
    couch_vrt: float
    mu_delivered: float
    beam_on: bool

    # MLC positions
    mlc_a: Optional[List[float]] = None
    mlc_b: Optional[List[float]] = None

    # Jaw positions
    jaw_x1: Optional[float] = None
    jaw_x2: Optional[float] = None
    jaw_y1: Optional[float] = None
    jaw_y2: Optional[float] = None

    # Dose rate
    dose_rate: Optional[float] = None


@dataclass
class LogSummary:
    """Summary statistics from log analysis."""
    total_time: float  # seconds
    beam_on_time: float
    total_mu: float
    num_records: int

    gantry_range: Tuple[float, float]
    couch_range: Tuple[float, float]

    mlc_errors: Dict[str, float] = field(default_factory=dict)
    max_mlc_error: float = 0.0
    mean_mlc_error: float = 0.0

    # Pass/fail based on tolerances
    passed: bool = True
    failures: List[str] = field(default_factory=list)


class MachineLogParser(ABC):
    """Abstract base class for machine log parsers."""

    def __init__(self):
        self._records: List[LogRecord] = []
        self._header: Dict[str, Any] = {}
        self._file_path: Optional[str] = None

    @abstractmethod
    def parse(self, file_path: str) -> bool:
        """Parse log file."""
        pass

    @abstractmethod
    def get_file_type(self) -> LogFileType:
        """Get log file type."""
        pass

    def get_records(self) -> List[LogRecord]:
        """Get parsed log records."""
        return self._records

    def get_header(self) -> Dict[str, Any]:
        """Get file header information."""
        return self._header

    def get_record_at_time(self, time_seconds: float) -> Optional[LogRecord]:
        """Get record closest to specified time."""
        if not self._records:
            return None

        closest = min(self._records, key=lambda r: abs(r.timestamp - time_seconds))
        return closest

    def get_mlc_positions_at_time(
        self,
        time_seconds: float
    ) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """Get MLC A and B positions at specified time."""
        record = self.get_record_at_time(time_seconds)
        if record:
            return record.mlc_a, record.mlc_b
        return None, None


class TrajectoryLog(MachineLogParser):
    """
    Varian Trajectory Log parser.

    Parses .bin trajectory log files from Varian TrueBeam/Halcyon.
    Records machine state at 20ms intervals.
    """

    # Trajectory log constants
    HEADER_SIZE = 1024
    SAMPLE_INTERVAL = 0.020  # 20 ms
    NUM_MLC_LEAVES = 60  # per bank (A and B)

    def __init__(self):
        super().__init__()
        self._expected_records: List[LogRecord] = []
        self._actual_records: List[LogRecord] = []

    def get_file_type(self) -> LogFileType:
        return LogFileType.TRAJECTORY_LOG

    def parse(self, file_path: str) -> bool:
        """
        Parse trajectory log file.

        Note: This is a simulation - actual parsing would require
        reading the binary file format.
        """
        self._file_path = file_path
        self._records = []

        # Simulated header
        self._header = {
            "signature": "VARIAN_TRAJECTORY_LOG",
            "version": "3.0",
            "header_size": self.HEADER_SIZE,
            "sampling_interval": self.SAMPLE_INTERVAL,
            "num_axes": 8,
            "num_mlc_leaves": self.NUM_MLC_LEAVES * 2,
            "treatment_machine": "TrueBeam",
            "patient_id": "SIMULATED",
            "plan_uid": "1.2.3.4.5",
        }

        # In real implementation, would read binary file here
        # For simulation, generate sample data
        self._generate_sample_data()

        return True

    def _generate_sample_data(self):
        """Generate sample log data for simulation."""
        num_samples = 500  # 10 seconds at 20ms intervals

        for i in range(num_samples):
            timestamp = i * self.SAMPLE_INTERVAL

            # Simulate arc delivery
            gantry_progress = i / num_samples
            gantry_angle = 180.0 + 180.0 * gantry_progress

            record = LogRecord(
                timestamp=timestamp,
                gantry_angle=gantry_angle,
                collimator_angle=0.0,
                couch_angle=0.0,
                couch_lat=0.0,
                couch_lng=100.0,
                couch_vrt=0.0,
                mu_delivered=200.0 * gantry_progress,
                beam_on=True,
                mlc_a=[-5.0 + np.random.normal(0, 0.1)] * self.NUM_MLC_LEAVES,
                mlc_b=[5.0 + np.random.normal(0, 0.1)] * self.NUM_MLC_LEAVES,
                jaw_x1=-50.0,
                jaw_x2=50.0,
                jaw_y1=-50.0,
                jaw_y2=50.0,
                dose_rate=600.0,
            )

            self._records.append(record)
            self._actual_records.append(record)

            # Create expected record (from plan)
            expected = LogRecord(
                timestamp=timestamp,
                gantry_angle=gantry_angle,
                collimator_angle=0.0,
                couch_angle=0.0,
                couch_lat=0.0,
                couch_lng=100.0,
                couch_vrt=0.0,
                mu_delivered=200.0 * gantry_progress,
                beam_on=True,
                mlc_a=[-5.0] * self.NUM_MLC_LEAVES,
                mlc_b=[5.0] * self.NUM_MLC_LEAVES,
                jaw_x1=-50.0,
                jaw_x2=50.0,
                jaw_y1=-50.0,
                jaw_y2=50.0,
            )
            self._expected_records.append(expected)

    def get_expected_records(self) -> List[LogRecord]:
        """Get expected (planned) records."""
        return self._expected_records

    def get_actual_records(self) -> List[LogRecord]:
        """Get actual (delivered) records."""
        return self._actual_records

    def calculate_mlc_errors(self) -> Dict[str, np.ndarray]:
        """Calculate MLC position errors (actual - expected)."""
        if not self._expected_records or not self._actual_records:
            return {}

        num_records = min(len(self._expected_records), len(self._actual_records))
        num_leaves = self.NUM_MLC_LEAVES

        errors_a = np.zeros((num_records, num_leaves))
        errors_b = np.zeros((num_records, num_leaves))

        for i in range(num_records):
            expected = self._expected_records[i]
            actual = self._actual_records[i]

            if expected.mlc_a and actual.mlc_a:
                for j in range(min(num_leaves, len(expected.mlc_a), len(actual.mlc_a))):
                    errors_a[i, j] = actual.mlc_a[j] - expected.mlc_a[j]

            if expected.mlc_b and actual.mlc_b:
                for j in range(min(num_leaves, len(expected.mlc_b), len(actual.mlc_b))):
                    errors_b[i, j] = actual.mlc_b[j] - expected.mlc_b[j]

        return {
            "bank_a": errors_a,
            "bank_b": errors_b,
            "combined": np.concatenate([errors_a, errors_b], axis=1),
        }

    def get_rms_error(self) -> float:
        """Calculate RMS MLC error across all leaves and snapshots."""
        errors = self.calculate_mlc_errors()
        if "combined" not in errors:
            return 0.0
        return float(np.sqrt(np.mean(errors["combined"] ** 2)))

    def get_max_error(self) -> float:
        """Get maximum absolute MLC error."""
        errors = self.calculate_mlc_errors()
        if "combined" not in errors:
            return 0.0
        return float(np.max(np.abs(errors["combined"])))


class DynaLog(MachineLogParser):
    """
    Varian Dynalog file parser.

    Parses .dlg files from older Varian machines.
    Contains MLC positions at 50ms intervals.
    """

    SAMPLE_INTERVAL = 0.050  # 50 ms

    def __init__(self):
        super().__init__()
        self._bank_a_file: Optional[str] = None
        self._bank_b_file: Optional[str] = None

    def get_file_type(self) -> LogFileType:
        return LogFileType.DYNALOG

    def set_file_pair(self, bank_a_file: str, bank_b_file: str):
        """Set both A and B bank log files."""
        self._bank_a_file = bank_a_file
        self._bank_b_file = bank_b_file

    def parse(self, file_path: str) -> bool:
        """
        Parse dynalog file.

        Note: Actual implementation would parse ASCII format.
        """
        self._file_path = file_path
        self._records = []

        # Simulated header
        self._header = {
            "version": "B",
            "patient_name": "SIMULATED",
            "plan_name": "Test Plan",
            "beam_number": 1,
            "tolerance": 0.5,  # mm
            "num_leaves": 60,
        }

        # Generate sample data
        self._generate_sample_dynalog()

        return True

    def _generate_sample_dynalog(self):
        """Generate sample dynalog data."""
        num_samples = 200

        for i in range(num_samples):
            timestamp = i * self.SAMPLE_INTERVAL

            record = LogRecord(
                timestamp=timestamp,
                gantry_angle=0.0,
                collimator_angle=0.0,
                couch_angle=0.0,
                couch_lat=0.0,
                couch_lng=0.0,
                couch_vrt=0.0,
                mu_delivered=i * 0.5,
                beam_on=True,
                mlc_a=[-10.0 + i * 0.1] * 60,
                mlc_b=[10.0 - i * 0.1] * 60,
            )

            self._records.append(record)

    def get_leaf_travel(self, leaf_index: int, bank: str = "A") -> float:
        """Calculate total travel for specified leaf."""
        if not self._records:
            return 0.0

        total_travel = 0.0
        for i in range(1, len(self._records)):
            prev = self._records[i - 1]
            curr = self._records[i]

            if bank == "A" and prev.mlc_a and curr.mlc_a:
                total_travel += abs(curr.mlc_a[leaf_index] - prev.mlc_a[leaf_index])
            elif bank == "B" and prev.mlc_b and curr.mlc_b:
                total_travel += abs(curr.mlc_b[leaf_index] - prev.mlc_b[leaf_index])

        return total_travel


class LogAnalyzer:
    """
    Analyzes machine log files for QA purposes.

    Calculates statistics and compares to tolerances.
    """

    # Default tolerances
    DEFAULT_TOLERANCES = {
        "mlc_position": 1.0,  # mm
        "gantry_angle": 1.0,  # degrees
        "collimator_angle": 1.0,  # degrees
        "couch_angle": 1.0,  # degrees
        "jaw_position": 1.0,  # mm
        "mu_deviation": 2.0,  # percent
    }

    def __init__(self, tolerances: Optional[Dict[str, float]] = None):
        self._tolerances = tolerances or self.DEFAULT_TOLERANCES.copy()
        self._parser: Optional[MachineLogParser] = None

    def set_tolerances(self, tolerances: Dict[str, float]):
        """Set analysis tolerances."""
        self._tolerances.update(tolerances)

    def load_log(self, file_path: str, log_type: LogFileType) -> bool:
        """Load and parse log file."""
        if log_type == LogFileType.TRAJECTORY_LOG:
            self._parser = TrajectoryLog()
        elif log_type == LogFileType.DYNALOG:
            self._parser = DynaLog()
        else:
            return False

        return self._parser.parse(file_path)

    def analyze(self) -> LogSummary:
        """Perform complete analysis of loaded log."""
        if not self._parser or not self._parser.get_records():
            return LogSummary(
                total_time=0,
                beam_on_time=0,
                total_mu=0,
                num_records=0,
                gantry_range=(0, 0),
                couch_range=(0, 0),
                passed=False,
                failures=["No log data loaded"]
            )

        records = self._parser.get_records()

        # Calculate basic statistics
        total_time = records[-1].timestamp if records else 0
        beam_on_time = sum(
            records[i].timestamp - records[i-1].timestamp
            for i in range(1, len(records))
            if records[i].beam_on
        )
        total_mu = max(r.mu_delivered for r in records) if records else 0

        gantry_angles = [r.gantry_angle for r in records]
        couch_angles = [r.couch_angle for r in records]

        gantry_range = (min(gantry_angles), max(gantry_angles))
        couch_range = (min(couch_angles), max(couch_angles))

        # MLC analysis
        mlc_errors = {}
        max_mlc_error = 0.0
        mean_mlc_error = 0.0

        if isinstance(self._parser, TrajectoryLog):
            errors = self._parser.calculate_mlc_errors()
            if "combined" in errors:
                max_mlc_error = float(np.max(np.abs(errors["combined"])))
                mean_mlc_error = float(np.mean(np.abs(errors["combined"])))
                mlc_errors = {
                    "rms": self._parser.get_rms_error(),
                    "max": max_mlc_error,
                    "mean": mean_mlc_error,
                }

        # Check tolerances
        failures = []
        passed = True

        if max_mlc_error > self._tolerances.get("mlc_position", 1.0):
            failures.append(
                f"MLC error exceeds tolerance: {max_mlc_error:.2f}mm > "
                f"{self._tolerances['mlc_position']}mm"
            )
            passed = False

        return LogSummary(
            total_time=total_time,
            beam_on_time=beam_on_time,
            total_mu=total_mu,
            num_records=len(records),
            gantry_range=gantry_range,
            couch_range=couch_range,
            mlc_errors=mlc_errors,
            max_mlc_error=max_mlc_error,
            mean_mlc_error=mean_mlc_error,
            passed=passed,
            failures=failures,
        )

    def generate_report(self) -> Dict[str, Any]:
        """Generate analysis report."""
        summary = self.analyze()

        return {
            "file": self._parser._file_path if self._parser else None,
            "file_type": self._parser.get_file_type().value if self._parser else None,
            "header": self._parser.get_header() if self._parser else {},
            "summary": {
                "total_time_seconds": summary.total_time,
                "beam_on_time_seconds": summary.beam_on_time,
                "total_mu": summary.total_mu,
                "num_records": summary.num_records,
                "gantry_range": summary.gantry_range,
                "couch_range": summary.couch_range,
            },
            "mlc_analysis": summary.mlc_errors,
            "tolerances": self._tolerances,
            "passed": summary.passed,
            "failures": summary.failures,
        }

    def compare_to_plan(
        self,
        expected_positions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compare log to planned positions."""
        if not self._parser:
            return {}

        records = self._parser.get_records()
        if not records or not expected_positions:
            return {}

        # Calculate deviations
        gantry_deviations = []
        mlc_deviations = []

        for i, (record, expected) in enumerate(zip(records, expected_positions)):
            if "gantry_angle" in expected:
                gantry_deviations.append(
                    abs(record.gantry_angle - expected["gantry_angle"])
                )

            if "mlc_a" in expected and record.mlc_a:
                for actual, exp in zip(record.mlc_a, expected["mlc_a"]):
                    mlc_deviations.append(abs(actual - exp))

            if "mlc_b" in expected and record.mlc_b:
                for actual, exp in zip(record.mlc_b, expected["mlc_b"]):
                    mlc_deviations.append(abs(actual - exp))

        return {
            "gantry_max_deviation": max(gantry_deviations) if gantry_deviations else 0,
            "gantry_mean_deviation": np.mean(gantry_deviations) if gantry_deviations else 0,
            "mlc_max_deviation": max(mlc_deviations) if mlc_deviations else 0,
            "mlc_mean_deviation": np.mean(mlc_deviations) if mlc_deviations else 0,
            "mlc_rms_deviation": np.sqrt(np.mean(np.array(mlc_deviations)**2)) if mlc_deviations else 0,
        }


class GammaAnalysis:
    """
    Gamma analysis for comparing dose distributions.

    Used for patient-specific QA comparing measured
    vs calculated dose distributions.
    """

    def __init__(
        self,
        dose_tolerance: float = 3.0,  # percent
        distance_tolerance: float = 3.0,  # mm
        threshold: float = 10.0,  # percent of max for analysis
        local_global: str = "global"  # "local" or "global"
    ):
        self.dose_tolerance = dose_tolerance
        self.distance_tolerance = distance_tolerance
        self.threshold = threshold
        self.local_global = local_global

    def calculate_gamma(
        self,
        reference: np.ndarray,
        evaluated: np.ndarray,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0)
    ) -> Tuple[np.ndarray, float]:
        """
        Calculate gamma map and passing rate.

        Returns (gamma_map, passing_rate).
        """
        if reference.shape != evaluated.shape:
            raise ValueError("Arrays must have same shape")

        # Threshold mask
        max_dose = np.max(reference)
        threshold_value = max_dose * self.threshold / 100
        mask = reference >= threshold_value

        # Dose normalization
        if self.local_global == "global":
            dose_norm = max_dose
        else:
            dose_norm = reference.copy()
            dose_norm[dose_norm < threshold_value] = threshold_value

        # Calculate gamma
        gamma_map = np.full(reference.shape, np.inf)

        # Simplified gamma calculation
        # Full implementation would use efficient search algorithm
        rows, cols = reference.shape

        for i in range(rows):
            for j in range(cols):
                if not mask[i, j]:
                    gamma_map[i, j] = 0
                    continue

                ref_dose = reference[i, j]
                eval_dose = evaluated[i, j]

                # Search radius based on distance tolerance
                search_radius = int(
                    self.distance_tolerance / min(pixel_spacing) * 2
                )

                min_gamma = np.inf

                for di in range(-search_radius, search_radius + 1):
                    for dj in range(-search_radius, search_radius + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            distance = np.sqrt(
                                (di * pixel_spacing[0]) ** 2 +
                                (dj * pixel_spacing[1]) ** 2
                            )

                            if distance <= self.distance_tolerance * 2:
                                local_norm = (
                                    dose_norm if isinstance(dose_norm, float)
                                    else dose_norm[ni, nj]
                                )

                                dose_diff = abs(eval_dose - reference[ni, nj])
                                dose_term = (
                                    dose_diff /
                                    (local_norm * self.dose_tolerance / 100)
                                ) ** 2
                                dist_term = (
                                    distance / self.distance_tolerance
                                ) ** 2

                                gamma = np.sqrt(dose_term + dist_term)
                                min_gamma = min(min_gamma, gamma)

                gamma_map[i, j] = min_gamma

        # Calculate passing rate
        valid_points = mask.sum()
        passing_points = np.sum((gamma_map <= 1.0) & mask)
        passing_rate = (passing_points / valid_points * 100) if valid_points > 0 else 0

        return gamma_map, float(passing_rate)

    def analyze(
        self,
        reference: np.ndarray,
        evaluated: np.ndarray,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0)
    ) -> Dict[str, Any]:
        """Perform complete gamma analysis."""
        gamma_map, passing_rate = self.calculate_gamma(
            reference, evaluated, pixel_spacing
        )

        # Statistics on gamma values
        valid_gamma = gamma_map[gamma_map < np.inf]

        return {
            "passing_rate": passing_rate,
            "mean_gamma": float(np.mean(valid_gamma)) if len(valid_gamma) > 0 else 0,
            "max_gamma": float(np.max(valid_gamma)) if len(valid_gamma) > 0 else 0,
            "min_gamma": float(np.min(valid_gamma)) if len(valid_gamma) > 0 else 0,
            "std_gamma": float(np.std(valid_gamma)) if len(valid_gamma) > 0 else 0,
            "criteria": f"{self.dose_tolerance}%/{self.distance_tolerance}mm",
            "threshold": self.threshold,
            "normalization": self.local_global,
            "gamma_map": gamma_map,
        }
