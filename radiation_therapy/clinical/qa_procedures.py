"""
Quality Assurance Procedures.

Implements QA protocols:
- Daily, monthly, annual machine QA
- Patient-specific QA (IMRT/VMAT verification)
- Commissioning procedures
- Output calibration (TG-51, TRS-398)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Dict
from datetime import datetime, date
import math
import numpy as np


class QAFrequency(Enum):
    """QA frequency categories."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ANNUAL = "annual"
    COMMISSIONING = "commissioning"


class QAStatus(Enum):
    """QA test status."""
    PASS = "pass"
    FAIL = "fail"
    CONDITIONAL = "conditional"
    NOT_TESTED = "not_tested"


@dataclass
class QAResult:
    """Result of a QA test."""
    test_name: str
    measured_value: float
    expected_value: float
    tolerance: float
    unit: str
    status: QAStatus
    timestamp: datetime = field(default_factory=datetime.now)
    notes: str = ""

    @property
    def deviation(self) -> float:
        """Calculate deviation from expected."""
        if self.expected_value == 0:
            return 0
        return (self.measured_value - self.expected_value) / self.expected_value * 100


class DailyQA:
    """
    Daily machine QA procedures.

    Per TG-142 recommendations for daily checks.
    """

    # TG-142 daily tolerances
    TOLERANCES = {
        "output": 3.0,  # % from baseline
        "laser_alignment": 2.0,  # mm
        "distance_indicator": 2.0,  # mm
        "door_interlock": None,  # Functional
        "audiovisual_monitor": None,  # Functional
    }

    def __init__(self, machine_id: str):
        self.machine_id = machine_id
        self._results: List[QAResult] = []
        self._baseline_output = 1.0

    def set_baseline(self, output: float):
        """Set baseline output for comparison."""
        self._baseline_output = output

    def check_output(self, measured_output: float) -> QAResult:
        """Check output constancy."""
        deviation = abs(measured_output - self._baseline_output) / self._baseline_output * 100
        status = QAStatus.PASS if deviation <= self.TOLERANCES["output"] else QAStatus.FAIL

        result = QAResult(
            test_name="Output Constancy",
            measured_value=measured_output,
            expected_value=self._baseline_output,
            tolerance=self.TOLERANCES["output"],
            unit="cGy/MU",
            status=status
        )
        self._results.append(result)
        return result

    def check_lasers(
        self,
        lateral: float,
        longitudinal: float,
        sagittal: float
    ) -> List[QAResult]:
        """Check laser alignment."""
        results = []
        tolerance = self.TOLERANCES["laser_alignment"]

        for name, offset in [("Lateral", lateral), ("Longitudinal", longitudinal), ("Sagittal", sagittal)]:
            status = QAStatus.PASS if abs(offset) <= tolerance else QAStatus.FAIL
            result = QAResult(
                test_name=f"{name} Laser",
                measured_value=offset,
                expected_value=0.0,
                tolerance=tolerance,
                unit="mm",
                status=status
            )
            results.append(result)
            self._results.append(result)

        return results

    def check_odi(self, at_100cm: float, at_ssd: float) -> QAResult:
        """Check optical distance indicator."""
        error = abs(at_100cm - 100.0)
        tolerance = self.TOLERANCES["distance_indicator"]
        status = QAStatus.PASS if error <= tolerance else QAStatus.FAIL

        result = QAResult(
            test_name="ODI at 100cm",
            measured_value=at_100cm,
            expected_value=100.0,
            tolerance=tolerance,
            unit="cm",
            status=status
        )
        self._results.append(result)
        return result

    def check_door_interlock(self, door_open_beam_off: bool) -> QAResult:
        """Check door interlock function."""
        status = QAStatus.PASS if door_open_beam_off else QAStatus.FAIL

        result = QAResult(
            test_name="Door Interlock",
            measured_value=1.0 if door_open_beam_off else 0.0,
            expected_value=1.0,
            tolerance=0.0,
            unit="functional",
            status=status
        )
        self._results.append(result)
        return result

    def run_complete_daily(self) -> Tuple[bool, List[QAResult]]:
        """
        Run complete daily QA checklist.

        Returns:
            Tuple of (all_passed, results)
        """
        # Simulate measurements
        self.check_output(np.random.normal(self._baseline_output, 0.01))
        self.check_lasers(
            np.random.normal(0, 0.5),
            np.random.normal(0, 0.5),
            np.random.normal(0, 0.5)
        )
        self.check_odi(np.random.normal(100, 0.5), 100.0)
        self.check_door_interlock(True)

        all_passed = all(r.status == QAStatus.PASS for r in self._results)
        return all_passed, self._results


class MonthlyQA:
    """
    Monthly machine QA procedures.

    Per TG-142 recommendations for monthly checks.
    """

    TOLERANCES = {
        "output": 2.0,  # %
        "energy": 2.0,  # %
        "flatness": 2.0,  # %
        "symmetry": 2.0,  # %
        "light_rad_coincidence": 2.0,  # mm
        "gantry_rotation": 1.0,  # degree
        "collimator_rotation": 1.0,  # degree
        "couch_position": 2.0,  # mm
        "mlc_position": 1.0,  # mm
    }

    def __init__(self, machine_id: str):
        self.machine_id = machine_id
        self._results: List[QAResult] = []

    def check_output_calibration(
        self,
        measured: float,
        expected: float
    ) -> QAResult:
        """Check output calibration."""
        deviation = abs(measured - expected) / expected * 100
        status = QAStatus.PASS if deviation <= self.TOLERANCES["output"] else QAStatus.FAIL

        result = QAResult(
            test_name="Output Calibration",
            measured_value=measured,
            expected_value=expected,
            tolerance=self.TOLERANCES["output"],
            unit="cGy/MU",
            status=status
        )
        self._results.append(result)
        return result

    def check_beam_flatness(
        self,
        measured_flatness: float
    ) -> QAResult:
        """Check beam flatness."""
        status = QAStatus.PASS if measured_flatness <= 3.0 else QAStatus.FAIL

        result = QAResult(
            test_name="Beam Flatness",
            measured_value=measured_flatness,
            expected_value=0.0,
            tolerance=3.0,
            unit="%",
            status=status
        )
        self._results.append(result)
        return result

    def check_beam_symmetry(
        self,
        inplane: float,
        crossplane: float
    ) -> List[QAResult]:
        """Check beam symmetry."""
        results = []
        tolerance = self.TOLERANCES["symmetry"]

        for name, value in [("In-plane", inplane), ("Cross-plane", crossplane)]:
            status = QAStatus.PASS if abs(value) <= tolerance else QAStatus.FAIL
            result = QAResult(
                test_name=f"{name} Symmetry",
                measured_value=value,
                expected_value=0.0,
                tolerance=tolerance,
                unit="%",
                status=status
            )
            results.append(result)
            self._results.append(result)

        return results

    def check_light_radiation_coincidence(
        self,
        difference: float
    ) -> QAResult:
        """Check light field to radiation field coincidence."""
        tolerance = self.TOLERANCES["light_rad_coincidence"]
        status = QAStatus.PASS if abs(difference) <= tolerance else QAStatus.FAIL

        result = QAResult(
            test_name="Light/Rad Coincidence",
            measured_value=difference,
            expected_value=0.0,
            tolerance=tolerance,
            unit="mm",
            status=status
        )
        self._results.append(result)
        return result

    def check_gantry_rotation(
        self,
        isocentricity: float
    ) -> QAResult:
        """Check gantry rotation isocentricity."""
        tolerance = self.TOLERANCES["gantry_rotation"]
        status = QAStatus.PASS if isocentricity <= tolerance else QAStatus.FAIL

        result = QAResult(
            test_name="Gantry Isocentricity",
            measured_value=isocentricity,
            expected_value=0.0,
            tolerance=tolerance,
            unit="mm",
            status=status
        )
        self._results.append(result)
        return result


class AnnualQA:
    """
    Annual machine QA and recommissioning.

    Comprehensive testing per TG-142.
    """

    def __init__(self, machine_id: str):
        self.machine_id = machine_id
        self._results: Dict[str, List[QAResult]] = {}

    def perform_output_calibration(
        self,
        chamber,
        phantom
    ) -> Dict[str, QAResult]:
        """Perform absolute output calibration."""
        # TG-51 calibration
        results = {}

        # Simulate calibration
        for energy in [6.0, 10.0, 15.0]:
            result = QAResult(
                test_name=f"Output {energy}MV",
                measured_value=1.001,
                expected_value=1.000,
                tolerance=1.0,
                unit="cGy/MU",
                status=QAStatus.PASS
            )
            results[f"{energy}MV"] = result

        self._results["output_calibration"] = list(results.values())
        return results

    def perform_pdd_verification(self, energies: List[float]) -> Dict:
        """Verify percentage depth dose for each energy."""
        results = {}

        for energy in energies:
            # Check dmax and d10 values
            results[energy] = {
                "dmax": np.random.uniform(14, 16),  # mm
                "d10": np.random.uniform(60, 75),  # %
            }

        return results


class PatientSpecificQA:
    """
    Patient-specific QA for IMRT/VMAT plans.

    Verifies deliverability of complex treatment plans.
    """

    def __init__(self, plan_id: str):
        self.plan_id = plan_id
        self._measurement_device = None
        self._gamma_criteria = (3.0, 3.0)  # (%, mm)
        self._passing_threshold = 95.0  # %

    def set_measurement_device(self, device):
        """Set measurement device (array, film, etc.)."""
        self._measurement_device = device

    def set_gamma_criteria(
        self,
        dose_diff: float,
        dta: float,
        threshold: float = 95.0
    ):
        """Set gamma analysis criteria."""
        self._gamma_criteria = (dose_diff, dta)
        self._passing_threshold = threshold

    def perform_measurement(self) -> np.ndarray:
        """Perform QA measurement."""
        if not self._measurement_device:
            return None

        return self._measurement_device.acquire()

    def compare_to_plan(
        self,
        measured: np.ndarray,
        calculated: np.ndarray
    ) -> Dict:
        """
        Compare measured to calculated dose.

        Performs gamma analysis.
        """
        # Simplified gamma calculation
        dose_diff = np.abs(measured - calculated) / np.max(calculated) * 100
        pass_rate = np.sum(dose_diff < self._gamma_criteria[0]) / measured.size * 100

        return {
            "gamma_pass_rate": pass_rate,
            "criteria": self._gamma_criteria,
            "threshold": self._passing_threshold,
            "passed": pass_rate >= self._passing_threshold,
            "max_gamma": float(np.max(dose_diff)),
            "mean_gamma": float(np.mean(dose_diff)),
        }


class CommissioningProcedure:
    """
    Machine commissioning procedure.

    Complete beam characterization for new machines.
    """

    def __init__(self, machine_id: str, machine_type: str):
        self.machine_id = machine_id
        self.machine_type = machine_type
        self._beam_data: Dict = {}

    def measure_pdd(
        self,
        energy: float,
        field_sizes: List[Tuple[float, float]],
        ssd: float = 100.0
    ) -> Dict:
        """
        Measure percentage depth dose curves.

        Args:
            energy: Beam energy (MV or MeV)
            field_sizes: List of field sizes
            ssd: Source-to-surface distance
        """
        pdd_data = {}

        for fs in field_sizes:
            depths = np.arange(0, 350, 5)  # mm
            # Simulate PDD curve
            dmax = 15 + energy  # Simplified
            pdd = 100 * np.exp(-0.05 * (depths - dmax))
            pdd[depths < dmax] = 100 * (depths[depths < dmax] / dmax)

            pdd_data[f"{fs[0]}x{fs[1]}"] = {
                "depths": depths.tolist(),
                "pdd": pdd.tolist(),
                "dmax": dmax,
            }

        self._beam_data[f"{energy}MV_PDD"] = pdd_data
        return pdd_data

    def measure_profiles(
        self,
        energy: float,
        field_size: Tuple[float, float],
        depths: List[float]
    ) -> Dict:
        """Measure beam profiles at various depths."""
        profile_data = {}

        for depth in depths:
            positions = np.arange(-200, 201, 2)  # mm
            # Simulate profile (Gaussian + flat top)
            half_width = field_size[0] * 5  # mm
            profile = np.where(
                np.abs(positions) < half_width,
                100,
                100 * np.exp(-((np.abs(positions) - half_width) ** 2) / 500)
            )

            profile_data[f"d{depth}mm"] = {
                "positions": positions.tolist(),
                "profile": profile.tolist(),
                "flatness": 2.5,
                "symmetry": 0.5,
            }

        self._beam_data[f"{energy}MV_profiles_{field_size}"] = profile_data
        return profile_data

    def measure_output_factors(
        self,
        energy: float,
        field_sizes: List[Tuple[float, float]]
    ) -> Dict:
        """Measure output factors for field sizes."""
        of_data = {}
        ref_size = (10, 10)

        for fs in field_sizes:
            # Simulate OF
            ratio = (fs[0] * fs[1]) / (ref_size[0] * ref_size[1])
            of = 0.9 + 0.1 * ratio ** 0.3

            of_data[f"{fs[0]}x{fs[1]}"] = of

        self._beam_data[f"{energy}MV_OF"] = of_data
        return of_data


class OutputCalibration:
    """
    Absolute output calibration procedure.

    Base class for TG-51 and TRS-398 protocols.
    """

    def __init__(self, chamber, electrometer):
        self.chamber = chamber
        self.electrometer = electrometer
        self._temperature = 22.0  # °C
        self._pressure = 101.325  # kPa
        self._humidity = 50.0  # %

    def set_environmental_conditions(
        self,
        temperature: float,
        pressure: float,
        humidity: float = 50.0
    ):
        """Set environmental conditions."""
        self._temperature = temperature
        self._pressure = pressure
        self._humidity = humidity

    def calculate_ptp_correction(self) -> float:
        """Calculate temperature-pressure correction factor."""
        T_ref = 295.15  # K (22°C)
        P_ref = 101.325  # kPa

        T = self._temperature + 273.15
        P = self._pressure

        return (P_ref / P) * (T / T_ref)


class TG51Calibration(OutputCalibration):
    """
    TG-51 absolute calibration protocol.

    AAPM protocol for photon and electron beams.
    """

    def __init__(self, chamber, electrometer):
        super().__init__(chamber, electrometer)
        self._n_d_w = 0.0  # Absorbed dose to water cal factor
        self._k_q = 1.0  # Beam quality correction

    def set_calibration_factor(self, n_d_w: float):
        """Set chamber calibration factor."""
        self._n_d_w = n_d_w

    def determine_beam_quality(
        self,
        pdd_10: float = None,
        tpr_20_10: float = None
    ) -> float:
        """
        Determine beam quality specifier.

        For photons: %dd(10)x or TPR20,10
        """
        if pdd_10:
            return pdd_10
        elif tpr_20_10:
            return tpr_20_10
        return 0.0

    def get_k_q(self, beam_quality: float) -> float:
        """Get beam quality correction factor."""
        # Simplified interpolation from TG-51 tables
        # For photons, kQ ≈ 1 for most chambers
        self._k_q = 1.0 - 0.001 * (beam_quality - 66)
        return self._k_q

    def calculate_dose(
        self,
        m_raw: float,
        mu: float
    ) -> float:
        """
        Calculate absorbed dose to water.

        D_w = M × N_D,w × k_Q × P_TP × k_pol × k_s × k_elec
        """
        p_tp = self.calculate_ptp_correction()
        k_pol = 1.0  # Polarity correction
        k_s = 1.0  # Saturation correction
        k_elec = 1.0  # Electrometer correction

        m_corrected = m_raw * p_tp * k_pol * k_s * k_elec

        d_w = m_corrected * self._n_d_w * self._k_q

        # Convert to cGy/MU
        return d_w / mu * 100


class TRS398Calibration(OutputCalibration):
    """
    IAEA TRS-398 absolute calibration protocol.

    International protocol for photon, electron, and proton beams.
    """

    def __init__(self, chamber, electrometer):
        super().__init__(chamber, electrometer)
        self._n_d_w_q0 = 0.0  # Cal factor at reference quality Q0
        self._k_q_q0 = 1.0  # Beam quality correction

    def set_calibration_factor(self, n_d_w_q0: float, q0: str = "Co-60"):
        """Set calibration factor at reference quality."""
        self._n_d_w_q0 = n_d_w_q0
        self._reference_quality = q0

    def get_k_q_q0(
        self,
        beam_quality: float,
        beam_type: str = "photon"
    ) -> float:
        """Get beam quality correction factor kQ,Q0."""
        if beam_type == "photon":
            # For photon beams
            self._k_q_q0 = 0.99 + 0.001 * (beam_quality - 60)
        elif beam_type == "electron":
            # For electron beams
            self._k_q_q0 = 0.90 + 0.01 * beam_quality / 10
        elif beam_type == "proton":
            # For proton beams - constant for most energies
            self._k_q_q0 = 1.0

        return self._k_q_q0

    def calculate_dose(
        self,
        m_raw: float,
        mu: float
    ) -> float:
        """Calculate absorbed dose to water per TRS-398."""
        # D_w,Q = M_Q × N_D,w,Q0 × k_Q,Q0

        p_tp = self.calculate_ptp_correction()
        k_pol = 1.0
        k_s = 1.0
        k_elec = 1.0

        m_q = m_raw * p_tp * k_pol * k_s * k_elec

        d_w_q = m_q * self._n_d_w_q0 * self._k_q_q0

        return d_w_q / mu * 100  # cGy/MU
