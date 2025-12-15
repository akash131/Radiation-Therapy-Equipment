"""
Biological Dosimetry Models.

Implements radiobiological models for:
- Linear-quadratic cell survival
- Tumor control probability (TCP)
- Normal tissue complication probability (NTCP)
- Biologically effective dose (BED)
- Equivalent dose in 2Gy fractions (EQD2)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
from abc import ABC, abstractmethod


class FractionationScheme(Enum):
    """Common fractionation schemes."""
    CONVENTIONAL = "conventional"  # 1.8-2 Gy/fx
    HYPOFRACTIONATED = "hypofractionated"  # >2 Gy/fx
    HYPERFRACTIONATED = "hyperfractionated"  # <1.8 Gy/fx
    SBRT = "sbrt"  # 6-20 Gy/fx
    SRS = "srs"  # Single fraction


@dataclass
class TissueParameters:
    """Radiobiological parameters for tissue."""
    name: str
    alpha: float  # Gy^-1
    beta: float  # Gy^-2
    alpha_beta: float  # Gy

    # For TCP/NTCP models
    d50: Optional[float] = None  # Gy
    gamma50: Optional[float] = None
    n: Optional[float] = None  # Volume effect parameter

    # Repair parameters
    repair_half_time: float = 1.5  # hours

    @classmethod
    def create(cls, name: str, alpha: float, beta: float) -> "TissueParameters":
        """Create tissue parameters from alpha and beta."""
        return cls(
            name=name,
            alpha=alpha,
            beta=beta,
            alpha_beta=alpha / beta if beta > 0 else float('inf')
        )


class BiologicalDoseModel(ABC):
    """Abstract base class for biological dose models."""

    @abstractmethod
    def calculate(self, *args, **kwargs) -> float:
        """Calculate biological dose metric."""
        pass


class LinearQuadraticModel(BiologicalDoseModel):
    """
    Linear-Quadratic (LQ) model for cell survival.

    S = exp(-alpha*D - beta*D^2)

    The most widely used model for radiation response.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.03
    ):
        self.alpha = alpha
        self.beta = beta

    @property
    def alpha_beta(self) -> float:
        """Calculate alpha/beta ratio."""
        if self.beta > 0:
            return self.alpha / self.beta
        return float('inf')

    def calculate(
        self,
        dose: float,
        fractions: int = 1
    ) -> float:
        """
        Calculate surviving fraction.

        Args:
            dose: Total dose in Gy
            fractions: Number of fractions

        Returns:
            Surviving fraction (0-1)
        """
        dose_per_fraction = dose / fractions
        survival = np.exp(
            -fractions * (self.alpha * dose_per_fraction +
                         self.beta * dose_per_fraction ** 2)
        )
        return float(survival)

    def calculate_survival_curve(
        self,
        doses: np.ndarray,
        fractions: int = 1
    ) -> np.ndarray:
        """Calculate survival curve for range of doses."""
        return np.array([self.calculate(d, fractions) for d in doses])

    def calculate_d10(self, fractions: int = 1) -> float:
        """Calculate dose for 10% survival (90% cell kill)."""
        # Solve for D where S = 0.1
        # -ln(0.1) = alpha*d + beta*d^2 (per fraction)
        target = -np.log(0.1) / fractions

        # Quadratic formula
        a = self.beta
        b = self.alpha
        c = -target

        if a > 0:
            d_per_fx = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        else:
            d_per_fx = target / self.alpha

        return d_per_fx * fractions

    def calculate_d37(self, fractions: int = 1) -> float:
        """Calculate dose for 37% survival (D0 equivalent)."""
        target = 1.0 / fractions  # ln(1/e)

        a = self.beta
        b = self.alpha
        c = -target

        if a > 0:
            d_per_fx = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        else:
            d_per_fx = target / self.alpha

        return d_per_fx * fractions


class BEDCalculator(BiologicalDoseModel):
    """
    Biologically Effective Dose (BED) calculator.

    BED = D * (1 + d/[alpha/beta])

    Where D is total dose and d is dose per fraction.
    """

    def __init__(
        self,
        alpha_beta: float = 10.0
    ):
        self.alpha_beta = alpha_beta

    def calculate(
        self,
        total_dose: float,
        fractions: int,
        alpha_beta: Optional[float] = None
    ) -> float:
        """
        Calculate BED.

        Args:
            total_dose: Total physical dose (Gy)
            fractions: Number of fractions
            alpha_beta: Optional override for alpha/beta

        Returns:
            BED in Gy
        """
        ab = alpha_beta or self.alpha_beta
        dose_per_fraction = total_dose / fractions
        bed = total_dose * (1 + dose_per_fraction / ab)
        return bed

    def calculate_with_time(
        self,
        total_dose: float,
        fractions: int,
        treatment_days: int,
        alpha_beta: float = 10.0,
        potential_doubling_time: float = 3.0,  # days
        kick_off_time: float = 21.0  # days
    ) -> float:
        """
        Calculate BED with time factor for tumor repopulation.

        BED = D*(1 + d/[α/β]) - K * (T - Tk)

        Where K accounts for repopulation.
        """
        bed_physical = self.calculate(total_dose, fractions, alpha_beta)

        # Repopulation factor
        if treatment_days > kick_off_time:
            k = 0.693 / (alpha_beta * potential_doubling_time)
            repopulation = k * (treatment_days - kick_off_time)
        else:
            repopulation = 0

        return bed_physical - repopulation

    def compare_schedules(
        self,
        schedule1: Tuple[float, int],  # (dose, fractions)
        schedule2: Tuple[float, int],
        alpha_beta: float = 10.0
    ) -> Dict[str, float]:
        """Compare two fractionation schedules."""
        bed1 = self.calculate(schedule1[0], schedule1[1], alpha_beta)
        bed2 = self.calculate(schedule2[0], schedule2[1], alpha_beta)

        return {
            "schedule1_bed": bed1,
            "schedule2_bed": bed2,
            "difference": bed1 - bed2,
            "ratio": bed1 / bed2 if bed2 > 0 else float('inf'),
            "equivalent": abs(bed1 - bed2) < 1.0,  # Within 1 Gy
        }


class EQD2Calculator(BiologicalDoseModel):
    """
    Equivalent Dose in 2Gy fractions (EQD2) calculator.

    EQD2 = D * (d + [α/β]) / (2 + [α/β])

    Converts any fractionation to equivalent dose at 2 Gy/fraction.
    """

    def __init__(
        self,
        alpha_beta: float = 10.0
    ):
        self.alpha_beta = alpha_beta

    def calculate(
        self,
        total_dose: float,
        fractions: int,
        alpha_beta: Optional[float] = None
    ) -> float:
        """
        Calculate EQD2.

        Args:
            total_dose: Total physical dose (Gy)
            fractions: Number of fractions
            alpha_beta: Optional override for alpha/beta

        Returns:
            EQD2 in Gy
        """
        ab = alpha_beta or self.alpha_beta
        dose_per_fraction = total_dose / fractions
        eqd2 = total_dose * (dose_per_fraction + ab) / (2 + ab)
        return eqd2

    def convert_to_schedule(
        self,
        target_eqd2: float,
        dose_per_fraction: float,
        alpha_beta: Optional[float] = None
    ) -> Tuple[float, int]:
        """
        Convert EQD2 to physical dose schedule.

        Returns (total_dose, fractions).
        """
        ab = alpha_beta or self.alpha_beta

        # EQD2 = D * (d + ab) / (2 + ab)
        # D = EQD2 * (2 + ab) / (d + ab)
        total_dose = target_eqd2 * (2 + ab) / (dose_per_fraction + ab)
        fractions = int(np.ceil(total_dose / dose_per_fraction))

        return (total_dose, fractions)


class TCPModel(BiologicalDoseModel):
    """
    Tumor Control Probability (TCP) models.

    Implements multiple TCP models:
    - Poisson model
    - Logistic model
    - Linear-quadratic based
    """

    def __init__(
        self,
        model_type: str = "poisson"
    ):
        self.model_type = model_type

        # Default tumor parameters
        self._n0 = 1e9  # Initial clonogenic cells
        self._alpha = 0.35  # Gy^-1
        self._beta = 0.035  # Gy^-2

        # For logistic model
        self._d50 = 50.0  # Gy
        self._gamma50 = 2.0  # Slope

    def set_tumor_parameters(
        self,
        n0: float = 1e9,
        alpha: float = 0.35,
        beta: float = 0.035
    ):
        """Set tumor radiosensitivity parameters."""
        self._n0 = n0
        self._alpha = alpha
        self._beta = beta

    def set_logistic_parameters(
        self,
        d50: float,
        gamma50: float
    ):
        """Set logistic model parameters."""
        self._d50 = d50
        self._gamma50 = gamma50

    def calculate(
        self,
        total_dose: float,
        fractions: int
    ) -> float:
        """Calculate TCP based on model type."""
        if self.model_type == "poisson":
            return self._calculate_poisson(total_dose, fractions)
        elif self.model_type == "logistic":
            return self._calculate_logistic(total_dose)
        else:
            return self._calculate_poisson(total_dose, fractions)

    def _calculate_poisson(
        self,
        total_dose: float,
        fractions: int
    ) -> float:
        """
        Poisson TCP model.

        TCP = exp(-N0 * S)
        where S is surviving fraction from LQ model.
        """
        dose_per_fraction = total_dose / fractions
        surviving_fraction = np.exp(
            -fractions * (self._alpha * dose_per_fraction +
                         self._beta * dose_per_fraction ** 2)
        )

        expected_survivors = self._n0 * surviving_fraction
        tcp = np.exp(-expected_survivors)

        return float(tcp)

    def _calculate_logistic(
        self,
        total_dose: float
    ) -> float:
        """
        Logistic TCP model.

        TCP = 1 / (1 + (D50/D)^(4*gamma50))
        """
        if total_dose <= 0:
            return 0.0

        exponent = 4 * self._gamma50
        tcp = 1 / (1 + (self._d50 / total_dose) ** exponent)

        return float(tcp)

    def calculate_tcp_curve(
        self,
        doses: np.ndarray,
        fractions: int = 1
    ) -> np.ndarray:
        """Calculate TCP curve for range of doses."""
        return np.array([self.calculate(d, fractions) for d in doses])

    def calculate_dose_for_tcp(
        self,
        target_tcp: float,
        fractions: int,
        tolerance: float = 0.01
    ) -> float:
        """Find dose required for target TCP."""
        # Binary search
        low, high = 0.0, 200.0

        while high - low > tolerance:
            mid = (low + high) / 2
            tcp = self.calculate(mid, fractions)

            if tcp < target_tcp:
                low = mid
            else:
                high = mid

        return (low + high) / 2


class NTCPModel(BiologicalDoseModel):
    """
    Normal Tissue Complication Probability (NTCP) models.

    Implements:
    - Lyman-Kutcher-Burman (LKB) model
    - Equivalent uniform dose (EUD) based model
    """

    # Organ-specific parameters (Emami parameters)
    ORGAN_PARAMETERS = {
        "brain": {"td5050": 60.0, "td505": 45.0, "n": 0.25, "m": 0.15},
        "brainstem": {"td5050": 65.0, "td505": 50.0, "n": 0.16, "m": 0.14},
        "spinal_cord": {"td5050": 50.0, "td505": 47.0, "n": 0.05, "m": 0.175},
        "lung": {"td5050": 24.5, "td505": 17.5, "n": 0.87, "m": 0.18},
        "heart": {"td5050": 48.0, "td505": 40.0, "n": 0.35, "m": 0.10},
        "liver": {"td5050": 45.0, "td505": 30.0, "n": 0.32, "m": 0.15},
        "kidney": {"td5050": 28.0, "td505": 23.0, "n": 0.70, "m": 0.10},
        "rectum": {"td5050": 80.0, "td505": 60.0, "n": 0.12, "m": 0.15},
        "bladder": {"td5050": 80.0, "td505": 65.0, "n": 0.50, "m": 0.11},
        "parotid": {"td5050": 46.0, "td505": 32.0, "n": 0.70, "m": 0.18},
        "larynx": {"td5050": 80.0, "td505": 70.0, "n": 0.08, "m": 0.17},
        "esophagus": {"td5050": 68.0, "td505": 55.0, "n": 0.06, "m": 0.11},
        "small_bowel": {"td5050": 55.0, "td505": 40.0, "n": 0.15, "m": 0.16},
        "optic_nerve": {"td5050": 65.0, "td505": 50.0, "n": 0.25, "m": 0.14},
        "cochlea": {"td5050": 65.0, "td505": 55.0, "n": 0.01, "m": 0.20},
    }

    def __init__(
        self,
        organ: str = "lung"
    ):
        self.organ = organ

        # Get organ parameters
        params = self.ORGAN_PARAMETERS.get(organ, self.ORGAN_PARAMETERS["lung"])
        self._td50 = params["td5050"]  # TD50/5 for whole organ
        self._n = params["n"]  # Volume effect parameter
        self._m = params["m"]  # Slope parameter

    def set_parameters(
        self,
        td50: float,
        n: float,
        m: float
    ):
        """Set model parameters directly."""
        self._td50 = td50
        self._n = n
        self._m = m

    def calculate(
        self,
        dose: float,
        volume_fraction: float = 1.0
    ) -> float:
        """
        Calculate NTCP using LKB model.

        Args:
            dose: Uniform dose to partial volume
            volume_fraction: Fraction of organ irradiated

        Returns:
            NTCP (0-1)
        """
        # Effective volume correction
        effective_td50 = self._td50 * (volume_fraction ** (-self._n))

        # Probit function
        t = (dose - effective_td50) / (self._m * effective_td50)
        ntcp = 0.5 * (1 + self._erf(t / np.sqrt(2)))

        return float(ntcp)

    def calculate_from_dvh(
        self,
        doses: np.ndarray,
        volumes: np.ndarray
    ) -> float:
        """
        Calculate NTCP from DVH using Kutcher-Burman reduction.

        Args:
            doses: Dose bins
            volumes: Differential volume at each dose

        Returns:
            NTCP
        """
        # Calculate effective volume
        total_volume = np.sum(volumes)
        if total_volume <= 0:
            return 0.0

        # Kutcher-Burman effective volume reduction
        d_ref = np.max(doses)
        if d_ref <= 0:
            return 0.0

        v_eff = np.sum(volumes * (doses / d_ref) ** (1 / self._n)) / total_volume

        return self.calculate(d_ref, v_eff)

    def calculate_eud(
        self,
        doses: np.ndarray,
        volumes: np.ndarray,
        a: Optional[float] = None
    ) -> float:
        """
        Calculate Equivalent Uniform Dose (EUD).

        EUD = (sum(vi * Di^a))^(1/a)
        """
        if a is None:
            a = 1 / self._n

        total_volume = np.sum(volumes)
        if total_volume <= 0:
            return 0.0

        normalized_volumes = volumes / total_volume
        eud = np.sum(normalized_volumes * (doses ** a)) ** (1 / a)

        return float(eud)

    def _erf(self, x: float) -> float:
        """Approximate error function."""
        sign = 1 if x >= 0 else -1
        x = abs(x)

        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

        return sign * y

    def dose_for_ntcp(
        self,
        target_ntcp: float,
        volume_fraction: float = 1.0
    ) -> float:
        """Find dose for target NTCP."""
        # Inverse of NTCP calculation
        # Numerical solution
        low, high = 0.0, 200.0

        while high - low > 0.1:
            mid = (low + high) / 2
            ntcp = self.calculate(mid, volume_fraction)

            if ntcp < target_ntcp:
                low = mid
            else:
                high = mid

        return (low + high) / 2


class CombinedBiologicalModel:
    """
    Combined TCP/NTCP optimization model.

    Calculates:
    - Uncomplicated tumor control probability (UTCP)
    - Probability of complication-free tumor control (P+)
    """

    def __init__(self):
        self._tcp_model = TCPModel()
        self._ntcp_models: Dict[str, NTCPModel] = {}

    def add_organ(self, organ: str):
        """Add organ for NTCP calculation."""
        self._ntcp_models[organ] = NTCPModel(organ)

    def calculate_utcp(
        self,
        tumor_dose: float,
        tumor_fractions: int,
        organ_doses: Dict[str, float],
        organ_volumes: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate uncomplicated tumor control probability.

        UTCP = TCP * product(1 - NTCP_i)
        """
        tcp = self._tcp_model.calculate(tumor_dose, tumor_fractions)

        complication_free = 1.0
        for organ, dose in organ_doses.items():
            if organ in self._ntcp_models:
                volume = organ_volumes.get(organ, 1.0) if organ_volumes else 1.0
                ntcp = self._ntcp_models[organ].calculate(dose, volume)
                complication_free *= (1 - ntcp)

        return tcp * complication_free

    def optimize_dose(
        self,
        organ_dose_ratios: Dict[str, float],
        target_tcp: float = 0.9,
        max_ntcp: float = 0.05
    ) -> Dict[str, Any]:
        """
        Find optimal dose given constraints.

        Args:
            organ_dose_ratios: Ratio of organ dose to tumor dose
            target_tcp: Target TCP (default 90%)
            max_ntcp: Maximum acceptable NTCP for any organ

        Returns:
            Optimization results
        """
        best_dose = 0.0
        best_utcp = 0.0

        for dose in np.linspace(10, 100, 181):
            # Check TCP
            tcp = self._tcp_model.calculate(dose, 30)
            if tcp < target_tcp:
                continue

            # Check NTCPs
            ntcp_ok = True
            organ_ntcps = {}
            for organ, ratio in organ_dose_ratios.items():
                if organ in self._ntcp_models:
                    organ_dose = dose * ratio
                    ntcp = self._ntcp_models[organ].calculate(organ_dose)
                    organ_ntcps[organ] = ntcp
                    if ntcp > max_ntcp:
                        ntcp_ok = False
                        break

            if not ntcp_ok:
                continue

            # Calculate UTCP
            utcp = tcp
            for ntcp in organ_ntcps.values():
                utcp *= (1 - ntcp)

            if utcp > best_utcp:
                best_utcp = utcp
                best_dose = dose

        return {
            "optimal_dose": best_dose,
            "utcp": best_utcp,
            "tcp": self._tcp_model.calculate(best_dose, 30),
            "organ_ntcps": {
                organ: self._ntcp_models[organ].calculate(best_dose * ratio)
                for organ, ratio in organ_dose_ratios.items()
                if organ in self._ntcp_models
            },
        }
