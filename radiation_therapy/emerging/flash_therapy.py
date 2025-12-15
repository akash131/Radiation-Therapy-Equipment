"""
FLASH Radiotherapy Systems.

Implements ultra-high dose rate (UHDR) delivery systems:
- Electron FLASH (clinical and research)
- Proton FLASH
- X-ray FLASH (emerging)

FLASH effect: >40 Gy/s spares normal tissue while
maintaining tumor control.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime
import numpy as np
from abc import ABC, abstractmethod

from ..base import Position3D


class FLASHModality(Enum):
    """FLASH delivery modalities."""
    ELECTRON = "electron"
    PROTON = "proton"
    XRAY = "xray"
    VHEE = "vhee"  # Very High Energy Electrons


class DoseRatePattern(Enum):
    """Dose rate delivery patterns."""
    CONTINUOUS = "continuous"
    PULSED = "pulsed"
    QUASI_CONTINUOUS = "quasi_continuous"


@dataclass
class FLASHBeamParameters:
    """Parameters for FLASH beam delivery."""
    modality: FLASHModality
    energy: float  # MeV
    dose_rate: float  # Gy/s (instantaneous)
    mean_dose_rate: float  # Gy/s (averaged over delivery)

    # Pulse parameters (for pulsed delivery)
    pulse_duration: float = 0.0  # microseconds
    pulse_repetition_rate: float = 0.0  # Hz
    dose_per_pulse: float = 0.0  # Gy

    # Beam characteristics
    field_size: Tuple[float, float] = (10.0, 10.0)  # cm
    applicator_size: Optional[float] = None  # cm (for electrons)
    source_distance: float = 100.0  # cm

    # FLASH thresholds
    minimum_dose_rate: float = 40.0  # Gy/s threshold for FLASH effect
    total_delivery_time: float = 0.0  # seconds

    def is_flash_dose_rate(self) -> bool:
        """Check if dose rate meets FLASH threshold."""
        return self.mean_dose_rate >= self.minimum_dose_rate

    def calculate_delivery_time(self, total_dose: float) -> float:
        """Calculate delivery time for given dose."""
        if self.mean_dose_rate > 0:
            self.total_delivery_time = total_dose / self.mean_dose_rate
        return self.total_delivery_time


@dataclass
class FLASHDosePoint:
    """Dose and dose rate at a point."""
    position: Position3D
    dose: float  # Gy
    dose_rate: float  # Gy/s
    delivery_time: float  # seconds
    is_flash: bool = False  # Meets FLASH criteria


class FLASHDeliverySystem(ABC):
    """
    Abstract base class for FLASH delivery systems.

    FLASH radiotherapy delivers ultra-high dose rates
    (>40 Gy/s) which spare normal tissue through the
    FLASH effect while maintaining tumor control.
    """

    # FLASH effect thresholds
    FLASH_DOSE_RATE_THRESHOLD = 40.0  # Gy/s minimum
    FLASH_DOSE_THRESHOLD = 4.0  # Gy minimum total dose
    FLASH_TIME_THRESHOLD = 0.5  # seconds maximum delivery time

    def __init__(
        self,
        name: str,
        modality: FLASHModality
    ):
        self.name = name
        self.modality = modality

        self._is_initialized = False
        self._is_calibrated = False
        self._current_params: Optional[FLASHBeamParameters] = None

        # Dose rate monitoring
        self._measured_dose_rates: List[float] = []
        self._dose_rate_log: List[Dict] = []

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize FLASH system."""
        pass

    @abstractmethod
    def calibrate(self) -> bool:
        """Calibrate FLASH system."""
        pass

    @abstractmethod
    def deliver_dose(
        self,
        dose: float,
        params: FLASHBeamParameters
    ) -> Dict[str, Any]:
        """Deliver prescribed dose at FLASH rate."""
        pass

    def verify_flash_conditions(
        self,
        params: FLASHBeamParameters,
        dose: float
    ) -> Tuple[bool, List[str]]:
        """
        Verify FLASH delivery conditions are met.

        Returns (conditions_met, list_of_issues).
        """
        issues = []

        # Check dose rate
        if params.mean_dose_rate < self.FLASH_DOSE_RATE_THRESHOLD:
            issues.append(
                f"Mean dose rate {params.mean_dose_rate:.1f} Gy/s < "
                f"{self.FLASH_DOSE_RATE_THRESHOLD} Gy/s threshold"
            )

        # Check total dose
        if dose < self.FLASH_DOSE_THRESHOLD:
            issues.append(
                f"Total dose {dose:.1f} Gy < "
                f"{self.FLASH_DOSE_THRESHOLD} Gy threshold"
            )

        # Check delivery time
        delivery_time = params.calculate_delivery_time(dose)
        if delivery_time > self.FLASH_TIME_THRESHOLD:
            issues.append(
                f"Delivery time {delivery_time:.3f}s > "
                f"{self.FLASH_TIME_THRESHOLD}s threshold"
            )

        return len(issues) == 0, issues

    def log_dose_rate(self, timestamp: datetime, dose_rate: float):
        """Log measured dose rate."""
        self._measured_dose_rates.append(dose_rate)
        self._dose_rate_log.append({
            "timestamp": timestamp,
            "dose_rate": dose_rate,
            "is_flash": dose_rate >= self.FLASH_DOSE_RATE_THRESHOLD,
        })

    def get_dose_rate_statistics(self) -> Dict[str, float]:
        """Get dose rate statistics from delivery."""
        if not self._measured_dose_rates:
            return {}

        rates = np.array(self._measured_dose_rates)
        return {
            "mean": float(np.mean(rates)),
            "max": float(np.max(rates)),
            "min": float(np.min(rates)),
            "std": float(np.std(rates)),
            "flash_fraction": float(np.mean(rates >= self.FLASH_DOSE_RATE_THRESHOLD)),
        }


class ElectronFLASH(FLASHDeliverySystem):
    """
    Electron FLASH delivery system.

    Most clinically advanced FLASH modality using
    high-energy electrons (typically 6-20 MeV).
    Limited depth penetration but high dose rates achievable.
    """

    # Typical electron FLASH parameters
    TYPICAL_ENERGIES = [6, 9, 12, 16, 20]  # MeV
    MAX_DOSE_RATE = 1000.0  # Gy/s achievable
    TYPICAL_FIELD_SIZES = [3, 4, 6, 10]  # cm diameter

    def __init__(self, name: str = "Electron FLASH"):
        super().__init__(name, FLASHModality.ELECTRON)

        self._available_energies = self.TYPICAL_ENERGIES.copy()
        self._available_applicators = self.TYPICAL_FIELD_SIZES.copy()

        # Machine parameters
        self._max_current = 100.0  # mA peak
        self._pulse_width = 4.0  # microseconds
        self._max_rep_rate = 360.0  # Hz

        # Output calibration (Gy/MU at reference conditions)
        self._output_factors: Dict[int, float] = {
            6: 1.0,
            9: 1.0,
            12: 1.0,
            16: 1.0,
            20: 1.0,
        }

    def initialize(self) -> bool:
        """Initialize electron FLASH system."""
        self._is_initialized = True
        return True

    def calibrate(self) -> bool:
        """Calibrate electron FLASH output."""
        self._is_calibrated = True
        return True

    def set_energy(self, energy: float) -> bool:
        """Set beam energy."""
        if int(energy) in self._available_energies:
            if self._current_params:
                self._current_params.energy = energy
            return True
        return False

    def set_applicator(self, size: float) -> bool:
        """Set electron applicator."""
        if int(size) in self._available_applicators:
            if self._current_params:
                self._current_params.applicator_size = size
            return True
        return False

    def calculate_dose_rate(
        self,
        energy: float,
        current: float,
        applicator: float,
        distance: float = 100.0
    ) -> float:
        """
        Calculate expected dose rate.

        Dose rate depends on beam current, energy,
        applicator size, and SSD.
        """
        # Simplified dose rate calculation
        # Real calculation would use measured data
        base_rate = 100.0  # Gy/s at reference conditions

        # Current factor
        current_factor = current / 10.0  # Reference 10 mA

        # Energy factor (higher energy = lower dose rate for same charge)
        energy_factor = 10.0 / energy

        # Applicator factor (smaller applicator = higher dose rate)
        applicator_factor = (10.0 / applicator) ** 2

        # Distance factor (inverse square)
        distance_factor = (100.0 / distance) ** 2

        return base_rate * current_factor * energy_factor * applicator_factor * distance_factor

    def get_depth_dose(
        self,
        energy: float,
        depth: float
    ) -> float:
        """Get percentage depth dose for electron beam."""
        # Simplified PDD model
        # R90 ~ E/4, R50 ~ E/2.4, Rp ~ E/2
        r90 = energy / 4.0  # cm
        rp = energy / 2.0  # cm

        if depth < r90:
            # Buildup and plateau
            pdd = 100.0 * (1 - np.exp(-depth / (r90 * 0.3)))
        elif depth < rp:
            # Falloff region
            pdd = 100.0 * np.exp(-((depth - r90) / (rp - r90)) ** 2 * 2)
        else:
            # Beyond practical range
            pdd = 5.0 * np.exp(-(depth - rp) / 0.5)

        return float(pdd)

    def deliver_dose(
        self,
        dose: float,
        params: FLASHBeamParameters
    ) -> Dict[str, Any]:
        """Deliver prescribed dose at FLASH rate."""
        self._current_params = params

        # Verify FLASH conditions
        conditions_met, issues = self.verify_flash_conditions(params, dose)

        # Calculate delivery parameters
        delivery_time = dose / params.mean_dose_rate

        # Simulate delivery
        num_pulses = int(delivery_time * params.pulse_repetition_rate)
        dose_per_pulse = dose / max(1, num_pulses)

        # Log dose rates
        for i in range(num_pulses):
            # Add some variation
            rate = params.dose_rate * (0.95 + 0.1 * np.random.random())
            self.log_dose_rate(datetime.now(), rate)

        return {
            "delivered_dose": dose,
            "delivery_time": delivery_time,
            "num_pulses": num_pulses,
            "dose_per_pulse": dose_per_pulse,
            "flash_conditions_met": conditions_met,
            "issues": issues,
            "dose_rate_stats": self.get_dose_rate_statistics(),
        }


class ProtonFLASH(FLASHDeliverySystem):
    """
    Proton FLASH delivery system.

    Uses ultra-high dose rate proton beams with
    Bragg peak for FLASH effect at depth.
    """

    # Proton FLASH parameters
    TYPICAL_ENERGIES = range(70, 251, 10)  # MeV
    MAX_DOSE_RATE = 100.0  # Gy/s (more challenging than electrons)

    def __init__(self, name: str = "Proton FLASH"):
        super().__init__(name, FLASHModality.PROTON)

        # Accelerator parameters
        self._accelerator_type = "cyclotron"  # or "synchrotron"
        self._max_current = 500.0  # nA
        self._extraction_efficiency = 0.8

        # Delivery mode
        self._transmission_mode = True  # vs scanning mode
        self._scatterer_in = True

    def initialize(self) -> bool:
        """Initialize proton FLASH system."""
        self._is_initialized = True
        return True

    def calibrate(self) -> bool:
        """Calibrate proton FLASH output."""
        self._is_calibrated = True
        return True

    def set_transmission_mode(self, enabled: bool):
        """
        Set transmission mode for FLASH.

        Transmission mode uses scatterers for uniform field
        at high dose rate. Scanning mode typically too slow
        for FLASH.
        """
        self._transmission_mode = enabled

    def calculate_dose_rate(
        self,
        energy: float,
        current_nA: float,
        field_size: Tuple[float, float]
    ) -> float:
        """
        Calculate expected dose rate at Bragg peak.

        Proton dose rate depends on beam current and
        spreading method.
        """
        # Simplified calculation
        # Reference: 1 nA at isocenter ~ 1 cGy/s for small field
        base_rate = current_nA * 0.01  # Gy/s

        # Field size factor (larger field = lower dose rate)
        area = field_size[0] * field_size[1]
        reference_area = 25.0  # 5x5 cm reference
        field_factor = reference_area / area

        # Energy factor
        energy_factor = np.sqrt(energy / 200.0)

        return base_rate * field_factor * energy_factor

    def get_bragg_peak_depth(self, energy: float) -> float:
        """Get Bragg peak depth for given energy."""
        # Approximate range in water (cm)
        # Range ~ 0.0022 * E^1.77 for protons in water
        return 0.0022 * (energy ** 1.77)

    def calculate_sobp(
        self,
        energy_weights: Dict[float, float],
        depth_range: Tuple[float, float]
    ) -> np.ndarray:
        """
        Calculate spread-out Bragg peak.

        For FLASH, SOBP delivery is challenging due to
        time constraints.
        """
        depths = np.linspace(0, depth_range[1] * 1.2, 100)
        sobp = np.zeros_like(depths)

        for energy, weight in energy_weights.items():
            peak_depth = self.get_bragg_peak_depth(energy)

            # Simplified Bragg peak
            for i, d in enumerate(depths):
                if d < peak_depth * 0.9:
                    sobp[i] += weight * 0.3
                elif d < peak_depth:
                    sobp[i] += weight * (0.3 + 0.7 * (d - peak_depth * 0.9) / (peak_depth * 0.1))
                elif d < peak_depth * 1.1:
                    sobp[i] += weight * np.exp(-((d - peak_depth) / (peak_depth * 0.05)) ** 2)

        return sobp

    def deliver_dose(
        self,
        dose: float,
        params: FLASHBeamParameters
    ) -> Dict[str, Any]:
        """Deliver prescribed dose at FLASH rate."""
        self._current_params = params

        # Verify FLASH conditions
        conditions_met, issues = self.verify_flash_conditions(params, dose)

        # Proton-specific challenges
        if not self._transmission_mode:
            issues.append("Scanning mode may not achieve FLASH dose rates")

        delivery_time = dose / params.mean_dose_rate

        # Simulate delivery
        for _ in range(10):  # Simplified
            rate = params.dose_rate * (0.9 + 0.2 * np.random.random())
            self.log_dose_rate(datetime.now(), rate)

        return {
            "delivered_dose": dose,
            "delivery_time": delivery_time,
            "transmission_mode": self._transmission_mode,
            "bragg_peak_depth": self.get_bragg_peak_depth(params.energy),
            "flash_conditions_met": conditions_met,
            "issues": issues,
            "dose_rate_stats": self.get_dose_rate_statistics(),
        }


class XrayFLASH(FLASHDeliverySystem):
    """
    X-ray FLASH delivery system (experimental).

    Achieving FLASH dose rates with X-rays is extremely
    challenging due to beam generation limitations.
    Requires novel accelerator designs or multiple sources.
    """

    def __init__(self, name: str = "X-ray FLASH"):
        super().__init__(name, FLASHModality.XRAY)

        # X-ray FLASH is experimental
        self._is_experimental = True

        # Possible approaches
        self._approach = "multi_source"  # or "high_current_linac"

        # Target dose rate (very challenging)
        self._target_dose_rate = 40.0  # Gy/s minimum

    def initialize(self) -> bool:
        """Initialize X-ray FLASH system."""
        self._is_initialized = True
        return True

    def calibrate(self) -> bool:
        """Calibrate X-ray FLASH output."""
        self._is_calibrated = True
        return True

    def calculate_theoretical_rate(
        self,
        peak_current: float,  # mA
        energy: float,  # MV
        field_size: Tuple[float, float],
        distance: float
    ) -> float:
        """
        Calculate theoretical maximum dose rate.

        X-ray FLASH requires orders of magnitude higher
        beam current than conventional LINAC.
        """
        # Conventional LINAC: ~0.01 Gy/s per mA at 100 cm
        # Need 40+ Gy/s = 4000+ times conventional
        base_rate = peak_current * 0.01 * (100.0 / distance) ** 2

        # Energy factor
        energy_factor = energy / 6.0

        # Field factor
        area = field_size[0] * field_size[1]
        field_factor = 100.0 / area  # Smaller field = higher rate

        return base_rate * energy_factor * field_factor

    def deliver_dose(
        self,
        dose: float,
        params: FLASHBeamParameters
    ) -> Dict[str, Any]:
        """Deliver dose (experimental system)."""
        self._current_params = params

        conditions_met, issues = self.verify_flash_conditions(params, dose)

        # Add experimental warning
        issues.insert(0, "X-ray FLASH is experimental technology")

        return {
            "delivered_dose": dose,
            "experimental": True,
            "flash_conditions_met": conditions_met,
            "issues": issues,
        }


class FLASHDoseCalculator:
    """
    Dose calculation for FLASH radiotherapy.

    Accounts for ultra-high dose rate effects on
    biological response.
    """

    def __init__(self):
        self._flash_threshold = 40.0  # Gy/s
        self._sparing_factor = 0.7  # Normal tissue sparing

    def calculate_effective_dose(
        self,
        physical_dose: float,
        dose_rate: float,
        tissue_type: str = "normal"
    ) -> float:
        """
        Calculate effective biological dose.

        FLASH spares normal tissue but maintains
        tumor control.
        """
        if tissue_type == "tumor":
            # Tumor response maintained
            return physical_dose
        else:
            # Normal tissue sparing at FLASH rates
            if dose_rate >= self._flash_threshold:
                return physical_dose * self._sparing_factor
            else:
                return physical_dose

    def calculate_therapeutic_ratio(
        self,
        dose: float,
        dose_rate: float,
        tumor_alpha_beta: float = 10.0,
        normal_alpha_beta: float = 3.0
    ) -> float:
        """
        Calculate therapeutic ratio with FLASH effect.

        FLASH improves therapeutic ratio by sparing
        normal tissue.
        """
        # Standard therapeutic ratio
        tumor_effect = dose * (1 + dose / tumor_alpha_beta)
        normal_effect = dose * (1 + dose / normal_alpha_beta)

        standard_ratio = tumor_effect / normal_effect

        # FLASH enhancement
        if dose_rate >= self._flash_threshold:
            flash_ratio = standard_ratio / self._sparing_factor
        else:
            flash_ratio = standard_ratio

        return flash_ratio


class FLASHBiologicalModel:
    """
    Biological modeling for FLASH effect.

    Models the differential response of normal tissue
    and tumor at ultra-high dose rates.
    """

    # Proposed mechanisms for FLASH effect
    MECHANISMS = [
        "oxygen_depletion",
        "radical_recombination",
        "immune_sparing",
        "stem_cell_protection",
    ]

    def __init__(self):
        self._oxygen_model = True
        self._flash_threshold = 40.0  # Gy/s

        # Tissue-specific parameters
        self._tissue_params: Dict[str, Dict] = {
            "normal": {
                "alpha": 0.3,
                "beta": 0.1,
                "flash_sparing": 0.7,
            },
            "tumor": {
                "alpha": 0.35,
                "beta": 0.035,
                "flash_sparing": 1.0,  # No sparing for tumor
            },
        }

    def set_tissue_parameters(
        self,
        tissue_type: str,
        alpha: float,
        beta: float,
        flash_sparing: float
    ):
        """Set tissue-specific biological parameters."""
        self._tissue_params[tissue_type] = {
            "alpha": alpha,
            "beta": beta,
            "flash_sparing": flash_sparing,
        }

    def calculate_survival(
        self,
        dose: float,
        dose_rate: float,
        tissue_type: str
    ) -> float:
        """
        Calculate cell survival fraction.

        Uses modified linear-quadratic model with
        FLASH effect.
        """
        params = self._tissue_params.get(tissue_type, self._tissue_params["normal"])

        alpha = params["alpha"]
        beta = params["beta"]
        flash_sparing = params["flash_sparing"]

        # Standard LQ survival
        lq_survival = np.exp(-alpha * dose - beta * dose ** 2)

        # Apply FLASH effect
        if dose_rate >= self._flash_threshold and tissue_type != "tumor":
            # FLASH spares normal tissue
            effective_dose = dose * flash_sparing
            flash_survival = np.exp(-alpha * effective_dose - beta * effective_dose ** 2)
        else:
            flash_survival = lq_survival

        return float(flash_survival)

    def calculate_oxygen_depletion(
        self,
        dose: float,
        dose_rate: float,
        initial_oxygen: float = 40.0  # mmHg
    ) -> float:
        """
        Model oxygen depletion during FLASH delivery.

        Rapid oxygen depletion at FLASH rates may
        contribute to normal tissue sparing.
        """
        # Oxygen consumption rate
        consumption_rate = 3.0  # mmHg per Gy

        # Delivery time
        if dose_rate > 0:
            delivery_time = dose / dose_rate
        else:
            return initial_oxygen

        # Oxygen re-supply rate
        resupply_rate = 10.0  # mmHg/s

        # At FLASH rates, delivery faster than resupply
        if delivery_time < 0.5:  # < 500 ms
            # Significant depletion
            final_oxygen = initial_oxygen - consumption_rate * dose
            final_oxygen = max(0, final_oxygen)
        else:
            # Resupply compensates
            net_change = consumption_rate * dose - resupply_rate * delivery_time
            final_oxygen = max(0, initial_oxygen - net_change)

        return final_oxygen

    def predict_ntcp(
        self,
        dose: float,
        dose_rate: float,
        organ: str,
        volume_fraction: float = 1.0
    ) -> float:
        """
        Predict normal tissue complication probability.

        FLASH should reduce NTCP compared to conventional
        dose rates.
        """
        # Organ-specific TD50 and slope
        organ_params = {
            "lung": {"td50": 30.0, "m": 0.3},
            "skin": {"td50": 50.0, "m": 0.2},
            "brain": {"td50": 60.0, "m": 0.15},
            "spinal_cord": {"td50": 50.0, "m": 0.175},
        }

        params = organ_params.get(organ, {"td50": 50.0, "m": 0.25})
        td50 = params["td50"]
        m = params["m"]

        # Apply FLASH sparing
        if dose_rate >= self._flash_threshold:
            effective_dose = dose * 0.7  # Sparing factor
        else:
            effective_dose = dose

        # Lyman-Kutcher-Burman model
        t = (effective_dose - td50) / (m * td50)
        ntcp = 0.5 * (1 + self._erf(t / np.sqrt(2)))

        return float(ntcp)

    def _erf(self, x: float) -> float:
        """Approximate error function."""
        # Approximation
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
