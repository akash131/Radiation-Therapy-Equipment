"""
Radiation Therapy Safety Systems.

Implements safety interlocks and monitoring systems:
- Safety interlocks (door, beam, motion)
- Dose monitoring and termination
- Emergency stop systems
- Radiation surveys
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Callable, Tuple
from datetime import datetime
import threading
import time


class InterlockType(Enum):
    """Types of safety interlocks."""
    DOOR = "door"
    BEAM = "beam"
    MOTION = "motion"
    COLLISION = "collision"
    DOSE = "dose"
    TEMPERATURE = "temperature"
    VACUUM = "vacuum"
    WATER = "water"
    EMERGENCY = "emergency"


class InterlockStatus(Enum):
    """Status of safety interlock."""
    CLEAR = "clear"
    ACTIVE = "active"
    BYPASSED = "bypassed"
    FAULT = "fault"


class SeverityLevel(Enum):
    """Severity level for safety events."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class InterlockEvent:
    """Record of an interlock event."""
    interlock_type: InterlockType
    status: InterlockStatus
    timestamp: datetime
    message: str
    severity: SeverityLevel
    auto_cleared: bool = False


@dataclass
class SafetyLimit:
    """Safety limit definition."""
    parameter: str
    lower_limit: Optional[float]
    upper_limit: Optional[float]
    action: str  # "warn", "pause", "terminate"
    unit: str = ""


class SafetyInterlock:
    """
    Individual safety interlock.

    Monitors a specific safety condition and can
    trigger beam hold or termination.
    """

    def __init__(
        self,
        name: str,
        interlock_type: InterlockType,
        severity: SeverityLevel = SeverityLevel.CRITICAL
    ):
        self.name = name
        self.interlock_type = interlock_type
        self.severity = severity

        self._status = InterlockStatus.CLEAR
        self._is_bypassed = False
        self._bypass_authorized_by: Optional[str] = None
        self._bypass_reason: Optional[str] = None
        self._bypass_expiry: Optional[datetime] = None

        self._check_function: Optional[Callable[[], bool]] = None
        self._event_history: List[InterlockEvent] = []

    def set_check_function(self, func: Callable[[], bool]):
        """Set function to check interlock condition."""
        self._check_function = func

    def check(self) -> InterlockStatus:
        """Check interlock status."""
        if self._is_bypassed:
            # Check bypass expiry
            if self._bypass_expiry and datetime.now() > self._bypass_expiry:
                self._is_bypassed = False
                self._log_event(
                    InterlockStatus.CLEAR,
                    "Bypass expired, interlock reactivated",
                    SeverityLevel.INFO
                )
            else:
                return InterlockStatus.BYPASSED

        if self._check_function:
            try:
                is_clear = self._check_function()
                new_status = InterlockStatus.CLEAR if is_clear else InterlockStatus.ACTIVE
            except Exception as e:
                new_status = InterlockStatus.FAULT
                self._log_event(
                    InterlockStatus.FAULT,
                    f"Interlock check failed: {str(e)}",
                    SeverityLevel.CRITICAL
                )

            if new_status != self._status:
                self._status = new_status
                self._log_event(new_status, f"Status changed to {new_status.value}")

        return self._status

    def activate(self, message: str = "Manual activation"):
        """Manually activate interlock."""
        self._status = InterlockStatus.ACTIVE
        self._log_event(InterlockStatus.ACTIVE, message)

    def clear(self, message: str = "Manual clear"):
        """Clear interlock (if allowed)."""
        self._status = InterlockStatus.CLEAR
        self._log_event(InterlockStatus.CLEAR, message, SeverityLevel.INFO)

    def bypass(
        self,
        authorized_by: str,
        reason: str,
        duration_minutes: int = 60
    ) -> bool:
        """
        Bypass interlock (requires authorization).

        Bypassing interlocks is a serious action requiring
        physicist authorization and documentation.
        """
        if self.severity == SeverityLevel.EMERGENCY:
            # Cannot bypass emergency interlocks
            return False

        self._is_bypassed = True
        self._bypass_authorized_by = authorized_by
        self._bypass_reason = reason
        self._bypass_expiry = datetime.now()

        self._log_event(
            InterlockStatus.BYPASSED,
            f"Bypassed by {authorized_by}: {reason}",
            SeverityLevel.WARNING
        )
        return True

    def _log_event(
        self,
        status: InterlockStatus,
        message: str,
        severity: Optional[SeverityLevel] = None
    ):
        """Log interlock event."""
        event = InterlockEvent(
            interlock_type=self.interlock_type,
            status=status,
            timestamp=datetime.now(),
            message=message,
            severity=severity or self.severity
        )
        self._event_history.append(event)

    def get_history(self) -> List[InterlockEvent]:
        """Get interlock event history."""
        return self._event_history.copy()

    def get_status(self) -> Dict:
        """Get current interlock status."""
        return {
            "name": self.name,
            "type": self.interlock_type.value,
            "status": self._status.value,
            "is_bypassed": self._is_bypassed,
            "bypass_authorized_by": self._bypass_authorized_by,
            "severity": self.severity.value,
        }


class InterlockSystem:
    """
    Complete interlock system managing all safety interlocks.

    Coordinates multiple interlocks and provides system-wide
    beam control.
    """

    def __init__(self, machine_name: str):
        self.machine_name = machine_name

        self._interlocks: Dict[str, SafetyInterlock] = {}
        self._is_beam_enabled = False
        self._global_fault = False

        # Callbacks for beam control
        self._beam_hold_callback: Optional[Callable] = None
        self._beam_off_callback: Optional[Callable] = None

        self._initialize_standard_interlocks()

    def _initialize_standard_interlocks(self):
        """Initialize standard set of interlocks."""
        standard_interlocks = [
            ("DOOR", InterlockType.DOOR, SeverityLevel.CRITICAL),
            ("EMERGENCY_STOP", InterlockType.EMERGENCY, SeverityLevel.EMERGENCY),
            ("COLLISION", InterlockType.COLLISION, SeverityLevel.CRITICAL),
            ("DOSE_RATE", InterlockType.DOSE, SeverityLevel.CRITICAL),
            ("DOSE_1", InterlockType.DOSE, SeverityLevel.CRITICAL),
            ("DOSE_2", InterlockType.DOSE, SeverityLevel.CRITICAL),
            ("GANTRY_MOTION", InterlockType.MOTION, SeverityLevel.WARNING),
            ("COUCH_MOTION", InterlockType.MOTION, SeverityLevel.WARNING),
            ("MLC_MOTION", InterlockType.MOTION, SeverityLevel.WARNING),
            ("TEMPERATURE", InterlockType.TEMPERATURE, SeverityLevel.WARNING),
            ("VACUUM", InterlockType.VACUUM, SeverityLevel.CRITICAL),
            ("WATER_FLOW", InterlockType.WATER, SeverityLevel.CRITICAL),
        ]

        for name, itype, severity in standard_interlocks:
            self._interlocks[name] = SafetyInterlock(name, itype, severity)

    def add_interlock(self, interlock: SafetyInterlock):
        """Add custom interlock to system."""
        self._interlocks[interlock.name] = interlock

    def remove_interlock(self, name: str) -> bool:
        """Remove interlock (if not critical)."""
        if name in self._interlocks:
            if self._interlocks[name].severity == SeverityLevel.EMERGENCY:
                return False
            del self._interlocks[name]
            return True
        return False

    def check_all(self) -> Dict[str, InterlockStatus]:
        """Check all interlocks and return status."""
        results = {}
        beam_hold_required = False
        beam_off_required = False

        for name, interlock in self._interlocks.items():
            status = interlock.check()
            results[name] = status

            if status == InterlockStatus.ACTIVE:
                if interlock.severity in [SeverityLevel.CRITICAL, SeverityLevel.EMERGENCY]:
                    beam_off_required = True
                elif interlock.severity == SeverityLevel.WARNING:
                    beam_hold_required = True
            elif status == InterlockStatus.FAULT:
                beam_off_required = True
                self._global_fault = True

        # Execute beam control
        if beam_off_required and self._beam_off_callback:
            self._beam_off_callback()
            self._is_beam_enabled = False
        elif beam_hold_required and self._beam_hold_callback:
            self._beam_hold_callback()

        return results

    def is_beam_allowed(self) -> Tuple[bool, List[str]]:
        """
        Check if beam is allowed.

        Returns tuple of (allowed, list of blocking interlocks).
        """
        blocking = []

        for name, interlock in self._interlocks.items():
            status = interlock.check()
            if status == InterlockStatus.ACTIVE:
                if interlock.severity in [SeverityLevel.CRITICAL, SeverityLevel.EMERGENCY]:
                    blocking.append(name)
            elif status == InterlockStatus.FAULT:
                blocking.append(f"{name} (FAULT)")

        return len(blocking) == 0, blocking

    def enable_beam(self) -> bool:
        """Enable beam (if all interlocks clear)."""
        allowed, blocking = self.is_beam_allowed()
        if allowed:
            self._is_beam_enabled = True
            return True
        return False

    def disable_beam(self):
        """Disable beam."""
        self._is_beam_enabled = False

    def set_beam_callbacks(
        self,
        hold_callback: Callable,
        off_callback: Callable
    ):
        """Set callbacks for beam control."""
        self._beam_hold_callback = hold_callback
        self._beam_off_callback = off_callback

    def get_interlock(self, name: str) -> Optional[SafetyInterlock]:
        """Get specific interlock."""
        return self._interlocks.get(name)

    def get_status(self) -> Dict:
        """Get system status."""
        statuses = self.check_all()
        return {
            "machine": self.machine_name,
            "beam_enabled": self._is_beam_enabled,
            "global_fault": self._global_fault,
            "interlocks": {
                name: status.value for name, status in statuses.items()
            },
            "critical_active": sum(
                1 for name, status in statuses.items()
                if status == InterlockStatus.ACTIVE
                and self._interlocks[name].severity in [
                    SeverityLevel.CRITICAL, SeverityLevel.EMERGENCY
                ]
            ),
        }


class TimeoutMonitor:
    """
    Treatment timeout monitoring.

    Monitors treatment duration and terminates if
    maximum time exceeded.
    """

    def __init__(
        self,
        max_beam_on_time: float = 600.0,  # seconds
        max_treatment_time: float = 1800.0,  # seconds
        max_fraction_mu: float = 2000.0
    ):
        self.max_beam_on_time = max_beam_on_time
        self.max_treatment_time = max_treatment_time
        self.max_fraction_mu = max_fraction_mu

        self._treatment_start: Optional[datetime] = None
        self._beam_on_start: Optional[datetime] = None
        self._total_beam_on_time: float = 0.0
        self._total_mu_delivered: float = 0.0

        self._is_monitoring = False
        self._timeout_callback: Optional[Callable[[str], None]] = None

    def start_treatment(self):
        """Start treatment timing."""
        self._treatment_start = datetime.now()
        self._is_monitoring = True
        self._total_beam_on_time = 0.0
        self._total_mu_delivered = 0.0

    def stop_treatment(self):
        """Stop treatment timing."""
        self._is_monitoring = False
        self._treatment_start = None
        self.beam_off()

    def beam_on(self):
        """Record beam on time."""
        if self._is_monitoring:
            self._beam_on_start = datetime.now()

    def beam_off(self):
        """Record beam off and accumulate time."""
        if self._beam_on_start:
            elapsed = (datetime.now() - self._beam_on_start).total_seconds()
            self._total_beam_on_time += elapsed
            self._beam_on_start = None

    def add_mu(self, mu: float):
        """Add delivered MU."""
        self._total_mu_delivered += mu

    def check_limits(self) -> Tuple[bool, Optional[str]]:
        """
        Check if any limits exceeded.

        Returns (within_limits, reason_if_exceeded).
        """
        if not self._is_monitoring:
            return True, None

        # Check treatment time
        if self._treatment_start:
            treatment_time = (datetime.now() - self._treatment_start).total_seconds()
            if treatment_time > self.max_treatment_time:
                return False, f"Treatment time exceeded ({treatment_time:.0f}s > {self.max_treatment_time:.0f}s)"

        # Check beam on time
        current_beam_time = self._total_beam_on_time
        if self._beam_on_start:
            current_beam_time += (datetime.now() - self._beam_on_start).total_seconds()
        if current_beam_time > self.max_beam_on_time:
            return False, f"Beam-on time exceeded ({current_beam_time:.0f}s > {self.max_beam_on_time:.0f}s)"

        # Check MU
        if self._total_mu_delivered > self.max_fraction_mu:
            return False, f"MU limit exceeded ({self._total_mu_delivered:.1f} > {self.max_fraction_mu:.1f})"

        return True, None

    def set_timeout_callback(self, callback: Callable[[str], None]):
        """Set callback for timeout events."""
        self._timeout_callback = callback

    def get_status(self) -> Dict:
        """Get timing status."""
        treatment_time = 0.0
        if self._treatment_start:
            treatment_time = (datetime.now() - self._treatment_start).total_seconds()

        current_beam_time = self._total_beam_on_time
        if self._beam_on_start:
            current_beam_time += (datetime.now() - self._beam_on_start).total_seconds()

        return {
            "is_monitoring": self._is_monitoring,
            "treatment_time": treatment_time,
            "beam_on_time": current_beam_time,
            "total_mu": self._total_mu_delivered,
            "treatment_time_remaining": max(0, self.max_treatment_time - treatment_time),
            "beam_time_remaining": max(0, self.max_beam_on_time - current_beam_time),
            "mu_remaining": max(0, self.max_fraction_mu - self._total_mu_delivered),
        }


class DoseMonitor:
    """
    Real-time dose monitoring system.

    Uses redundant ion chambers to monitor dose
    and terminate if limits exceeded.
    """

    def __init__(
        self,
        dose_rate_limit: float = 600.0,  # MU/min
        dose_per_pulse_limit: float = 0.005,  # MU/pulse
        dose_tolerance: float = 3.0  # percent
    ):
        self.dose_rate_limit = dose_rate_limit
        self.dose_per_pulse_limit = dose_per_pulse_limit
        self.dose_tolerance = dose_tolerance

        # Dual ion chamber readings
        self._chamber_1_reading: float = 0.0
        self._chamber_2_reading: float = 0.0

        # Expected dose
        self._expected_dose: float = 0.0
        self._delivered_dose: float = 0.0

        # Dose rate monitoring
        self._current_dose_rate: float = 0.0
        self._dose_history: List[Tuple[datetime, float]] = []

        # Calibration factors
        self._chamber_1_cal: float = 1.0
        self._chamber_2_cal: float = 1.0

        self._terminate_callback: Optional[Callable[[str], None]] = None

    def set_expected_dose(self, dose: float):
        """Set expected dose for current beam."""
        self._expected_dose = dose
        self._delivered_dose = 0.0

    def update_readings(
        self,
        chamber_1: float,
        chamber_2: float,
        pulse_count: int = 1
    ) -> Tuple[bool, Optional[str]]:
        """
        Update chamber readings and check limits.

        Returns (ok, reason_if_not_ok).
        """
        self._chamber_1_reading = chamber_1 * self._chamber_1_cal
        self._chamber_2_reading = chamber_2 * self._chamber_2_cal

        # Check chamber agreement
        if self._chamber_1_reading > 0 and self._chamber_2_reading > 0:
            agreement = abs(
                self._chamber_1_reading - self._chamber_2_reading
            ) / self._chamber_1_reading * 100

            if agreement > 2.0:  # 2% disagreement
                return False, f"Chamber disagreement: {agreement:.1f}%"

        # Average dose
        avg_dose = (self._chamber_1_reading + self._chamber_2_reading) / 2
        self._delivered_dose += avg_dose

        # Update history for rate calculation
        self._dose_history.append((datetime.now(), avg_dose))

        # Keep only last 10 seconds
        cutoff = datetime.now()
        self._dose_history = [
            (t, d) for t, d in self._dose_history
            if (cutoff - t).total_seconds() < 10
        ]

        # Calculate dose rate
        if len(self._dose_history) >= 2:
            time_span = (
                self._dose_history[-1][0] - self._dose_history[0][0]
            ).total_seconds()
            if time_span > 0:
                total_dose = sum(d for _, d in self._dose_history)
                self._current_dose_rate = total_dose / time_span * 60  # MU/min

        # Check dose rate limit
        if self._current_dose_rate > self.dose_rate_limit:
            return False, f"Dose rate exceeded: {self._current_dose_rate:.1f} MU/min"

        # Check dose per pulse
        dose_per_pulse = avg_dose / max(1, pulse_count)
        if dose_per_pulse > self.dose_per_pulse_limit:
            return False, f"Dose per pulse exceeded: {dose_per_pulse:.5f} MU"

        # Check against expected dose
        if self._expected_dose > 0:
            overdose_limit = self._expected_dose * (1 + self.dose_tolerance / 100)
            if self._delivered_dose > overdose_limit:
                return False, f"Overdose: {self._delivered_dose:.1f} MU > {overdose_limit:.1f} MU"

        return True, None

    def set_calibration(self, chamber_1_cal: float, chamber_2_cal: float):
        """Set chamber calibration factors."""
        self._chamber_1_cal = chamber_1_cal
        self._chamber_2_cal = chamber_2_cal

    def set_terminate_callback(self, callback: Callable[[str], None]):
        """Set callback for dose termination."""
        self._terminate_callback = callback

    def get_status(self) -> Dict:
        """Get dose monitor status."""
        return {
            "chamber_1": self._chamber_1_reading,
            "chamber_2": self._chamber_2_reading,
            "delivered_dose": self._delivered_dose,
            "expected_dose": self._expected_dose,
            "current_dose_rate": self._current_dose_rate,
            "percent_delivered": (
                self._delivered_dose / self._expected_dose * 100
                if self._expected_dose > 0 else 0
            ),
        }

    def reset(self):
        """Reset dose monitor."""
        self._chamber_1_reading = 0.0
        self._chamber_2_reading = 0.0
        self._delivered_dose = 0.0
        self._expected_dose = 0.0
        self._dose_history.clear()


class EmergencyStop:
    """
    Emergency stop system.

    Manages emergency stop buttons and automatic
    emergency response.
    """

    def __init__(self):
        self._is_active = False
        self._activation_time: Optional[datetime] = None
        self._activation_reason: Optional[str] = None
        self._activation_location: Optional[str] = None

        self._stop_buttons: Dict[str, bool] = {
            "console": False,
            "door": False,
            "gantry": False,
            "couch": False,
            "maze": False,
        }

        self._callbacks: List[Callable[[], None]] = []
        self._event_log: List[Dict] = []

    def add_stop_button(self, location: str):
        """Add emergency stop button location."""
        self._stop_buttons[location] = False

    def activate(self, location: str, reason: str = "Manual activation"):
        """Activate emergency stop."""
        if not self._is_active:
            self._is_active = True
            self._activation_time = datetime.now()
            self._activation_reason = reason
            self._activation_location = location

            if location in self._stop_buttons:
                self._stop_buttons[location] = True

            self._log_event("ACTIVATED", location, reason)

            # Execute callbacks
            for callback in self._callbacks:
                try:
                    callback()
                except Exception:
                    pass

    def reset(self, authorized_by: str) -> bool:
        """
        Reset emergency stop.

        Requires all physical buttons to be reset first.
        """
        # Check all buttons are reset
        if any(self._stop_buttons.values()):
            return False

        self._is_active = False
        self._log_event("RESET", authorized_by, "Emergency stop reset")
        return True

    def reset_button(self, location: str) -> bool:
        """Reset specific button."""
        if location in self._stop_buttons:
            self._stop_buttons[location] = False
            return True
        return False

    def add_callback(self, callback: Callable[[], None]):
        """Add callback for emergency stop activation."""
        self._callbacks.append(callback)

    def _log_event(self, event_type: str, location: str, message: str):
        """Log emergency stop event."""
        self._event_log.append({
            "timestamp": datetime.now(),
            "type": event_type,
            "location": location,
            "message": message,
        })

    def is_active(self) -> bool:
        """Check if emergency stop is active."""
        return self._is_active

    def get_status(self) -> Dict:
        """Get emergency stop status."""
        return {
            "is_active": self._is_active,
            "activation_time": self._activation_time,
            "activation_reason": self._activation_reason,
            "activation_location": self._activation_location,
            "buttons": self._stop_buttons.copy(),
        }

    def get_event_log(self) -> List[Dict]:
        """Get emergency stop event log."""
        return self._event_log.copy()


class SurveyMeterType(Enum):
    """Types of radiation survey meters."""
    ION_CHAMBER = "ion_chamber"
    GM = "geiger_mueller"
    SCINTILLATOR = "scintillator"
    ELECTRONIC = "electronic_dosimeter"


@dataclass
class SurveyPoint:
    """Single radiation survey measurement point."""
    location: str
    description: str
    x: float  # meters from isocenter
    y: float
    z: float
    reading: float  # mR/hr or mSv/hr
    background: float
    net_reading: float
    occupancy_factor: float = 1.0
    shielding_adequate: bool = True


class RadiationSurvey:
    """
    Radiation survey management.

    Performs and documents radiation surveys for
    regulatory compliance and safety verification.
    """

    # Regulatory limits (example values)
    LIMITS = {
        "controlled_area": 5.0,  # mSv/week
        "uncontrolled_area": 0.1,  # mSv/week
        "public_area": 0.02,  # mSv/week
        "instantaneous_controlled": 2.0,  # mSv/hr
        "instantaneous_uncontrolled": 0.02,  # mSv/hr
    }

    WORKLOAD_ASSUMPTIONS = {
        "weekly_patients": 50,
        "fields_per_patient": 4,
        "mu_per_field": 250,
        "hours_per_week": 40,
    }

    def __init__(
        self,
        facility_name: str,
        room_name: str,
        survey_type: str = "routine"
    ):
        self.facility_name = facility_name
        self.room_name = room_name
        self.survey_type = survey_type

        self._survey_date: Optional[datetime] = None
        self._surveyor: Optional[str] = None
        self._meter_type: Optional[SurveyMeterType] = None
        self._meter_serial: Optional[str] = None
        self._meter_cal_date: Optional[datetime] = None

        self._survey_points: List[SurveyPoint] = []
        self._machine_settings: Dict = {}

        self._is_complete = False
        self._all_points_pass = True

    def start_survey(
        self,
        surveyor: str,
        meter_type: SurveyMeterType,
        meter_serial: str,
        meter_cal_date: datetime
    ):
        """Start radiation survey."""
        self._survey_date = datetime.now()
        self._surveyor = surveyor
        self._meter_type = meter_type
        self._meter_serial = meter_serial
        self._meter_cal_date = meter_cal_date

    def set_machine_settings(
        self,
        energy: str,
        field_size: Tuple[float, float],
        gantry_angle: float,
        mu_rate: float
    ):
        """Set machine settings for survey."""
        self._machine_settings = {
            "energy": energy,
            "field_size": field_size,
            "gantry_angle": gantry_angle,
            "mu_rate": mu_rate,
        }

    def add_measurement(
        self,
        location: str,
        description: str,
        position: Tuple[float, float, float],
        reading: float,
        background: float,
        area_type: str = "controlled_area",
        occupancy_factor: float = 1.0
    ) -> SurveyPoint:
        """Add survey measurement point."""
        net_reading = max(0, reading - background)

        # Check against limits
        limit = self.LIMITS.get(f"instantaneous_{area_type.replace('_area', '')}", 2.0)
        shielding_adequate = net_reading * occupancy_factor <= limit

        if not shielding_adequate:
            self._all_points_pass = False

        point = SurveyPoint(
            location=location,
            description=description,
            x=position[0],
            y=position[1],
            z=position[2],
            reading=reading,
            background=background,
            net_reading=net_reading,
            occupancy_factor=occupancy_factor,
            shielding_adequate=shielding_adequate,
        )

        self._survey_points.append(point)
        return point

    def calculate_weekly_dose(
        self,
        point: SurveyPoint,
        use_factor: float = 1.0
    ) -> float:
        """
        Calculate weekly dose at survey point.

        Accounts for workload, use factor, and occupancy.
        """
        # Weekly workload in MU
        weekly_mu = (
            self.WORKLOAD_ASSUMPTIONS["weekly_patients"] *
            self.WORKLOAD_ASSUMPTIONS["fields_per_patient"] *
            self.WORKLOAD_ASSUMPTIONS["mu_per_field"]
        )

        # Hours per week at maximum output
        hours_at_max = weekly_mu / (self._machine_settings.get("mu_rate", 400) * 60)

        # Weekly dose = instantaneous rate * hours * use * occupancy
        weekly_dose = (
            point.net_reading *
            hours_at_max *
            use_factor *
            point.occupancy_factor
        )

        return weekly_dose

    def complete_survey(self) -> bool:
        """Complete survey and verify all points pass."""
        self._is_complete = True
        return self._all_points_pass

    def generate_report(self) -> Dict:
        """Generate survey report."""
        return {
            "facility": self.facility_name,
            "room": self.room_name,
            "survey_type": self.survey_type,
            "date": self._survey_date,
            "surveyor": self._surveyor,
            "meter": {
                "type": self._meter_type.value if self._meter_type else None,
                "serial": self._meter_serial,
                "calibration_date": self._meter_cal_date,
            },
            "machine_settings": self._machine_settings,
            "measurements": [
                {
                    "location": p.location,
                    "description": p.description,
                    "position": (p.x, p.y, p.z),
                    "reading": p.reading,
                    "background": p.background,
                    "net": p.net_reading,
                    "occupancy_factor": p.occupancy_factor,
                    "pass": p.shielding_adequate,
                }
                for p in self._survey_points
            ],
            "all_pass": self._all_points_pass,
            "complete": self._is_complete,
        }

    def get_failed_points(self) -> List[SurveyPoint]:
        """Get list of failed measurement points."""
        return [p for p in self._survey_points if not p.shielding_adequate]


class CollisionDetection:
    """
    Collision detection and avoidance system.

    Monitors machine geometry to prevent collisions
    between gantry, couch, and patient.
    """

    def __init__(self):
        # Minimum clearances (mm)
        self._clearance_limits = {
            "gantry_couch": 20.0,
            "gantry_patient": 50.0,
            "couch_wall": 10.0,
            "accessory_patient": 30.0,
        }

        # Current positions
        self._gantry_angle: float = 0.0
        self._couch_angle: float = 0.0
        self._couch_position: Tuple[float, float, float] = (0, 0, 0)
        self._patient_contour: Optional[List] = None

        self._collision_zones: List[Dict] = []
        self._interlock_callback: Optional[Callable[[], None]] = None

    def set_positions(
        self,
        gantry_angle: float,
        couch_angle: float,
        couch_position: Tuple[float, float, float]
    ):
        """Update current positions."""
        self._gantry_angle = gantry_angle
        self._couch_angle = couch_angle
        self._couch_position = couch_position

    def set_patient_contour(self, contour: List):
        """Set patient body contour for collision checking."""
        self._patient_contour = contour

    def add_collision_zone(
        self,
        name: str,
        gantry_range: Tuple[float, float],
        couch_range: Tuple[float, float]
    ):
        """Add known collision zone."""
        self._collision_zones.append({
            "name": name,
            "gantry_range": gantry_range,
            "couch_range": couch_range,
        })

    def check_collision(
        self,
        target_gantry: float,
        target_couch: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if target position would cause collision.

        Returns (safe, collision_description).
        """
        # Check collision zones
        for zone in self._collision_zones:
            gantry_in_zone = (
                zone["gantry_range"][0] <= target_gantry <= zone["gantry_range"][1]
            )
            couch_in_zone = (
                zone["couch_range"][0] <= target_couch <= zone["couch_range"][1]
            )

            if gantry_in_zone and couch_in_zone:
                return False, f"Collision zone: {zone['name']}"

        # Simplified geometric check
        # In practice, this would use 3D models
        clearance = self._calculate_clearance(target_gantry, target_couch)

        if clearance < self._clearance_limits["gantry_couch"]:
            return False, f"Insufficient clearance: {clearance:.1f}mm"

        return True, None

    def _calculate_clearance(
        self,
        gantry_angle: float,
        couch_angle: float
    ) -> float:
        """Calculate minimum clearance (simplified)."""
        # This is a simplified model
        # Real implementation would use 3D geometry
        import math

        # Higher risk at lateral gantry angles with rotated couch
        gantry_rad = math.radians(gantry_angle)
        couch_rad = math.radians(couch_angle)

        # Simplified clearance model
        base_clearance = 500.0  # mm
        gantry_factor = 1.0 - 0.3 * abs(math.sin(gantry_rad))
        couch_factor = 1.0 - 0.2 * abs(math.sin(couch_rad))

        return base_clearance * gantry_factor * couch_factor

    def check_path(
        self,
        start_gantry: float,
        end_gantry: float,
        start_couch: float,
        end_couch: float,
        steps: int = 10
    ) -> Tuple[bool, Optional[str]]:
        """Check if motion path is collision-free."""
        for i in range(steps + 1):
            t = i / steps
            gantry = start_gantry + t * (end_gantry - start_gantry)
            couch = start_couch + t * (end_couch - start_couch)

            safe, reason = self.check_collision(gantry, couch)
            if not safe:
                return False, f"At gantry={gantry:.1f}, couch={couch:.1f}: {reason}"

        return True, None

    def set_interlock_callback(self, callback: Callable[[], None]):
        """Set callback for collision interlock."""
        self._interlock_callback = callback

    def get_status(self) -> Dict:
        """Get collision detection status."""
        safe, reason = self.check_collision(self._gantry_angle, self._couch_angle)
        return {
            "gantry_angle": self._gantry_angle,
            "couch_angle": self._couch_angle,
            "couch_position": self._couch_position,
            "is_safe": safe,
            "warning": reason,
            "clearance": self._calculate_clearance(self._gantry_angle, self._couch_angle),
        }
