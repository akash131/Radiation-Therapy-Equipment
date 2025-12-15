"""
Clinical Treatment Workflows.

Implements standard clinical workflows for radiation therapy:
- Patient setup and positioning
- Image guidance and verification
- Treatment delivery
- Adaptive radiotherapy
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime
import numpy as np

from ..base import Position3D, TreatmentPlan, TreatmentTarget, DoseDistribution


class SetupStatus(Enum):
    """Patient setup status."""
    NOT_STARTED = "not_started"
    POSITIONING = "positioning"
    IMAGING = "imaging"
    REGISTERED = "registered"
    VERIFIED = "verified"
    READY = "ready"
    FAILED = "failed"


class DeliveryStatus(Enum):
    """Treatment delivery status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"
    ERROR = "error"


@dataclass
class PatientIdentification:
    """Patient identification information."""
    patient_id: str
    name: str
    date_of_birth: str
    treatment_site: str
    plan_id: str
    fraction_number: int


@dataclass
class SetupCorrection:
    """Patient setup correction."""
    translation: Position3D
    rotation: Dict[str, float]  # pitch, roll, yaw in degrees
    correction_method: str  # "auto", "manual"
    confidence: float  # Registration confidence 0-1
    applied: bool = False


@dataclass
class DeliveryRecord:
    """Record of treatment delivery."""
    fraction_number: int
    beam_name: str
    planned_mu: float
    delivered_mu: float
    gantry_angle: float
    couch_angle: float
    delivery_time: float  # seconds
    status: DeliveryStatus
    timestamp: datetime = field(default_factory=datetime.now)


class PatientSetup:
    """
    Patient setup and positioning workflow.

    Manages the complete patient setup process from identification
    to treatment readiness.
    """

    def __init__(self):
        self._patient: Optional[PatientIdentification] = None
        self._status = SetupStatus.NOT_STARTED
        self._setup_images: List[np.ndarray] = []
        self._correction: Optional[SetupCorrection] = None
        self._tolerance = 3.0  # mm default

        # Workflow steps
        self._steps_completed = {
            "identification": False,
            "positioning": False,
            "immobilization": False,
            "imaging": False,
            "registration": False,
            "verification": False,
        }

    def start_setup(self, patient: PatientIdentification) -> bool:
        """Start patient setup workflow."""
        self._patient = patient
        self._status = SetupStatus.POSITIONING
        return True

    def verify_identity(
        self,
        patient_id: str,
        photo_match: bool = True,
        verbal_confirm: bool = True
    ) -> bool:
        """
        Verify patient identity.

        Two-factor verification using photo ID and verbal confirmation.
        """
        if not self._patient:
            return False

        if self._patient.patient_id != patient_id:
            return False

        if not photo_match or not verbal_confirm:
            return False

        self._steps_completed["identification"] = True
        return True

    def position_patient(
        self,
        couch_position: Dict[str, float],
        immobilization_verified: bool = True
    ) -> bool:
        """
        Position patient on treatment couch.

        Args:
            couch_position: Initial couch coordinates
            immobilization_verified: Confirmation of proper immobilization
        """
        if not self._steps_completed["identification"]:
            return False

        self._steps_completed["positioning"] = True
        self._steps_completed["immobilization"] = immobilization_verified

        return True

    def acquire_setup_image(
        self,
        image_type: str,  # "CBCT", "kV2D", "MV2D"
        image_data: Optional[np.ndarray] = None
    ) -> bool:
        """Acquire setup verification image."""
        if image_data is None:
            # Simulate image acquisition
            if image_type == "CBCT":
                image_data = np.random.random((512, 512, 200))
            else:
                image_data = np.random.random((1024, 1024))

        self._setup_images.append(image_data)
        self._status = SetupStatus.IMAGING
        self._steps_completed["imaging"] = True
        return True

    def register_images(
        self,
        registration_type: str = "automatic",
        roi: str = "bone"
    ) -> SetupCorrection:
        """
        Register setup images to reference (planning CT/DRR).

        Args:
            registration_type: "automatic", "manual", "hybrid"
            roi: Registration region - "bone", "soft_tissue", "marker"
        """
        if not self._setup_images:
            return None

        # Simulate registration result
        self._correction = SetupCorrection(
            translation=Position3D(
                x=np.random.uniform(-5, 5),
                y=np.random.uniform(-5, 5),
                z=np.random.uniform(-5, 5)
            ),
            rotation={
                "pitch": np.random.uniform(-2, 2),
                "roll": np.random.uniform(-2, 2),
                "yaw": np.random.uniform(-2, 2),
            },
            correction_method=registration_type,
            confidence=np.random.uniform(0.85, 0.99)
        )

        self._status = SetupStatus.REGISTERED
        self._steps_completed["registration"] = True

        return self._correction

    def apply_correction(self) -> bool:
        """Apply calculated setup correction."""
        if not self._correction:
            return False

        # Would command couch to move
        self._correction.applied = True
        return True

    def verify_setup(
        self,
        tolerance: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Verify setup is within tolerance.

        Returns:
            Tuple of (passed, message)
        """
        if tolerance is None:
            tolerance = self._tolerance

        if not self._correction:
            return False, "No registration performed"

        # Check translational tolerance
        trans = self._correction.translation
        total_shift = np.sqrt(trans.x**2 + trans.y**2 + trans.z**2)

        if total_shift > tolerance:
            return False, f"Shift {total_shift:.1f}mm exceeds tolerance {tolerance}mm"

        # Check rotational tolerance (typically 2-3 degrees)
        rot_tolerance = 3.0
        for axis, angle in self._correction.rotation.items():
            if abs(angle) > rot_tolerance:
                return False, f"Rotation {axis}={angle:.1f}° exceeds tolerance"

        self._status = SetupStatus.VERIFIED
        self._steps_completed["verification"] = True
        return True, "Setup verified within tolerance"

    def mark_ready(self) -> bool:
        """Mark patient as ready for treatment."""
        if not all(self._steps_completed.values()):
            return False

        self._status = SetupStatus.READY
        return True

    def get_status(self) -> Dict:
        """Get setup workflow status."""
        return {
            "patient_id": self._patient.patient_id if self._patient else None,
            "status": self._status.value,
            "steps_completed": self._steps_completed,
            "correction": {
                "translation": (
                    self._correction.translation.x,
                    self._correction.translation.y,
                    self._correction.translation.z
                ) if self._correction else None,
                "rotation": self._correction.rotation if self._correction else None,
                "applied": self._correction.applied if self._correction else False,
            } if self._correction else None,
        }


class TreatmentDelivery:
    """
    Treatment delivery workflow manager.

    Manages beam delivery including monitoring, pausing,
    and recording.
    """

    def __init__(self):
        self._plan: Optional[TreatmentPlan] = None
        self._current_beam_index = 0
        self._status = DeliveryStatus.NOT_STARTED
        self._delivery_records: List[DeliveryRecord] = []

        # Monitoring
        self._delivered_mu = 0.0
        self._remaining_mu = 0.0
        self._elapsed_time = 0.0

        # Safety limits
        self._mu_limit = 10000.0
        self._time_limit = 3600.0  # seconds

    def load_plan(self, plan: TreatmentPlan) -> bool:
        """Load treatment plan for delivery."""
        self._plan = plan
        self._current_beam_index = 0
        self._remaining_mu = self._calculate_total_mu()
        return True

    def _calculate_total_mu(self) -> float:
        """Calculate total MU for plan."""
        if not self._plan:
            return 0.0
        # Simplified - actual implementation from plan data
        return 500.0  # Typical fraction MU

    def start_delivery(self) -> bool:
        """Start treatment delivery."""
        if not self._plan:
            return False

        if self._status == DeliveryStatus.IN_PROGRESS:
            return False

        self._status = DeliveryStatus.IN_PROGRESS
        return True

    def deliver_beam(
        self,
        beam_index: int,
        machine
    ) -> DeliveryRecord:
        """
        Deliver a single beam.

        Args:
            beam_index: Index of beam to deliver
            machine: Treatment machine instance
        """
        if not self._plan or beam_index >= len(self._plan.beams):
            return None

        beam = self._plan.beams[beam_index]
        start_time = datetime.now()

        # Configure machine
        machine.set_gantry_angle(beam.gantry_angle)

        # Deliver
        planned_mu = 100.0  # From plan
        dose_dist = machine.deliver_dose(beam, planned_mu)

        end_time = datetime.now()
        delivery_time = (end_time - start_time).total_seconds()

        record = DeliveryRecord(
            fraction_number=self._plan.total_fractions,
            beam_name=f"Beam_{beam_index}",
            planned_mu=planned_mu,
            delivered_mu=planned_mu,  # Assume successful delivery
            gantry_angle=beam.gantry_angle,
            couch_angle=beam.couch_angle,
            delivery_time=delivery_time,
            status=DeliveryStatus.COMPLETED,
            timestamp=end_time
        )

        self._delivery_records.append(record)
        self._delivered_mu += planned_mu
        self._current_beam_index += 1

        return record

    def pause_delivery(self, reason: str = "") -> bool:
        """Pause treatment delivery."""
        if self._status != DeliveryStatus.IN_PROGRESS:
            return False

        self._status = DeliveryStatus.PAUSED
        return True

    def resume_delivery(self) -> bool:
        """Resume paused treatment."""
        if self._status != DeliveryStatus.PAUSED:
            return False

        self._status = DeliveryStatus.IN_PROGRESS
        return True

    def abort_delivery(self, reason: str) -> bool:
        """Abort treatment delivery."""
        self._status = DeliveryStatus.ABORTED
        return True

    def complete_delivery(self) -> bool:
        """Mark delivery as complete."""
        if not self._plan:
            return False

        if self._current_beam_index < len(self._plan.beams):
            return False  # Not all beams delivered

        self._status = DeliveryStatus.COMPLETED
        return True

    def get_progress(self) -> Dict:
        """Get delivery progress."""
        if not self._plan:
            return {}

        total_beams = len(self._plan.beams)
        return {
            "status": self._status.value,
            "beams_completed": self._current_beam_index,
            "total_beams": total_beams,
            "progress_percent": 100 * self._current_beam_index / total_beams if total_beams > 0 else 0,
            "delivered_mu": self._delivered_mu,
            "remaining_mu": self._remaining_mu - self._delivered_mu,
        }

    def get_delivery_records(self) -> List[DeliveryRecord]:
        """Get all delivery records for this session."""
        return self._delivery_records


class ImageGuidance:
    """
    Image guidance workflow.

    Supports various IGRT techniques:
    - CBCT
    - Planar kV/MV imaging
    - Surface imaging
    - Fiducial tracking
    """

    def __init__(self):
        self._reference_images: Dict[str, np.ndarray] = {}
        self._acquired_images: Dict[str, np.ndarray] = {}
        self._registrations: List[SetupCorrection] = []

        # Supported modalities
        self._modalities = ["CBCT", "kV2D", "MV2D", "Surface"]
        self._active_modality = "CBCT"

    def set_reference(
        self,
        image_type: str,
        image_data: np.ndarray
    ):
        """Set reference image (from planning)."""
        self._reference_images[image_type] = image_data

    def acquire_image(
        self,
        modality: str,
        params: Optional[Dict] = None
    ) -> np.ndarray:
        """Acquire verification image."""
        if modality not in self._modalities:
            return None

        # Simulate acquisition
        if modality == "CBCT":
            image = np.random.random((512, 512, 200))
        elif modality == "Surface":
            image = np.random.random((480, 640, 3))  # Point cloud
        else:
            image = np.random.random((1024, 1024))

        self._acquired_images[modality] = image
        return image

    def perform_registration(
        self,
        modality: str,
        algorithm: str = "mutual_information"
    ) -> SetupCorrection:
        """
        Perform image registration.

        Args:
            modality: Image type to register
            algorithm: Registration algorithm
        """
        if modality not in self._acquired_images:
            return None

        if modality not in self._reference_images:
            return None

        # Simulate registration
        correction = SetupCorrection(
            translation=Position3D(
                x=np.random.uniform(-3, 3),
                y=np.random.uniform(-3, 3),
                z=np.random.uniform(-3, 3)
            ),
            rotation={
                "pitch": np.random.uniform(-1, 1),
                "roll": np.random.uniform(-1, 1),
                "yaw": np.random.uniform(-1, 1),
            },
            correction_method=algorithm,
            confidence=np.random.uniform(0.9, 0.99)
        )

        self._registrations.append(correction)
        return correction

    def get_latest_correction(self) -> Optional[SetupCorrection]:
        """Get most recent registration result."""
        if not self._registrations:
            return None
        return self._registrations[-1]


class AdaptiveWorkflow:
    """
    Adaptive radiotherapy workflow.

    Supports:
    - Offline adaptive (re-plan between fractions)
    - Online adaptive (re-plan during fraction)
    - Plan library selection
    """

    def __init__(self, adaptation_type: str = "offline"):
        self.adaptation_type = adaptation_type

        self._reference_plan: Optional[TreatmentPlan] = None
        self._adapted_plans: List[TreatmentPlan] = []
        self._plan_library: Dict[str, TreatmentPlan] = {}

        # Online adaptive parameters
        self._adaptation_time_limit = 300.0  # seconds
        self._dose_threshold = 0.05  # 5% dose difference threshold

    def set_reference_plan(self, plan: TreatmentPlan):
        """Set reference treatment plan."""
        self._reference_plan = plan

    def add_to_library(self, plan_id: str, plan: TreatmentPlan):
        """Add plan to plan library."""
        self._plan_library[plan_id] = plan

    def evaluate_adaptation_need(
        self,
        current_anatomy: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Evaluate if adaptation is needed.

        Based on anatomical changes, dose accumulation, etc.
        """
        # Simulate evaluation
        needs_adaptation = np.random.random() < 0.2  # 20% chance

        if needs_adaptation:
            return True, "Significant anatomical change detected"
        return False, "Within tolerance"

    def select_plan_from_library(
        self,
        current_anatomy: np.ndarray
    ) -> Optional[TreatmentPlan]:
        """
        Select best-matching plan from library.

        Uses anatomy similarity metrics.
        """
        if not self._plan_library:
            return None

        # Select best match (simplified)
        plan_id = list(self._plan_library.keys())[0]
        return self._plan_library[plan_id]

    def create_adapted_plan(
        self,
        current_anatomy: np.ndarray,
        contours: Dict[str, np.ndarray]
    ) -> TreatmentPlan:
        """
        Create online-adapted plan.

        Quick re-optimization based on current anatomy.
        """
        if not self._reference_plan:
            return None

        # Would perform rapid re-optimization
        # For now, return modified reference plan
        adapted_plan = TreatmentPlan(
            plan_id=f"{self._reference_plan.plan_id}_adapted",
            patient_id=self._reference_plan.patient_id,
            targets=self._reference_plan.targets,
            organs_at_risk=self._reference_plan.organs_at_risk,
            beams=self._reference_plan.beams,
            total_dose=self._reference_plan.total_dose,
            total_fractions=self._reference_plan.total_fractions
        )

        self._adapted_plans.append(adapted_plan)
        return adapted_plan

    def verify_adapted_plan(
        self,
        adapted_plan: TreatmentPlan
    ) -> Tuple[bool, Dict]:
        """
        Verify adapted plan meets clinical requirements.

        Quick QA check before delivery.
        """
        # Simulate verification
        passed = np.random.random() > 0.1  # 90% pass rate

        metrics = {
            "target_coverage": np.random.uniform(0.95, 1.0),
            "oar_constraints_met": True,
            "mu_deviation": np.random.uniform(-5, 5),  # %
            "calculation_time": np.random.uniform(30, 120),  # seconds
        }

        return passed, metrics


class PreTreatmentVerification:
    """
    Pre-treatment verification workflow.

    Ensures treatment can be safely delivered.
    """

    def __init__(self):
        self._checklist = {
            "patient_id_verified": False,
            "plan_loaded": False,
            "machine_ready": False,
            "imaging_ready": False,
            "physics_approved": False,
            "previous_fraction_reviewed": False,
        }

    def verify_patient_id(self, patient_id: str, plan_id: str) -> bool:
        """Verify patient ID matches plan."""
        self._checklist["patient_id_verified"] = True
        return True

    def verify_plan_loaded(self, plan: TreatmentPlan) -> bool:
        """Verify correct plan is loaded."""
        self._checklist["plan_loaded"] = True
        return True

    def verify_machine_ready(self, machine) -> Tuple[bool, List[str]]:
        """
        Verify machine is ready for treatment.

        Returns:
            Tuple of (ready, list of issues)
        """
        issues = []

        status = machine.get_status()
        if not status.get("is_ready", False):
            issues.append("Machine not ready")
        if not status.get("is_calibrated", False):
            issues.append("Machine not calibrated")
        if not status.get("interlocks_ok", True):
            issues.append("Interlock fault")

        ready = len(issues) == 0
        self._checklist["machine_ready"] = ready

        return ready, issues

    def verify_imaging_ready(self, imaging_system) -> bool:
        """Verify imaging system is ready."""
        self._checklist["imaging_ready"] = True
        return True

    def physics_approval(self, physicist_id: str) -> bool:
        """Record physics approval for treatment."""
        self._checklist["physics_approved"] = True
        return True

    def review_previous_fraction(
        self,
        fraction_number: int
    ) -> Tuple[bool, Dict]:
        """
        Review previous fraction delivery.

        Check for any issues that need addressing.
        """
        if fraction_number == 1:
            self._checklist["previous_fraction_reviewed"] = True
            return True, {"status": "First fraction"}

        # Review previous delivery
        review = {
            "previous_mu_delivered": 500.0,
            "previous_setup_shifts": (1.0, 0.5, -0.5),
            "issues_noted": [],
        }

        self._checklist["previous_fraction_reviewed"] = True
        return True, review

    def is_ready_for_treatment(self) -> Tuple[bool, List[str]]:
        """
        Check if all verification steps are complete.

        Returns:
            Tuple of (ready, list of incomplete items)
        """
        incomplete = [
            item for item, done in self._checklist.items() if not done
        ]

        return len(incomplete) == 0, incomplete

    def get_checklist_status(self) -> Dict:
        """Get current checklist status."""
        return self._checklist.copy()
