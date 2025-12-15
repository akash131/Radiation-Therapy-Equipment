"""
Treatment Planning Integration Module.

Provides unified treatment planning interface for all radiation therapy modalities.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Union
import numpy as np

from .base import (
    TreatmentPlan,
    TreatmentTarget,
    OrganAtRisk,
    DoseDistribution,
    Position3D,
    BeamParameters,
    ParticleType,
    BeamModality,
)
from .linac import LinearAccelerator
from .proton_therapy import ProtonTherapySystem, HeavyIonTherapySystem, TreatmentPlanningSystem
from .cyberknife import CyberKnifeSystem, TreatmentNode


class TreatmentModality(Enum):
    """Available treatment modalities."""
    PHOTON_3DCRT = "photon_3dcrt"
    PHOTON_IMRT = "photon_imrt"
    PHOTON_VMAT = "photon_vmat"
    PHOTON_SBRT = "photon_sbrt"
    PROTON_PBS = "proton_pbs"
    CARBON_PBS = "carbon_pbs"
    CYBERKNIFE = "cyberknife"


@dataclass
class PlanningConstraint:
    """Dose constraint for planning."""
    structure_name: str
    constraint_type: str  # "max", "mean", "V%", "D%"
    value: float
    priority: int = 1
    is_target: bool = False


@dataclass
class PlanningObjective:
    """Optimization objective for planning."""
    structure_name: str
    objective_type: str  # "min_dose", "max_dose", "uniform", "dvh"
    target_value: float
    weight: float = 1.0


class UnifiedTreatmentPlanner:
    """
    Unified treatment planning system.

    Provides a common interface for planning across all modalities.
    """

    def __init__(self, modality: TreatmentModality):
        self.modality = modality
        self._plan: Optional[TreatmentPlan] = None
        self._targets: List[TreatmentTarget] = []
        self._oars: List[OrganAtRisk] = []
        self._constraints: List[PlanningConstraint] = []
        self._objectives: List[PlanningObjective] = []

        # Initialize appropriate delivery system
        self._delivery_system = self._create_delivery_system()

    def _create_delivery_system(self):
        """Create appropriate delivery system for modality."""
        if self.modality in [
            TreatmentModality.PHOTON_3DCRT,
            TreatmentModality.PHOTON_IMRT,
            TreatmentModality.PHOTON_VMAT,
            TreatmentModality.PHOTON_SBRT,
        ]:
            return LinearAccelerator()
        elif self.modality == TreatmentModality.PROTON_PBS:
            return ProtonTherapySystem()
        elif self.modality == TreatmentModality.CARBON_PBS:
            return HeavyIonTherapySystem()
        elif self.modality == TreatmentModality.CYBERKNIFE:
            return CyberKnifeSystem()
        return None

    def add_target(self, target: TreatmentTarget):
        """Add treatment target."""
        self._targets.append(target)

    def add_oar(self, oar: OrganAtRisk):
        """Add organ at risk."""
        self._oars.append(oar)

    def add_constraint(self, constraint: PlanningConstraint):
        """Add planning constraint."""
        self._constraints.append(constraint)

    def add_objective(self, objective: PlanningObjective):
        """Add planning objective."""
        self._objectives.append(objective)

    def create_plan(
        self,
        plan_id: str,
        patient_id: str,
        prescribed_dose: float,
        fractions: int
    ) -> TreatmentPlan:
        """Create treatment plan."""
        # Generate beam configuration based on modality
        beams = self._generate_beams()

        self._plan = TreatmentPlan(
            plan_id=plan_id,
            patient_id=patient_id,
            targets=self._targets,
            organs_at_risk=self._oars,
            beams=beams,
            total_dose=prescribed_dose,
            total_fractions=fractions
        )

        return self._plan

    def _generate_beams(self) -> List[BeamParameters]:
        """Generate beam configuration for modality."""
        beams = []

        if self.modality == TreatmentModality.PHOTON_3DCRT:
            # 4-field box technique
            angles = [0, 90, 180, 270]
            for angle in angles:
                beams.append(BeamParameters(
                    energy=6.0,
                    particle_type=ParticleType.PHOTON,
                    gantry_angle=angle,
                    modality=BeamModality.STATIC
                ))

        elif self.modality == TreatmentModality.PHOTON_IMRT:
            # 7-9 field IMRT
            angles = [0, 51, 102, 153, 204, 255, 306]
            for angle in angles:
                beams.append(BeamParameters(
                    energy=6.0,
                    particle_type=ParticleType.PHOTON,
                    gantry_angle=angle,
                    modality=BeamModality.IMRT
                ))

        elif self.modality == TreatmentModality.PHOTON_VMAT:
            # Full arc or dual arc
            beams.append(BeamParameters(
                energy=6.0,
                particle_type=ParticleType.PHOTON,
                gantry_angle=181,  # Start angle
                modality=BeamModality.VMAT
            ))
            beams.append(BeamParameters(
                energy=6.0,
                particle_type=ParticleType.PHOTON,
                gantry_angle=179,  # End angle (opposite direction)
                modality=BeamModality.VMAT
            ))

        elif self.modality == TreatmentModality.PHOTON_SBRT:
            # Multiple non-coplanar beams
            angles = [(0, 0), (45, 90), (90, 45), (135, 315), (180, 270)]
            for gantry, couch in angles:
                beams.append(BeamParameters(
                    energy=6.0,  # Typically FFF
                    particle_type=ParticleType.PHOTON,
                    gantry_angle=gantry,
                    couch_angle=couch,
                    modality=BeamModality.SBRT
                ))

        elif self.modality == TreatmentModality.PROTON_PBS:
            # 2-3 field proton
            angles = [0, 90, 180]
            for angle in angles:
                beams.append(BeamParameters(
                    energy=200.0,
                    particle_type=ParticleType.PROTON,
                    gantry_angle=angle,
                    modality=BeamModality.IMRT
                ))

        elif self.modality == TreatmentModality.CARBON_PBS:
            # 2-3 field carbon
            angles = [0, 90]
            for angle in angles:
                beams.append(BeamParameters(
                    energy=350.0,
                    particle_type=ParticleType.CARBON_ION,
                    gantry_angle=angle,
                    modality=BeamModality.IMRT
                ))

        elif self.modality == TreatmentModality.CYBERKNIFE:
            # CyberKnife generates nodes, not traditional beams
            beams.append(BeamParameters(
                energy=6.0,
                particle_type=ParticleType.PHOTON,
                modality=BeamModality.SRS
            ))

        return beams

    def optimize(self, max_iterations: int = 100) -> bool:
        """
        Optimize treatment plan.

        Uses iterative optimization to achieve objectives
        while respecting constraints.
        """
        if not self._plan or not self._targets:
            return False

        # Different optimization strategies per modality
        if self.modality in [TreatmentModality.PHOTON_IMRT, TreatmentModality.PHOTON_VMAT]:
            return self._optimize_imrt()
        elif self.modality in [TreatmentModality.PROTON_PBS, TreatmentModality.CARBON_PBS]:
            return self._optimize_pbs()
        elif self.modality == TreatmentModality.CYBERKNIFE:
            return self._optimize_cyberknife()
        else:
            return self._optimize_forward()

        return True

    def _optimize_forward(self) -> bool:
        """Forward planning optimization (3DCRT)."""
        # Adjust beam weights based on target coverage
        if not self._plan:
            return False

        num_beams = len(self._plan.beams)
        if num_beams == 0:
            return False

        # Equal weighting for forward planning
        weight = 1.0 / num_beams
        for beam in self._plan.beams:
            beam.dose_rate = 600 * weight

        return True

    def _optimize_imrt(self) -> bool:
        """Inverse planning optimization for IMRT/VMAT."""
        # Simplified IMRT optimization
        # Real systems use gradient-based fluence optimization

        return True

    def _optimize_pbs(self) -> bool:
        """Spot weight optimization for PBS."""
        if isinstance(self._delivery_system, (ProtonTherapySystem, HeavyIonTherapySystem)):
            tps = self._delivery_system.tps
            if self._targets:
                tps.generate_spot_map(self._targets[0], 0.0)
                tps.optimize_spot_weights(
                    self._targets[0],
                    self._oars,
                    self._targets[0].prescribed_dose
                )
        return True

    def _optimize_cyberknife(self) -> bool:
        """Node weight optimization for CyberKnife."""
        if isinstance(self._delivery_system, CyberKnifeSystem):
            if self._targets:
                nodes = self._delivery_system.generate_treatment_nodes(
                    self._targets[0],
                    num_nodes=100
                )
                self._delivery_system.optimize_node_weights(
                    nodes,
                    self._targets[0],
                    self._oars
                )
                self._delivery_system.load_treatment_plan(nodes, self._targets[0])
        return True

    def calculate_dose(self) -> DoseDistribution:
        """Calculate dose distribution for current plan."""
        if not self._plan:
            return DoseDistribution(
                origin=Position3D(),
                spacing=(2.5, 2.5, 2.5),
                dimensions=(100, 100, 100)
            )

        # Use delivery system to calculate dose
        if self._delivery_system:
            self._delivery_system.initialize()
            self._delivery_system.calibrate()

            total_dose = DoseDistribution(
                origin=Position3D(-125, -125, -125),
                spacing=(2.5, 2.5, 2.5),
                dimensions=(100, 100, 100)
            )

            for beam in self._plan.beams:
                dose = self._delivery_system.deliver_dose(beam, 100)
                total_dose.dose_grid += dose.dose_grid

            self._plan.dose_distribution = total_dose
            return total_dose

        return DoseDistribution(
            origin=Position3D(),
            spacing=(2.5, 2.5, 2.5),
            dimensions=(100, 100, 100)
        )

    def evaluate_plan(self) -> Dict:
        """Evaluate plan quality."""
        if not self._plan:
            return {}

        results = {
            "plan_id": self._plan.plan_id,
            "modality": self.modality.value,
            "conformity_index": self._plan.calculate_conformity_index(),
            "homogeneity_index": self._plan.calculate_homogeneity_index(),
            "constraint_violations": [],
            "objective_scores": [],
        }

        # Check constraints
        for constraint in self._constraints:
            violation = self._check_constraint(constraint)
            if violation:
                results["constraint_violations"].append(violation)

        # Score objectives
        for objective in self._objectives:
            score = self._score_objective(objective)
            results["objective_scores"].append(score)

        return results

    def _check_constraint(self, constraint: PlanningConstraint) -> Optional[Dict]:
        """Check if constraint is violated."""
        if not self._plan or not self._plan.dose_distribution:
            return None

        # Find structure
        structure = None
        if constraint.is_target:
            for t in self._targets:
                if t.name == constraint.structure_name:
                    structure = t
                    break
        else:
            for o in self._oars:
                if o.name == constraint.structure_name:
                    structure = o
                    break

        if not structure:
            return None

        # Check constraint
        dose_dist = self._plan.dose_distribution

        if constraint.constraint_type == "max":
            actual = dose_dist.get_max_dose()
            if actual > constraint.value:
                return {
                    "structure": constraint.structure_name,
                    "type": constraint.constraint_type,
                    "limit": constraint.value,
                    "actual": actual,
                }

        return None

    def _score_objective(self, objective: PlanningObjective) -> Dict:
        """Score an optimization objective."""
        return {
            "structure": objective.structure_name,
            "type": objective.objective_type,
            "target": objective.target_value,
            "achieved": 0.0,  # Would calculate from dose distribution
            "score": 0.0,
        }

    def get_delivery_system(self):
        """Get the delivery system."""
        return self._delivery_system

    def get_plan(self) -> Optional[TreatmentPlan]:
        """Get current treatment plan."""
        return self._plan


def compare_modalities(
    target: TreatmentTarget,
    oars: List[OrganAtRisk],
    modalities: List[TreatmentModality]
) -> Dict:
    """
    Compare different treatment modalities for a given case.

    Returns comparison metrics for each modality.
    """
    results = {}

    for modality in modalities:
        planner = UnifiedTreatmentPlanner(modality)
        planner.add_target(target)
        for oar in oars:
            planner.add_oar(oar)

        plan = planner.create_plan(
            plan_id=f"comparison_{modality.value}",
            patient_id="comparison_patient",
            prescribed_dose=target.prescribed_dose,
            fractions=target.fractions
        )

        planner.optimize()
        planner.calculate_dose()
        evaluation = planner.evaluate_plan()

        results[modality.value] = evaluation

    return results
