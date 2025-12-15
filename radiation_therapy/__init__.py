"""
Radiation Therapy Equipment Simulation Package

This package provides comprehensive simulation and control systems for:
- Linear Accelerators (LINACs)
- Proton/Heavy Ion Therapy Systems
- CyberKnife-Type Robotic Systems
"""

from .base import (
    RadiationSource,
    BeamParameters,
    TreatmentTarget,
    DoseDistribution,
    Position3D,
    Vector3D,
)
from .linac import (
    ElectronGun,
    AcceleratingWaveguide,
    BendingMagnet,
    BeamSteering,
    MultiLeafCollimator,
    TumorTracker,
    LinearAccelerator,
)
from .proton_therapy import (
    Cyclotron,
    Synchrotron,
    BeamTransport,
    GantrySystem,
    BraggPeakOptimizer,
    TreatmentPlanningSystem,
    ProtonTherapySystem,
    HeavyIonTherapySystem,
)
from .cyberknife import (
    RoboticArm,
    MotionCompensation,
    ImageGuidedTargeting,
    CyberKnifeSystem,
)
from .planning import (
    TreatmentModality,
    PlanningConstraint,
    PlanningObjective,
    UnifiedTreatmentPlanner,
    compare_modalities,
)

__version__ = "1.0.0"
__all__ = [
    # Base classes
    "RadiationSource",
    "BeamParameters",
    "TreatmentTarget",
    "DoseDistribution",
    "Position3D",
    "Vector3D",
    # LINAC components
    "ElectronGun",
    "AcceleratingWaveguide",
    "BendingMagnet",
    "BeamSteering",
    "MultiLeafCollimator",
    "TumorTracker",
    "LinearAccelerator",
    # Proton/Heavy Ion Therapy
    "Cyclotron",
    "Synchrotron",
    "BeamTransport",
    "GantrySystem",
    "BraggPeakOptimizer",
    "TreatmentPlanningSystem",
    "ProtonTherapySystem",
    "HeavyIonTherapySystem",
    # CyberKnife
    "RoboticArm",
    "MotionCompensation",
    "ImageGuidedTargeting",
    "CyberKnifeSystem",
    # Planning
    "TreatmentModality",
    "PlanningConstraint",
    "PlanningObjective",
    "UnifiedTreatmentPlanner",
    "compare_modalities",
]
