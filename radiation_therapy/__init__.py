"""
Radiation Therapy Equipment Simulation Package

This package provides comprehensive simulation and control systems for:
- Linear Accelerators (LINACs) with vendor-specific implementations
- Proton/Heavy Ion Therapy Systems
- CyberKnife-Type Robotic Systems
- MR-Linac and Adaptive Radiotherapy
- Hardware components (detectors, positioning, QA equipment)
- Clinical workflows and QA procedures
- DICOM RT data format support
- Emerging technologies (FLASH, AI planning)
"""

from .base import (
    RadiationSource,
    BeamParameters,
    TreatmentTarget,
    DoseDistribution,
    Position3D,
    Vector3D,
    OrganAtRisk,
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

# Vendor-specific implementations
from .vendors import (
    # Varian
    VarianTrueBeam,
    VarianHalcyon,
    VarianClinac,
    VarianEdge,
    VarianProBeam,
    # Elekta
    ElektaVersaHD,
    ElektaInfinity,
    ElektaUnity,
    # Siemens
    SiemensArtiste,
    SiemensOncor,
    # Particle therapy vendors
    IBAProteusPlus,
    IBAProteusOne,
    HitachiPROBEAT,
    MevionS250i,
    # Accuray
    AccurayCyberKnifeM6,
    AccurayCyberKnifeS7,
    AccurayTomoTherapy,
    AccurayRadixact,
)

# Hardware components
from .hardware import (
    # Detectors
    IonChamber,
    FarmerChamber,
    PinPointChamber,
    DiodeDosimeter,
    FilmDosimeter,
    MatriXX,
    ArcCHECK,
    Delta4,
    PortalImager,
    # Positioning
    TreatmentCouch,
    SixDOFCouch,
    HexaPOD,
    HeadAndNeckMask,
    BodyFrame,
    VacuumCushion,
    # QA Equipment
    WaterPhantom,
    SolidWaterPhantom,
    DailyQADevice,
    WinstonLutz,
    PicketFence,
    # Accessories
    PhysicalWedge,
    ElectronApplicator,
    BlockTray,
    Bolus,
    LeadShield,
)

# Clinical workflows
from .clinical import (
    # Workflows
    PatientSetup,
    TreatmentDelivery,
    ImageGuidance,
    AdaptiveWorkflow,
    PreTreatmentVerification,
    # QA Procedures
    DailyQA,
    MonthlyQA,
    AnnualQA,
    PatientSpecificQA,
    CommissioningProcedure,
    TG51Calibration,
    TRS398Calibration,
    # Safety
    SafetyInterlock,
    InterlockSystem,
    TimeoutMonitor,
    DoseMonitor,
    EmergencyStop,
    RadiationSurvey,
)

# Data formats
from .data_formats import (
    # DICOM RT
    DICOMRTHandler,
    RTStructureSet,
    RTPlan,
    RTDose,
    RTImage,
    # Machine logs
    MachineLogParser,
    TrajectoryLog,
    DynaLog,
    LogAnalyzer,
    # TPS formats
    PinnacleFormat,
    EclipseFormat,
    RayStationFormat,
    MonacoFormat,
)

# Emerging technologies
from .emerging import (
    # FLASH therapy
    FLASHDeliverySystem,
    FLASHBeamParameters,
    ElectronFLASH,
    ProtonFLASH,
    FLASHBiologicalModel,
    # Spatial fractionation
    GRIDTherapy,
    LATTICETherapy,
    MinibeamTherapy,
    # Biological models
    LinearQuadraticModel,
    BEDCalculator,
    EQD2Calculator,
    TCPModel,
    NTCPModel,
    # AI planning
    AutoContouring,
    KnowledgeBasedPlanning,
    DeepLearningDosePrediction,
    AutomaticPlanOptimization,
)

__version__ = "2.0.0"
__all__ = [
    # Base classes
    "RadiationSource",
    "BeamParameters",
    "TreatmentTarget",
    "DoseDistribution",
    "Position3D",
    "Vector3D",
    "OrganAtRisk",
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
    # Vendor-specific (Varian)
    "VarianTrueBeam",
    "VarianHalcyon",
    "VarianClinac",
    "VarianEdge",
    "VarianProBeam",
    # Vendor-specific (Elekta)
    "ElektaVersaHD",
    "ElektaInfinity",
    "ElektaUnity",
    # Vendor-specific (Siemens)
    "SiemensArtiste",
    "SiemensOncor",
    # Vendor-specific (Particle therapy)
    "IBAProteusPlus",
    "IBAProteusOne",
    "HitachiPROBEAT",
    "MevionS250i",
    # Vendor-specific (Accuray)
    "AccurayCyberKnifeM6",
    "AccurayCyberKnifeS7",
    "AccurayTomoTherapy",
    "AccurayRadixact",
    # Hardware - Detectors
    "IonChamber",
    "FarmerChamber",
    "PinPointChamber",
    "DiodeDosimeter",
    "FilmDosimeter",
    "MatriXX",
    "ArcCHECK",
    "Delta4",
    "PortalImager",
    # Hardware - Positioning
    "TreatmentCouch",
    "SixDOFCouch",
    "HexaPOD",
    "HeadAndNeckMask",
    "BodyFrame",
    "VacuumCushion",
    # Hardware - QA Equipment
    "WaterPhantom",
    "SolidWaterPhantom",
    "DailyQADevice",
    "WinstonLutz",
    "PicketFence",
    # Hardware - Accessories
    "PhysicalWedge",
    "ElectronApplicator",
    "BlockTray",
    "Bolus",
    "LeadShield",
    # Clinical - Workflows
    "PatientSetup",
    "TreatmentDelivery",
    "ImageGuidance",
    "AdaptiveWorkflow",
    "PreTreatmentVerification",
    # Clinical - QA Procedures
    "DailyQA",
    "MonthlyQA",
    "AnnualQA",
    "PatientSpecificQA",
    "CommissioningProcedure",
    "TG51Calibration",
    "TRS398Calibration",
    # Clinical - Safety
    "SafetyInterlock",
    "InterlockSystem",
    "TimeoutMonitor",
    "DoseMonitor",
    "EmergencyStop",
    "RadiationSurvey",
    # Data Formats - DICOM RT
    "DICOMRTHandler",
    "RTStructureSet",
    "RTPlan",
    "RTDose",
    "RTImage",
    # Data Formats - Machine Logs
    "MachineLogParser",
    "TrajectoryLog",
    "DynaLog",
    "LogAnalyzer",
    # Data Formats - TPS
    "PinnacleFormat",
    "EclipseFormat",
    "RayStationFormat",
    "MonacoFormat",
    # Emerging - FLASH
    "FLASHDeliverySystem",
    "FLASHBeamParameters",
    "ElectronFLASH",
    "ProtonFLASH",
    "FLASHBiologicalModel",
    # Emerging - Spatial Fractionation
    "GRIDTherapy",
    "LATTICETherapy",
    "MinibeamTherapy",
    # Emerging - Biological Models
    "LinearQuadraticModel",
    "BEDCalculator",
    "EQD2Calculator",
    "TCPModel",
    "NTCPModel",
    # Emerging - AI Planning
    "AutoContouring",
    "KnowledgeBasedPlanning",
    "DeepLearningDosePrediction",
    "AutomaticPlanOptimization",
]
