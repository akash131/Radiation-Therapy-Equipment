"""
Hardware Components for Radiation Therapy.

This module provides implementations of:
- Radiation detectors (ion chambers, diodes, arrays)
- Patient positioning systems (6-DOF couches, immobilization)
- Quality assurance equipment (phantoms, dosimeters)
- Accessory devices (blocks, wedges, applicators)
"""

from .detectors import (
    IonChamber,
    FarmerChamber,
    PinPointChamber,
    ParallelPlateChamber,
    DiodeDosimeter,
    DiamondDetector,
    MOSFETDosimeter,
    OSLDosimeter,
    FilmDosimeter,
    DetectorArray,
    MatriXX,
    MapCHECK,
    ArcCHECK,
    Delta4,
    PortalImager,
)

from .positioning import (
    TreatmentCouch,
    SixDOFCouch,
    HexaPOD,
    RoboticCouch,
    ImmobilizationDevice,
    HeadAndNeckMask,
    BodyFrame,
    VacuumCushion,
    BellyBoard,
    BreastBoard,
    Indexing,
)

from .qa_equipment import (
    Phantom,
    WaterPhantom,
    SolidWaterPhantom,
    CylindricalPhantom,
    AnthropomorphicPhantom,
    DailyQADevice,
    QuickCheck,
    DailyQA3,
    StarShot,
    WinstonLutz,
    PicketFence,
)

from .accessories import (
    PhysicalWedge,
    ElectronApplicator,
    BlockTray,
    Bolus,
    LeadShield,
)

__all__ = [
    # Detectors
    "IonChamber",
    "FarmerChamber",
    "PinPointChamber",
    "ParallelPlateChamber",
    "DiodeDosimeter",
    "DiamondDetector",
    "MOSFETDosimeter",
    "OSLDosimeter",
    "FilmDosimeter",
    "DetectorArray",
    "MatriXX",
    "MapCHECK",
    "ArcCHECK",
    "Delta4",
    "PortalImager",
    # Positioning
    "TreatmentCouch",
    "SixDOFCouch",
    "HexaPOD",
    "RoboticCouch",
    "ImmobilizationDevice",
    "HeadAndNeckMask",
    "BodyFrame",
    "VacuumCushion",
    "BellyBoard",
    "BreastBoard",
    "Indexing",
    # QA Equipment
    "Phantom",
    "WaterPhantom",
    "SolidWaterPhantom",
    "CylindricalPhantom",
    "AnthropomorphicPhantom",
    "DailyQADevice",
    "QuickCheck",
    "DailyQA3",
    "StarShot",
    "WinstonLutz",
    "PicketFence",
    # Accessories
    "PhysicalWedge",
    "ElectronApplicator",
    "BlockTray",
    "Bolus",
    "LeadShield",
]
