"""
Emerging Technologies in Radiation Therapy.

This module provides implementations for cutting-edge
and experimental radiation therapy technologies:
- FLASH radiotherapy (ultra-high dose rate)
- Spatially fractionated radiotherapy (GRID/LATTICE)
- Radioimmunotherapy combinations
- Biological dosimetry
- AI-assisted treatment planning
"""

from .flash_therapy import (
    FLASHDeliverySystem,
    FLASHBeamParameters,
    FLASHDoseCalculator,
    FLASHBiologicalModel,
    ElectronFLASH,
    ProtonFLASH,
    XrayFLASH,
)

from .spatial_fractionation import (
    GRIDTherapy,
    LATTICETherapy,
    MinibeamTherapy,
    SpatiallyFractionatedPlan,
)

from .biological_models import (
    BiologicalDoseModel,
    LinearQuadraticModel,
    TCPModel,
    NTCPModel,
    BEDCalculator,
    EQD2Calculator,
)

from .ai_planning import (
    AutoContouring,
    KnowledgeBasedPlanning,
    DeepLearningDosePrediction,
    AutomaticPlanOptimization,
)

__all__ = [
    # FLASH Therapy
    "FLASHDeliverySystem",
    "FLASHBeamParameters",
    "FLASHDoseCalculator",
    "FLASHBiologicalModel",
    "ElectronFLASH",
    "ProtonFLASH",
    "XrayFLASH",
    # Spatial Fractionation
    "GRIDTherapy",
    "LATTICETherapy",
    "MinibeamTherapy",
    "SpatiallyFractionatedPlan",
    # Biological Models
    "BiologicalDoseModel",
    "LinearQuadraticModel",
    "TCPModel",
    "NTCPModel",
    "BEDCalculator",
    "EQD2Calculator",
    # AI Planning
    "AutoContouring",
    "KnowledgeBasedPlanning",
    "DeepLearningDosePrediction",
    "AutomaticPlanOptimization",
]
