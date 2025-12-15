"""
Vendor-Specific Radiation Therapy Equipment Implementations.

This module provides accurate simulations of commercial radiation therapy
systems from major vendors including:

LINAC Systems:
- Varian (TrueBeam, Halcyon, Clinac, Edge)
- Elekta (Versa HD, Infinity, Unity MR-Linac)
- Siemens (Artiste, Oncor)
- Accuray (TomoTherapy, Radixact)

Particle Therapy:
- IBA (Proteus Plus, Proteus One)
- Varian (ProBeam)
- Hitachi (PROBEAT)
- Mevion (S250i HYPERSCAN)

Robotic Systems:
- Accuray (CyberKnife M6, S7)
- Brainlab (ExacTrac)
- ViewRay (MRIdian)
"""

from .varian import (
    VarianTrueBeam,
    VarianHalcyon,
    VarianClinac,
    VarianEdge,
    VarianProBeam,
    VarianMLCHD120,
    VarianMLCMillennium,
)

from .elekta import (
    ElektaVersaHD,
    ElektaInfinity,
    ElektaUnity,
    ElektaAgility,
    ElektaMLCAgility,
)

from .siemens import (
    SiemensArtiste,
    SiemensOncor,
    Siemens160MLC,
)

from .particle_vendors import (
    IBAProteusPlus,
    IBAProteusOne,
    HitachiPROBEAT,
    MevionS250i,
)

from .accuray import (
    AccurayCyberKnifeM6,
    AccurayCyberKnifeS7,
    AccurayTomoTherapy,
    AccurayRadixact,
)

__all__ = [
    # Varian
    "VarianTrueBeam",
    "VarianHalcyon",
    "VarianClinac",
    "VarianEdge",
    "VarianProBeam",
    "VarianMLCHD120",
    "VarianMLCMillennium",
    # Elekta
    "ElektaVersaHD",
    "ElektaInfinity",
    "ElektaUnity",
    "ElektaAgility",
    "ElektaMLCAgility",
    # Siemens
    "SiemensArtiste",
    "SiemensOncor",
    "Siemens160MLC",
    # Particle Therapy
    "IBAProteusPlus",
    "IBAProteusOne",
    "HitachiPROBEAT",
    "MevionS250i",
    # Accuray
    "AccurayCyberKnifeM6",
    "AccurayCyberKnifeS7",
    "AccurayTomoTherapy",
    "AccurayRadixact",
]
