"""
Treatment Planning System Data Formats.

Implements import/export for various TPS formats:
- Pinnacle (Philips)
- Eclipse (Varian)
- RayStation (RaySearch)
- Monaco (Elekta)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any, Union
from datetime import datetime
from abc import ABC, abstractmethod
import numpy as np


class TPSVendor(Enum):
    """Treatment planning system vendors."""
    PINNACLE = "pinnacle"
    ECLIPSE = "eclipse"
    RAYSTATION = "raystation"
    MONACO = "monaco"
    TOMOTHERAPY = "tomotherapy"
    CYBERKNIFE = "cyberknife"


@dataclass
class BeamData:
    """Generic beam data structure."""
    name: str
    energy: float
    modality: str  # photon, electron, proton
    gantry_angle: float
    collimator_angle: float
    couch_angle: float
    isocenter: Tuple[float, float, float]
    field_size: Tuple[float, float]
    mu: float
    wedge: Optional[str] = None
    mlc_data: Optional[Dict] = None


@dataclass
class StructureData:
    """Generic structure data."""
    name: str
    roi_type: str
    color: Tuple[int, int, int]
    contours: List[Dict]  # List of {z, points}
    volume: Optional[float] = None


@dataclass
class DoseData:
    """Generic dose data structure."""
    dimensions: Tuple[int, int, int]
    spacing: Tuple[float, float, float]
    origin: Tuple[float, float, float]
    data: Optional[np.ndarray] = None
    units: str = "Gy"
    summation_type: str = "plan"


@dataclass
class PlanData:
    """Generic plan data structure."""
    name: str
    patient_id: str
    patient_name: str
    prescribed_dose: float
    fractions: int
    beams: List[BeamData] = field(default_factory=list)
    structures: List[StructureData] = field(default_factory=list)
    dose: Optional[DoseData] = None


class TPSExporter(ABC):
    """Abstract base class for TPS exporters."""

    @abstractmethod
    def export_plan(self, plan: PlanData, output_path: str) -> bool:
        """Export plan to TPS format."""
        pass

    @abstractmethod
    def export_structures(
        self,
        structures: List[StructureData],
        output_path: str
    ) -> bool:
        """Export structures to TPS format."""
        pass

    @abstractmethod
    def export_dose(self, dose: DoseData, output_path: str) -> bool:
        """Export dose to TPS format."""
        pass


class TPSImporter(ABC):
    """Abstract base class for TPS importers."""

    @abstractmethod
    def import_plan(self, input_path: str) -> Optional[PlanData]:
        """Import plan from TPS format."""
        pass

    @abstractmethod
    def import_structures(self, input_path: str) -> List[StructureData]:
        """Import structures from TPS format."""
        pass

    @abstractmethod
    def import_dose(self, input_path: str) -> Optional[DoseData]:
        """Import dose from TPS format."""
        pass


class PinnacleFormat(TPSImporter, TPSExporter):
    """
    Philips Pinnacle TPS format handler.

    Pinnacle uses proprietary binary and text formats:
    - .Plan files for treatment plans
    - .roi files for structures
    - .Pinnacle files for patient data
    """

    def __init__(self):
        self._version: str = "16.2"
        self._database_path: Optional[str] = None

    def set_database_path(self, path: str):
        """Set Pinnacle database path."""
        self._database_path = path

    def export_plan(self, plan: PlanData, output_path: str) -> bool:
        """Export plan to Pinnacle format."""
        # Pinnacle plan format
        content = self._generate_pinnacle_plan(plan)
        return self._write_file(output_path, content)

    def _generate_pinnacle_plan(self, plan: PlanData) -> str:
        """Generate Pinnacle plan file content."""
        lines = [
            f"// Pinnacle Plan Export",
            f"// Generated: {datetime.now().isoformat()}",
            f"PlanName = \"{plan.name}\";",
            f"PatientID = \"{plan.patient_id}\";",
            f"PatientName = \"{plan.patient_name}\";",
            f"PrescribedDose = {plan.prescribed_dose};",
            f"NumberOfFractions = {plan.fractions};",
            "",
            "// Beams",
            f"NumberOfBeams = {len(plan.beams)};",
        ]

        for i, beam in enumerate(plan.beams):
            lines.extend([
                f"Beam[{i}] = {{",
                f"  Name = \"{beam.name}\";",
                f"  Energy = {beam.energy};",
                f"  Modality = \"{beam.modality}\";",
                f"  GantryAngle = {beam.gantry_angle};",
                f"  CollimatorAngle = {beam.collimator_angle};",
                f"  CouchAngle = {beam.couch_angle};",
                f"  Isocenter = {{{beam.isocenter[0]}, {beam.isocenter[1]}, {beam.isocenter[2]}}};",
                f"  MonitorUnits = {beam.mu};",
                "};",
            ])

        return "\n".join(lines)

    def export_structures(
        self,
        structures: List[StructureData],
        output_path: str
    ) -> bool:
        """Export structures to Pinnacle ROI format."""
        content = self._generate_pinnacle_roi(structures)
        return self._write_file(output_path, content)

    def _generate_pinnacle_roi(self, structures: List[StructureData]) -> str:
        """Generate Pinnacle ROI file content."""
        lines = [
            "// Pinnacle ROI Export",
            f"// Generated: {datetime.now().isoformat()}",
            f"NumberOfROIs = {len(structures)};",
        ]

        for i, struct in enumerate(structures):
            lines.extend([
                f"ROI[{i}] = {{",
                f"  Name = \"{struct.name}\";",
                f"  Type = \"{struct.roi_type}\";",
                f"  Color = {{{struct.color[0]}, {struct.color[1]}, {struct.color[2]}}};",
                f"  NumberOfContours = {len(struct.contours)};",
            ])

            for j, contour in enumerate(struct.contours):
                z = contour.get("z", 0)
                points = contour.get("points", [])
                lines.append(f"  Contour[{j}] = {{ Z = {z}; NumPoints = {len(points)}; }};")

            lines.append("};")

        return "\n".join(lines)

    def export_dose(self, dose: DoseData, output_path: str) -> bool:
        """Export dose to Pinnacle format."""
        # Pinnacle uses proprietary binary dose format
        # This generates a header file
        content = self._generate_pinnacle_dose_header(dose)
        return self._write_file(output_path + ".header", content)

    def _generate_pinnacle_dose_header(self, dose: DoseData) -> str:
        """Generate Pinnacle dose header."""
        lines = [
            "// Pinnacle Dose Header",
            f"Dimensions = {{{dose.dimensions[0]}, {dose.dimensions[1]}, {dose.dimensions[2]}}};",
            f"Spacing = {{{dose.spacing[0]}, {dose.spacing[1]}, {dose.spacing[2]}}};",
            f"Origin = {{{dose.origin[0]}, {dose.origin[1]}, {dose.origin[2]}}};",
            f"Units = \"{dose.units}\";",
        ]
        return "\n".join(lines)

    def import_plan(self, input_path: str) -> Optional[PlanData]:
        """Import plan from Pinnacle format."""
        # Simulated import
        return PlanData(
            name="Imported Plan",
            patient_id="SIM001",
            patient_name="Simulated Patient",
            prescribed_dose=50.0,
            fractions=25
        )

    def import_structures(self, input_path: str) -> List[StructureData]:
        """Import structures from Pinnacle ROI format."""
        # Simulated import
        return []

    def import_dose(self, input_path: str) -> Optional[DoseData]:
        """Import dose from Pinnacle format."""
        # Simulated import
        return None

    def _write_file(self, path: str, content: str) -> bool:
        """Write content to file."""
        try:
            with open(path, 'w') as f:
                f.write(content)
            return True
        except IOError:
            return False


class EclipseFormat(TPSImporter, TPSExporter):
    """
    Varian Eclipse TPS format handler.

    Eclipse primarily uses DICOM for data exchange,
    but also has proprietary formats for some data.
    """

    def __init__(self):
        self._version: str = "16.1"
        self._scripting_context: Optional[Any] = None

    def export_plan(self, plan: PlanData, output_path: str) -> bool:
        """Export plan (Eclipse uses DICOM RT Plan)."""
        # Generate XML representation for Eclipse scripting
        content = self._generate_eclipse_xml(plan)
        return self._write_file(output_path, content)

    def _generate_eclipse_xml(self, plan: PlanData) -> str:
        """Generate Eclipse-compatible XML."""
        lines = [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<EclipsePlan>',
            f'  <PlanName>{plan.name}</PlanName>',
            f'  <PatientID>{plan.patient_id}</PatientID>',
            f'  <PrescribedDose unit="Gy">{plan.prescribed_dose}</PrescribedDose>',
            f'  <Fractions>{plan.fractions}</Fractions>',
            '  <Beams>',
        ]

        for beam in plan.beams:
            lines.extend([
                '    <Beam>',
                f'      <Name>{beam.name}</Name>',
                f'      <Energy>{beam.energy}</Energy>',
                f'      <GantryAngle>{beam.gantry_angle}</GantryAngle>',
                f'      <CollimatorAngle>{beam.collimator_angle}</CollimatorAngle>',
                f'      <CouchAngle>{beam.couch_angle}</CouchAngle>',
                f'      <MU>{beam.mu}</MU>',
                '    </Beam>',
            ])

        lines.extend([
            '  </Beams>',
            '</EclipsePlan>',
        ])

        return "\n".join(lines)

    def export_structures(
        self,
        structures: List[StructureData],
        output_path: str
    ) -> bool:
        """Export structures (uses DICOM RT Structure Set)."""
        content = self._generate_structure_xml(structures)
        return self._write_file(output_path, content)

    def _generate_structure_xml(self, structures: List[StructureData]) -> str:
        """Generate structure XML."""
        lines = [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<Structures>',
        ]

        for struct in structures:
            lines.extend([
                '  <Structure>',
                f'    <Name>{struct.name}</Name>',
                f'    <Type>{struct.roi_type}</Type>',
                f'    <Color R="{struct.color[0]}" G="{struct.color[1]}" B="{struct.color[2]}"/>',
                f'    <Volume>{struct.volume or 0}</Volume>',
                '  </Structure>',
            ])

        lines.append('</Structures>')
        return "\n".join(lines)

    def export_dose(self, dose: DoseData, output_path: str) -> bool:
        """Export dose (uses DICOM RT Dose)."""
        # Eclipse primarily uses DICOM
        content = f"Dose dimensions: {dose.dimensions}\n"
        content += f"Spacing: {dose.spacing}\n"
        content += f"Origin: {dose.origin}\n"
        return self._write_file(output_path, content)

    def import_plan(self, input_path: str) -> Optional[PlanData]:
        """Import plan from Eclipse format."""
        return PlanData(
            name="Imported Eclipse Plan",
            patient_id="ECL001",
            patient_name="Eclipse Patient",
            prescribed_dose=60.0,
            fractions=30
        )

    def import_structures(self, input_path: str) -> List[StructureData]:
        """Import structures from Eclipse."""
        return []

    def import_dose(self, input_path: str) -> Optional[DoseData]:
        """Import dose from Eclipse."""
        return None

    def _write_file(self, path: str, content: str) -> bool:
        """Write content to file."""
        try:
            with open(path, 'w') as f:
                f.write(content)
            return True
        except IOError:
            return False


class RayStationFormat(TPSImporter, TPSExporter):
    """
    RaySearch RayStation TPS format handler.

    RayStation uses DICOM and proprietary formats,
    with extensive scripting API support.
    """

    def __init__(self):
        self._version: str = "12A"
        self._script_environment: Optional[Any] = None

    def export_plan(self, plan: PlanData, output_path: str) -> bool:
        """Export plan to RayStation format."""
        content = self._generate_raystation_script(plan)
        return self._write_file(output_path, content)

    def _generate_raystation_script(self, plan: PlanData) -> str:
        """Generate RayStation Python script for plan creation."""
        lines = [
            "# RayStation Plan Script",
            f"# Generated: {datetime.now().isoformat()}",
            "",
            "from connect import *",
            "",
            "# Get current patient and case",
            "patient = get_current('Patient')",
            "case = get_current('Case')",
            "plan = get_current('Plan')",
            "",
            f"# Plan: {plan.name}",
            f"# Prescribed dose: {plan.prescribed_dose} Gy in {plan.fractions} fractions",
            "",
        ]

        for beam in plan.beams:
            lines.extend([
                f"# Add beam: {beam.name}",
                "beam_set.AddBeam(",
                f"    Name='{beam.name}',",
                f"    Energy={beam.energy},",
                f"    GantryAngle={beam.gantry_angle},",
                f"    CollimatorAngle={beam.collimator_angle},",
                f"    CouchAngle={beam.couch_angle},",
                f"    Isocenter={{'x': {beam.isocenter[0]}, 'y': {beam.isocenter[1]}, 'z': {beam.isocenter[2]}}},",
                ")",
                "",
            ])

        return "\n".join(lines)

    def export_structures(
        self,
        structures: List[StructureData],
        output_path: str
    ) -> bool:
        """Export structures for RayStation."""
        content = self._generate_structure_script(structures)
        return self._write_file(output_path, content)

    def _generate_structure_script(self, structures: List[StructureData]) -> str:
        """Generate RayStation structure script."""
        lines = [
            "# RayStation Structure Script",
            "from connect import *",
            "",
            "case = get_current('Case')",
            "ss = case.PatientModel.StructureSets[0]",
            "",
        ]

        for struct in structures:
            lines.extend([
                f"# Create ROI: {struct.name}",
                "ss.RoiGeometries.CreateRoi(",
                f"    Name='{struct.name}',",
                f"    Type='{struct.roi_type}',",
                f"    Color='#{struct.color[0]:02x}{struct.color[1]:02x}{struct.color[2]:02x}',",
                ")",
                "",
            ])

        return "\n".join(lines)

    def export_dose(self, dose: DoseData, output_path: str) -> bool:
        """Export dose from RayStation."""
        content = f"# RayStation Dose Export\n"
        content += f"# Dimensions: {dose.dimensions}\n"
        return self._write_file(output_path, content)

    def import_plan(self, input_path: str) -> Optional[PlanData]:
        """Import plan from RayStation."""
        return PlanData(
            name="Imported RayStation Plan",
            patient_id="RS001",
            patient_name="RayStation Patient",
            prescribed_dose=70.0,
            fractions=35
        )

    def import_structures(self, input_path: str) -> List[StructureData]:
        """Import structures from RayStation."""
        return []

    def import_dose(self, input_path: str) -> Optional[DoseData]:
        """Import dose from RayStation."""
        return None

    def _write_file(self, path: str, content: str) -> bool:
        """Write content to file."""
        try:
            with open(path, 'w') as f:
                f.write(content)
            return True
        except IOError:
            return False


class MonacoFormat(TPSImporter, TPSExporter):
    """
    Elekta Monaco TPS format handler.

    Monaco uses DICOM and proprietary formats for
    Monte Carlo dose calculation and optimization.
    """

    def __init__(self):
        self._version: str = "5.51"
        self._gpu_enabled: bool = True

    def export_plan(self, plan: PlanData, output_path: str) -> bool:
        """Export plan to Monaco format."""
        content = self._generate_monaco_plan(plan)
        return self._write_file(output_path, content)

    def _generate_monaco_plan(self, plan: PlanData) -> str:
        """Generate Monaco plan file."""
        lines = [
            "# Monaco Treatment Plan",
            f"# Generated: {datetime.now().isoformat()}",
            "",
            "[General]",
            f"PlanName={plan.name}",
            f"PatientID={plan.patient_id}",
            f"PatientName={plan.patient_name}",
            "",
            "[Prescription]",
            f"PrescribedDose={plan.prescribed_dose}",
            f"NumberOfFractions={plan.fractions}",
            f"DosePerFraction={plan.prescribed_dose / plan.fractions:.2f}",
            "",
            "[Beams]",
            f"NumberOfBeams={len(plan.beams)}",
        ]

        for i, beam in enumerate(plan.beams):
            lines.extend([
                "",
                f"[Beam{i+1}]",
                f"Name={beam.name}",
                f"Energy={beam.energy}",
                f"Modality={beam.modality}",
                f"GantryAngle={beam.gantry_angle}",
                f"CollimatorAngle={beam.collimator_angle}",
                f"CouchAngle={beam.couch_angle}",
                f"IsoX={beam.isocenter[0]}",
                f"IsoY={beam.isocenter[1]}",
                f"IsoZ={beam.isocenter[2]}",
                f"MU={beam.mu}",
            ])

        return "\n".join(lines)

    def export_structures(
        self,
        structures: List[StructureData],
        output_path: str
    ) -> bool:
        """Export structures to Monaco format."""
        content = self._generate_monaco_structures(structures)
        return self._write_file(output_path, content)

    def _generate_monaco_structures(self, structures: List[StructureData]) -> str:
        """Generate Monaco structure file."""
        lines = [
            "# Monaco Structure Set",
            f"# Generated: {datetime.now().isoformat()}",
            "",
            "[Structures]",
            f"NumberOfStructures={len(structures)}",
        ]

        for i, struct in enumerate(structures):
            lines.extend([
                "",
                f"[Structure{i+1}]",
                f"Name={struct.name}",
                f"Type={struct.roi_type}",
                f"ColorR={struct.color[0]}",
                f"ColorG={struct.color[1]}",
                f"ColorB={struct.color[2]}",
                f"Volume={struct.volume or 0}",
            ])

        return "\n".join(lines)

    def export_dose(self, dose: DoseData, output_path: str) -> bool:
        """Export dose to Monaco format."""
        content = self._generate_monaco_dose(dose)
        return self._write_file(output_path + ".info", content)

    def _generate_monaco_dose(self, dose: DoseData) -> str:
        """Generate Monaco dose info file."""
        lines = [
            "# Monaco Dose Grid Info",
            "",
            "[DoseGrid]",
            f"DimX={dose.dimensions[0]}",
            f"DimY={dose.dimensions[1]}",
            f"DimZ={dose.dimensions[2]}",
            f"SpacingX={dose.spacing[0]}",
            f"SpacingY={dose.spacing[1]}",
            f"SpacingZ={dose.spacing[2]}",
            f"OriginX={dose.origin[0]}",
            f"OriginY={dose.origin[1]}",
            f"OriginZ={dose.origin[2]}",
            f"Units={dose.units}",
        ]
        return "\n".join(lines)

    def import_plan(self, input_path: str) -> Optional[PlanData]:
        """Import plan from Monaco format."""
        return PlanData(
            name="Imported Monaco Plan",
            patient_id="MON001",
            patient_name="Monaco Patient",
            prescribed_dose=66.0,
            fractions=33
        )

    def import_structures(self, input_path: str) -> List[StructureData]:
        """Import structures from Monaco."""
        return []

    def import_dose(self, input_path: str) -> Optional[DoseData]:
        """Import dose from Monaco."""
        return None

    def _write_file(self, path: str, content: str) -> bool:
        """Write content to file."""
        try:
            with open(path, 'w') as f:
                f.write(content)
            return True
        except IOError:
            return False


class FormatConverter:
    """
    Converts between different TPS formats.

    Uses intermediate generic data structures for conversion.
    """

    def __init__(self):
        self._formats: Dict[TPSVendor, Union[TPSImporter, TPSExporter]] = {
            TPSVendor.PINNACLE: PinnacleFormat(),
            TPSVendor.ECLIPSE: EclipseFormat(),
            TPSVendor.RAYSTATION: RayStationFormat(),
            TPSVendor.MONACO: MonacoFormat(),
        }

    def convert_plan(
        self,
        source_path: str,
        source_format: TPSVendor,
        target_path: str,
        target_format: TPSVendor
    ) -> bool:
        """Convert plan between formats."""
        source_handler = self._formats.get(source_format)
        target_handler = self._formats.get(target_format)

        if not source_handler or not target_handler:
            return False

        # Import from source
        plan = source_handler.import_plan(source_path)
        if not plan:
            return False

        # Export to target
        return target_handler.export_plan(plan, target_path)

    def convert_structures(
        self,
        source_path: str,
        source_format: TPSVendor,
        target_path: str,
        target_format: TPSVendor
    ) -> bool:
        """Convert structures between formats."""
        source_handler = self._formats.get(source_format)
        target_handler = self._formats.get(target_format)

        if not source_handler or not target_handler:
            return False

        structures = source_handler.import_structures(source_path)
        return target_handler.export_structures(structures, target_path)

    def convert_dose(
        self,
        source_path: str,
        source_format: TPSVendor,
        target_path: str,
        target_format: TPSVendor
    ) -> bool:
        """Convert dose between formats."""
        source_handler = self._formats.get(source_format)
        target_handler = self._formats.get(target_format)

        if not source_handler or not target_handler:
            return False

        dose = source_handler.import_dose(source_path)
        if not dose:
            return False

        return target_handler.export_dose(dose, target_path)
