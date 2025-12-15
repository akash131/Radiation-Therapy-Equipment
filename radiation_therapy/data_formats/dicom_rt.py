"""
DICOM RT Data Format Support.

Implements DICOM Radiation Therapy objects:
- RT Structure Set (RTSTRUCT)
- RT Plan (RTPLAN)
- RT Dose (RTDOSE)
- RT Image (RTIMAGE)

Based on DICOM PS3.3 standards for radiation therapy.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any, Union
from datetime import datetime, date, time
import numpy as np
import struct
import uuid


class DICOMModality(Enum):
    """DICOM modalities for radiation therapy."""
    CT = "CT"
    MR = "MR"
    PT = "PT"  # PET
    RTIMAGE = "RTIMAGE"
    RTDOSE = "RTDOSE"
    RTPLAN = "RTPLAN"
    RTSTRUCT = "RTSTRUCT"
    RTRECORD = "RTRECORD"


class RTROIInterpretedType(Enum):
    """ROI interpreted types per DICOM standard."""
    GTV = "GTV"
    CTV = "CTV"
    PTV = "PTV"
    ORGAN = "ORGAN"
    EXTERNAL = "EXTERNAL"
    AVOIDANCE = "AVOIDANCE"
    TREATED_VOLUME = "TREATED_VOLUME"
    IRRAD_VOLUME = "IRRAD_VOLUME"
    BOLUS = "BOLUS"
    BRACHY_CHANNEL = "BRACHY_CHANNEL"
    MARKER = "MARKER"
    REGISTRATION = "REGISTRATION"
    ISOCENTER = "ISOCENTER"
    CONTRAST_AGENT = "CONTRAST_AGENT"
    CAVITY = "CAVITY"
    SUPPORT = "SUPPORT"
    FIXATION = "FIXATION"


class BeamType(Enum):
    """Beam types in RT Plan."""
    STATIC = "STATIC"
    DYNAMIC = "DYNAMIC"


class TreatmentDeliveryType(Enum):
    """Treatment delivery types."""
    TREATMENT = "TREATMENT"
    CONTINUATION = "CONTINUATION"
    OPEN_PORTFILM = "OPEN_PORTFILM"
    TRMT_PORTFILM = "TRMT_PORTFILM"
    SETUP = "SETUP"


class RadiationType(Enum):
    """Radiation types."""
    PHOTON = "PHOTON"
    ELECTRON = "ELECTRON"
    PROTON = "PROTON"
    ION = "ION"
    NEUTRON = "NEUTRON"


class DoseUnits(Enum):
    """Dose units."""
    GY = "GY"
    CGY = "CGY"
    RELATIVE = "RELATIVE"


class DoseType(Enum):
    """Dose type."""
    PHYSICAL = "PHYSICAL"
    EFFECTIVE = "EFFECTIVE"
    ERROR = "ERROR"


class DoseSummationType(Enum):
    """Dose summation type."""
    PLAN = "PLAN"
    BEAM = "BEAM"
    BRACHY = "BRACHY"
    FRACTION = "FRACTION"
    CONTROL_POINT = "CONTROL_POINT"


@dataclass
class DICOMPatient:
    """DICOM Patient module attributes."""
    patient_id: str
    patient_name: str
    birth_date: Optional[date] = None
    sex: Optional[str] = None
    other_patient_ids: List[str] = field(default_factory=list)
    ethnic_group: Optional[str] = None
    comments: Optional[str] = None

    def to_dicom_dict(self) -> Dict[str, Any]:
        """Convert to DICOM attribute dictionary."""
        return {
            "PatientID": self.patient_id,
            "PatientName": self.patient_name,
            "PatientBirthDate": self.birth_date.strftime("%Y%m%d") if self.birth_date else "",
            "PatientSex": self.sex or "",
            "OtherPatientIDs": self.other_patient_ids,
            "EthnicGroup": self.ethnic_group or "",
            "PatientComments": self.comments or "",
        }


@dataclass
class DICOMStudy:
    """DICOM Study module attributes."""
    study_instance_uid: str
    study_date: Optional[date] = None
    study_time: Optional[time] = None
    study_id: Optional[str] = None
    study_description: Optional[str] = None
    accession_number: Optional[str] = None
    referring_physician: Optional[str] = None

    @classmethod
    def generate_uid(cls) -> str:
        """Generate a DICOM UID."""
        # DICOM UID root for test/demo purposes
        root = "1.2.826.0.1.3680043.8.498"
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        random_part = uuid.uuid4().int % 1000000
        return f"{root}.{timestamp}.{random_part}"

    def to_dicom_dict(self) -> Dict[str, Any]:
        """Convert to DICOM attribute dictionary."""
        return {
            "StudyInstanceUID": self.study_instance_uid,
            "StudyDate": self.study_date.strftime("%Y%m%d") if self.study_date else "",
            "StudyTime": self.study_time.strftime("%H%M%S") if self.study_time else "",
            "StudyID": self.study_id or "",
            "StudyDescription": self.study_description or "",
            "AccessionNumber": self.accession_number or "",
            "ReferringPhysicianName": self.referring_physician or "",
        }


@dataclass
class DICOMSeries:
    """DICOM Series module attributes."""
    series_instance_uid: str
    modality: DICOMModality
    series_number: Optional[int] = None
    series_description: Optional[str] = None
    series_date: Optional[date] = None
    series_time: Optional[time] = None
    body_part_examined: Optional[str] = None
    patient_position: Optional[str] = None  # HFS, HFP, FFS, FFP, etc.

    def to_dicom_dict(self) -> Dict[str, Any]:
        """Convert to DICOM attribute dictionary."""
        return {
            "SeriesInstanceUID": self.series_instance_uid,
            "Modality": self.modality.value,
            "SeriesNumber": self.series_number,
            "SeriesDescription": self.series_description or "",
            "SeriesDate": self.series_date.strftime("%Y%m%d") if self.series_date else "",
            "SeriesTime": self.series_time.strftime("%H%M%S") if self.series_time else "",
            "BodyPartExamined": self.body_part_examined or "",
            "PatientPosition": self.patient_position or "",
        }


@dataclass
class ROIContour:
    """ROI contour data for structure set."""
    roi_number: int
    roi_name: str
    roi_description: Optional[str] = None
    roi_interpreted_type: RTROIInterpretedType = RTROIInterpretedType.ORGAN
    roi_color: Tuple[int, int, int] = (255, 0, 0)  # RGB

    # Contour data: list of (z_position, contour_points) tuples
    # contour_points is Nx3 array of (x, y, z) coordinates
    contours: List[Tuple[float, np.ndarray]] = field(default_factory=list)

    # Reference frame of reference UID
    referenced_frame_uid: Optional[str] = None

    # Volume and centroid (calculated)
    volume_cc: Optional[float] = None
    centroid: Optional[Tuple[float, float, float]] = None

    def add_contour(self, z_position: float, points: np.ndarray):
        """Add contour at specified z position."""
        if points.shape[1] != 3:
            raise ValueError("Contour points must be Nx3 array")
        self.contours.append((z_position, points))

    def get_contour_at_z(self, z: float, tolerance: float = 0.5) -> Optional[np.ndarray]:
        """Get contour at specified z position."""
        for z_pos, points in self.contours:
            if abs(z_pos - z) <= tolerance:
                return points
        return None

    def calculate_volume(self, slice_thickness: float = 2.5) -> float:
        """Calculate ROI volume using slice-by-slice method."""
        if not self.contours:
            return 0.0

        total_area = 0.0
        sorted_contours = sorted(self.contours, key=lambda x: x[0])

        for z_pos, points in sorted_contours:
            # Calculate area using shoelace formula
            x = points[:, 0]
            y = points[:, 1]
            area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            total_area += area

        # Volume = sum of areas * slice thickness (convert to cc)
        self.volume_cc = total_area * slice_thickness / 1000.0
        return self.volume_cc

    def calculate_centroid(self) -> Tuple[float, float, float]:
        """Calculate ROI centroid."""
        if not self.contours:
            return (0.0, 0.0, 0.0)

        total_x, total_y, total_z = 0.0, 0.0, 0.0
        total_points = 0

        for z_pos, points in self.contours:
            total_x += np.sum(points[:, 0])
            total_y += np.sum(points[:, 1])
            total_z += len(points) * z_pos
            total_points += len(points)

        if total_points > 0:
            self.centroid = (
                total_x / total_points,
                total_y / total_points,
                total_z / total_points
            )
        else:
            self.centroid = (0.0, 0.0, 0.0)

        return self.centroid

    def to_dicom_dict(self) -> Dict[str, Any]:
        """Convert to DICOM attribute dictionary."""
        contour_sequence = []
        for z_pos, points in self.contours:
            contour_data = points.flatten().tolist()
            contour_sequence.append({
                "ContourGeometricType": "CLOSED_PLANAR",
                "NumberOfContourPoints": len(points),
                "ContourData": contour_data,
            })

        return {
            "ROINumber": self.roi_number,
            "ROIName": self.roi_name,
            "ROIDescription": self.roi_description or "",
            "RTROIInterpretedType": self.roi_interpreted_type.value,
            "ROIDisplayColor": list(self.roi_color),
            "ContourSequence": contour_sequence,
            "ReferencedFrameOfReferenceUID": self.referenced_frame_uid or "",
        }


class RTStructureSet:
    """
    DICOM RT Structure Set object.

    Contains ROI definitions for treatment planning.
    """

    def __init__(
        self,
        patient: DICOMPatient,
        study: DICOMStudy,
        label: str = "Structure Set"
    ):
        self.patient = patient
        self.study = study
        self.label = label

        self.series = DICOMSeries(
            series_instance_uid=DICOMStudy.generate_uid(),
            modality=DICOMModality.RTSTRUCT,
            series_description="RT Structure Set"
        )

        self.sop_instance_uid = DICOMStudy.generate_uid()
        self.sop_class_uid = "1.2.840.10008.5.1.4.1.1.481.3"  # RT Structure Set Storage

        self._rois: Dict[int, ROIContour] = {}
        self._next_roi_number = 1

        # Reference to CT series
        self._referenced_series_uid: Optional[str] = None
        self._frame_of_reference_uid: Optional[str] = None

    def set_referenced_series(self, series_uid: str, frame_uid: str):
        """Set referenced CT/MR series."""
        self._referenced_series_uid = series_uid
        self._frame_of_reference_uid = frame_uid

    def add_roi(
        self,
        name: str,
        interpreted_type: RTROIInterpretedType = RTROIInterpretedType.ORGAN,
        color: Tuple[int, int, int] = (255, 0, 0),
        description: Optional[str] = None
    ) -> ROIContour:
        """Add new ROI to structure set."""
        roi = ROIContour(
            roi_number=self._next_roi_number,
            roi_name=name,
            roi_description=description,
            roi_interpreted_type=interpreted_type,
            roi_color=color,
            referenced_frame_uid=self._frame_of_reference_uid
        )

        self._rois[self._next_roi_number] = roi
        self._next_roi_number += 1
        return roi

    def get_roi(self, name_or_number: Union[str, int]) -> Optional[ROIContour]:
        """Get ROI by name or number."""
        if isinstance(name_or_number, int):
            return self._rois.get(name_or_number)
        else:
            for roi in self._rois.values():
                if roi.roi_name == name_or_number:
                    return roi
        return None

    def get_all_rois(self) -> List[ROIContour]:
        """Get all ROIs."""
        return list(self._rois.values())

    def remove_roi(self, name_or_number: Union[str, int]) -> bool:
        """Remove ROI from structure set."""
        roi = self.get_roi(name_or_number)
        if roi:
            del self._rois[roi.roi_number]
            return True
        return False

    def get_target_rois(self) -> List[ROIContour]:
        """Get all target ROIs (GTV, CTV, PTV)."""
        target_types = [
            RTROIInterpretedType.GTV,
            RTROIInterpretedType.CTV,
            RTROIInterpretedType.PTV,
        ]
        return [
            roi for roi in self._rois.values()
            if roi.roi_interpreted_type in target_types
        ]

    def get_organ_rois(self) -> List[ROIContour]:
        """Get all organ ROIs."""
        return [
            roi for roi in self._rois.values()
            if roi.roi_interpreted_type == RTROIInterpretedType.ORGAN
        ]

    def to_dicom_dict(self) -> Dict[str, Any]:
        """Convert to DICOM attribute dictionary."""
        result = {}
        result.update(self.patient.to_dicom_dict())
        result.update(self.study.to_dicom_dict())
        result.update(self.series.to_dicom_dict())

        result["SOPClassUID"] = self.sop_class_uid
        result["SOPInstanceUID"] = self.sop_instance_uid
        result["StructureSetLabel"] = self.label
        result["StructureSetDate"] = datetime.now().strftime("%Y%m%d")
        result["StructureSetTime"] = datetime.now().strftime("%H%M%S")

        # Structure Set ROI Sequence
        result["StructureSetROISequence"] = [
            {
                "ROINumber": roi.roi_number,
                "ReferencedFrameOfReferenceUID": roi.referenced_frame_uid or "",
                "ROIName": roi.roi_name,
                "ROIDescription": roi.roi_description or "",
                "ROIGenerationAlgorithm": "MANUAL",
            }
            for roi in self._rois.values()
        ]

        # ROI Contour Sequence
        result["ROIContourSequence"] = [
            roi.to_dicom_dict() for roi in self._rois.values()
        ]

        # RT ROI Observations Sequence
        result["RTROIObservationsSequence"] = [
            {
                "ObservationNumber": roi.roi_number,
                "ReferencedROINumber": roi.roi_number,
                "RTROIInterpretedType": roi.roi_interpreted_type.value,
                "ROIInterpreter": "",
            }
            for roi in self._rois.values()
        ]

        return result


@dataclass
class ControlPoint:
    """Control point in beam sequence."""
    control_point_index: int
    nominal_beam_energy: float
    gantry_angle: float
    gantry_rotation_direction: str = "NONE"  # CW, CC, NONE
    beam_limiting_device_angle: float = 0.0
    patient_support_angle: float = 0.0
    table_top_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    isocenter_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    source_to_surface_distance: Optional[float] = None
    cumulative_meterset_weight: float = 0.0

    # MLC positions (pairs of A/B leaf positions)
    mlc_positions_a: Optional[List[float]] = None
    mlc_positions_b: Optional[List[float]] = None

    # Jaw positions
    jaw_x: Tuple[float, float] = (-50.0, 50.0)  # (X1, X2)
    jaw_y: Tuple[float, float] = (-50.0, 50.0)  # (Y1, Y2)

    def to_dicom_dict(self) -> Dict[str, Any]:
        """Convert to DICOM attribute dictionary."""
        result = {
            "ControlPointIndex": self.control_point_index,
            "NominalBeamEnergy": self.nominal_beam_energy,
            "GantryAngle": self.gantry_angle,
            "GantryRotationDirection": self.gantry_rotation_direction,
            "BeamLimitingDeviceAngle": self.beam_limiting_device_angle,
            "PatientSupportAngle": self.patient_support_angle,
            "TableTopVerticalPosition": self.table_top_position[2],
            "TableTopLongitudinalPosition": self.table_top_position[1],
            "TableTopLateralPosition": self.table_top_position[0],
            "IsocenterPosition": list(self.isocenter_position),
            "CumulativeMetersetWeight": self.cumulative_meterset_weight,
        }

        if self.source_to_surface_distance:
            result["SourceToSurfaceDistance"] = self.source_to_surface_distance

        # Beam limiting device positions
        bld_sequence = []

        # Jaws
        bld_sequence.append({
            "RTBeamLimitingDeviceType": "ASYMX",
            "LeafJawPositions": list(self.jaw_x),
        })
        bld_sequence.append({
            "RTBeamLimitingDeviceType": "ASYMY",
            "LeafJawPositions": list(self.jaw_y),
        })

        # MLC
        if self.mlc_positions_a and self.mlc_positions_b:
            mlc_positions = self.mlc_positions_a + self.mlc_positions_b
            bld_sequence.append({
                "RTBeamLimitingDeviceType": "MLCX",
                "LeafJawPositions": mlc_positions,
            })

        result["BeamLimitingDevicePositionSequence"] = bld_sequence

        return result


@dataclass
class BeamSequence:
    """Beam in RT Plan."""
    beam_number: int
    beam_name: str
    beam_type: BeamType = BeamType.STATIC
    radiation_type: RadiationType = RadiationType.PHOTON
    treatment_machine_name: str = "Unknown"

    # Beam parameters
    primary_dosimeter_unit: str = "MU"
    final_cumulative_meterset_weight: float = 1.0
    number_of_control_points: int = 2

    # Source parameters
    source_axis_distance: float = 1000.0  # mm (SAD)

    # Control points
    control_points: List[ControlPoint] = field(default_factory=list)

    # Beam limiting devices
    mlc_type: Optional[str] = None
    num_mlc_leaves: int = 120

    # References
    referenced_dose_reference_number: Optional[int] = None

    def add_control_point(self, cp: ControlPoint):
        """Add control point to beam."""
        self.control_points.append(cp)
        self.number_of_control_points = len(self.control_points)

    def to_dicom_dict(self) -> Dict[str, Any]:
        """Convert to DICOM attribute dictionary."""
        return {
            "BeamNumber": self.beam_number,
            "BeamName": self.beam_name,
            "BeamType": self.beam_type.value,
            "RadiationType": self.radiation_type.value,
            "TreatmentMachineName": self.treatment_machine_name,
            "PrimaryDosimeterUnit": self.primary_dosimeter_unit,
            "SourceAxisDistance": self.source_axis_distance,
            "FinalCumulativeMetersetWeight": self.final_cumulative_meterset_weight,
            "NumberOfControlPoints": self.number_of_control_points,
            "ControlPointSequence": [
                cp.to_dicom_dict() for cp in self.control_points
            ],
        }


class RTPlan:
    """
    DICOM RT Plan object.

    Contains treatment plan information including beams,
    dose references, and machine parameters.
    """

    def __init__(
        self,
        patient: DICOMPatient,
        study: DICOMStudy,
        plan_label: str = "Treatment Plan"
    ):
        self.patient = patient
        self.study = study
        self.plan_label = plan_label

        self.series = DICOMSeries(
            series_instance_uid=DICOMStudy.generate_uid(),
            modality=DICOMModality.RTPLAN,
            series_description="RT Plan"
        )

        self.sop_instance_uid = DICOMStudy.generate_uid()
        self.sop_class_uid = "1.2.840.10008.5.1.4.1.1.481.5"  # RT Plan Storage

        self.plan_name: str = plan_label
        self.plan_description: Optional[str] = None
        self.plan_intent: str = "CURATIVE"  # CURATIVE, PALLIATIVE, PROPHYLACTIC

        # Prescription
        self.prescribed_dose: float = 0.0
        self.number_of_fractions: int = 1
        self.dose_reference_sequence: List[Dict] = []

        # Beams
        self._beams: Dict[int, BeamSequence] = {}
        self._next_beam_number = 1

        # Fraction group
        self.fraction_group_sequence: List[Dict] = []

        # Referenced structure set
        self._referenced_structure_set_uid: Optional[str] = None

        # Approval status
        self.approval_status: str = "UNAPPROVED"

    def set_referenced_structure_set(self, structure_set_uid: str):
        """Set referenced structure set."""
        self._referenced_structure_set_uid = structure_set_uid

    def add_dose_reference(
        self,
        dose_reference_number: int,
        dose_reference_type: str,  # TARGET, ORGAN_AT_RISK
        target_prescription_dose: float,
        target_minimum_dose: Optional[float] = None,
        target_maximum_dose: Optional[float] = None,
        dose_reference_description: str = ""
    ):
        """Add dose reference (prescription)."""
        self.dose_reference_sequence.append({
            "DoseReferenceNumber": dose_reference_number,
            "DoseReferenceType": dose_reference_type,
            "DoseReferenceDescription": dose_reference_description,
            "TargetPrescriptionDose": target_prescription_dose,
            "TargetMinimumDose": target_minimum_dose,
            "TargetMaximumDose": target_maximum_dose,
        })

    def add_beam(
        self,
        name: str,
        beam_type: BeamType = BeamType.STATIC,
        radiation_type: RadiationType = RadiationType.PHOTON,
        machine_name: str = "TrueBeam"
    ) -> BeamSequence:
        """Add beam to plan."""
        beam = BeamSequence(
            beam_number=self._next_beam_number,
            beam_name=name,
            beam_type=beam_type,
            radiation_type=radiation_type,
            treatment_machine_name=machine_name
        )

        self._beams[self._next_beam_number] = beam
        self._next_beam_number += 1
        return beam

    def get_beam(self, name_or_number: Union[str, int]) -> Optional[BeamSequence]:
        """Get beam by name or number."""
        if isinstance(name_or_number, int):
            return self._beams.get(name_or_number)
        else:
            for beam in self._beams.values():
                if beam.beam_name == name_or_number:
                    return beam
        return None

    def get_all_beams(self) -> List[BeamSequence]:
        """Get all beams."""
        return list(self._beams.values())

    def set_fraction_group(
        self,
        number_of_fractions: int,
        beam_meterset_values: Dict[int, float]
    ):
        """Set fraction group with beam meterset values."""
        self.number_of_fractions = number_of_fractions

        referenced_beam_sequence = [
            {
                "ReferencedBeamNumber": beam_num,
                "BeamMeterset": meterset,
            }
            for beam_num, meterset in beam_meterset_values.items()
        ]

        self.fraction_group_sequence = [{
            "FractionGroupNumber": 1,
            "NumberOfFractionsPlanned": number_of_fractions,
            "NumberOfBeams": len(beam_meterset_values),
            "ReferencedBeamSequence": referenced_beam_sequence,
        }]

    def approve(self, reviewer_name: str):
        """Approve the plan."""
        self.approval_status = "APPROVED"

    def to_dicom_dict(self) -> Dict[str, Any]:
        """Convert to DICOM attribute dictionary."""
        result = {}
        result.update(self.patient.to_dicom_dict())
        result.update(self.study.to_dicom_dict())
        result.update(self.series.to_dicom_dict())

        result["SOPClassUID"] = self.sop_class_uid
        result["SOPInstanceUID"] = self.sop_instance_uid
        result["RTPlanLabel"] = self.plan_label
        result["RTPlanName"] = self.plan_name
        result["RTPlanDescription"] = self.plan_description or ""
        result["RTPlanDate"] = datetime.now().strftime("%Y%m%d")
        result["RTPlanTime"] = datetime.now().strftime("%H%M%S")
        result["RTPlanGeometry"] = "PATIENT"
        result["PlanIntent"] = self.plan_intent
        result["ApprovalStatus"] = self.approval_status

        # Dose Reference Sequence
        result["DoseReferenceSequence"] = self.dose_reference_sequence

        # Fraction Group Sequence
        result["FractionGroupSequence"] = self.fraction_group_sequence

        # Beam Sequence
        result["BeamSequence"] = [
            beam.to_dicom_dict() for beam in self._beams.values()
        ]

        # Referenced Structure Set
        if self._referenced_structure_set_uid:
            result["ReferencedStructureSetSequence"] = [{
                "ReferencedSOPClassUID": "1.2.840.10008.5.1.4.1.1.481.3",
                "ReferencedSOPInstanceUID": self._referenced_structure_set_uid,
            }]

        return result


@dataclass
class DoseGrid:
    """3D dose grid data."""
    dimensions: Tuple[int, int, int]  # (columns, rows, frames)
    pixel_spacing: Tuple[float, float]  # mm
    slice_thickness: float  # mm
    image_position: Tuple[float, float, float]  # mm
    image_orientation: Tuple[float, ...] = (1, 0, 0, 0, 1, 0)

    # Dose data (3D numpy array)
    dose_data: Optional[np.ndarray] = None

    # Scaling
    dose_grid_scaling: float = 1.0

    def set_dose_data(self, data: np.ndarray):
        """Set dose data array."""
        self.dose_data = data
        self.dimensions = data.shape

    def get_dose_at_point(self, x: float, y: float, z: float) -> float:
        """Get dose at physical point (trilinear interpolation)."""
        if self.dose_data is None:
            return 0.0

        # Convert physical coordinates to grid indices
        ix = (x - self.image_position[0]) / self.pixel_spacing[0]
        iy = (y - self.image_position[1]) / self.pixel_spacing[1]
        iz = (z - self.image_position[2]) / self.slice_thickness

        # Bounds check
        if (ix < 0 or iy < 0 or iz < 0 or
            ix >= self.dimensions[0] - 1 or
            iy >= self.dimensions[1] - 1 or
            iz >= self.dimensions[2] - 1):
            return 0.0

        # Trilinear interpolation
        x0, y0, z0 = int(ix), int(iy), int(iz)
        xd, yd, zd = ix - x0, iy - y0, iz - z0

        c000 = self.dose_data[z0, y0, x0]
        c100 = self.dose_data[z0, y0, x0 + 1]
        c010 = self.dose_data[z0, y0 + 1, x0]
        c110 = self.dose_data[z0, y0 + 1, x0 + 1]
        c001 = self.dose_data[z0 + 1, y0, x0]
        c101 = self.dose_data[z0 + 1, y0, x0 + 1]
        c011 = self.dose_data[z0 + 1, y0 + 1, x0]
        c111 = self.dose_data[z0 + 1, y0 + 1, x0 + 1]

        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        dose = c0 * (1 - zd) + c1 * zd

        return dose * self.dose_grid_scaling

    def get_max_dose(self) -> float:
        """Get maximum dose in grid."""
        if self.dose_data is None:
            return 0.0
        return float(np.max(self.dose_data)) * self.dose_grid_scaling

    def get_mean_dose_in_roi(self, roi: ROIContour) -> float:
        """Calculate mean dose in ROI (simplified)."""
        if self.dose_data is None:
            return 0.0

        total_dose = 0.0
        num_points = 0

        for z_pos, points in roi.contours:
            for point in points:
                dose = self.get_dose_at_point(point[0], point[1], z_pos)
                total_dose += dose
                num_points += 1

        return total_dose / max(1, num_points)


class RTDose:
    """
    DICOM RT Dose object.

    Contains 3D dose distribution data.
    """

    def __init__(
        self,
        patient: DICOMPatient,
        study: DICOMStudy
    ):
        self.patient = patient
        self.study = study

        self.series = DICOMSeries(
            series_instance_uid=DICOMStudy.generate_uid(),
            modality=DICOMModality.RTDOSE,
            series_description="RT Dose"
        )

        self.sop_instance_uid = DICOMStudy.generate_uid()
        self.sop_class_uid = "1.2.840.10008.5.1.4.1.1.481.2"  # RT Dose Storage

        self.dose_units: DoseUnits = DoseUnits.GY
        self.dose_type: DoseType = DoseType.PHYSICAL
        self.dose_summation_type: DoseSummationType = DoseSummationType.PLAN

        self.dose_grid: Optional[DoseGrid] = None

        # Referenced plan
        self._referenced_plan_uid: Optional[str] = None

    def set_referenced_plan(self, plan_uid: str):
        """Set referenced RT Plan."""
        self._referenced_plan_uid = plan_uid

    def set_dose_grid(self, grid: DoseGrid):
        """Set dose grid."""
        self.dose_grid = grid

    def create_dose_grid(
        self,
        dimensions: Tuple[int, int, int],
        pixel_spacing: Tuple[float, float],
        slice_thickness: float,
        origin: Tuple[float, float, float]
    ) -> DoseGrid:
        """Create empty dose grid."""
        self.dose_grid = DoseGrid(
            dimensions=dimensions,
            pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness,
            image_position=origin
        )
        self.dose_grid.dose_data = np.zeros(dimensions[::-1], dtype=np.float32)
        return self.dose_grid

    def add_beam_dose(self, beam_dose: np.ndarray, weight: float = 1.0):
        """Add dose contribution from a beam."""
        if self.dose_grid and self.dose_grid.dose_data is not None:
            self.dose_grid.dose_data += beam_dose * weight

    def to_dicom_dict(self) -> Dict[str, Any]:
        """Convert to DICOM attribute dictionary."""
        result = {}
        result.update(self.patient.to_dicom_dict())
        result.update(self.study.to_dicom_dict())
        result.update(self.series.to_dicom_dict())

        result["SOPClassUID"] = self.sop_class_uid
        result["SOPInstanceUID"] = self.sop_instance_uid
        result["DoseUnits"] = self.dose_units.value
        result["DoseType"] = self.dose_type.value
        result["DoseSummationType"] = self.dose_summation_type.value

        if self.dose_grid:
            result["Columns"] = self.dose_grid.dimensions[0]
            result["Rows"] = self.dose_grid.dimensions[1]
            result["NumberOfFrames"] = self.dose_grid.dimensions[2]
            result["PixelSpacing"] = list(self.dose_grid.pixel_spacing)
            result["SliceThickness"] = self.dose_grid.slice_thickness
            result["ImagePositionPatient"] = list(self.dose_grid.image_position)
            result["ImageOrientationPatient"] = list(self.dose_grid.image_orientation)
            result["DoseGridScaling"] = self.dose_grid.dose_grid_scaling

        if self._referenced_plan_uid:
            result["ReferencedRTPlanSequence"] = [{
                "ReferencedSOPClassUID": "1.2.840.10008.5.1.4.1.1.481.5",
                "ReferencedSOPInstanceUID": self._referenced_plan_uid,
            }]

        return result


class RTImage:
    """
    DICOM RT Image object.

    Contains portal images, DRRs, and other RT images.
    """

    def __init__(
        self,
        patient: DICOMPatient,
        study: DICOMStudy,
        image_type: str = "DRR"  # DRR, PORTAL, SIMULATOR
    ):
        self.patient = patient
        self.study = study
        self.image_type = image_type

        self.series = DICOMSeries(
            series_instance_uid=DICOMStudy.generate_uid(),
            modality=DICOMModality.RTIMAGE,
            series_description=f"RT Image ({image_type})"
        )

        self.sop_instance_uid = DICOMStudy.generate_uid()
        self.sop_class_uid = "1.2.840.10008.5.1.4.1.1.481.1"  # RT Image Storage

        # Image parameters
        self.rows: int = 512
        self.columns: int = 512
        self.pixel_spacing: Tuple[float, float] = (1.0, 1.0)

        # RT Image specific
        self.rt_image_label: str = ""
        self.rt_image_description: str = ""
        self.rt_image_plane: str = "NORMAL"  # NORMAL, NON_NORMAL
        self.x_ray_image_receptor_angle: float = 0.0
        self.image_plane_pixel_spacing: Tuple[float, float] = (1.0, 1.0)
        self.rt_image_position: Tuple[float, float] = (0.0, 0.0)
        self.rt_image_sid: float = 1000.0  # Source to image distance

        # Gantry angle
        self.gantry_angle: float = 0.0
        self.beam_limiting_device_angle: float = 0.0
        self.patient_support_angle: float = 0.0

        # Pixel data
        self._pixel_data: Optional[np.ndarray] = None

        # Referenced plan/beam
        self._referenced_plan_uid: Optional[str] = None
        self._referenced_beam_number: Optional[int] = None

    def set_pixel_data(self, data: np.ndarray):
        """Set image pixel data."""
        self._pixel_data = data
        self.rows, self.columns = data.shape

    def get_pixel_data(self) -> Optional[np.ndarray]:
        """Get image pixel data."""
        return self._pixel_data

    def set_referenced_beam(self, plan_uid: str, beam_number: int):
        """Set referenced plan and beam."""
        self._referenced_plan_uid = plan_uid
        self._referenced_beam_number = beam_number

    def to_dicom_dict(self) -> Dict[str, Any]:
        """Convert to DICOM attribute dictionary."""
        result = {}
        result.update(self.patient.to_dicom_dict())
        result.update(self.study.to_dicom_dict())
        result.update(self.series.to_dicom_dict())

        result["SOPClassUID"] = self.sop_class_uid
        result["SOPInstanceUID"] = self.sop_instance_uid
        result["RTImageLabel"] = self.rt_image_label
        result["RTImageDescription"] = self.rt_image_description
        result["RTImagePlane"] = self.rt_image_plane
        result["XRayImageReceptorAngle"] = self.x_ray_image_receptor_angle
        result["ImagePlanePixelSpacing"] = list(self.image_plane_pixel_spacing)
        result["RTImagePosition"] = list(self.rt_image_position)
        result["RTImageSID"] = self.rt_image_sid
        result["GantryAngle"] = self.gantry_angle
        result["BeamLimitingDeviceAngle"] = self.beam_limiting_device_angle
        result["PatientSupportAngle"] = self.patient_support_angle
        result["Rows"] = self.rows
        result["Columns"] = self.columns
        result["PixelSpacing"] = list(self.pixel_spacing)

        if self._referenced_plan_uid and self._referenced_beam_number:
            result["ReferencedRTPlanSequence"] = [{
                "ReferencedSOPClassUID": "1.2.840.10008.5.1.4.1.1.481.5",
                "ReferencedSOPInstanceUID": self._referenced_plan_uid,
                "ReferencedFractionGroupSequence": [{
                    "ReferencedFractionGroupNumber": 1,
                    "ReferencedBeamSequence": [{
                        "ReferencedBeamNumber": self._referenced_beam_number,
                    }],
                }],
            }]

        return result


class DICOMRTHandler:
    """
    Handler for DICOM RT file operations.

    Provides import/export functionality for DICOM RT objects.
    Note: This is a simulation - actual DICOM file I/O would
    require pydicom or similar library.
    """

    def __init__(self):
        self._loaded_objects: Dict[str, Any] = {}

    def create_patient(
        self,
        patient_id: str,
        patient_name: str,
        birth_date: Optional[date] = None,
        sex: Optional[str] = None
    ) -> DICOMPatient:
        """Create new patient."""
        return DICOMPatient(
            patient_id=patient_id,
            patient_name=patient_name,
            birth_date=birth_date,
            sex=sex
        )

    def create_study(
        self,
        study_description: str = "RT Treatment"
    ) -> DICOMStudy:
        """Create new study."""
        return DICOMStudy(
            study_instance_uid=DICOMStudy.generate_uid(),
            study_date=date.today(),
            study_time=datetime.now().time(),
            study_description=study_description
        )

    def create_structure_set(
        self,
        patient: DICOMPatient,
        study: DICOMStudy,
        label: str = "Structure Set"
    ) -> RTStructureSet:
        """Create new RT Structure Set."""
        return RTStructureSet(patient, study, label)

    def create_plan(
        self,
        patient: DICOMPatient,
        study: DICOMStudy,
        label: str = "Treatment Plan"
    ) -> RTPlan:
        """Create new RT Plan."""
        return RTPlan(patient, study, label)

    def create_dose(
        self,
        patient: DICOMPatient,
        study: DICOMStudy
    ) -> RTDose:
        """Create new RT Dose."""
        return RTDose(patient, study)

    def create_image(
        self,
        patient: DICOMPatient,
        study: DICOMStudy,
        image_type: str = "DRR"
    ) -> RTImage:
        """Create new RT Image."""
        return RTImage(patient, study, image_type)

    def export_to_dict(self, obj: Any) -> Dict[str, Any]:
        """Export DICOM RT object to dictionary."""
        if hasattr(obj, 'to_dicom_dict'):
            return obj.to_dicom_dict()
        raise ValueError(f"Cannot export object of type {type(obj)}")

    def validate_structure_set(self, ss: RTStructureSet) -> List[str]:
        """Validate RT Structure Set."""
        errors = []

        if not ss.patient.patient_id:
            errors.append("Missing Patient ID")
        if not ss.patient.patient_name:
            errors.append("Missing Patient Name")
        if not ss._rois:
            errors.append("No ROIs defined")

        for roi in ss.get_all_rois():
            if not roi.contours:
                errors.append(f"ROI '{roi.roi_name}' has no contours")

        return errors

    def validate_plan(self, plan: RTPlan) -> List[str]:
        """Validate RT Plan."""
        errors = []

        if not plan.patient.patient_id:
            errors.append("Missing Patient ID")
        if not plan._beams:
            errors.append("No beams defined")
        if not plan.dose_reference_sequence:
            errors.append("No dose references defined")
        if not plan.fraction_group_sequence:
            errors.append("No fraction group defined")

        for beam in plan.get_all_beams():
            if len(beam.control_points) < 2:
                errors.append(f"Beam '{beam.beam_name}' has insufficient control points")

        return errors

    def link_objects(
        self,
        structure_set: RTStructureSet,
        plan: RTPlan,
        dose: RTDose
    ):
        """Link DICOM RT objects together."""
        plan.set_referenced_structure_set(structure_set.sop_instance_uid)
        dose.set_referenced_plan(plan.sop_instance_uid)
