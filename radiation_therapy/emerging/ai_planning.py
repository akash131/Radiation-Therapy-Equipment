"""
AI-Assisted Treatment Planning.

Implements artificial intelligence components for:
- Automatic contouring (auto-segmentation)
- Knowledge-based planning
- Deep learning dose prediction
- Automatic plan optimization
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any, Callable
import numpy as np
from abc import ABC, abstractmethod


class SegmentationModel(Enum):
    """Deep learning segmentation models."""
    UNET = "unet"
    UNET_PLUS_PLUS = "unet++"
    ATTENTION_UNET = "attention_unet"
    NNUNET = "nnunet"
    SWIN_UNETR = "swin_unetr"


class PlanningApproach(Enum):
    """Planning optimization approaches."""
    KNOWLEDGE_BASED = "knowledge_based"
    DEEP_LEARNING = "deep_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    HYBRID = "hybrid"


@dataclass
class ContourResult:
    """Result of auto-contouring."""
    structure_name: str
    contours: List[Dict]  # List of {z, points}
    confidence: float  # 0-1
    dice_predicted: float  # Predicted Dice coefficient
    needs_review: bool = False


@dataclass
class DVHPrediction:
    """Predicted DVH for a structure."""
    structure_name: str
    doses: np.ndarray  # Dose values
    volumes: np.ndarray  # Volume percentages
    confidence_lower: np.ndarray
    confidence_upper: np.ndarray
    d_mean: float
    d_max: float
    v_prescription: float  # Volume at prescription dose


@dataclass
class DosePrediction:
    """3D dose prediction result."""
    dose_grid: np.ndarray
    uncertainty: np.ndarray
    confidence: float
    predicted_metrics: Dict[str, float]


class AutoContouring:
    """
    Automatic structure contouring using deep learning.

    Supports multiple anatomical sites and model architectures.
    """

    # Supported anatomical sites
    SUPPORTED_SITES = [
        "head_and_neck",
        "thorax",
        "abdomen",
        "pelvis",
        "brain",
        "breast",
    ]

    # Organs available for each site
    SITE_ORGANS = {
        "head_and_neck": [
            "brain", "brainstem", "spinal_cord", "parotid_l", "parotid_r",
            "submandibular_l", "submandibular_r", "mandible", "oral_cavity",
            "larynx", "pharynx", "esophagus", "thyroid", "optic_nerve_l",
            "optic_nerve_r", "eye_l", "eye_r", "lens_l", "lens_r",
            "cochlea_l", "cochlea_r",
        ],
        "thorax": [
            "lung_l", "lung_r", "heart", "esophagus", "spinal_cord",
            "trachea", "great_vessels", "brachial_plexus",
        ],
        "abdomen": [
            "liver", "kidney_l", "kidney_r", "spleen", "stomach",
            "bowel", "spinal_cord", "pancreas",
        ],
        "pelvis": [
            "bladder", "rectum", "femur_l", "femur_r", "bowel",
            "prostate", "seminal_vesicles", "penile_bulb",
        ],
        "brain": [
            "brain", "brainstem", "optic_chiasm", "optic_nerve_l",
            "optic_nerve_r", "hippocampus_l", "hippocampus_r",
            "cochlea_l", "cochlea_r", "pituitary", "lens_l", "lens_r",
        ],
        "breast": [
            "lung_l", "lung_r", "heart", "breast_l", "breast_r",
            "chest_wall", "lat_l", "lat_r", "spinal_cord",
        ],
    }

    def __init__(
        self,
        model_type: SegmentationModel = SegmentationModel.NNUNET,
        site: str = "head_and_neck"
    ):
        self.model_type = model_type
        self.site = site

        self._model_loaded = False
        self._model_weights: Optional[str] = None

        # Performance metrics from validation
        self._validation_metrics: Dict[str, Dict] = {}

    def load_model(self, weights_path: str) -> bool:
        """Load pre-trained model weights."""
        self._model_weights = weights_path
        self._model_loaded = True
        return True

    def get_available_structures(self) -> List[str]:
        """Get structures available for current site."""
        return self.SITE_ORGANS.get(self.site, [])

    def segment(
        self,
        ct_volume: np.ndarray,
        structures: Optional[List[str]] = None
    ) -> List[ContourResult]:
        """
        Perform automatic segmentation.

        Args:
            ct_volume: 3D CT array (HU values)
            structures: Optional list of structures to segment

        Returns:
            List of ContourResult objects
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded")

        if structures is None:
            structures = self.get_available_structures()

        results = []

        for structure in structures:
            # Simulate segmentation result
            result = self._segment_structure(ct_volume, structure)
            results.append(result)

        return results

    def _segment_structure(
        self,
        ct_volume: np.ndarray,
        structure: str
    ) -> ContourResult:
        """Segment single structure (simulated)."""
        # Simulated confidence based on structure type
        confidence_map = {
            "brain": 0.95,
            "lung_l": 0.94,
            "lung_r": 0.94,
            "heart": 0.92,
            "liver": 0.91,
            "kidney_l": 0.90,
            "kidney_r": 0.90,
            "bladder": 0.88,
            "rectum": 0.85,
            "parotid_l": 0.87,
            "parotid_r": 0.87,
        }

        confidence = confidence_map.get(structure, 0.80)

        # Generate simulated contours
        contours = []
        num_slices = ct_volume.shape[0]

        for z in range(num_slices // 3, 2 * num_slices // 3):
            # Simple circular contour for simulation
            center = (ct_volume.shape[1] // 2, ct_volume.shape[2] // 2)
            radius = 20 + np.random.randint(-5, 5)
            angles = np.linspace(0, 2 * np.pi, 36)
            points = np.column_stack([
                center[0] + radius * np.cos(angles),
                center[1] + radius * np.sin(angles),
                np.full(36, z)
            ])
            contours.append({"z": z, "points": points.tolist()})

        return ContourResult(
            structure_name=structure,
            contours=contours,
            confidence=confidence,
            dice_predicted=confidence - 0.05 + np.random.random() * 0.1,
            needs_review=confidence < 0.85,
        )

    def evaluate(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate segmentation quality."""
        # Dice coefficient
        intersection = np.sum(prediction & ground_truth)
        dice = 2 * intersection / (np.sum(prediction) + np.sum(ground_truth) + 1e-8)

        # Hausdorff distance (simplified)
        hd95 = np.random.uniform(1, 5)  # Simulated

        # Surface dice
        surface_dice = dice - 0.05 + np.random.random() * 0.1

        return {
            "dice": float(dice),
            "hd95_mm": float(hd95),
            "surface_dice": float(surface_dice),
        }


class KnowledgeBasedPlanning:
    """
    Knowledge-Based Planning (KBP).

    Uses historical plan database to predict achievable
    DVH objectives for new patients.
    """

    def __init__(self):
        self._plan_database: List[Dict] = []
        self._feature_extractor: Optional[Callable] = None

        # Model parameters (trained on database)
        self._model_trained = False
        self._structure_models: Dict[str, Dict] = {}

    def add_plan_to_database(
        self,
        patient_features: Dict[str, float],
        dvh_metrics: Dict[str, Dict[str, float]]
    ):
        """Add historical plan to database."""
        self._plan_database.append({
            "features": patient_features,
            "dvh_metrics": dvh_metrics,
        })

    def extract_features(
        self,
        target_volume: float,
        target_location: Tuple[float, float, float],
        oar_distances: Dict[str, float],
        oar_volumes: Dict[str, float]
    ) -> Dict[str, float]:
        """Extract geometric features for KBP."""
        features = {
            "target_volume": target_volume,
            "target_x": target_location[0],
            "target_y": target_location[1],
            "target_z": target_location[2],
        }

        for organ, distance in oar_distances.items():
            features[f"{organ}_distance"] = distance
            features[f"{organ}_overlap_volume"] = oar_volumes.get(organ, 0)

        return features

    def train_model(self):
        """Train KBP model on database."""
        if len(self._plan_database) < 10:
            raise ValueError("Insufficient plans in database (need >= 10)")

        # Simplified training - would use regression models
        self._model_trained = True

        # Calculate mean DVH metrics per structure
        structure_metrics: Dict[str, List] = {}

        for plan in self._plan_database:
            for structure, metrics in plan["dvh_metrics"].items():
                if structure not in structure_metrics:
                    structure_metrics[structure] = []
                structure_metrics[structure].append(metrics)

        for structure, metrics_list in structure_metrics.items():
            d_means = [m.get("d_mean", 0) for m in metrics_list]
            d_maxs = [m.get("d_max", 0) for m in metrics_list]

            self._structure_models[structure] = {
                "d_mean_avg": np.mean(d_means),
                "d_mean_std": np.std(d_means),
                "d_max_avg": np.mean(d_maxs),
                "d_max_std": np.std(d_maxs),
            }

    def predict_dvh(
        self,
        patient_features: Dict[str, float],
        structures: List[str]
    ) -> Dict[str, DVHPrediction]:
        """Predict achievable DVH for new patient."""
        if not self._model_trained:
            raise RuntimeError("Model not trained")

        predictions = {}

        for structure in structures:
            if structure in self._structure_models:
                model = self._structure_models[structure]

                # Generate predicted DVH
                doses = np.linspace(0, 80, 81)
                volumes = 100 * np.exp(-doses / model["d_mean_avg"])

                predictions[structure] = DVHPrediction(
                    structure_name=structure,
                    doses=doses,
                    volumes=volumes,
                    confidence_lower=volumes - 10,
                    confidence_upper=volumes + 10,
                    d_mean=model["d_mean_avg"],
                    d_max=model["d_max_avg"],
                    v_prescription=0.0,
                )

        return predictions

    def generate_optimization_objectives(
        self,
        predictions: Dict[str, DVHPrediction],
        prescription_dose: float
    ) -> List[Dict]:
        """Generate optimization objectives from predictions."""
        objectives = []

        for structure, prediction in predictions.items():
            # Mean dose objective
            objectives.append({
                "structure": structure,
                "type": "mean_dose",
                "limit": prediction.d_mean * 1.05,  # 5% margin
                "weight": 50,
                "priority": 2,
            })

            # Max dose objective
            objectives.append({
                "structure": structure,
                "type": "max_dose",
                "limit": prediction.d_max * 1.05,
                "weight": 100,
                "priority": 1,
            })

        return objectives


class DeepLearningDosePrediction:
    """
    Deep learning-based 3D dose prediction.

    Predicts dose distribution directly from CT and structures.
    """

    def __init__(
        self,
        model_architecture: str = "3d_unet"
    ):
        self.model_architecture = model_architecture

        self._model_loaded = False
        self._input_channels: int = 0
        self._normalization_params: Dict = {}

    def load_model(self, weights_path: str) -> bool:
        """Load pre-trained dose prediction model."""
        self._model_loaded = True
        return True

    def preprocess_input(
        self,
        ct_volume: np.ndarray,
        structure_masks: Dict[str, np.ndarray],
        prescription_dose: float
    ) -> np.ndarray:
        """Prepare input for dose prediction."""
        # Stack CT and structure masks
        num_structures = len(structure_masks)
        self._input_channels = 1 + num_structures

        # Normalize CT
        ct_normalized = (ct_volume + 1000) / 2000  # -1000 to 1000 HU -> 0 to 1

        # Create multi-channel input
        input_volume = np.zeros(
            (self._input_channels,) + ct_volume.shape,
            dtype=np.float32
        )

        input_volume[0] = ct_normalized

        for i, (name, mask) in enumerate(structure_masks.items()):
            input_volume[i + 1] = mask.astype(np.float32)

        return input_volume

    def predict(
        self,
        ct_volume: np.ndarray,
        structure_masks: Dict[str, np.ndarray],
        prescription_dose: float
    ) -> DosePrediction:
        """Predict 3D dose distribution."""
        if not self._model_loaded:
            raise RuntimeError("Model not loaded")

        # Preprocess
        input_volume = self.preprocess_input(
            ct_volume, structure_masks, prescription_dose
        )

        # Simulated prediction
        predicted_dose = self._simulate_dose_prediction(
            ct_volume, structure_masks, prescription_dose
        )

        # Uncertainty estimation (Monte Carlo dropout simulation)
        uncertainty = np.abs(predicted_dose) * 0.05  # 5% uncertainty

        # Calculate metrics
        target_mask = structure_masks.get("PTV", np.ones_like(ct_volume))
        target_doses = predicted_dose[target_mask > 0]

        metrics = {
            "d_mean_target": float(np.mean(target_doses)),
            "d_max": float(np.max(predicted_dose)),
            "d_min_target": float(np.min(target_doses)),
            "homogeneity_index": float(
                (np.percentile(target_doses, 98) - np.percentile(target_doses, 2)) /
                np.median(target_doses)
            ) if np.median(target_doses) > 0 else 0,
        }

        return DosePrediction(
            dose_grid=predicted_dose,
            uncertainty=uncertainty,
            confidence=0.9,
            predicted_metrics=metrics,
        )

    def _simulate_dose_prediction(
        self,
        ct_volume: np.ndarray,
        structure_masks: Dict[str, np.ndarray],
        prescription_dose: float
    ) -> np.ndarray:
        """Simulate dose prediction (for demonstration)."""
        # Create simple dose distribution
        dose = np.zeros_like(ct_volume, dtype=np.float32)

        # Get target if available
        if "PTV" in structure_masks:
            target = structure_masks["PTV"]
        else:
            # Create synthetic target in center
            center = [s // 2 for s in ct_volume.shape]
            target = np.zeros_like(ct_volume)
            target[
                center[0]-10:center[0]+10,
                center[1]-10:center[1]+10,
                center[2]-10:center[2]+10
            ] = 1

        # Set target dose
        dose[target > 0] = prescription_dose

        # Add falloff
        from scipy.ndimage import gaussian_filter
        dose = gaussian_filter(dose, sigma=3)

        return dose


class AutomaticPlanOptimization:
    """
    Fully automatic treatment plan optimization.

    Combines KBP, dose prediction, and iterative optimization.
    """

    def __init__(self):
        self._kbp = KnowledgeBasedPlanning()
        self._dose_predictor = DeepLearningDosePrediction()

        # Optimization parameters
        self._max_iterations = 100
        self._convergence_threshold = 0.001

    def initialize_kbp(self, database_path: str):
        """Initialize KBP with plan database."""
        # Would load database from file
        # For now, create sample data
        for i in range(20):
            self._kbp.add_plan_to_database(
                patient_features={"target_volume": 100 + i * 10},
                dvh_metrics={
                    "bladder": {"d_mean": 30 + np.random.randn() * 5, "d_max": 60 + np.random.randn() * 5},
                    "rectum": {"d_mean": 35 + np.random.randn() * 5, "d_max": 65 + np.random.randn() * 5},
                }
            )
        self._kbp.train_model()

    def create_automatic_plan(
        self,
        ct_volume: np.ndarray,
        structures: Dict[str, np.ndarray],
        target_name: str,
        prescription_dose: float,
        fractions: int
    ) -> Dict[str, Any]:
        """
        Create fully automatic treatment plan.

        Args:
            ct_volume: CT image array
            structures: Dictionary of structure masks
            target_name: Name of target structure
            prescription_dose: Total prescribed dose
            fractions: Number of fractions

        Returns:
            Complete plan dictionary
        """
        results = {}

        # Step 1: Feature extraction
        target_mask = structures.get(target_name)
        if target_mask is None:
            raise ValueError(f"Target '{target_name}' not found in structures")

        target_volume = np.sum(target_mask > 0)  # voxels
        target_centroid = np.mean(np.where(target_mask > 0), axis=1)

        oar_distances = {}
        oar_volumes = {}
        for name, mask in structures.items():
            if name != target_name:
                if np.sum(mask) > 0:
                    oar_centroid = np.mean(np.where(mask > 0), axis=1)
                    oar_distances[name] = np.linalg.norm(target_centroid - oar_centroid)
                    oar_volumes[name] = np.sum(mask > 0)

        features = self._kbp.extract_features(
            target_volume,
            tuple(target_centroid),
            oar_distances,
            oar_volumes
        )
        results["features"] = features

        # Step 2: KBP DVH prediction
        oar_names = [n for n in structures.keys() if n != target_name]
        dvh_predictions = self._kbp.predict_dvh(features, oar_names)
        results["predicted_dvh"] = {
            name: {"d_mean": pred.d_mean, "d_max": pred.d_max}
            for name, pred in dvh_predictions.items()
        }

        # Step 3: Generate objectives
        objectives = self._kbp.generate_optimization_objectives(
            dvh_predictions, prescription_dose
        )

        # Add target objectives
        objectives.extend([
            {
                "structure": target_name,
                "type": "min_dose",
                "limit": prescription_dose * 0.95,
                "weight": 100,
                "priority": 1,
            },
            {
                "structure": target_name,
                "type": "max_dose",
                "limit": prescription_dose * 1.07,
                "weight": 80,
                "priority": 1,
            },
        ])
        results["objectives"] = objectives

        # Step 4: Dose prediction
        dose_prediction = self._dose_predictor.predict(
            ct_volume, structures, prescription_dose
        )
        results["predicted_dose"] = dose_prediction.predicted_metrics

        # Step 5: Plan quality assessment
        results["quality_metrics"] = self._assess_plan_quality(
            dose_prediction.dose_grid,
            structures,
            target_name,
            prescription_dose,
            objectives
        )

        results["status"] = "completed"
        results["auto_generated"] = True

        return results

    def _assess_plan_quality(
        self,
        dose: np.ndarray,
        structures: Dict[str, np.ndarray],
        target_name: str,
        prescription_dose: float,
        objectives: List[Dict]
    ) -> Dict[str, Any]:
        """Assess plan quality against objectives."""
        metrics = {}
        objective_results = []

        for obj in objectives:
            structure = obj["structure"]
            mask = structures.get(structure)

            if mask is None or np.sum(mask) == 0:
                continue

            structure_doses = dose[mask > 0]

            if obj["type"] == "mean_dose":
                achieved = float(np.mean(structure_doses))
            elif obj["type"] == "max_dose":
                achieved = float(np.max(structure_doses))
            elif obj["type"] == "min_dose":
                achieved = float(np.min(structure_doses))
            else:
                continue

            passed = achieved <= obj["limit"] if "max" in obj["type"] or "mean" in obj["type"] else achieved >= obj["limit"]

            objective_results.append({
                "structure": structure,
                "type": obj["type"],
                "limit": obj["limit"],
                "achieved": achieved,
                "passed": passed,
            })

        # Calculate overall scores
        target_doses = dose[structures[target_name] > 0]

        metrics["conformity_index"] = self._calculate_ci(
            dose, structures[target_name], prescription_dose
        )
        metrics["homogeneity_index"] = (
            np.percentile(target_doses, 98) - np.percentile(target_doses, 2)
        ) / prescription_dose
        metrics["objectives_passed"] = sum(o["passed"] for o in objective_results)
        metrics["objectives_total"] = len(objective_results)
        metrics["pass_rate"] = metrics["objectives_passed"] / max(1, metrics["objectives_total"])
        metrics["objective_results"] = objective_results

        return metrics

    def _calculate_ci(
        self,
        dose: np.ndarray,
        target: np.ndarray,
        prescription_dose: float
    ) -> float:
        """Calculate conformity index."""
        # Volume receiving >= prescription dose
        treated_volume = np.sum(dose >= prescription_dose * 0.95)
        target_volume = np.sum(target > 0)

        if target_volume == 0:
            return 0.0

        # RTOG conformity index
        ci = treated_volume / target_volume

        return float(ci)

    def optimize_iteratively(
        self,
        initial_plan: Dict,
        target_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Iteratively optimize plan to meet targets.

        Uses reinforcement learning-style optimization.
        """
        current_plan = initial_plan.copy()
        history = []

        for iteration in range(self._max_iterations):
            # Evaluate current plan
            current_metrics = current_plan.get("quality_metrics", {})

            # Calculate objective function
            objective_value = self._calculate_objective(
                current_metrics, target_metrics
            )

            history.append({
                "iteration": iteration,
                "objective": objective_value,
                "metrics": current_metrics,
            })

            # Check convergence
            if iteration > 0:
                improvement = history[-2]["objective"] - objective_value
                if improvement < self._convergence_threshold:
                    break

            # Adjust plan parameters (simplified)
            current_plan = self._adjust_plan(current_plan, target_metrics)

        return {
            "final_plan": current_plan,
            "iterations": len(history),
            "history": history,
            "converged": len(history) < self._max_iterations,
        }

    def _calculate_objective(
        self,
        current: Dict,
        target: Dict
    ) -> float:
        """Calculate scalar objective function."""
        total = 0.0

        for metric, target_value in target.items():
            current_value = current.get(metric, 0)
            deviation = abs(current_value - target_value) / max(target_value, 1e-6)
            total += deviation ** 2

        return np.sqrt(total)

    def _adjust_plan(
        self,
        plan: Dict,
        targets: Dict
    ) -> Dict:
        """Adjust plan parameters to improve metrics."""
        # Simplified - would actually modify beam weights, etc.
        adjusted = plan.copy()

        # Random perturbation for demonstration
        if "predicted_dose" in adjusted:
            for key in adjusted["predicted_dose"]:
                adjusted["predicted_dose"][key] *= (0.98 + np.random.random() * 0.04)

        return adjusted
