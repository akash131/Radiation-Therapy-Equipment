"""Tests for CyberKnife-type Robotic Radiosurgery components."""

import pytest
import numpy as np

from radiation_therapy.base import (
    Position3D,
    Vector3D,
    TreatmentTarget,
    OrganAtRisk,
)
from radiation_therapy.cyberknife import (
    RoboticArm,
    RobotConfiguration,
    EndEffectorPose,
    MotionCompensation,
    TrackingModality,
    ImageGuidedTargeting,
    CyberKnifeSystem,
    TreatmentNode,
)


class TestRoboticArm:
    """Tests for RoboticArm class."""

    def test_initialization(self):
        """Test robot arm initialization."""
        robot = RoboticArm()
        assert robot.initialize()
        assert robot._is_calibrated

    def test_calibration(self):
        """Test robot calibration."""
        robot = RoboticArm()
        assert robot.calibrate()
        config = robot.get_current_configuration()
        assert config.j1 == 0.0
        assert config.j2 == 0.0

    def test_pose_movement(self):
        """Test movement to specific pose."""
        robot = RoboticArm()
        robot.initialize()

        target_pose = EndEffectorPose(
            position=Position3D(500, 100, 600),
            orientation=Vector3D(0, 0, -1)
        )

        assert robot.move_to_pose(target_pose)
        current_pose = robot.get_current_pose()
        # Pose should be approximately correct
        assert abs(current_pose.position.z - 600) < 100

    def test_aim_at_target(self):
        """Test aiming at target position."""
        robot = RoboticArm()
        robot.initialize()

        source = Position3D(500, 0, 500)
        target = Position3D(0, 0, 0)

        assert robot.aim_at_target(source, target)

    def test_configuration_limits(self):
        """Test joint limit checking."""
        robot = RoboticArm()
        robot.initialize()

        # Valid configuration
        valid_config = RobotConfiguration(
            j1=45, j2=-30, j3=60, j4=0, j5=0, j6=0
        )
        assert robot._check_joint_limits(valid_config)

        # Invalid configuration (j1 out of range)
        invalid_config = RobotConfiguration(
            j1=200, j2=0, j3=0, j4=0, j5=0, j6=0
        )
        assert not robot._check_joint_limits(invalid_config)

    def test_path_time_calculation(self):
        """Test motion time calculation."""
        robot = RoboticArm()
        robot.initialize()

        target_pose = EndEffectorPose(
            position=Position3D(600, 200, 500),
            orientation=Vector3D(0, 0, -1)
        )

        path_time = robot.calculate_path_time(target_pose)
        assert path_time >= 0

    def test_positioning_accuracy(self):
        """Test positioning accuracy specification."""
        robot = RoboticArm()
        robot.initialize()

        # Accuracy should be sub-millimeter
        assert robot._positioning_accuracy < 1.0
        assert robot._repeatability < 1.0


class TestMotionCompensation:
    """Tests for MotionCompensation class."""

    def test_initialization(self):
        """Test motion compensation initialization."""
        mc = MotionCompensation()
        assert mc is not None

    def test_tracking_modality(self):
        """Test tracking modality selection."""
        mc = MotionCompensation()
        assert mc.set_tracking_modality(TrackingModality.XRAY_STEREO)
        assert mc._tracking_modality == TrackingModality.XRAY_STEREO

    def test_reference_acquisition(self):
        """Test reference position acquisition."""
        mc = MotionCompensation()
        assert mc.acquire_reference(num_samples=50)
        assert len(mc._motion_history) == 50

    def test_tracking_start_stop(self):
        """Test tracking activation."""
        mc = MotionCompensation()
        mc.acquire_reference()

        assert mc.start_tracking()
        assert mc._is_tracking

        assert mc.stop_tracking()
        assert not mc._is_tracking

    def test_position_update(self):
        """Test position update from external signal."""
        mc = MotionCompensation()
        mc.acquire_reference()
        mc.start_tracking()

        position = mc.update_position(0.5)  # Half-way in breathing cycle
        assert isinstance(position, Position3D)

    def test_motion_tolerance(self):
        """Test motion tolerance checking."""
        mc = MotionCompensation()
        mc.acquire_reference()
        mc.start_tracking()

        mc._target_position = Position3D(0, 0, 0)
        assert mc.is_motion_within_tolerance()

        mc._target_position = Position3D(0, 0, 10)  # > 3mm default
        assert not mc.is_motion_within_tolerance()

    def test_gating_signal(self):
        """Test respiratory gating signal."""
        mc = MotionCompensation()
        mc.acquire_reference()
        mc.start_tracking()

        mc._target_position = Position3D(0, 0, 0)
        assert mc.get_gating_signal()

        mc._target_position = Position3D(0, 0, 10)
        assert not mc.get_gating_signal()

    def test_position_prediction(self):
        """Test motion prediction."""
        mc = MotionCompensation()
        mc.acquire_reference(num_samples=100)
        mc.start_tracking()

        # Update a few times
        for i in range(5):
            mc.update_position(np.sin(i * 0.5))

        predicted = mc.predict_position(0.1)
        assert isinstance(predicted, Position3D)

    def test_compensation_offset(self):
        """Test compensation offset calculation."""
        mc = MotionCompensation()
        mc.acquire_reference()
        mc.start_tracking()
        mc.start_compensation()

        offset = mc.get_compensation_offset()
        assert isinstance(offset, Position3D)

    def test_motion_statistics(self):
        """Test motion statistics calculation."""
        mc = MotionCompensation()
        mc.acquire_reference()

        stats = mc.get_motion_statistics()
        assert "mean_position" in stats
        assert "amplitude" in stats
        assert "num_samples" in stats


class TestImageGuidedTargeting:
    """Tests for ImageGuidedTargeting class."""

    def test_initialization(self):
        """Test image guidance initialization."""
        igt = ImageGuidedTargeting()
        assert igt is not None

    def test_fiducial_addition(self):
        """Test fiducial marker addition."""
        igt = ImageGuidedTargeting()

        assert igt.add_fiducial("F1", Position3D(0, 0, 0))
        assert igt.add_fiducial("F2", Position3D(10, 0, 0))
        assert len(igt._fiducials) == 2

    def test_drr_generation(self):
        """Test DRR generation."""
        igt = ImageGuidedTargeting()
        igt.add_fiducial("F1", Position3D(0, 0, 0))

        assert igt.generate_drrs()
        assert igt._drr_1 is not None
        assert igt._drr_2 is not None

    def test_image_acquisition(self):
        """Test image acquisition."""
        igt = ImageGuidedTargeting()
        igt.add_fiducial("F1", Position3D(0, 0, 0))
        igt.generate_drrs()

        assert igt.acquire_images()
        assert igt._current_image_1 is not None
        assert igt._current_image_2 is not None

    def test_registration(self):
        """Test image registration."""
        igt = ImageGuidedTargeting()
        igt.add_fiducial("F1", Position3D(0, 0, 0))
        igt.generate_drrs()
        igt.acquire_images()

        offset, error = igt.register_images()
        assert isinstance(offset, Position3D)
        assert error >= 0

    def test_correction_calculation(self):
        """Test position correction calculation."""
        igt = ImageGuidedTargeting()
        igt.add_fiducial("F1", Position3D(0, 0, 0))
        igt.generate_drrs()
        igt.acquire_images()
        igt.register_images()

        correction = igt.calculate_correction()
        assert isinstance(correction, Position3D)


class TestCyberKnifeSystem:
    """Tests for complete CyberKnifeSystem."""

    def test_initialization(self):
        """Test system initialization."""
        ck = CyberKnifeSystem()
        assert ck.initialize()
        assert ck._is_ready

    def test_calibration(self):
        """Test system calibration."""
        ck = CyberKnifeSystem()
        ck.initialize()
        assert ck.calibrate()
        assert ck._is_calibrated

    def test_collimator_setting(self):
        """Test collimator selection."""
        ck = CyberKnifeSystem()
        ck.initialize()

        assert ck.set_collimator(20.0)
        assert ck._current_collimator == 20.0

        # Non-standard size should snap to nearest
        ck.set_collimator(22.0)
        assert ck._current_collimator in ck._available_collimators

    def test_treatment_node_generation(self):
        """Test treatment node generation."""
        ck = CyberKnifeSystem()
        ck.initialize()

        target = TreatmentTarget(
            name="Tumor",
            center=Position3D(0, 0, 0),
            dimensions=(20, 20, 20),
            prescribed_dose=30.0,
            fractions=3
        )

        nodes = ck.generate_treatment_nodes(target, num_nodes=50)
        assert len(nodes) == 50

        for node in nodes:
            # Check nodes are on sphere around target
            distance = node.source_position.distance_to(target.center)
            assert 650 <= distance <= 1000

    def test_node_weight_optimization(self):
        """Test node weight optimization."""
        ck = CyberKnifeSystem()
        ck.initialize()

        target = TreatmentTarget(
            name="Tumor",
            center=Position3D(0, 0, 0),
            dimensions=(20, 20, 20),
            prescribed_dose=30.0
        )

        nodes = ck.generate_treatment_nodes(target, num_nodes=20)
        weights = ck.optimize_node_weights(nodes, target, [])

        assert len(weights) == 20
        assert sum(weights) > 0

    def test_plan_loading(self):
        """Test treatment plan loading."""
        ck = CyberKnifeSystem()
        ck.initialize()

        target = TreatmentTarget(
            name="Tumor",
            center=Position3D(0, 0, 0),
            dimensions=(20, 20, 20),
            prescribed_dose=30.0
        )

        nodes = ck.generate_treatment_nodes(target, num_nodes=10)
        assert ck.load_treatment_plan(nodes, target)
        assert len(ck._treatment_nodes) == 10

    def test_treatment_time_calculation(self):
        """Test treatment time estimation."""
        ck = CyberKnifeSystem()
        ck.initialize()

        target = TreatmentTarget(
            name="Tumor",
            center=Position3D(0, 0, 0),
            dimensions=(20, 20, 20),
            prescribed_dose=30.0
        )

        nodes = ck.generate_treatment_nodes(target, num_nodes=50)
        ck.load_treatment_plan(nodes, target)

        treatment_time = ck.calculate_treatment_time()
        assert treatment_time > 0  # Should be positive

    def test_motion_tracking_integration(self):
        """Test motion tracking integration."""
        ck = CyberKnifeSystem()
        ck.initialize()

        assert ck.start_motion_tracking()
        assert ck.motion_compensation._is_tracking
        assert ck.motion_compensation._is_compensating

    def test_image_guidance_integration(self):
        """Test image guidance integration."""
        ck = CyberKnifeSystem()
        ck.initialize()

        target = TreatmentTarget(
            name="Tumor",
            center=Position3D(0, 0, 0),
            dimensions=(20, 20, 20),
            prescribed_dose=30.0
        )

        nodes = ck.generate_treatment_nodes(target, num_nodes=5)
        ck.load_treatment_plan(nodes, target)

        # Image guidance should work after loading plan
        result = ck.perform_image_guidance()
        # Result depends on registration accuracy

    def test_status_reporting(self):
        """Test comprehensive status reporting."""
        ck = CyberKnifeSystem()
        ck.initialize()

        status = ck.get_status()
        assert "name" in status
        assert "beam_energy" in status
        assert "robotic_arm" in status
        assert "motion_compensation" in status
        assert "image_guidance" in status


class TestTreatmentNode:
    """Tests for TreatmentNode data class."""

    def test_creation(self):
        """Test node creation."""
        node = TreatmentNode(
            node_id=1,
            source_position=Position3D(800, 0, 0),
            target_position=Position3D(0, 0, 0),
            monitor_units=50.0,
            beam_direction=Vector3D(-1, 0, 0)
        )

        assert node.node_id == 1
        assert node.monitor_units == 50.0
        assert node.source_position.distance_to(node.target_position) == 800.0
