"""Tests for Proton and Heavy Ion Therapy components."""

import pytest
import numpy as np

from radiation_therapy.base import (
    Position3D,
    BeamParameters,
    TreatmentTarget,
    OrganAtRisk,
    ParticleType,
)
from radiation_therapy.proton_therapy import (
    Cyclotron,
    Synchrotron,
    BeamTransport,
    BeamlineElement,
    GantrySystem,
    GantryType,
    BraggPeakOptimizer,
    TreatmentPlanningSystem,
    ProtonTherapySystem,
    HeavyIonTherapySystem,
)


class TestCyclotron:
    """Tests for Cyclotron class."""

    def test_initialization(self):
        """Test cyclotron initialization."""
        cyclotron = Cyclotron(max_energy=250.0)
        assert cyclotron.initialize()
        assert cyclotron.parameters.magnetic_field > 0
        assert cyclotron.parameters.rf_frequency > 0

    def test_operation(self):
        """Test cyclotron operation."""
        cyclotron = Cyclotron()
        cyclotron.initialize()

        assert cyclotron.start_operation()
        assert cyclotron._is_operating

        assert cyclotron.beam_on()
        assert cyclotron.is_active

        cyclotron.beam_off()
        assert not cyclotron.is_active

    def test_beam_current(self):
        """Test beam current control."""
        cyclotron = Cyclotron()
        cyclotron.initialize()

        assert cyclotron.set_beam_current(100.0)
        assert cyclotron.parameters.beam_current == 100.0

        # Above max should fail
        assert not cyclotron.set_beam_current(1000.0)

    def test_beam_power(self):
        """Test beam power calculation."""
        cyclotron = Cyclotron(max_energy=250.0)
        cyclotron.initialize()
        cyclotron.set_beam_current(100.0)

        power = cyclotron.calculate_beam_power()
        assert power > 0


class TestSynchrotron:
    """Tests for Synchrotron class."""

    def test_initialization(self):
        """Test synchrotron initialization."""
        synchrotron = Synchrotron(
            max_energy=430.0,
            particle_type=ParticleType.CARBON_ION
        )
        assert synchrotron.initialize()

    def test_energy_setting(self):
        """Test variable energy capability."""
        synchrotron = Synchrotron()
        synchrotron.initialize()

        assert synchrotron.set_energy(300.0)
        assert synchrotron._current_energy == 300.0

        assert synchrotron.set_energy(400.0)
        assert synchrotron._current_energy == 400.0

    def test_acceleration_cycle(self):
        """Test acceleration cycle."""
        synchrotron = Synchrotron()
        synchrotron.initialize()
        synchrotron.set_energy(350.0)

        assert synchrotron.run_cycle()
        assert synchrotron._cycle_phase == "extraction"

    def test_range_calculation(self):
        """Test range in water calculation."""
        synchrotron = Synchrotron(particle_type=ParticleType.PROTON)
        synchrotron.initialize()
        synchrotron.set_energy(200.0)

        range_mm = synchrotron.calculate_range_in_water()
        assert range_mm > 0
        assert range_mm < 400  # Reasonable range for 200 MeV protons


class TestBeamTransport:
    """Tests for BeamTransport class."""

    def test_initialization(self):
        """Test beam transport initialization."""
        transport = BeamTransport()
        assert transport is not None

    def test_element_addition(self):
        """Test adding beamline elements."""
        transport = BeamTransport()

        element = BeamlineElement(
            name="Degrader",
            element_type="degrader",
            position=5.0,
            length=0.5
        )
        transport.add_element(element)
        assert len(transport._elements) == 1

    def test_energy_configuration(self):
        """Test energy degradation configuration."""
        transport = BeamTransport()
        transport.add_element(BeamlineElement(
            "Degrader", "degrader", 5.0, 0.5
        ))
        transport.add_element(BeamlineElement(
            "Q1", "quadrupole", 10.0, 1.0
        ))

        assert transport.configure_for_energy(200.0, 250.0)
        assert transport._is_configured

    def test_beam_transport_simulation(self):
        """Test beam transport through beamline."""
        transport = BeamTransport()
        transport.add_element(BeamlineElement(
            "Q1", "quadrupole", 5.0, 1.0, strength=1.0
        ))
        transport.configure_for_energy(200.0, 250.0)

        position, size = transport.transport_beam()
        assert isinstance(position, Position3D)
        assert len(size) == 2


class TestGantrySystem:
    """Tests for GantrySystem class."""

    def test_initialization(self):
        """Test gantry initialization."""
        gantry = GantrySystem()
        assert gantry.initialize()
        assert gantry._is_calibrated

    def test_rotation(self):
        """Test gantry rotation."""
        gantry = GantrySystem()
        gantry.initialize()

        assert gantry.set_angle(90.0)
        rotation_time = gantry.rotate_to_angle()
        assert gantry._current_angle == 90.0
        assert rotation_time > 0

    def test_scanning(self):
        """Test pencil beam scanning."""
        gantry = GantrySystem()
        gantry.initialize()

        assert gantry.set_scan_position(50.0, -30.0)
        spot = gantry.calculate_spot_position()
        assert spot.x == 50.0
        assert spot.y == -30.0

    def test_scan_range_limits(self):
        """Test scanning range limits."""
        gantry = GantrySystem()
        gantry.initialize()

        # Within range
        assert gantry.set_scan_position(100.0, 100.0)

        # Outside range
        assert not gantry.set_scan_position(300.0, 0.0)


class TestBraggPeakOptimizer:
    """Tests for BraggPeakOptimizer class."""

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = BraggPeakOptimizer(ParticleType.PROTON)
        assert optimizer is not None

    def test_range_calculation(self):
        """Test range calculation."""
        optimizer = BraggPeakOptimizer(ParticleType.PROTON)

        range_150 = optimizer.calculate_range(150.0)
        range_200 = optimizer.calculate_range(200.0)

        assert range_200 > range_150

    def test_energy_for_range(self):
        """Test energy calculation for given range."""
        optimizer = BraggPeakOptimizer(ParticleType.PROTON)

        target_range = 200.0  # mm
        energy = optimizer.calculate_energy_for_range(target_range)
        calculated_range = optimizer.calculate_range(energy)

        assert abs(calculated_range - target_range) < 1.0  # Within 1mm

    def test_bragg_peak_calculation(self):
        """Test Bragg peak curve calculation."""
        optimizer = BraggPeakOptimizer(ParticleType.PROTON)

        depths = np.linspace(0, 300, 300)
        bragg = optimizer.calculate_bragg_peak(200.0, depths)

        assert len(bragg) == 300
        assert bragg.max() == pytest.approx(1.0)
        # Peak should be near calculated range
        peak_depth = depths[np.argmax(bragg)]
        expected_range = optimizer.calculate_range(200.0)
        assert abs(peak_depth - expected_range) < 20  # Within 20mm

    def test_sobp_calculation(self):
        """Test Spread-Out Bragg Peak calculation."""
        optimizer = BraggPeakOptimizer(ParticleType.PROTON)

        depths = np.linspace(0, 200, 200)
        sobp, weights = optimizer.calculate_sobp(100.0, 150.0, depths)

        assert len(sobp) == 200
        assert len(weights) > 0

        # Check uniformity in target region
        target_mask = (depths >= 100.0) & (depths <= 150.0)
        target_dose = sobp[target_mask]
        uniformity = target_dose.std() / target_dose.mean()
        assert uniformity < 0.3  # Reasonable uniformity

    def test_rbe_calculation(self):
        """Test RBE calculation."""
        proton_opt = BraggPeakOptimizer(ParticleType.PROTON)
        carbon_opt = BraggPeakOptimizer(ParticleType.CARBON_ION)

        proton_rbe = proton_opt.calculate_rbe(10.0)
        carbon_rbe = carbon_opt.calculate_rbe(50.0)

        assert proton_rbe == pytest.approx(1.1)
        assert carbon_rbe > proton_rbe  # Carbon has higher RBE


class TestTreatmentPlanningSystem:
    """Tests for TreatmentPlanningSystem class."""

    def test_initialization(self):
        """Test TPS initialization."""
        tps = TreatmentPlanningSystem(ParticleType.PROTON)
        assert tps is not None

    def test_plan_creation(self):
        """Test treatment plan creation."""
        tps = TreatmentPlanningSystem()

        target = TreatmentTarget(
            name="PTV",
            center=Position3D(0, 0, 100),
            dimensions=(30, 30, 40),
            prescribed_dose=60.0,
            fractions=30
        )

        plan = tps.create_plan(
            plan_id="TEST001",
            patient_id="PATIENT001",
            targets=[target],
            oars=[],
            prescribed_dose=60.0,
            fractions=30
        )

        assert plan is not None
        assert plan.plan_id == "TEST001"
        assert len(plan.beams) > 0

    def test_spot_map_generation(self):
        """Test spot map generation for PBS."""
        tps = TreatmentPlanningSystem()

        target = TreatmentTarget(
            name="PTV",
            center=Position3D(0, 0, 100),
            dimensions=(30, 30, 40),
            prescribed_dose=60.0
        )

        spots = tps.generate_spot_map(target, 0.0)
        assert len(spots) > 0


class TestProtonTherapySystem:
    """Tests for complete ProtonTherapySystem."""

    def test_initialization(self):
        """Test system initialization."""
        system = ProtonTherapySystem()
        assert system.initialize()
        assert system._is_ready

    def test_calibration(self):
        """Test system calibration."""
        system = ProtonTherapySystem()
        system.initialize()
        assert system.calibrate()
        assert system._is_calibrated

    def test_energy_setting(self):
        """Test beam energy setting."""
        system = ProtonTherapySystem(max_energy=250.0)
        system.initialize()

        assert system.set_energy(200.0)
        assert system._current_energy == 200.0

        assert not system.set_energy(300.0)  # Above max

    def test_gantry_control(self):
        """Test gantry angle control."""
        system = ProtonTherapySystem()
        system.initialize()

        assert system.set_gantry_angle(45.0)
        assert system.gantry._current_angle == 45.0

    def test_dose_delivery(self):
        """Test dose delivery."""
        system = ProtonTherapySystem()
        system.initialize()
        system.calibrate()

        beam_params = BeamParameters(
            energy=200.0,
            particle_type=ParticleType.PROTON,
            gantry_angle=0.0
        )

        dose = system.deliver_dose(beam_params, 100.0)
        assert dose is not None
        assert dose.get_max_dose() > 0


class TestHeavyIonTherapySystem:
    """Tests for HeavyIonTherapySystem."""

    def test_initialization(self):
        """Test system initialization."""
        system = HeavyIonTherapySystem()
        assert system.initialize()
        assert system.particle_type == ParticleType.CARBON_ION

    def test_rbe_weighted_dose(self):
        """Test RBE-weighted dose calculation."""
        system = HeavyIonTherapySystem()
        system.initialize()
        system.calibrate()
        system.set_energy(300.0)

        beam_params = BeamParameters(
            energy=300.0,
            particle_type=ParticleType.CARBON_ION
        )

        physical_dose = system.deliver_dose(beam_params, 100.0)
        rbe_dose = system.calculate_rbe_weighted_dose(physical_dose)

        # RBE-weighted dose should be higher than physical
        assert rbe_dose.get_max_dose() > physical_dose.get_max_dose()
