"""Tests for Linear Accelerator (LINAC) components."""

import pytest
import numpy as np

from radiation_therapy.base import (
    Position3D,
    Vector3D,
    BeamParameters,
    TreatmentTarget,
    ParticleType,
    BeamModality,
)
from radiation_therapy.linac import (
    ElectronGun,
    ElectronGunState,
    AcceleratingWaveguide,
    WaveguideType,
    BendingMagnet,
    BeamSteering,
    MultiLeafCollimator,
    TumorTracker,
    ImagingModality,
    LinearAccelerator,
)


class TestElectronGun:
    """Tests for ElectronGun class."""

    def test_initialization(self):
        """Test electron gun initialization."""
        gun = ElectronGun()
        assert gun.initialize()
        assert gun.state == ElectronGunState.STANDBY

    def test_warmup(self):
        """Test cathode warmup."""
        gun = ElectronGun()
        gun.initialize()
        assert gun.warmup(2500.0)
        assert gun.state == ElectronGunState.READY
        assert gun.parameters.cathode_temperature == 2500.0

    def test_beam_current_setting(self):
        """Test beam current control."""
        gun = ElectronGun()
        gun.initialize()
        gun.warmup()

        assert gun.set_beam_current(100.0)
        assert gun._beam_current == 100.0

        # Should fail for current above maximum
        assert not gun.set_beam_current(500.0)

    def test_beam_on_off(self):
        """Test beam activation."""
        gun = ElectronGun()
        gun.initialize()
        gun.warmup()

        assert gun.beam_on()
        assert gun.is_active
        assert gun.state == ElectronGunState.ACTIVE

        assert gun.beam_off()
        assert not gun.is_active
        assert gun.state == ElectronGunState.READY

    def test_emission_current_calculation(self):
        """Test thermionic emission calculation."""
        gun = ElectronGun()
        gun.initialize()
        gun.warmup(2500.0)

        emission = gun.get_emission_current()
        assert emission > 0


class TestAcceleratingWaveguide:
    """Tests for AcceleratingWaveguide class."""

    def test_initialization(self):
        """Test waveguide initialization."""
        waveguide = AcceleratingWaveguide()
        assert waveguide.initialize()

    def test_energy_setting(self):
        """Test beam energy configuration."""
        waveguide = AcceleratingWaveguide(max_energy=25.0)
        waveguide.initialize()

        assert waveguide.set_energy(6.0)
        waveguide.power_on()
        assert waveguide.get_beam_energy() == pytest.approx(6.0, rel=0.1)

        # Should fail above max energy
        assert not waveguide.set_energy(30.0)

    def test_rf_power_control(self):
        """Test RF power setting."""
        waveguide = AcceleratingWaveguide()
        waveguide.initialize()

        assert waveguide.set_rf_power(2.5)
        assert waveguide.parameters.rf_power == 2.5


class TestBendingMagnet:
    """Tests for BendingMagnet class."""

    def test_initialization(self):
        """Test bending magnet initialization."""
        magnet = BendingMagnet()
        assert magnet.initialize()

    def test_energy_configuration(self):
        """Test magnetic field for beam energy."""
        magnet = BendingMagnet()
        magnet.initialize()

        assert magnet.set_for_energy(6.0)
        assert magnet.parameters.field_strength > 0

        assert magnet.energize()
        assert magnet._is_energized

    def test_energy_filter_window(self):
        """Test energy filtering calculation."""
        magnet = BendingMagnet()
        magnet.initialize()
        magnet.set_for_energy(6.0)

        min_e, max_e = magnet.calculate_energy_filter_window()
        assert min_e < 6.0 < max_e


class TestBeamSteering:
    """Tests for BeamSteering class."""

    def test_initialization(self):
        """Test steering system initialization."""
        steering = BeamSteering()
        assert steering.initialize()

    def test_steering_setting(self):
        """Test beam steering offset."""
        steering = BeamSteering()
        steering.initialize()

        assert steering.set_steering(2.0, -1.5)
        steering.enable()

        offset = steering.get_current_offset()
        assert offset == (2.0, -1.5)

    def test_steering_limits(self):
        """Test steering range limits."""
        steering = BeamSteering()
        steering.initialize()

        # Should fail for offset > 5mm
        assert not steering.set_steering(10.0, 0.0)


class TestMultiLeafCollimator:
    """Tests for MultiLeafCollimator class."""

    def test_initialization(self):
        """Test MLC initialization and calibration."""
        mlc = MultiLeafCollimator(num_leaf_pairs=60)
        assert mlc.initialize()
        assert mlc._is_calibrated

    def test_rectangular_field(self):
        """Test rectangular field setting."""
        mlc = MultiLeafCollimator(num_leaf_pairs=60)
        mlc.initialize()

        assert mlc.set_rectangular_field(100.0, 100.0)
        aperture = mlc.get_aperture()
        assert len(aperture) == 60

    def test_leaf_position(self):
        """Test individual leaf positioning."""
        mlc = MultiLeafCollimator(num_leaf_pairs=60)
        mlc.initialize()

        assert mlc.set_leaf_position("A", 30, 10.0)
        assert mlc._leaves_a[30].position == 10.0

    def test_transmission_calculation(self):
        """Test beam transmission calculation."""
        mlc = MultiLeafCollimator(num_leaf_pairs=60)
        mlc.initialize()
        mlc.set_rectangular_field(100.0, 100.0)

        # Open field center
        transmission = mlc.calculate_transmission(0.0, 0.0)
        assert transmission == 1.0

        # Blocked region
        transmission = mlc.calculate_transmission(0.0, 200.0)
        assert transmission < 0.1  # Leakage only


class TestTumorTracker:
    """Tests for TumorTracker class."""

    def test_initialization(self):
        """Test tracking system initialization."""
        tracker = TumorTracker()
        assert tracker.initialize([ImagingModality.CBCT])

    def test_reference_position(self):
        """Test reference position setting."""
        tracker = TumorTracker()
        tracker.initialize([ImagingModality.CBCT])

        ref_pos = Position3D(10.0, 20.0, 30.0)
        assert tracker.set_reference_position(ref_pos)

    def test_tracking(self):
        """Test tracking operation."""
        tracker = TumorTracker()
        tracker.initialize([ImagingModality.CBCT])
        tracker.set_reference_position(Position3D(0, 0, 0))

        assert tracker.start_tracking()
        assert tracker._is_tracking

        position = tracker.update_position()
        assert isinstance(position, Position3D)

        tracker.stop_tracking()
        assert not tracker._is_tracking


class TestLinearAccelerator:
    """Tests for complete LinearAccelerator class."""

    def test_initialization(self):
        """Test LINAC initialization."""
        linac = LinearAccelerator()
        assert linac.initialize()
        assert linac._is_ready

    def test_calibration(self):
        """Test LINAC calibration."""
        linac = LinearAccelerator()
        linac.initialize()
        assert linac.calibrate()
        assert linac._is_calibrated

    def test_mode_setting(self):
        """Test photon/electron mode selection."""
        linac = LinearAccelerator()
        linac.initialize()

        assert linac.set_mode(ParticleType.PHOTON)
        assert linac._current_mode == ParticleType.PHOTON

        assert linac.set_mode(ParticleType.ELECTRON)
        assert linac._current_mode == ParticleType.ELECTRON

    def test_energy_setting(self):
        """Test energy selection."""
        linac = LinearAccelerator(max_photon_energy=18.0)
        linac.initialize()
        linac.set_mode(ParticleType.PHOTON)

        assert linac.set_energy(6.0)
        assert linac._current_energy == 6.0

        assert linac.set_energy(15.0)
        assert linac._current_energy == 15.0

        # Should fail above max
        assert not linac.set_energy(25.0)

    def test_gantry_angle(self):
        """Test gantry angle setting."""
        linac = LinearAccelerator()
        linac.initialize()

        assert linac.set_gantry_angle(90.0)
        assert linac._gantry_angle == 90.0

        assert linac.set_gantry_angle(450.0)  # Should normalize
        assert linac._gantry_angle == 90.0

    def test_beam_delivery(self):
        """Test basic beam delivery."""
        linac = LinearAccelerator()
        linac.initialize()
        linac.calibrate()

        beam_params = BeamParameters(
            energy=6.0,
            particle_type=ParticleType.PHOTON,
            gantry_angle=0.0,
            field_size_x=10.0,
            field_size_y=10.0
        )

        dose_dist = linac.deliver_dose(beam_params, 100.0)
        assert dose_dist is not None
        assert dose_dist.get_max_dose() > 0

    def test_interlock_system(self):
        """Test safety interlock functionality."""
        linac = LinearAccelerator()
        linac.initialize()
        linac.calibrate()

        # Normal operation
        assert linac.check_interlocks()

        # Emergency stop
        linac.emergency_stop()
        assert linac._emergency_stop
        assert not linac._beam_on

    def test_status_reporting(self):
        """Test status information."""
        linac = LinearAccelerator()
        linac.initialize()

        status = linac.get_status()
        assert "name" in status
        assert "energy" in status
        assert "electron_gun" in status
        assert "mlc" in status
