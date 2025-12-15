"""
Example: Linear Accelerator (LINAC) Treatment Simulation

This example demonstrates a typical photon therapy treatment
using a medical linear accelerator with IMRT delivery.
"""

from radiation_therapy import (
    LinearAccelerator,
    BeamParameters,
    TreatmentTarget,
    OrganAtRisk,
    Position3D,
    ParticleType,
    BeamModality,
)


def main():
    """Run LINAC treatment simulation."""
    print("=" * 60)
    print("Linear Accelerator (LINAC) Treatment Simulation")
    print("=" * 60)

    # Initialize LINAC
    print("\n1. Initializing LINAC...")
    linac = LinearAccelerator(
        name="Varian TrueBeam",
        max_photon_energy=18.0,
        max_electron_energy=22.0
    )

    linac.initialize()
    linac.calibrate()
    print(f"   LINAC initialized: {linac.name}")
    print(f"   Max photon energy: {linac.max_photon_energy} MV")

    # Define treatment target
    print("\n2. Defining treatment target...")
    target = TreatmentTarget(
        name="Prostate PTV",
        center=Position3D(0, 0, 100),  # 100mm deep
        dimensions=(40, 50, 60),  # 4cm x 5cm x 6cm
        prescribed_dose=78.0,  # Gy
        fractions=39,
        margin_ptv=5.0
    )
    print(f"   Target: {target.name}")
    print(f"   Center: ({target.center.x}, {target.center.y}, {target.center.z}) mm")
    print(f"   Prescription: {target.prescribed_dose} Gy in {target.fractions} fractions")
    print(f"   Dose/fraction: {target.dose_per_fraction:.2f} Gy")

    # Configure 7-field IMRT
    print("\n3. Configuring 7-field IMRT...")
    gantry_angles = [0, 51, 102, 153, 204, 255, 306]

    for i, angle in enumerate(gantry_angles):
        linac.set_mode(ParticleType.PHOTON)
        linac.set_energy(6.0)  # 6 MV photons
        linac.set_gantry_angle(angle)
        linac.set_field(100, 120)  # Field size in mm
        print(f"   Field {i+1}: Gantry {angle}°, 6 MV, 10x12 cm")

    # Check system status
    print("\n4. System Status Check...")
    status = linac.get_status()
    print(f"   Mode: {status['mode']}")
    print(f"   Energy: {status['energy']} MV")
    print(f"   Dose rate: {status['dose_rate']} MU/min")
    print(f"   Interlocks OK: {status['interlocks_ok']}")

    # Simulate IMRT delivery for one field
    print("\n5. Simulating IMRT Delivery (first field)...")
    beam_params = BeamParameters(
        energy=6.0,
        particle_type=ParticleType.PHOTON,
        dose_rate=600.0,
        field_size_x=10.0,
        field_size_y=12.0,
        gantry_angle=0.0,
        modality=BeamModality.IMRT
    )

    dose_dist = linac.deliver_dose(beam_params, monitor_units=100.0)

    print(f"   Monitor units delivered: {linac._monitor_units_delivered}")
    print(f"   Max dose: {dose_dist.get_max_dose():.3f} Gy")
    print(f"   Mean dose: {dose_dist.get_mean_dose():.6f} Gy")

    # MLC demonstration
    print("\n6. Multi-Leaf Collimator Configuration...")
    mlc = linac.mlc
    print(f"   Number of leaf pairs: {mlc.num_leaf_pairs}")
    print(f"   Leaf width at isocenter: {mlc.leaf_width} mm")

    # Create irregular field shape
    aperture = mlc.get_aperture()
    print(f"   Current aperture: {len(aperture)} leaf pairs")

    # Tumor tracking
    print("\n7. Tumor Tracking Status...")
    tracker = linac.tumor_tracker
    tracker.initialize([])
    tracker.set_reference_position(target.center)
    tracking_status = tracker.get_status()
    print(f"   Tracking active: {tracking_status['is_tracking']}")
    print(f"   Threshold: {tracking_status['tracking_threshold']} mm")

    print("\n" + "=" * 60)
    print("LINAC treatment simulation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
