"""
Example: Proton Therapy Treatment Simulation

This example demonstrates proton therapy treatment planning
and delivery using pencil beam scanning (PBS).
"""

import numpy as np

from radiation_therapy import (
    ProtonTherapySystem,
    TreatmentPlanningSystem,
    BraggPeakOptimizer,
    TreatmentTarget,
    OrganAtRisk,
    BeamParameters,
    Position3D,
    ParticleType,
)


def main():
    """Run proton therapy simulation."""
    print("=" * 60)
    print("Proton Therapy Treatment Simulation")
    print("=" * 60)

    # Initialize proton therapy system
    print("\n1. Initializing Proton Therapy System...")
    proton_system = ProtonTherapySystem(
        name="IBA Proteus Plus",
        max_energy=250.0  # MeV
    )

    proton_system.initialize()
    proton_system.calibrate()

    status = proton_system.get_status()
    print(f"   System: {status['name']}")
    print(f"   Max energy: {proton_system.max_energy} MeV")
    print(f"   Delivery mode: {status['delivery_mode']}")

    # Define treatment target (pediatric brain tumor)
    print("\n2. Defining Treatment Target...")
    target = TreatmentTarget(
        name="Ependymoma PTV",
        center=Position3D(0, 20, 80),  # 80mm depth
        dimensions=(35, 40, 45),  # Irregular shape
        prescribed_dose=54.0,  # Gy (RBE)
        fractions=30,
        margin_ptv=3.0
    )
    print(f"   Target: {target.name}")
    print(f"   Volume: {target.volume / 1000:.1f} cm³")
    print(f"   Prescription: {target.prescribed_dose} Gy(RBE) in {target.fractions} fx")

    # Define OARs
    print("\n3. Defining Organs at Risk...")
    brainstem = OrganAtRisk(
        name="Brainstem",
        center=Position3D(0, 0, 100),
        dimensions=(25, 25, 40),
        max_dose=54.0,  # Gy
        serial=True
    )

    cochlea_l = OrganAtRisk(
        name="Cochlea_L",
        center=Position3D(-40, 30, 70),
        dimensions=(8, 8, 8),
        max_dose=35.0,  # Gy
        mean_dose_constraint=25.0
    )

    print(f"   {brainstem.name}: Max dose constraint {brainstem.max_dose} Gy")
    print(f"   {cochlea_l.name}: Max dose constraint {cochlea_l.max_dose} Gy")

    # Bragg peak analysis
    print("\n4. Bragg Peak Analysis...")
    optimizer = BraggPeakOptimizer(ParticleType.PROTON)

    # Calculate range for target depth
    target_depth = 80  # mm
    required_energy = optimizer.calculate_energy_for_range(target_depth + 20)
    print(f"   Target depth: {target_depth} mm")
    print(f"   Required energy: {required_energy:.1f} MeV")

    # Calculate pristine Bragg peak
    depths = np.linspace(0, 150, 150)
    bragg_curve = optimizer.calculate_bragg_peak(required_energy, depths)
    peak_depth = depths[np.argmax(bragg_curve)]
    print(f"   Bragg peak depth: {peak_depth:.1f} mm")

    # Calculate SOBP for target coverage
    print("\n5. Spread-Out Bragg Peak (SOBP) Calculation...")
    min_depth = target.center.z - target.dimensions[2] / 2
    max_depth = target.center.z + target.dimensions[2] / 2

    sobp, weights = optimizer.calculate_sobp(min_depth, max_depth, depths, num_peaks=15)
    print(f"   SOBP range: {min_depth:.1f} - {max_depth:.1f} mm")
    print(f"   Number of energy layers: {len(weights)}")

    # Treatment planning
    print("\n6. Treatment Planning (PBS/IMPT)...")
    tps = TreatmentPlanningSystem(ParticleType.PROTON)

    plan = tps.create_plan(
        plan_id="PROTON001",
        patient_id="PEDS001",
        targets=[target],
        oars=[brainstem, cochlea_l],
        prescribed_dose=54.0,
        fractions=30
    )

    print(f"   Plan ID: {plan.plan_id}")
    print(f"   Number of beams: {len(plan.beams)}")

    # Generate spot map
    print("\n7. Generating Spot Map...")
    spots = tps.generate_spot_map(target, gantry_angle=0.0, spot_spacing=4.0)
    print(f"   Total spots: {len(spots)}")

    # Group by energy
    energies = set(s.energy for s in spots)
    print(f"   Energy layers: {len(energies)}")
    print(f"   Energy range: {min(energies):.1f} - {max(energies):.1f} MeV")

    # Optimize spot weights
    print("\n8. Optimizing Spot Weights...")
    weights = tps.optimize_spot_weights(target, [brainstem, cochlea_l], 54.0)
    total_mu = sum(s.monitor_units for s in tps._spot_map)
    print(f"   Total MU: {total_mu:.1f}")

    # System configuration for delivery
    print("\n9. Configuring Delivery System...")
    proton_system.set_gantry_angle(0.0)

    # Check accelerator status
    accel_status = proton_system.accelerator.get_status()
    print(f"   Cyclotron extraction energy: {accel_status['extraction_energy']} MeV")
    print(f"   Beam current: {accel_status['beam_current']} nA")

    # Gantry status
    gantry_status = proton_system.gantry.get_status()
    print(f"   Gantry angle: {gantry_status['current_angle']}°")
    print(f"   Source to isocenter: {gantry_status['source_to_isocenter']} mm")

    # Simulate delivery of one layer
    print("\n10. Simulating Beam Delivery...")
    proton_system.set_energy(150.0)
    beam_params = BeamParameters(
        energy=150.0,
        particle_type=ParticleType.PROTON,
        gantry_angle=0.0
    )

    dose = proton_system.deliver_dose(beam_params, 50.0)
    print(f"   Delivery complete")
    print(f"   Max dose: {dose.get_max_dose():.3f} Gy")

    # Final status
    print("\n11. Final System Status...")
    final_status = proton_system.get_status()
    print(f"   Beam on: {final_status['beam_on']}")
    print(f"   Current energy: {final_status['current_energy']} MeV")

    print("\n" + "=" * 60)
    print("Proton therapy simulation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
