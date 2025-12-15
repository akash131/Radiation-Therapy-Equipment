"""
Example: CyberKnife Robotic Radiosurgery Simulation

This example demonstrates stereotactic radiosurgery treatment
using a CyberKnife-type robotic system with real-time tracking.
"""

import numpy as np

from radiation_therapy import (
    CyberKnifeSystem,
    RoboticArm,
    MotionCompensation,
    ImageGuidedTargeting,
    TreatmentTarget,
    OrganAtRisk,
    Position3D,
)
from radiation_therapy.cyberknife import TrackingModality, TreatmentNode


def main():
    """Run CyberKnife radiosurgery simulation."""
    print("=" * 60)
    print("CyberKnife Robotic Radiosurgery Simulation")
    print("=" * 60)

    # Initialize CyberKnife system
    print("\n1. Initializing CyberKnife System...")
    cyberknife = CyberKnifeSystem(
        name="CyberKnife M6",
        beam_energy=6.0  # 6 MV fixed
    )

    cyberknife.initialize()
    cyberknife.calibrate()

    print(f"   System: {cyberknife.name}")
    print(f"   Beam energy: {cyberknife.beam_energy} MV")
    print(f"   Available collimators: {cyberknife._available_collimators} mm")

    # Robotic arm status
    print("\n2. Robotic Arm Configuration...")
    robot = cyberknife.robotic_arm
    robot_status = robot.get_status()
    print(f"   6-DOF positioning accuracy: {robot_status['positioning_accuracy']} mm")
    print(f"   Payload capacity: {robot.payload_capacity} kg")
    print(f"   Reach: {robot.reach} mm")

    # Define lung tumor target
    print("\n3. Defining Treatment Target...")
    target = TreatmentTarget(
        name="Lung Lesion",
        center=Position3D(50, -30, 120),  # Right lower lobe
        dimensions=(25, 22, 28),  # Small lesion
        prescribed_dose=54.0,  # Gy total
        fractions=3,  # SBRT (3 fractions)
        margin_ptv=2.0
    )
    print(f"   Target: {target.name}")
    print(f"   Location: ({target.center.x}, {target.center.y}, {target.center.z}) mm")
    print(f"   Size: {target.dimensions[0]}x{target.dimensions[1]}x{target.dimensions[2]} mm")
    print(f"   Volume: {target.volume / 1000:.2f} cm³")
    print(f"   Prescription: {target.prescribed_dose} Gy in {target.fractions} fx")
    print(f"   Dose per fraction: {target.dose_per_fraction:.1f} Gy")

    # Define critical structures
    print("\n4. Defining Critical Structures...")
    spinal_cord = OrganAtRisk(
        name="Spinal Cord",
        center=Position3D(0, 0, 120),
        dimensions=(12, 12, 200),
        max_dose=18.0,  # Gy for 3 fractions
        serial=True
    )

    esophagus = OrganAtRisk(
        name="Esophagus",
        center=Position3D(10, 0, 100),
        dimensions=(15, 15, 100),
        max_dose=27.0,
        serial=True
    )

    print(f"   {spinal_cord.name}: Max {spinal_cord.max_dose} Gy")
    print(f"   {esophagus.name}: Max {esophagus.max_dose} Gy")

    # Motion compensation setup (for lung tracking)
    print("\n5. Setting Up Motion Compensation...")
    motion_comp = cyberknife.motion_compensation
    motion_comp.set_tracking_modality(TrackingModality.LUNG)

    # Acquire breathing reference
    print("   Acquiring breathing model (simulated)...")
    motion_comp.acquire_reference(num_samples=100)

    motion_stats = motion_comp.get_motion_statistics()
    print(f"   Breathing amplitude: {motion_stats['amplitude']}")
    print(f"   Correlation coefficient: {motion_stats['correlation_coefficient']:.3f}")

    # Image guidance setup
    print("\n6. Setting Up Image Guidance...")
    image_guidance = cyberknife.image_guided_targeting

    # Add fiducial markers (typically 3-4 gold seeds)
    image_guidance.add_fiducial("Seed1", Position3D(target.center.x - 10,
                                                     target.center.y,
                                                     target.center.z))
    image_guidance.add_fiducial("Seed2", Position3D(target.center.x + 10,
                                                     target.center.y,
                                                     target.center.z))
    image_guidance.add_fiducial("Seed3", Position3D(target.center.x,
                                                     target.center.y + 10,
                                                     target.center.z))

    # Generate reference DRRs
    image_guidance.generate_drrs()
    print(f"   Fiducials placed: {len(image_guidance._fiducials)}")
    print("   Reference DRRs generated")

    # Select collimator
    print("\n7. Collimator Selection...")
    # Choose collimator based on target size
    min_dimension = min(target.dimensions)
    optimal_collimator = min(
        cyberknife._available_collimators,
        key=lambda x: abs(x - min_dimension * 0.8)
    )
    cyberknife.set_collimator(optimal_collimator)
    print(f"   Selected collimator: {cyberknife._current_collimator} mm")

    # Generate treatment nodes
    print("\n8. Generating Treatment Nodes...")
    nodes = cyberknife.generate_treatment_nodes(
        target,
        num_nodes=120,
        min_source_distance=700.0,
        max_source_distance=1000.0
    )
    print(f"   Total nodes: {len(nodes)}")

    # Analyze node distribution
    distances = [n.source_position.distance_to(target.center) for n in nodes]
    print(f"   Source distance range: {min(distances):.0f} - {max(distances):.0f} mm")

    # Optimize node weights
    print("\n9. Optimizing Node Weights...")
    weights = cyberknife.optimize_node_weights(
        nodes, target, [spinal_cord, esophagus]
    )

    total_mu = sum(n.monitor_units for n in nodes)
    print(f"   Total MU: {total_mu:.1f}")
    print(f"   Average MU per node: {total_mu / len(nodes):.1f}")

    # Non-zero nodes
    active_nodes = sum(1 for n in nodes if n.monitor_units > 0.1)
    print(f"   Active nodes: {active_nodes}")

    # Load treatment plan
    print("\n10. Loading Treatment Plan...")
    cyberknife.load_treatment_plan(nodes, target)
    print(f"   Plan loaded with {len(cyberknife._treatment_nodes)} nodes")

    # Calculate treatment time
    treatment_time = cyberknife.calculate_treatment_time()
    print(f"   Estimated treatment time: {treatment_time / 60:.1f} minutes")

    # Simulate initial image guidance
    print("\n11. Performing Image Guidance...")
    image_guidance.acquire_images()
    offset, error = image_guidance.register_images()
    print(f"   Registration error: {error:.2f} mm")
    print(f"   Position offset: ({offset.x:.1f}, {offset.y:.1f}, {offset.z:.1f}) mm")

    # Start motion tracking
    print("\n12. Starting Motion Tracking...")
    cyberknife.start_motion_tracking()
    motion_status = motion_comp.get_status()
    print(f"   Tracking active: {motion_status['is_tracking']}")
    print(f"   Compensation active: {motion_status['is_compensating']}")

    # Simulate delivery of first few nodes
    print("\n13. Simulating Treatment Delivery...")
    print("   Delivering first 5 nodes as demonstration...")

    for i in range(min(5, len(cyberknife._treatment_nodes))):
        node = cyberknife._treatment_nodes[i]

        # Update motion
        breathing_signal = np.sin(2 * np.pi * i / 20)  # Simulated breathing
        motion_comp.update_position(breathing_signal)

        # Check gating
        if motion_comp.get_gating_signal():
            # Move robot and deliver
            pose = robot.get_current_pose()
            print(f"   Node {i+1}: MU={node.monitor_units:.1f}, "
                  f"Src=({node.source_position.x:.0f}, {node.source_position.y:.0f}, "
                  f"{node.source_position.z:.0f})")
        else:
            print(f"   Node {i+1}: Gated (motion out of tolerance)")

    # Final status
    print("\n14. System Status Summary...")
    status = cyberknife.get_status()
    print(f"   Total MU delivered: {status['total_mu_delivered']:.1f}")
    print(f"   Current node: {status['current_node']} / {status['total_nodes']}")
    print(f"   Beam on: {status['beam_on']}")

    # Motion statistics after treatment
    print("\n15. Motion Statistics During Treatment...")
    final_motion_stats = motion_comp.get_motion_statistics()
    print(f"   Samples collected: {final_motion_stats['num_samples']}")

    print("\n" + "=" * 60)
    print("CyberKnife radiosurgery simulation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
