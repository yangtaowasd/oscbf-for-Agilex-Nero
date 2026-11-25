"""Nero collision envelopes and workspace constraint geometry."""

import jax.numpy as jnp

# Capsules fitted to both visual and collision meshes, in each link's frame.
# Links 6 and 7 keep spherical envelopes, enlarged by 10 percent.
CAPSULE_STARTS = (
    (0., -0.002, -0.02906209),
    (0.00332374, -0.19182503, 0.),
    (-0.00031859, 0., -0.16215795),
    (-0.00000003, -0.07517342, -0.00000250),
    (0.00000001, 0.00001394, -0.16413465),
    (-0.00123, -0.00212, -0.00010),
    (0.01301, 0., -0.02350),
)
CAPSULE_ENDS = (
    (0., -0.002, -0.00001657),
    (0.00332374, -0.00041257, 0.),
    (-0.00031859, 0., -0.00397434),
    (-0.00000003, -0.00001580, -0.00000250),
    (0.00000001, 0.00001394, -0.01942975),
    CAPSULE_STARTS[5],
    CAPSULE_STARTS[6],
)
CAPSULE_RADII = (
    0.04871226,
    0.05563700,
    0.04413593,
    0.04821160,
    0.04155978,
    0.03658 * 1.10,
    0.04036 * 1.10,
)
LINK_CLEARANCES = (0., 0., 0., 0., 0., 0.006, 0.006)


def capsule_data(robot, q):
    """Return capsule endpoints in the base frame and padded radii, in meters."""
    transforms = robot.joint_to_world_transforms(q)
    rotation, translation = transforms[:, :3, :3], transforms[:, :3, 3]
    starts = jnp.einsum("nij,nj->ni", rotation, jnp.asarray(CAPSULE_STARTS)) + translation
    ends = jnp.einsum("nij,nj->ni", rotation, jnp.asarray(CAPSULE_ENDS)) + translation
    radii = jnp.asarray(CAPSULE_RADII) + jnp.asarray(LINK_CLEARANCES)
    return starts, ends, radii


def geometry_margins(robot, q, pos_min, pos_max, obstacle_center, obstacle_radius):
    """Return workspace, obstacle, manipulability, and joint-limit margins."""
    starts, ends, radii = capsule_data(robot, q)
    # Both end spheres inside the box imply containment of the entire capsule.
    endpoints = jnp.stack([starts[2:], ends[2:]], axis=1)
    extents = radii[2:, None, None]
    box = jnp.concatenate(
        [
            (pos_max - endpoints - extents).ravel(),
            (endpoints - extents - pos_min).ravel(),
        ]
    )
    segments = ends - starts
    length_squared = jnp.sum(segments**2, axis=1)
    fraction = jnp.clip(
        jnp.sum((obstacle_center - starts) * segments, axis=1)
        / jnp.maximum(length_squared, 1.e-12), 0., 1.
    )
    closest = starts + fraction[:, None] * segments
    obstacle = jnp.linalg.norm(closest - obstacle_center, axis=1) - radii - obstacle_radius
    joints = jnp.concatenate(
        [
            q - jnp.asarray(robot.joint_lower_limits),
            jnp.asarray(robot.joint_upper_limits) - q,
        ]
    )
    linear_jacobian = robot.ee_jacobian(q)[:3]
    # Scale the translational manipulability barrier to useful QP units.
    singularity = 100. * (jnp.linalg.det(linear_jacobian @ linear_jacobian.T) - 1.e-4)
    return jnp.concatenate([box, obstacle, jnp.atleast_1d(singularity), joints])
