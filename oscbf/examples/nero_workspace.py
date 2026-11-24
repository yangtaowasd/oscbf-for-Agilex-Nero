"""Collision geometry and singularity recovery for the interactive Nero simulation."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

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
CAPSULE_RADII = (0.04871226, 0.05563700, 0.04413593, 0.04821160,
                 0.04155978, 0.03658 * 1.10, 0.04036 * 1.10)
LINK_CLEARANCES = (0., 0., 0., 0., 0., 0.006, 0.006)


def capsule_data(robot, q):
    transforms = robot.joint_to_world_transforms(q)
    rotation, translation = transforms[:, :3, :3], transforms[:, :3, 3]
    starts = jnp.einsum("nij,nj->ni", rotation, jnp.asarray(CAPSULE_STARTS)) + translation
    ends = jnp.einsum("nij,nj->ni", rotation, jnp.asarray(CAPSULE_ENDS)) + translation
    radii = jnp.asarray(CAPSULE_RADII) + jnp.asarray(LINK_CLEARANCES)
    return starts, ends, radii


def geometry_margins(robot, q, pos_min, pos_max, obstacle_center, obstacle_radius):
    starts, ends, radii = capsule_data(robot, q)
    # Both end spheres inside the box imply containment of the entire capsule.
    endpoints = jnp.stack([starts[2:], ends[2:]], axis=1)
    extents = radii[2:, None, None]
    box = jnp.concatenate([
        (pos_max - endpoints - extents).ravel(),
        (endpoints - extents - pos_min).ravel(),
    ])
    segments = ends - starts
    length_squared = jnp.sum(segments**2, axis=1)
    fraction = jnp.clip(
        jnp.sum((obstacle_center - starts) * segments, axis=1)
        / jnp.maximum(length_squared, 1.e-12), 0., 1.
    )
    closest = starts + fraction[:, None] * segments
    obstacle = jnp.linalg.norm(closest - obstacle_center, axis=1) - radii - obstacle_radius
    joints = jnp.concatenate([
        q - jnp.asarray(robot.joint_lower_limits),
        jnp.asarray(robot.joint_upper_limits) - q,
    ])
    linear_jacobian = robot.ee_jacobian(q)[:3]
    # Scale the translational manipulability barrier to useful QP units.
    singularity = 100. * (jnp.linalg.det(linear_jacobian @ linear_jacobian.T) - 1.e-4)
    return jnp.concatenate([box, obstacle, jnp.atleast_1d(singularity), joints])


def posture_velocity(robot, q, constraint_fn):
    """Use redundant motion to recover clearance and avoid joint/singular limits."""
    def clearance_cost(configuration):
        margins = constraint_fn(configuration)
        # Geometric and manipulability margins precede the joint-limit margins.
        distances = margins[:-14]
        joint_gaps = margins[-14:]
        return (
            jnp.sum(jnp.maximum(0.04 - distances, 0.)**2)
            + 0.01 * jnp.sum(jnp.maximum(0.20 - joint_gaps, 0.)**2)
        )

    def manipulability(configuration):
        linear_jacobian = robot.ee_jacobian(configuration)[:3]
        return jnp.linalg.slogdet(
            linear_jacobian @ linear_jacobian.T + 1.e-4 * jnp.eye(3)
        )[1]

    lower = jnp.asarray(robot.joint_lower_limits)
    upper = jnp.asarray(robot.joint_upper_limits)
    centering = 0.1 * ((lower + upper) * 0.5 - q) / (upper - lower)
    velocity = (
        -20. * jax.grad(clearance_cost)(q)
        + 0.02 * jax.grad(manipulability)(q)
        + centering
    )
    return jnp.clip(velocity, -0.4, 0.4)


def manual_torque_control(robot, controller, cbf, state, target, constraint_fn, posture_reference):
    q, qdot = jnp.split(state, 2)
    M, M_inv, g, c, J, transform = robot.torque_control_matrices(q, qdot)
    secondary_velocity = posture_velocity(robot, q, constraint_fn)
    nominal = controller(
        q, qdot, transform[:3, 3], target[:3], target[12:15], jnp.zeros(3),
        q, secondary_velocity, J[:3], M, M_inv, g, c,
    )
    recovery_acceleration = 8. * jnp.clip(posture_reference - q, -0.25, 0.25) - 5. * qdot
    recovery_nominal = M @ recovery_acceleration + g + c
    nominal = jnp.where(jnp.linalg.norm(posture_reference - q) > 1.e-6, recovery_nominal, nominal)
    return cbf.safety_filter(state, nominal)


def manual_velocity_control(robot, controller, cbf, q, target, constraint_fn, posture_reference):
    M_inv, J, transform = robot.dynamically_consistent_velocity_control_matrices(q)
    secondary_velocity = posture_velocity(robot, q, constraint_fn)
    nominal = controller(
        q, transform[:3, 3], target[:3], target[12:15],
        q + secondary_velocity / controller.kp_joint, J[:3], M_inv,
    )
    recovery_nominal = jnp.clip(1.5 * (posture_reference - q), -0.25, 0.25)
    nominal = jnp.where(jnp.linalg.norm(posture_reference - q) > 1.e-6, recovery_nominal, nominal)
    return cbf.safety_filter(q, nominal)


class PostureRecovery:
    """Find another feasible joint posture when position tracking stops progressing."""

    def __init__(self, robot, constraint_fn, initial_posture):
        self.robot = robot
        self.constraints = jax.jit(constraint_fn)
        self.constraint_jacobian = jax.jit(jax.jacfwd(constraint_fn))
        self.position = jax.jit(robot.ee_position)
        self.jacobian = jax.jit(robot.ee_jacobian)
        self.initial_posture = np.asarray(initial_posture)
        self.reference = None
        self.goal = None
        self.stalled_since = None
        self.next_attempt = 0.

    def update(self, state, target, time):
        q, qdot = np.split(state, 2)
        goal = np.asarray(target[:3])
        error = np.linalg.norm(np.asarray(self.position(q)) - goal)
        if self.goal is not None and np.linalg.norm(goal - self.goal) > 0.005:
            self.reference = None
        if error < 0.002:
            self.reference = None
            self.stalled_since = None
        speed = np.linalg.norm(np.asarray(self.jacobian(q))[:3] @ qdot)
        if error < 0.005 or speed > 0.003 or np.linalg.norm(target[12:15]) > 0.005:
            self.stalled_since = None
            return q if self.reference is None else self.reference
        if self.stalled_since is None:
            self.stalled_since = time
        if time - self.stalled_since < 0.6 or time < self.next_attempt:
            return q if self.reference is None else self.reference
        self.next_attempt = time + 2.

        def objective(candidate):
            displacement = np.asarray(self.position(candidate)) - goal
            delta = candidate - q
            value = 0.5 * displacement @ displacement + 5.e-6 * (delta @ delta)
            gradient = np.asarray(self.jacobian(candidate))[:3].T @ displacement + 1.e-5 * delta
            return value, gradient

        constraints = {
            "type": "ineq",
            "fun": lambda candidate: np.asarray(self.constraints(candidate)),
            "jac": lambda candidate: np.asarray(self.constraint_jacobian(candidate)),
        }
        bounds = list(zip(
            np.asarray(self.robot.joint_lower_limits) + 1.e-4,
            np.asarray(self.robot.joint_upper_limits) - 1.e-4,
        ))
        for seed in (q, self.initial_posture):
            result = minimize(
                objective, seed, jac=True, method="SLSQP", bounds=bounds,
                constraints=constraints, options={"maxiter": 100, "ftol": 1.e-10},
            )
            remaining = np.linalg.norm(np.asarray(self.position(result.x)) - goal)
            if (np.isfinite(result.x).all() and remaining < 0.003
                    and np.min(self.constraints(result.x)) >= -1.e-6):
                self.reference = result.x
                self.goal = goal.copy()
                print(f"Nero posture recovery found a feasible solution, error {remaining * 1000:.2f} mm.",
                      flush=True)
                break
        return q if self.reference is None else self.reference
