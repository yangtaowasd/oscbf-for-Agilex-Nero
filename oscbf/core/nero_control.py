"""Nero task controllers, safety filters, and posture recovery."""

from functools import partial

from cbfpy import CBF
import jax
import jax.numpy as jnp
import numpy as np
import qpax
from scipy.optimize import minimize

from oscbf.core.controllers import PoseTaskTorqueController
from oscbf.core.oscbf_configs import OSCBFTorqueConfig

# Intersection of the default and v1.11+ Nero SDK MIT feed-forward ranges.
MIT_LIMIT = (16., 16., 16., 16., 8., 8., 8.)
WORKSPACE_MIN = (-0.8, -0.8, 0.0)
WORKSPACE_MAX = (0.8, 0.8, 0.9)


class NeroConfig(OSCBFTorqueConfig):
    def h_1(self, z):
        dq = z[self.num_joints:]
        vmax = jnp.asarray(self.robot.joint_max_velocities)
        return jnp.concatenate([vmax - dq, vmax + dq])

    def h_2(self, z):
        q = z[:self.num_joints]
        spheres = self.robot.link_collision_data(q)
        centers, radii = spheres[:, :3], spheres[:, 3:]
        singular_values = jnp.linalg.svd(self.robot.ee_jacobian(q), compute_uv=False)
        return jnp.concatenate([
            q - jnp.asarray(self.robot.joint_lower_limits),
            jnp.asarray(self.robot.joint_upper_limits) - q,
            (centers - radii - jnp.asarray(WORKSPACE_MIN)).ravel(),
            (jnp.asarray(WORKSPACE_MAX) - centers - radii).ravel(),
            jnp.array([jnp.prod(singular_values) - 0.001]),
        ])

    def alpha(self, h):
        return 8.0 * h

    def alpha_2(self, h):
        return 8.0 * h


class NeroController:
    def __init__(self, robot):
        self.robot = robot
        config = NeroConfig(robot, torque_limits=MIT_LIMIT, relax_cbf=False)
        self.limits = np.asarray(config.u_max)
        self.cbf = CBF.from_config(config)
        self.pose = PoseTaskTorqueController(
            robot.num_joints, (40., 40., 40., 16., 16., 16.),
            (13., 13., 13., 8., 8., 8.), 4., 4., None, None,
        )

    @partial(jax.jit, static_argnums=0)
    def _solve(self, state, target, posture):
        q, dq = jnp.split(state, 2)
        M, M_inv, g, c, J, tf = self.robot.torque_control_matrices(q, dq)
        nominal = self.pose(
            q, dq, tf[:3, 3], tf[:3, :3], target[:3], target[3:12].reshape(3, 3),
            target[12:15], target[15:18], jnp.zeros(3), jnp.zeros(3), posture,
            jnp.zeros_like(q), J, M, M_inv, g, c,
        )
        P, _, A, b, G, h = self.cbf.qp_data(state, nominal)
        # Solve in acceleration corrections, tau = nominal + M @ delta.
        # Nero's wrist inertias span several orders of magnitude.
        H = M.T @ P @ M
        delta, _, _, _, converged, _ = qpax.solve_qp(
            (H + H.T) * 0.5, jnp.zeros_like(q), A, b, G @ M, h - G @ nominal,
            solver_tol=1e-5,
        )
        tau = jnp.clip(nominal + M @ delta, -self.limits, self.limits)
        valid = converged & (jnp.max(G @ tau - h) <= 1e-5)
        return tau, valid

    def torque(self, state, target, posture):
        """Return total joint torque in N·m, or reject an infeasible command."""
        tau, valid = self._solve(state, target, posture)
        tau = np.asarray(tau)
        if not bool(valid) or not np.all(np.isfinite(tau)):
            raise ValueError("Nero OSCBF torque QP failed or violated a constraint")
        return tau

    def check_start(self, state):
        margins = np.concatenate([self.cbf.h_2(state), self.cbf.h(state)])
        if not np.all(np.isfinite(margins)) or margins.min() < 0:
            raise ValueError("Initial Nero state is outside the configured OSCBF safe set")


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


def manual_torque_control(
    robot, controller, cbf, state, target, constraint_fn, posture_reference,
):
    """Apply position tracking or posture recovery through the torque safety filter."""
    q, qdot = jnp.split(state, 2)
    M, M_inv, g, c, J, transform = robot.torque_control_matrices(q, qdot)
    secondary_velocity = posture_velocity(robot, q, constraint_fn)
    nominal = controller(
        q, qdot, transform[:3, 3], target[:3], target[12:15], jnp.zeros(3),
        q, secondary_velocity, J[:3], M, M_inv, g, c,
    )
    recovery_acceleration = 8. * jnp.clip(posture_reference - q, -0.25, 0.25) - 5. * qdot
    recovery_nominal = M @ recovery_acceleration + g + c
    nominal = jnp.where(
        jnp.linalg.norm(posture_reference - q) > 1.e-6, recovery_nominal, nominal,
    )
    return cbf.safety_filter(state, nominal)


def manual_velocity_control(
    robot, controller, cbf, q, target, constraint_fn, posture_reference,
):
    """Apply position tracking or posture recovery through the velocity safety filter."""
    M_inv, J, transform = robot.dynamically_consistent_velocity_control_matrices(q)
    secondary_velocity = posture_velocity(robot, q, constraint_fn)
    nominal = controller(
        q, transform[:3, 3], target[:3], target[12:15],
        q + secondary_velocity / controller.kp_joint, J[:3], M_inv,
    )
    recovery_nominal = jnp.clip(1.5 * (posture_reference - q), -0.25, 0.25)
    nominal = jnp.where(
        jnp.linalg.norm(posture_reference - q) > 1.e-6, recovery_nominal, nominal,
    )
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
        bounds = list(
            zip(
                np.asarray(self.robot.joint_lower_limits) + 1.e-4,
                np.asarray(self.robot.joint_upper_limits) - 1.e-4,
            )
        )
        for seed in (q, self.initial_posture):
            result = minimize(
                objective, seed, jac=True, method="SLSQP", bounds=bounds,
                constraints=constraints, options={"maxiter": 100, "ftol": 1.e-10},
            )
            remaining = np.linalg.norm(np.asarray(self.position(result.x)) - goal)
            if (
                np.isfinite(result.x).all()
                and remaining < 0.003
                and np.min(self.constraints(result.x)) >= -1.e-6
            ):
                self.reference = result.x
                self.goal = goal.copy()
                print(
                    f"Nero posture recovery found a feasible solution, error {remaining * 1000:.2f} mm.",
                    flush=True,
                )
                break
        return q if self.reference is None else self.reference
