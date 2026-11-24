"""Flange pose tracking with Nero torque, joint and workspace constraints."""

from functools import partial

from cbfpy import CBF
import jax
import jax.numpy as jnp
import numpy as np
import qpax

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
