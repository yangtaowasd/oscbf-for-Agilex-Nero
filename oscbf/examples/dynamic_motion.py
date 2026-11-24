"""Testing the performance of OSCBF during dynamic motions and input constraints

In general, we will command a rapid motion of the end-effector into the unsafe set,
and observe the controller's behavior under velocity control and torque control.

The reduced-order (velocity-control) OSCBF has no lower-level understanding of torque
limits, so the full-order (torque-control) OSCBF should perform better in this case.
"""

import argparse
from itertools import count

import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


from cbfpy import CBF
from oscbf.core.manipulator import Manipulator, NERO_FLANGE_URDF, NERO_READY, load_nero, load_panda
from oscbf.core.manipulation_env import FrankaTorqueControlEnv, FrankaVelocityControlEnv, ManipulationEnv
from oscbf.core.nero_control import MIT_LIMIT
from oscbf.core.oscbf_configs import OSCBFTorqueConfig, OSCBFVelocityConfig
from oscbf.utils.trajectory import SinusoidalTaskTrajectory
from oscbf.utils.visualization import visualize_3D_sphere
from oscbf.examples.nero_workspace import (
    geometry_margins, manual_torque_control, manual_velocity_control, PostureRecovery,
)
from oscbf.core.controllers import (
    PoseTaskTorqueController,
    PoseTaskVelocityController,
    PositionTaskTorqueController,
    PositionTaskVelocityController,
)

DATA_DIR = "oscbf/experiments/data/"
SAVE_DATA = False
PAUSE_FOR_PICTURES = False
RECORD_VIDEO = False
PICTURE_IDXS = [1000, 1250, 1600, 1900, 2200]
PANDA_POSTURE = (0., -np.pi / 6, 0., -3 * np.pi / 4, 0., 5 * np.pi / 9, 0.)


# Keep the flange TCP 1 mm inside the requested 700 mm radial limit.
MANUAL_RADIUS = 0.699
MANUAL_MIN = np.array([-0.470, -0.250, 0.150])
MANUAL_MAX = np.array([0.470, 0.250, 0.570])
BLUE_SPHERE_CENTER = np.array([-0.300, 0.000, 0.250])
BLUE_SPHERE_RADIUS = 0.035
# Start near (-400, -100, 400) mm with clearance from the blue obstacle.
MANUAL_POSTURE = (0.06657889, 0.39073832, 0.30844563, 1.28576442,
                  -0.31945470, 0.47671622, 0.04082224)


def position_margins(robot, q, pos_min, pos_max, radius):
    pos = robot.ee_position(q)
    margins = jnp.concatenate([pos_max - pos, pos - pos_min])
    if radius is not None:
        margins = jnp.concatenate([
            margins,
            jnp.atleast_1d(radius - jnp.linalg.norm(pos)),
            geometry_margins(robot, q, pos_min, pos_max, BLUE_SPHERE_CENTER, BLUE_SPHERE_RADIUS),
        ])
    return margins


@jax.tree_util.register_static
class EESafeSetTorqueConfig(OSCBFTorqueConfig):

    def __init__(
        self,
        robot: Manipulator,
        pos_min: ArrayLike,
        pos_max: ArrayLike,
        compensate_centrifugal_coriolis: bool,
        torque_limits=None,
        radius=None,
    ):
        self.radius = radius
        self.pos_min = np.asarray(pos_min)
        self.pos_max = np.asarray(pos_max)
        super().__init__(
            robot, compensate_centrifugal_coriolis=compensate_centrifugal_coriolis,
            torque_limits=torque_limits,
        )

    def h_2(self, z, **kwargs):
        q = z[: self.num_joints]
        return position_margins(self.robot, q, self.pos_min, self.pos_max, self.radius)

    def _P(self, z, **kwargs):
        if self.radius is None:
            return super()._P(z, **kwargs)
        q = z[:self.num_joints]
        M_inv = jnp.linalg.inv(self.robot.mass_matrix(q))
        J = self.robot.ee_jacobian(q)[:3]
        return M_inv.T @ (J.T @ J + 0.01 * jnp.eye(self.num_joints)) @ M_inv

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2


@jax.tree_util.register_static
class EESafeSetVelocityConfig(OSCBFVelocityConfig):

    def __init__(self, robot: Manipulator, pos_min: ArrayLike, pos_max: ArrayLike, radius=None):
        self.radius = radius
        self.pos_min = np.asarray(pos_min)
        self.pos_max = np.asarray(pos_max)
        super().__init__(robot)

    def h_1(self, z, **kwargs):
        q = z[: self.num_joints]
        return position_margins(self.robot, q, self.pos_min, self.pos_max, self.radius)

    def _P(self, z, **kwargs):
        if self.radius is None:
            return super()._P(z, **kwargs)
        J = self.robot.ee_jacobian(z)[:3]
        return J.T @ J + 0.01 * jnp.eye(self.num_joints)

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2


# @partial(jax.jit, static_argnums=(0, 1, 2, 3))
def compute_torque_control(
    robot: Manipulator,
    osc_controller: PoseTaskTorqueController,
    cbf: CBF,
    compensate_centrifugal_coriolis: bool,
    z: ArrayLike,
    z_ee_des: ArrayLike,
    posture=PANDA_POSTURE,
):
    q = z[: robot.num_joints]
    qdot = z[robot.num_joints :]
    M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)

    if not compensate_centrifugal_coriolis:
        c = jnp.zeros(robot.num_joints)

    # Compute nominal control
    u_nom = osc_controller(
        q,
        qdot,
        pos=ee_tmat[:3, 3],
        rot=ee_tmat[:3, :3],
        des_pos=z_ee_des[:3],
        des_rot=jnp.reshape(z_ee_des[3:12], (3, 3)),
        des_vel=z_ee_des[12:15],
        des_omega=z_ee_des[15:18],
        des_accel=jnp.zeros(3),
        des_alpha=jnp.zeros(3),
        des_q=jnp.asarray(posture),
        des_qdot=jnp.zeros(robot.num_joints),
        J=J,
        M=M,
        M_inv=M_inv,
        g=g,
        c=c,
    )
    # Apply the CBF safety filter
    return cbf.safety_filter(z, u_nom)


# @partial(jax.jit, static_argnums=(0, 1, 2))
def compute_velocity_control(
    robot: Manipulator,
    osc_controller: PoseTaskVelocityController,
    cbf: CBF,
    z: ArrayLike,
    z_ee_des: ArrayLike,
    posture=PANDA_POSTURE,
):
    q = z[: robot.num_joints]
    M_inv, J, ee_tmat = robot.dynamically_consistent_velocity_control_matrices(q)
    pos = ee_tmat[:3, 3]
    rot = ee_tmat[:3, :3]
    des_pos = z_ee_des[:3]
    des_rot = jnp.reshape(z_ee_des[3:12], (3, 3))
    des_vel = z_ee_des[12:15]
    des_omega = z_ee_des[15:18]
    u_nom = osc_controller(
        q, pos, rot, des_pos, des_rot, des_vel, des_omega, jnp.asarray(posture), J, M_inv
    )
    return cbf.safety_filter(q, u_nom)


class ManualTarget:
    """Base-frame target coordinates for Nero, relative displacements otherwise."""

    def __init__(self, env, origin, orientation, base_coordinates=False):
        self.base_coordinates = base_coordinates
        self.env = env
        self.origin = np.asarray(origin)
        self.position = self.origin.copy()
        self.orientation = orientation
        client = env.client
        client.configureDebugVisualizer(client.COV_ENABLE_GUI, 1)
        for preview in (client.COV_ENABLE_RGB_BUFFER_PREVIEW, client.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                        client.COV_ENABLE_SEGMENTATION_MARK_PREVIEW):
            client.configureDebugVisualizer(preview, 0)
        self.sliders = []
        widths = env.xyz_max - env.xyz_min
        lower, upper = -widths, widths
        initial = np.zeros(3)
        label = "Target"
        if base_coordinates:
            # Allow the requested target beyond each TCP workspace face.
            lower, upper = env.xyz_min - 0.100, env.xyz_max + 0.100
            initial = self.origin
            label = "Base"
        for axis, low, high, value in zip("XYZ", lower, upper, initial):
            self.sliders.append(client.addUserDebugParameter(
                f"{label} {axis} (mm)", 1000 * low, 1000 * high, 1000 * value
            ))

    def read(self):
        coordinates_mm = np.array([self.env.client.readUserDebugParameter(i) for i in self.sliders])
        goal = 0.001 * coordinates_mm
        if not self.base_coordinates:
            goal = self.origin + goal
        delta = goal - self.position
        # Limit target speed even when a slider moves from end to end in one frame.
        distance = np.linalg.norm(delta)
        step = delta * min(1., 0.08 * self.env.dt / distance) if distance else delta
        self.position = self.position + step
        return self.env.set_target_state(
            self.position, self.orientation, step / self.env.dt, np.zeros(3)
        )


def main(control_method="torque", robot_name="panda", headless=False, steps=None, target=None):
    if control_method not in ("torque", "velocity") or robot_name not in ("panda", "nero"):
        raise ValueError("Unsupported robot or control method")
    if steps is not None and steps < 1:
        raise ValueError("steps must be positive")

    if target is None:
        target = "manual" if robot_name == "nero" and not headless else "trajectory"
    if target not in ("manual", "trajectory"):
        raise ValueError("Unsupported target mode")
    if target == "manual" and headless:
        raise ValueError("Manual target sliders require a PyBullet GUI")

    wide_manual = robot_name == "nero" and target == "manual"
    radius = MANUAL_RADIUS if wide_manual else None
    if robot_name == "nero":
        robot = load_nero()
        posture = MANUAL_POSTURE if wide_manual else NERO_READY
        tf = np.asarray(robot.ee_transform(np.asarray(posture)))
        origin, orientation = tf[:3, 3], tf[:3, :3]
        half_width = np.array([0.04, 0.08, 0.08])
        pos_min, pos_max = (MANUAL_MIN, MANUAL_MAX) if wide_manual else (origin - half_width, origin + half_width)
        amplitude = (0.08, 0, 0)
        torque_limits = MIT_LIMIT
        camera_target = origin * 0.5
    else:
        robot = load_panda()
        posture = PANDA_POSTURE
        pos_min, pos_max = (0.25, -0.25, 0.25), (0.65, 0.25, 0.65)
        origin, orientation = (0.55, 0, 0.45), np.diag([1., -1., -1.])
        amplitude = (0.25, 0, 0)
        torque_limits = None
        camera_target = (0.44, 0.16, 0.28)

    compensate_centrifugal_coriolis = wide_manual
    if control_method == "torque":
        config = EESafeSetTorqueConfig(
            robot, pos_min, pos_max, compensate_centrifugal_coriolis, torque_limits, radius
        )
    else:
        config = EESafeSetVelocityConfig(robot, pos_min, pos_max, radius)
    cbf = CBF.from_config(config)
    traj = SinusoidalTaskTrajectory(
        init_pos=origin, init_rot=orientation, amplitude=amplitude,
        angular_freq=(5, 0, 0), phase=(0, 0, 0),
    )
    env_options = dict(
        xyz_min=pos_min, xyz_max=pos_max, traj=traj if target == "trajectory" else None, real_time=not headless,
        gui=not headless, bg_color=(1, 1, 1), load_floor=False, timestep=0.001,
    )
    if robot_name == "nero":
        env = ManipulationEnv(
            str(NERO_FLANGE_URDF), control_method, q_init=posture, target_pos=origin,
            tau_max=np.asarray(torque_limits), qdot_max=np.asarray(robot.joint_max_velocities),
            motor_friction=0., merge_fixed_links=False, target_scale=0.045, **env_options,
        )
    else:
        env_type = FrankaTorqueControlEnv if control_method == "torque" else FrankaVelocityControlEnv
        env = env_type(**env_options)

    if wide_manual:
        visualize_3D_sphere(
            BLUE_SPHERE_CENTER, BLUE_SPHERE_RADIUS, rgba=(0.1, 0.25, 0.95, 1.),
            client=env.client,
        )

    env.client.resetDebugVisualizerCamera(
        cameraDistance=1.00, cameraYaw=12, cameraPitch=-2.6, cameraTargetPosition=camera_target
    )

    kp_pos = 50.0
    kp_rot = 20.0
    kd_pos = 20.0
    kd_rot = 10.0
    kp_joint = 10.0
    kd_joint = 5.0
    osc_torque_controller = PoseTaskTorqueController(
        n_joints=robot.num_joints,
        kp_task=np.concatenate([kp_pos * np.ones(3), kp_rot * np.ones(3)]),
        kd_task=np.concatenate([kd_pos * np.ones(3), kd_rot * np.ones(3)]),
        kp_joint=kp_joint,
        kd_joint=kd_joint,
        # Note: torque limits will be enforced via the QP. We'll set them to None here
        # because we don't want to clip the values before the QP
        tau_min=None,
        tau_max=None,
    )

    osc_velocity_controller = PoseTaskVelocityController(
        n_joints=robot.num_joints,
        kp_task=np.array([kp_pos, kp_pos, kp_pos, kp_rot, kp_rot, kp_rot]),
        kp_joint=kp_joint,
        # Note: velocity limits will be enforced via the QP
        qdot_min=None,
        qdot_max=None,
    )

    position_torque_controller = PositionTaskTorqueController(
        robot.num_joints, 50., 20., 10., 5., None, None, damping=0.02,
    )
    position_velocity_controller = PositionTaskVelocityController(
        robot.num_joints, 20., 10., None, None, damping=0.02,
    )

    def constraints(q):
        return position_margins(robot, q, jnp.asarray(pos_min), jnp.asarray(pos_max), radius)

    @jax.jit
    def compute_torque_control_jit(z, z_ee_des, recovery_posture):
        if wide_manual:
            return manual_torque_control(
                robot, position_torque_controller, cbf, z, z_ee_des, constraints, recovery_posture,
            )
        return compute_torque_control(
            robot,
            osc_torque_controller,
            cbf,
            compensate_centrifugal_coriolis,
            z,
            z_ee_des,
            posture,
        )

    @jax.jit
    def compute_velocity_control_jit(z, z_ee_des, recovery_posture):
        if wide_manual:
            return manual_velocity_control(
                robot, position_velocity_controller, cbf, z[:robot.num_joints], z_ee_des, constraints, recovery_posture,
            )
        return compute_velocity_control(
            robot, osc_velocity_controller, cbf, z, z_ee_des, posture
        )

    if control_method == "torque":
        compute_control = compute_torque_control_jit
    elif control_method == "velocity":
        compute_control = compute_velocity_control_jit
    else:
        raise ValueError(f"Invalid control method: {control_method}")

    recovery = PostureRecovery(robot, constraints, posture) if wide_manual else None
    read_target = env.get_desired_ee_state
    if target == "manual":
        read_target = ManualTarget(env, origin, orientation, base_coordinates=wide_manual).read
        print("Move the X/Y/Z (mm) sliders in the PyBullet Parameters panel.", flush=True)

    reference_excess = 0.0
    flange_excess = 0.0
    min_x, max_x = np.inf, -np.inf

    def workspace_excess(pos):
        radial_excess = np.linalg.norm(pos) - radius if radius is not None else 0.
        return max(0., radial_excess, np.max(pos_min - pos), np.max(pos - pos_max))

    try:
        iterations = count() if steps is None else range(steps)
        for _ in iterations:
            if not env.client.isConnected():
                break
            state = env.get_joint_state()
            desired = read_target()
            recovery_posture = recovery.update(state, desired, env.t) if recovery else np.asarray(posture)
            command = np.asarray(compute_control(state, desired, recovery_posture))
            if not np.isfinite(command).all():
                raise RuntimeError(f"Nonfinite {control_method} command at t={env.t:.3f} s")
            env.apply_control(command)
            env.step()
            if steps is not None:
                flange = np.asarray(robot.ee_position(env.get_joint_state()[:robot.num_joints]))
                if not np.isfinite(flange).all():
                    raise RuntimeError(f"Nonfinite flange position at t={env.t:.3f} s")
                min_x, max_x = min(min_x, flange[0]), max(max_x, flange[0])
                reference_excess = max(reference_excess, workspace_excess(desired[:3]))
                flange_excess = max(flange_excess, workspace_excess(flange))
        if min_x <= max_x:
            print(f"{robot_name} {control_method}, {env.t:.3f} s: max workspace violation "
                  f"(target / flange) = {reference_excess * 1000:.3f} / {flange_excess * 1000:.3f} mm, "
                  f"flange x travel = {(max_x - min_x) * 1000:.2f} mm")
    finally:
        if env.client.isConnected():
            env.client.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run end-effector safe-set containment experiment."
    )
    parser.add_argument(
        "--control_method",
        type=str,
        choices=["torque", "velocity"],
        default="torque",
        help="Control method to use (default: torque)",
    )
    parser.add_argument("--robot", choices=["panda", "nero"], default="panda")
    parser.add_argument("--headless", action="store_true", help="Run PyBullet without a GUI.")
    parser.add_argument("--steps", type=int, help="Stop after this many 1 ms steps and report box violations.")
    parser.add_argument("--target", choices=["manual", "trajectory"],
                        help="Nero GUI defaults to manual XYZ sliders; headless runs use the trajectory.")
    args = parser.parse_args()
    if args.target == "manual" and args.headless:
        parser.error("--target manual requires a GUI; omit --headless")
    main(args.control_method, args.robot, args.headless, args.steps, args.target)
