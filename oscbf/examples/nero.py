"""Run a finite Nero flange tracking trajectory in PyBullet."""

import argparse

import numpy as np
from oscbf.core.manipulation_env import ManipulationEnv
from oscbf.core.manipulator import load_nero, NERO_FLANGE_URDF, NERO_READY
from oscbf.core.nero_control import MIT_LIMIT, NeroController


def make_env(origin, gui=False):
    return ManipulationEnv(
        str(NERO_FLANGE_URDF), "torque", q_init=NERO_READY, target_pos=origin,
        tau_max=np.asarray(MIT_LIMIT), gui=gui, motor_friction=0., merge_fixed_links=False, timestep=0.002,
        real_time=gui,
    )


def flange_target(tf, t):
    """Translate 2 cm along base y and return, starting at rest."""
    offset = 0.01 * (1 - np.cos(np.pi * t))
    pos = tf[:3, 3] + [0, offset, 0]
    vel = [0, 0.01 * np.pi * np.sin(np.pi * t), 0]
    return np.r_[pos, tf[:3, :3].ravel(), vel, np.zeros(3)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    robot = load_nero()
    controller = NeroController(robot)
    posture = np.asarray(NERO_READY)
    tf = np.asarray(robot.ee_transform(posture))
    env = make_env(tf[:3, 3], args.gui)
    try:
        controller.check_start(env.get_joint_state())
        errors = []
        for _ in range(args.steps):
            target = flange_target(tf, env.t)
            pos = target[:3]
            state = env.get_joint_state()
            errors.append(np.linalg.norm(robot.ee_position(state[:7]) - pos))
            env.client.resetBasePositionAndOrientation(env.target, pos, [0, 0, 0, 1])
            env.apply_control(controller.torque(state, target, posture))
            env.step()
        print(f"{args.steps} steps, max flange tracking error {1000 * max(errors):.2f} mm")
    finally:
        env.client.disconnect()


if __name__ == "__main__":
    main()
