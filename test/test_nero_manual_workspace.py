"""Geometry coverage and singularity regression checks for the manual simulation."""

import unittest
from xml.etree import ElementTree

import numpy as np
import trimesh

from oscbf.core.controllers import PositionTaskTorqueController, PositionTaskVelocityController
from oscbf.core.manipulator import load_nero, NERO_FLANGE_URDF
from oscbf.examples import dynamic_motion as demo
from oscbf.examples.nero_workspace import CAPSULE_STARTS, CAPSULE_ENDS, CAPSULE_RADII, PostureRecovery
from oscbf.utils.urdfpy.utils import parse_origin


class NeroManualWorkspaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = load_nero()

    def test_capsules_cover_visual_and_collision_meshes(self):
        root = ElementTree.parse(NERO_FLANGE_URDF)
        flange_transform = parse_origin(root.find("joint[@name='flange_joint']"))
        for index in range(7):
            names = [f"link{index + 1}"]
            if index == 6:
                names.append("flange_link")
            start = np.asarray(CAPSULE_STARTS[index])
            end = np.asarray(CAPSULE_ENDS[index])
            direction = end - start
            for name in names:
                for kind in ("visual", "collision"):
                    with self.subTest(link=name, kind=kind):
                        mesh = root.find(f"link[@name='{name}']/{kind}/geometry/mesh")
                        vertices = np.asarray(trimesh.load(
                            NERO_FLANGE_URDF.parent / mesh.attrib["filename"], force="mesh",
                        ).vertices)
                        if name == "flange_link":
                            vertices = vertices @ flange_transform[:3, :3].T + flange_transform[:3, 3]
                        fraction = np.clip(
                            (vertices - start) @ direction / max(direction @ direction, 1.e-12),
                            0., 1.,
                        )
                        closest = start + fraction[:, None] * direction
                        self.assertLessEqual(
                            np.linalg.norm(vertices - closest, axis=1).max(),
                            CAPSULE_RADII[index],
                        )

    def test_initial_posture_satisfies_all_constraints(self):
        margins = demo.position_margins(
            self.robot, np.asarray(demo.MANUAL_POSTURE),
            demo.MANUAL_MIN, demo.MANUAL_MAX, demo.MANUAL_RADIUS,
        )
        self.assertGreater(np.asarray(margins).min(), 0.)

    def test_stalled_posture_has_a_feasible_recovery(self):
        q = np.array([0.64414861, 0.54795236, 2.43074506, -1.00999962,
                      -2.64134084, 0.87264381, 1.00315929])
        state = np.r_[q, np.zeros(7)]
        target = np.r_[[-0.4, -0.1, 0.4], np.eye(3).ravel(), np.zeros(6)]
        def constraints(candidate):
            return demo.position_margins(
                self.robot, candidate, demo.MANUAL_MIN, demo.MANUAL_MAX, demo.MANUAL_RADIUS,
            )
        recovery = PostureRecovery(self.robot, constraints, demo.MANUAL_POSTURE)
        recovery.update(state, target, 0.)
        reference = recovery.update(state, target, 1.)
        self.assertIsNotNone(recovery.reference)
        self.assertLess(np.linalg.norm(self.robot.ee_position(reference) - target[:3]), 0.003)
        self.assertGreaterEqual(np.min(constraints(reference)), -1.e-6)

    def test_position_controllers_are_finite_at_straight_arm(self):
        q = np.zeros(7)
        M, M_inv, g, c, J, tf = self.robot.torque_control_matrices(q, q)
        goal = np.asarray(tf[:3, 3]) + np.array([0.02, 0., -0.02])
        torque = PositionTaskTorqueController(7, 50., 20., 10., 5., None, None, damping=0.02)
        velocity = PositionTaskVelocityController(7, 20., 10., None, None, damping=0.02)
        tau = torque(q, q, tf[:3, 3], goal, np.zeros(3), np.zeros(3), q, q, J[:3], M, M_inv, g, c)
        dq = velocity(q, tf[:3, 3], goal, np.zeros(3), q, J[:3], M_inv)
        self.assertTrue(np.isfinite(np.asarray(tau)).all())
        self.assertTrue(np.isfinite(np.asarray(dq)).all())


if __name__ == "__main__":
    unittest.main()
