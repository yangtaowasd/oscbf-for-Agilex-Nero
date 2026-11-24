"""Compare the seven-axis flange model to the independent PyBullet URDF loader."""

import unittest

import numpy as np
import pybullet as pb
from pybullet_utils.bullet_client import BulletClient

from oscbf.core.manipulator import NERO_FLANGE_URDF, NERO_READY, load_nero


class NeroDynamicsTest(unittest.TestCase):
    def test_flange_kinematics_and_dynamics(self):
        robot = load_nero()
        client = BulletClient(pb.DIRECT)
        self.addCleanup(lambda: client.disconnect())
        body = client.loadURDF(str(NERO_FLANGE_URDF), useFixedBase=True,
                               flags=pb.URDF_USE_INERTIA_FROM_FILE)
        client.setGravity(0, 0, -9.81)
        info = [client.getJointInfo(body, i) for i in range(client.getNumJoints(body))]
        joints = [j for j in info if j[2] == pb.JOINT_REVOLUTE]
        self.assertEqual([j[1].decode() for j in joints], [f"joint{i}" for i in range(1, 8)])
        self.assertEqual(robot.num_joints, 7)
        np.testing.assert_allclose(robot.joint_lower_limits, [j[8] for j in joints])
        np.testing.assert_allclose(robot.joint_upper_limits, [j[9] for j in joints])
        flange = next(j[0] for j in info if j[12] == b"flange_link")
        zeros = [0.] * 7
        dq = [0.1, 0.2, -0.1, 0.3, 0.2, -0.1, 0.1]
        poses = [zeros, NERO_READY, [0.3, -0.7, 0.5, 0.8, 0.2, -0.35, 0.4],
                 [0.5, 1., -0.5, 1.5, 0.5, 0.5, 0.]]
        for angles in poses:
            with self.subTest(angles=angles):
                q = np.array(angles)
                for joint, angle in zip(joints, q):
                    client.resetJointState(body, joint[0], angle)
                state = client.getLinkState(body, flange, computeForwardKinematics=1)
                transform = np.asarray(robot.ee_transform(q))
                rotation = np.array(client.getMatrixFromQuaternion(state[5])).reshape(3, 3)
                np.testing.assert_allclose(transform[:3, 3], state[4], atol=1e-7)
                np.testing.assert_allclose(transform[:3, :3], rotation, atol=2e-7)
                J = client.calculateJacobian(body, flange, [0., 0., 0.], angles, zeros, zeros)
                np.testing.assert_allclose(robot.ee_jacobian(q), np.vstack(J), atol=1e-8)
                M = np.asarray(robot.mass_matrix(q))
                np.testing.assert_allclose(M, client.calculateMassMatrix(body, angles), atol=1e-9)
                np.testing.assert_allclose(M, M.T, atol=1e-12)
                self.assertGreater(np.linalg.eigvalsh(M).min(), 0)
                g = np.asarray(client.calculateInverseDynamics(body, angles, zeros, zeros))
                bias = client.calculateInverseDynamics(body, angles, dq, zeros)
                np.testing.assert_allclose(robot.gravity_vector(q), g, atol=1e-9)
                np.testing.assert_allclose(robot.centrifugal_coriolis_vector(q, np.array(dq)),
                                           bias - g, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
