"""Regression tests for the Nero task, torque convention and active barriers."""

import unittest
import numpy as np

from oscbf.core.manipulator import NERO_READY, load_nero
from oscbf.core.nero_control import NeroController


class NeroControlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = load_nero()
        cls.controller = NeroController(cls.robot)

    def test_hold_and_singular_start(self):
        q = np.asarray(NERO_READY)
        state = np.r_[q, np.zeros(7)]
        self.controller.check_start(state)
        tf = np.asarray(self.robot.ee_transform(q))
        target = np.r_[tf[:3, 3], tf[:3, :3].ravel(), np.zeros(6)]
        tau = self.controller.torque(state, target, q)
        np.testing.assert_allclose(tau, self.robot.gravity_vector(q), atol=5e-5)
        with self.assertRaises(ValueError):
            self.controller.check_start(np.zeros(14))

    def test_outward_target_near_wrist_limit(self):
        q = np.asarray(NERO_READY)
        q[-1] = self.robot.joint_upper_limits[-1] - 0.001
        dq = np.zeros(7)
        state = np.r_[q, dq]
        self.controller.check_start(state)
        # Ask for task twist from a positive wrist velocity, into its upper bound.
        M, _, g, _, J, tf = map(np.asarray, self.robot.torque_control_matrices(q, dq))
        twist = J[:, -1]
        target = np.r_[tf[:3, 3], tf[:3, :3].ravel(), twist]
        tau = self.controller.torque(state, target, q)
        ddq = np.linalg.solve(M, tau - g)
        # At rest the second-order position barrier requires ddq <= alpha^2 * gap.
        self.assertLessEqual(ddq[-1], 8.0**2 * 0.001 + 1e-5)
        self.assertLess(abs(ddq[-1] - 0.064), 0.01)
        self.assertTrue(np.all(np.abs(tau) <= self.controller.limits))
