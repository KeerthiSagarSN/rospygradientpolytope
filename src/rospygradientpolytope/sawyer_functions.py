import numpy as np
from rospygradientpolytope.robot_functions import screw_transform


def transform_70(q):
    # pos70 Compute sigmoid functoon
    #  J = pos70(q) computes the position of the end effector

    right_j0 = q[0]
    right_j1 = q[1]
    right_j2 = q[2]
    right_j3 = q[3]
    right_j4 = q[4]
    right_j5 = q[5]
    right_j6 = q[6]
    T_right_arm_base_link_right_l6 = np.eye(5)

    T_right_arm_base_link_right_l6[1, 1] = (-0.984808260963545 * np.sin(right_j6) + 0.173645296907107 * np.cos(
        right_j6)) * (-0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                              right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                  -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) + 0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) + 0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) + (-0.173645296902421 * np.sin(
        right_j6) - 0.98480826093697 * np.cos(right_j6)) * (0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2) - 0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.sin(
        right_j5) + 1.0 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                   1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                               right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.cos(
        right_j5) - 1.0 * (-0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (
                                       -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                   1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                               right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
                                   right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
                                   right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j5)) + (-0 * np.sin(right_j6) - 0 * np.cos(right_j6)) * (0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2) - 0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) - 1.0 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                   1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                               right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) - 1.0 * (-0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (
                                       -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                   1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                               right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
                                   right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
                                   right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j5))
    T_right_arm_base_link_right_l6[1, 2] = (-0.173645296907107 * np.sin(right_j6) - 0.984808260963545 * np.cos(
        right_j6)) * (-0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                              right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                  -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) + 0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) + 0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) + (
                                                       0 * np.sin(right_j6) - 0 * np.cos(right_j6)) * (0 * (-1.0 * (
                0 * (-1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                 1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
                             right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                 -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
            right_j3) + 1.0 * (1.0 * (
                    1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
                right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                           1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                       right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                           -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
            right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
            right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) - 1.0 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                   1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                               right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) - 1.0 * (-0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (
                                       -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                   1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                               right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
                                   right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
                                   right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j5)) + (0.98480826093697 * np.sin(right_j6) - 0.173645296902421 * np.cos(right_j6)) * (0 * (-1.0 * (0 * (
                -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                            1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
                        right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                            -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.sin(
        right_j5) + 1.0 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                   1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                               right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.cos(
        right_j5) - 1.0 * (-0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (
                                       -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                   1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                               right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
                                   right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
                                   right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j5))
    T_right_arm_base_link_right_l6[1, 3] = -0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) - 0 * (-1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                 right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                     -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) + 0 * (1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                 right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                     -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                              right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.sin(
        right_j5) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                              right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) + 0.999999999973015 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.cos(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.sin(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2) - 0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) + 0 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                 1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                             right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.cos(
        right_j5) + 0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3) + 0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2) - 0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4) - 0 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                                                                                          -0 * np.sin(
                                                                                                                      right_j1) - 1.0 * np.cos(
                                                                                                                      right_j1)) * np.cos(
        right_j0) - 0 * (-1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                 right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                     -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j5) + 0.999999999973015 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                 right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                     -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) - 0 * (-1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                 right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                     -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j5) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)
    T_right_arm_base_link_right_l6[1, 4] = 0.168499999998653 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) - 0 * (
                                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) - 0.168499999998653 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2) - 0.399999999999381 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (
                                                       -1.0 * (1.0 * (
                                                           -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(
                                                   right_j0) - 0 * np.sin(right_j0) * np.sin(right_j1)) * np.sin(
                                                   right_j2) + 0 * (1.0 * (
                                                           -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
                                                   right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(
                                                   right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) + 0.400000000000516 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                 right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 0.1363 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                               1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                           right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                              right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) + 0.11 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) + 0.400000000000516 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0.1363 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                          -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j4) + 0.11 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                        1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                    right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                        -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                        -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (
                                        -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(
                                    right_j0) - 0 * np.sin(right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
                                    right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
                                    right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j5) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0.192500000001273 * np.sin(
        right_j0) + 0.0810000000006978 * np.cos(right_j0)
    T_right_arm_base_link_right_l6[2, 1] = (-0.984808260963545 * np.sin(right_j6) + 0.173645296907107 * np.cos(
        right_j6)) * (-0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                  0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) + 0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) + 0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j4) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) + (-0.173645296902421 * np.sin(
        right_j6) - 0.98480826093697 * np.cos(right_j6)) * (0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                           -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j4)) * np.sin(right_j5) + 1.0 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.cos(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.sin(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                           -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j4)) * np.cos(right_j5) - 1.0 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                           -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) - 0 * (-1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                 right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                     0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j5)) + (
                                                       -0 * np.sin(right_j6) - 0 * np.cos(right_j6)) * (0 * (-1.0 * (
                0 * (-1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                 1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(
                             right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                 0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
            right_j3) + 1.0 * (1.0 * (
                    1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
                right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                           1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                       right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                           0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
            right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) - 1.0 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                   1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                               right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) - 1.0 * (-0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * (
                                       -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.sin(right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                   1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                               right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
                                   right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
                                   right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j5))
    T_right_arm_base_link_right_l6[2, 2] = (-0.173645296907107 * np.sin(right_j6) - 0.984808260963545 * np.cos(
        right_j6)) * (-0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                  0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) + 0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) + 0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j4) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) + (
                                                       0 * np.sin(right_j6) - 0 * np.cos(right_j6)) * (0 * (-1.0 * (
                0 * (-1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                 1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(
                             right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                 0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
            right_j3) + 1.0 * (1.0 * (
                    1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
                right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                           1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                       right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                           0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
            right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) - 1.0 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                   1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                               right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) - 1.0 * (-0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * (
                                       -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.sin(right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                   1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                               right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
                                   right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
                                   right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j5)) + (0.98480826093697 * np.sin(right_j6) - 0.173645296902421 * np.cos(right_j6)) * (0 * (-1.0 * (0 * (
                -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                            1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(
                        right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                            0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.sin(
        right_j5) + 1.0 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                   1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                               right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.cos(
        right_j5) - 1.0 * (-0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * (
                                       -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.sin(right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                   1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                               right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
                                   right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
                                   right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j5))
    T_right_arm_base_link_right_l6[2, 3] = -0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) - 0 * (-1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                 right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                     0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) + 0 * (1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                 right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                     0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.sin(
        right_j5) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) + 0.999999999973015 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.cos(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.sin(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(

  right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                           -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j4)) * np.sin(right_j5) + 0 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.cos(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.sin(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                           -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j4)) * np.cos(right_j5) + 0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * (-1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                 right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                     0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                     -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j4) - 0 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                                                                                          -0 * np.sin(
                                                                                                                      right_j1) - 1.0 * np.cos(
                                                                                                                      right_j1)) * np.sin(
        right_j0) - 0 * (-1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                 right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                     0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j5) + 0.999999999973015 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                 right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                     -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) - 0 * (-1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                 right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                     0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j5) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)
    T_right_arm_base_link_right_l6[2, 4] = 0.168499999998653 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) - 0 * (
                                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) - 0.168499999998653 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2) - 0.399999999999381 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * (
                                                       -1.0 * (1.0 * (
                                                           -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(
                                                   right_j0) + 0 * np.sin(right_j1) * np.cos(right_j0)) * np.sin(
                                                   right_j2) + 0 * (1.0 * (
                                                           -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
                                                   right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(
                                                   right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) + 0.400000000000516 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                 right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 0.1363 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                               1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                           right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) + 0.11 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) + 0.400000000000516 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0.1363 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                          -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j4) + 0.11 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                        1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                    right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                        0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                        -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * (
                                        -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(
                                    right_j0) + 0 * np.sin(right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
                                    right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
                                    right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j5) + 0.0810000000006978 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(
        right_j1) + 0.192500000001273 * np.cos(right_j0)
    T_right_arm_base_link_right_l6[3, 1] = (-0.984808260963545 * np.sin(right_j6) + 0.173645296907107 * np.cos(
        right_j6)) * (-0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * (
                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                      1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(
                              right_j2)) * np.cos(right_j3) + 0 * (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                      1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(
                              right_j2)) * np.sin(right_j3) - 1.0 * (0 * (
                -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(
        right_j4) + 0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3) + 1.0 * (
                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                      1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(right_j2) + 0) * np.cos(
        right_j4) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(right_j2) + 0) + (
                                                       -0.173645296902421 * np.sin(
                                                   right_j6) - 0.98480826093697 * np.cos(right_j6)) * (0 * (-1.0 * (
                0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
            right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
            right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.cos(right_j4)) * np.sin(right_j5) + 1.0 * (1.0 * (0 * (
                -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.sin(right_j4)) * np.cos(right_j5) - 1.0 * (-0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (1.0 * np.sin(
        right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * (-1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                                                                1.0 * np.sin(right_j1) - 0 * np.cos(
                                                                            right_j1)) * np.cos(right_j2) + 0 * np.cos(
        right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
        right_j3) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(right_j2) + 0) * np.sin(right_j5)) + (
                                                       -0 * np.sin(right_j6) - 0 * np.cos(right_j6)) * (0 * (-1.0 * (
                0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
            right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
            right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.cos(right_j4)) * np.cos(right_j5) - 1.0 * (1.0 * (0 * (
                -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.sin(right_j4)) * np.sin(right_j5) - 1.0 * (-0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (1.0 * np.sin(
        right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * (-1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                                                                1.0 * np.sin(right_j1) - 0 * np.cos(
                                                                            right_j1)) * np.cos(right_j2) + 0 * np.cos(
        right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
        right_j3) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(right_j2) + 0) * np.cos(right_j5))
    T_right_arm_base_link_right_l6[3, 2] = (-0.173645296907107 * np.sin(right_j6) - 0.984808260963545 * np.cos(
        right_j6)) * (-0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * (
                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                      1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(
                              right_j2)) * np.cos(right_j3) + 0 * (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                      1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(
                              right_j2)) * np.sin(right_j3) - 1.0 * (0 * (
                -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(
        right_j4) + 0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3) + 1.0 * (
                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                      1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(right_j2) + 0) * np.cos(
        right_j4) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(right_j2) + 0) + (
                                                       0 * np.sin(right_j6) - 0 * np.cos(right_j6)) * (0 * (-1.0 * (
                0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
            right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
            right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.cos(right_j4)) * np.cos(right_j5) - 1.0 * (1.0 * (0 * (
                -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.sin(right_j4)) * np.sin(right_j5) - 1.0 * (-0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (1.0 * np.sin(
        right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * (-1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                                                                1.0 * np.sin(right_j1) - 0 * np.cos(
                                                                            right_j1)) * np.cos(right_j2) + 0 * np.cos(
        right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
        right_j3) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(right_j2) + 0) * np.cos(right_j5)) + (
                                                       0.98480826093697 * np.sin(right_j6) - 0.173645296902421 * np.cos(
                                                   right_j6)) * (0 * (-1.0 * (0 * (
                -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.cos(right_j4)) * np.sin(right_j5) + 1.0 * (1.0 * (0 * (
                -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.sin(right_j4)) * np.cos(right_j5) - 1.0 * (-0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (1.0 * np.sin(
        right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * (-1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                                                                1.0 * np.sin(right_j1) - 0 * np.cos(
                                                                            right_j1)) * np.cos(right_j2) + 0 * np.cos(
        right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
        right_j3) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(right_j2) + 0) * np.sin(right_j5))
    T_right_arm_base_link_right_l6[3, 3] = -0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * (
                                                       -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                   right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(
                                                   right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) + 0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(
        right_j4) + 0 * (-1.0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.cos(right_j4)) * np.sin(right_j5) - 0 * (-1.0 * (0 * (
                -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.cos(right_j4)) * np.cos(right_j5) + 0.999999999973015 * (1.0 * (0 * (
                -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.sin(right_j4)) * np.sin(right_j5) + 0 * (1.0 * (0 * (
                -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.sin(right_j4)) * np.cos(right_j5) + 0 * (
                                                       -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
        right_j3) + 0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.cos(right_j4) - 0 * (
                                                       -0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                   right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(
                                                   right_j1)) * np.cos(right_j2) - 0 * (-1.0 * (
                                                           -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                   right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(
                                                   right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
                                                   right_j3) + 1.0 * (1.0 * (
                                                           -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                   right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(
                                                   right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
                                                   right_j3) + 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(
                                                   right_j1) + 0) * np.cos(right_j3) - 0 * np.sin(
                                                   right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                                                   right_j2) + 0) * np.sin(right_j5) + 0.999999999973015 * (
                                                       -0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                   right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(
                                                   right_j1)) * np.cos(right_j2) - 0 * (-1.0 * (
                                                           -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                   right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(
                                                   right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
                                                   right_j3) + 1.0 * (1.0 * (
                                                           -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                   right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(
                                                   right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
                                                   right_j3) + 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(
                                                   right_j1) + 0) * np.cos(right_j3) - 0 * np.sin(
                                                   right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                                                   right_j2) + 0) * np.cos(right_j5) - 0 * np.sin(
        right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(right_j2) + 0
    T_right_arm_base_link_right_l6[3, 4] = 0.168499999998653 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) - 0 * (
                                                       1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
        right_j2) - 0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) + 0.400000000000516 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(
        right_j2)) * np.sin(right_j3) - 0.1363 * (0 * (
                -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(
        right_j4) - 0 * (-1.0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.cos(right_j4)) * np.cos(right_j5) + 0.11 * (1.0 * (0 * (
                -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.sin(right_j4)) * np.sin(right_j5) + 0.400000000000516 * (
                                                       -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
        right_j3) + 0.1363 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.cos(right_j4) + 0.11 * (
                                                       -0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                   right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(
                                                   right_j1)) * np.cos(right_j2) - 0 * (-1.0 * (
                                                           -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                   right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(
                                                   right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
                                                   right_j3) + 1.0 * (1.0 * (
                                                           -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                   right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(
                                                   right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
                                                   right_j3) + 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(
                                                   right_j1) + 0) * np.cos(right_j3) - 0 * np.sin(
                                                   right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                                                   right_j2) + 0) * np.cos(right_j5) - 0.399999999999381 * np.sin(
        right_j1) + 0 * np.cos(right_j1) - 0 * np.cos(right_j2) + 0.317
    T_right_arm_base_link_right_l6[4, 1] = 0
    T_right_arm_base_link_right_l6[4, 2] = 0
    T_right_arm_base_link_right_l6[4, 3] = 0
    T_right_arm_base_link_right_l6[4, 4] = 1.00000000000000

    T = np.delete(T_right_arm_base_link_right_l6, 0, 0)
    T = np.delete(T, 0, 1)
    return T


def position_70(q):
    T=transform_70(q)
    P = T[0:3, 3]
    return P


def jacobian70(q):
    # jacobian70 Compute Jacobian functoon
    #   J=jacobian70(q)computes the 6x7 Jaocbian matrix of the end effector

    TnE=np.eye(4)
    TnE[2,3]=0.15 # end effector offset
    T=transform_70(q)
    T0E=np.matmul(T,TnE)
    P=TnE[0:3,3]
    L=np.matmul(T[0:3,0:3],P)
    screw_transform_6n=screw_transform(L)

    right_j0 = q[0]
    right_j1 = q[1]
    right_j2 = q[2]
    right_j3 = q[3]
    right_j4 = q[4]
    right_j5 = q[5]
    right_j6 = q[6]
    jaco_b_l6 = np.zeros([7,8])

    jaco_b_l6[1, 1] = -0.168499999998653 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 0.168499999998653 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2) + 0.399999999999381 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * (
                                  -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(
                              right_j0) + 0 * np.sin(right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                              1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
                              right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
                              right_j2)) * np.cos(right_j3) - 0.400000000000516 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 0.1363 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) - 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) + 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                           -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j4)) * np.cos(right_j5) - 0.11 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.cos(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.sin(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                           -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j4)) * np.sin(right_j5) - 0.400000000000516 * (
                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(
                              right_j0)) * np.cos(right_j3) - 0.1363 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                                                    -0 * np.sin(
                                                                                right_j1) - 1.0 * np.cos(
                                                                                right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j4) - 0.11 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                        1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                    right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                        0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                        -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * (
                                        -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(
                                    right_j0) + 0 * np.sin(right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
                                    right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
                                    right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j5) - 0.0810000000006978 * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1) - 0.192500000001273 * np.cos(right_j0);
    jaco_b_l6[1, 2] = -0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + (
                                  0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * (
                                  0.168499999998653 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j2) - 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * (
                                              -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) + 0.400000000000516 * (
                                              1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 0.1363 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(right_j4) + 0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(right_j4) - 0 * (
                                              -1.0 * (0 * (
                                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                              right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (
                                                                  -1.0 * np.sin(right_j1) + 0 * np.cos(
                                                              right_j1) + 0) * np.sin(right_j3)) * np.sin(
                                          right_j4) + 0 * (0 * (
                                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                              right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (
                                                                       -1.0 * np.sin(right_j1) + 0 * np.cos(
                                                                   right_j1) + 0) * np.cos(right_j3)) * np.cos(
                                          right_j4) + 1.0 * (-1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                                          right_j2) + 0) * np.cos(right_j4)) * np.cos(right_j5) + 0.11 * (1.0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.cos(right_j4) + 0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.sin(right_j4) + 1.0 * (
                                                                                                                      -1.0 * (
                                                                                                                          -0 * np.sin(
                                                                                                                      right_j1) - 1.0 * np.cos(
                                                                                                                      right_j1)) * np.sin(
                                                                                                                  right_j2) + 0 * (
                                                                                                                                  1.0 * np.sin(
                                                                                                                              right_j1) - 0 * np.cos(
                                                                                                                              right_j1)) * np.cos(
                                                                                                                  right_j2) - 0 * np.sin(
                                                                                                                  right_j1) + 0 * np.cos(
                                                                                                                  right_j1) + 0 * np.cos(
                                                                                                                  right_j2) + 0) * np.sin(
                              right_j4)) * np.sin(right_j5) + 0.400000000000516 * (
                                              -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
                              right_j3) + 0.1363 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                              right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                              right_j2) + 0) * np.cos(right_j4) + 0.11 * (
                                              -0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) - 0 * (-1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) + 1.0 * (
                                                          -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
                                          right_j3) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                                          right_j2) + 0) * np.cos(right_j5) - 0.399999999999381 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) - 0 * np.cos(right_j2)) + 0 * (
                                  0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) - 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) + 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                           -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j4)) * np.cos(right_j5) - 0 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.cos(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.sin(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                           -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j4)) * np.sin(right_j5) - 0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) - 0 * (-1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                 right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                     0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                     -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j4) - 0 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                                                                                          -0 * np.sin(
                                                                                                                      right_j1) - 1.0 * np.cos(
                                                                                                                      right_j1)) * np.sin(
        right_j0) - 0 * (-1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                 right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                     0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j5) - 0 * np.sin(
        right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1) - 0 * np.cos(right_j0);
    jaco_b_l6[1, 3] = (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1) - 0) * (0.168499999998653 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) - 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) - 0.168499999998653 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2) - 0.259999999999381 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * (
                                                                                         -1.0 * (1.0 * (-1.0 * np.sin(
                                                                                     right_j1) + 0 * np.cos(
                                                                                     right_j1)) * np.sin(
                                                                                     right_j0) + 0 * np.sin(
                                                                                     right_j1) * np.cos(
                                                                                     right_j0)) * np.sin(
                                                                                     right_j2) + 0 * (1.0 * (
                                                                                             -0 * np.sin(
                                                                                         right_j1) - 1.0 * np.cos(
                                                                                         right_j1)) * np.sin(
                                                                                     right_j0) + 0 * np.cos(
                                                                                     right_j0) * np.cos(
                                                                                     right_j1)) * np.cos(
                                                                                     right_j2) + 1.0 * (0 * np.sin(
                                                                                     right_j0) + 1.0 * np.cos(
                                                                                     right_j0)) * np.cos(
                                                                                     right_j2)) * np.cos(
        right_j3) + 0.400000000000516 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                 right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 0.1363 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                               1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                           right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) + 0.11 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) + 0.400000000000516 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0.1363 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                          -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j4) + 0.11 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                        1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                    right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                        0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                        -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * (
                                        -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(
                                    right_j0) + 0 * np.sin(right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
                                    right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
                                    right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j5) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) + (
                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(
                              right_j0)) * (
                                  0.168499999998653 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j2) - 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * (
                                              -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) + 0.400000000000516 * (
                                              1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 0.1363 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(right_j4) + 0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(right_j4) - 0 * (
                                              -1.0 * (0 * (
                                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                              right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (
                                                                  -1.0 * np.sin(right_j1) + 0 * np.cos(
                                                              right_j1) + 0) * np.sin(right_j3)) * np.sin(
                                          right_j4) + 0 * (0 * (
                                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                              right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (
                                                                       -1.0 * np.sin(right_j1) + 0 * np.cos(
                                                                   right_j1) + 0) * np.cos(right_j3)) * np.cos(
                                          right_j4) + 1.0 * (-1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                                          right_j2) + 0) * np.cos(right_j4)) * np.cos(right_j5) + 0.11 * (1.0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.cos(right_j4) + 0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.sin(right_j4) + 1.0 * (
                                                                                                                      -1.0 * (
                                                                                                                          -0 * np.sin(
                                                                                                                      right_j1) - 1.0 * np.cos(
                                                                                                                      right_j1)) * np.sin(
                                                                                                                  right_j2) + 0 * (
                                                                                                                                  1.0 * np.sin(
                                                                                                                              right_j1) - 0 * np.cos(
                                                                                                                              right_j1)) * np.cos(
                                                                                                                  right_j2) - 0 * np.sin(
                                                                                                                  right_j1) + 0 * np.cos(
                                                                                                                  right_j1) + 0 * np.cos(
                                                                                                                  right_j2) + 0) * np.sin(
                              right_j4)) * np.sin(right_j5) + 0.400000000000516 * (
                                              -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
                              right_j3) + 0.1363 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                              right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                              right_j2) + 0) * np.cos(right_j4) + 0.11 * (
                                              -0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) - 0 * (-1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) + 1.0 * (
                                                          -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
                                          right_j3) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                                          right_j2) + 0) * np.cos(right_j5) - 0.259999999999381 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) - 0 * np.cos(right_j2));
    jaco_b_l6[1, 4] = (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) - 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.sin(right_j1) - 0 * np.cos(
        right_j1) - 0 * np.cos(right_j2) - 0) * (0.126499999998653 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) - 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) - 0.126499999998653 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2) + 0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) + 0.400000000000516 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 0.1363 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                           -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j4)) * np.cos(right_j5) + 0.11 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.cos(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.sin(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                           -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j4)) * np.sin(right_j5) + 0.400000000000516 * (-1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(
        right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3) + 0.1363 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                                                                        -0 * np.sin(
                                                                                                    right_j1) - 1.0 * np.cos(
                                                                                                    right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j4) + 0.11 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                        1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                    right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                        0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                        -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * (
                                        -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(
                                    right_j0) + 0 * np.sin(right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
                                    right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
                                    right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j5) + 0 * np.cos(right_j0) * np.cos(right_j1)) + (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                                            -0 * np.sin(right_j1) - 1.0 * np.cos(
                                                                        right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * (
                                  0.126499999998653 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j2) - 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * (
                                              -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) + 0.400000000000516 * (
                                              1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 0.1363 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(right_j4) + 0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(right_j4) - 0 * (
                                              -1.0 * (0 * (
                                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                              right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (
                                                                  -1.0 * np.sin(right_j1) + 0 * np.cos(
                                                              right_j1) + 0) * np.sin(right_j3)) * np.sin(
                                          right_j4) + 0 * (0 * (
                                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                              right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (
                                                                       -1.0 * np.sin(right_j1) + 0 * np.cos(
                                                                   right_j1) + 0) * np.cos(right_j3)) * np.cos(
                                          right_j4) + 1.0 * (-1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                                          right_j2) + 0) * np.cos(right_j4)) * np.cos(right_j5) + 0.11 * (1.0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.cos(right_j4) + 0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.sin(right_j4) + 1.0 * (
                                                                                                                      -1.0 * (
                                                                                                                          -0 * np.sin(
                                                                                                                      right_j1) - 1.0 * np.cos(
                                                                                                                      right_j1)) * np.sin(
                                                                                                                  right_j2) + 0 * (
                                                                                                                                  1.0 * np.sin(
                                                                                                                              right_j1) - 0 * np.cos(
                                                                                                                              right_j1)) * np.cos(
                                                                                                                  right_j2) - 0 * np.sin(
                                                                                                                  right_j1) + 0 * np.cos(
                                                                                                                  right_j1) + 0 * np.cos(
                                                                                                                  right_j2) + 0) * np.sin(
                              right_j4)) * np.sin(right_j5) + 0.400000000000516 * (
                                              -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
                              right_j3) + 0.1363 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                              right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                              right_j2) + 0) * np.cos(right_j4) + 0.11 * (
                                              -0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) - 0 * (-1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) + 1.0 * (
                                                          -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
                                          right_j3) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                                          right_j2) + 0) * np.cos(right_j5) + 0 * np.sin(right_j1) - 0 * np.cos(
                              right_j1) - 0 * np.cos(right_j2));
    jaco_b_l6[1, 5] = (0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) - 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                   -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(
                               right_j2)) * np.cos(right_j3) - 1.0 * (
                                   1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(
                               right_j2)) * np.sin(right_j3) - 1.0 * (
                                   -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3) + 0 * np.sin(
        right_j1) - 0 * np.cos(right_j1) - 0 * np.cos(right_j2) - 0) * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                                                    -1.0 * (1.0 * (-1.0 * np.sin(
                                                                                right_j1) + 0 * np.cos(
                                                                                right_j1)) * np.sin(
                                                                                right_j0) + 0 * np.sin(
                                                                                right_j1) * np.cos(right_j0)) * np.sin(
                                                                                right_j2) + 0 * (1.0 * (-0 * np.sin(
                                                                                right_j1) - 1.0 * np.cos(
                                                                                right_j1)) * np.sin(
                                                                                right_j0) + 0 * np.cos(
                                                                                right_j0) * np.cos(right_j1)) * np.cos(
                                                                                right_j2) + 1.0 * (0 * np.sin(
                                                                                right_j0) + 1.0 * np.cos(
                                                                                right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) + 0.275000000000516 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                 right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 0.1363 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                               1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                           right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) + 0.11 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) + 0.275000000000516 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0.1363 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                          -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j4) + 0.11 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                        1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                    right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                        0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                        -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * (
                                        -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(
                                    right_j0) + 0 * np.sin(right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
                                    right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
                                    right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j5)) + (-0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                  0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * (
                                  -0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                      1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * (
                                              -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) + 0.275000000000516 * (
                                              1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 0.1363 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(right_j4) + 0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(right_j4) - 0 * (
                                              -1.0 * (0 * (
                                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                              right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (
                                                                  -1.0 * np.sin(right_j1) + 0 * np.cos(
                                                              right_j1) + 0) * np.sin(right_j3)) * np.sin(
                                          right_j4) + 0 * (0 * (
                                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                              right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (
                                                                       -1.0 * np.sin(right_j1) + 0 * np.cos(
                                                                   right_j1) + 0) * np.cos(right_j3)) * np.cos(
                                          right_j4) + 1.0 * (-1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                                          right_j2) + 0) * np.cos(right_j4)) * np.cos(right_j5) + 0.11 * (1.0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.cos(right_j4) + 0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.sin(right_j4) + 1.0 * (
                                                                                                                      -1.0 * (
                                                                                                                          -0 * np.sin(
                                                                                                                      right_j1) - 1.0 * np.cos(
                                                                                                                      right_j1)) * np.sin(
                                                                                                                  right_j2) + 0 * (
                                                                                                                                  1.0 * np.sin(
                                                                                                                              right_j1) - 0 * np.cos(
                                                                                                                              right_j1)) * np.cos(
                                                                                                                  right_j2) - 0 * np.sin(
                                                                                                                  right_j1) + 0 * np.cos(
                                                                                                                  right_j1) + 0 * np.cos(
                                                                                                                  right_j2) + 0) * np.sin(
                              right_j4)) * np.sin(right_j5) + 0.275000000000516 * (
                                              -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
                              right_j3) + 0.1363 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                              right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                              right_j2) + 0) * np.cos(right_j4) + 0.11 * (
                                              -0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) - 0 * (-1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) + 1.0 * (
                                                          -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
                                          right_j3) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                                          right_j2) + 0) * np.cos(right_j5) + 0 * np.cos(right_j2));
    jaco_b_l6[1, 6] = (-0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                         1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                     right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                         0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) + 0 * (1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                 right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                     0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 0.1053 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                               1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                           right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) + 0.11 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) + 0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3) + 0.1053 * (
                                   -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(
                               right_j0) + 0 * np.sin(right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                               1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                           right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
                               right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
                               right_j2) - 0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                               right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(
                               right_j0)) * np.cos(right_j4) + 0.11 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                                                   -0 * np.sin(right_j1) - 1.0 * np.cos(
                                                                               right_j1)) * np.sin(right_j0) - 0 * (
                                                                                   -1.0 * (1.0 * (-1.0 * np.sin(
                                                                               right_j1) + 0 * np.cos(
                                                                               right_j1)) * np.sin(
                                                                               right_j0) + 0 * np.sin(
                                                                               right_j1) * np.cos(right_j0)) * np.sin(
                                                                               right_j2) + 0 * (1.0 * (-0 * np.sin(
                                                                               right_j1) - 1.0 * np.cos(
                                                                               right_j1)) * np.sin(
                                                                               right_j0) + 0 * np.cos(
                                                                               right_j0) * np.cos(right_j1)) * np.cos(
                                                                               right_j2) + 1.0 * (0 * np.sin(
                                                                               right_j0) + 1.0 * np.cos(
                                                                               right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j5)) * (
                                  0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) - 0 * (
                                      1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                              -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 0 * (
                                              1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) + 1.0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(right_j4) - 0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(right_j4) - 0 * (
                                              -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
                              right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                              right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                              right_j2) + 0) * np.cos(right_j4) + 0 * np.sin(right_j1) - 0 * np.cos(
                              right_j1) - 0 * np.cos(right_j2) - 0) + (-0 * (
                -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) + 0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 0.1053 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(
        right_j4) - 0 * (-1.0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.cos(right_j4)) * np.cos(right_j5) + 0.11 * (1.0 * (0 * (
                -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.sin(right_j4)) * np.sin(right_j5) + 0 * (
                                                                                   -1.0 * np.sin(right_j1) + 0 * np.cos(
                                                                               right_j1) + 0) * np.cos(
        right_j3) + 0.1053 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.cos(right_j4) + 0.11 * (-0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (1.0 * np.sin(
        right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * (-1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                                                                1.0 * np.sin(right_j1) - 0 * np.cos(
                                                                            right_j1)) * np.cos(right_j2) + 0 * np.cos(
        right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
        right_j3) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(right_j2) + 0) * np.cos(right_j5)) * (
                                  -0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(
                              right_j0) + 0 * np.sin(right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                              1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
                              right_j2) + 0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                              -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * (
                                              -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(
                                          right_j0) + 0 * np.sin(right_j1) * np.cos(right_j0)) * np.sin(
                                          right_j2) + 0 * (1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
                                          right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
                                          right_j2)) * np.cos(right_j3) + 0 * (1.0 * (
                                      1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(
                                  right_j0) + 0 * np.sin(right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(
                              right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(
                              right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (0 * (
                                      -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(
                                  right_j0) + 0 * np.sin(right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
                                  right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
                                  right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                      1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(
                                  right_j0) + 0 * np.sin(right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(
                              right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(
                              right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (
                                                                                                                              -1.0 * (
                                                                                                                                  -0 * np.sin(
                                                                                                                              right_j1) - 1.0 * np.cos(
                                                                                                                              right_j1)) * np.sin(
                                                                                                                          right_j0) + 0 * np.sin(
                                                                                                                          right_j0) - 0 * np.cos(
                                                                                                                          right_j0) * np.cos(
                                                                                                                          right_j1) + 0 * np.cos(
                                                                                                                          right_j0)) * np.sin(
                              right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                                      1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(
                                  right_j0) + 0 * np.sin(right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(
                              right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                                                                    0 * np.sin(right_j0) + 1.0 * np.cos(
                                                                                right_j0)) * np.cos(right_j2)) * np.cos(
                              right_j3) - 1.0 * (1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(
                              right_j0) + 0 * np.sin(right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(
                              right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                                             0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
                              right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
                              right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
                              right_j3)) * np.cos(right_j4) + 0 * (
                                              -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(
                                          right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3) + 1.0 * (-1.0 * (
                                      1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(
                                  right_j0) + 0 * np.sin(right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(
                              right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(
                              right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (-0 * np.sin(
                              right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
                              right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
                              right_j4) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(
                              right_j0));
    jaco_b_l6[1, 7] = 0;
    jaco_b_l6[2, 1] = 0.168499999998653 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) - 0 * (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                              right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) - 0.168499999998653 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2) - 0.399999999999381 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (
                                  -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(
                              right_j0) - 0 * np.sin(right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                              1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
                              right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
                              right_j2)) * np.cos(right_j3) + 0.400000000000516 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 0.1363 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2) - 0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) + 0.11 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) + 0.400000000000516 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0.1363 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                          -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j4) + 0.11 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                        1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                    right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                        -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                        -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (
                                        -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(
                                    right_j0) - 0 * np.sin(right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
                                    right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
                                    right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j5) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0.192500000001273 * np.sin(
        right_j0) + 0.0810000000006978 * np.cos(right_j0);
    jaco_b_l6[2, 2] = 0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.sin(right_j1)) * np.sin(right_j2) - 0 * (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                              right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) - 0 * (
                                  -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) + (
                                  1.0 * np.sin(right_j0) - 0 * np.cos(right_j0)) * (
                                  0.168499999998653 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j2) - 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * (
                                              -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) + 0.400000000000516 * (
                                              1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 0.1363 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(right_j4) + 0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(right_j4) - 0 * (
                                              -1.0 * (0 * (
                                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                              right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (
                                                                  -1.0 * np.sin(right_j1) + 0 * np.cos(
                                                              right_j1) + 0) * np.sin(right_j3)) * np.sin(
                                          right_j4) + 0 * (0 * (
                                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                              right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (
                                                                       -1.0 * np.sin(right_j1) + 0 * np.cos(
                                                                   right_j1) + 0) * np.cos(right_j3)) * np.cos(
                                          right_j4) + 1.0 * (-1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                                          right_j2) + 0) * np.cos(right_j4)) * np.cos(right_j5) + 0.11 * (1.0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.cos(right_j4) + 0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.sin(right_j4) + 1.0 * (
                                                                                                                      -1.0 * (
                                                                                                                          -0 * np.sin(
                                                                                                                      right_j1) - 1.0 * np.cos(
                                                                                                                      right_j1)) * np.sin(
                                                                                                                  right_j2) + 0 * (
                                                                                                                                  1.0 * np.sin(
                                                                                                                              right_j1) - 0 * np.cos(
                                                                                                                              right_j1)) * np.cos(
                                                                                                                  right_j2) - 0 * np.sin(
                                                                                                                  right_j1) + 0 * np.cos(
                                                                                                                  right_j1) + 0 * np.cos(
                                                                                                                  right_j2) + 0) * np.sin(
                              right_j4)) * np.sin(right_j5) + 0.400000000000516 * (
                                              -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
                              right_j3) + 0.1363 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                              right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                              right_j2) + 0) * np.cos(right_j4) + 0.11 * (
                                              -0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) - 0 * (-1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) + 1.0 * (
                                                          -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
                                          right_j3) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                                          right_j2) + 0) * np.cos(right_j5) - 0.399999999999381 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) - 0 * np.cos(right_j2)) - 0 * (
                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) + 0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2) - 0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) + 0 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                 1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                             right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) + 0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3) + 0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2) - 0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4) + 0 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                                                                                          -0 * np.sin(
                                                                                                                      right_j1) - 1.0 * np.cos(
                                                                                                                      right_j1)) * np.cos(
        right_j0) - 0 * (-1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                 right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                     -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j5) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0);
    jaco_b_l6[2, 3] = (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * (0.168499999998653 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) - 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) - 0.168499999998653 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2) - 0.259999999999381 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (
                                                                                          -1.0 * (1.0 * (-1.0 * np.sin(
                                                                                      right_j1) + 0 * np.cos(
                                                                                      right_j1)) * np.cos(
                                                                                      right_j0) - 0 * np.sin(
                                                                                      right_j0) * np.sin(
                                                                                      right_j1)) * np.sin(
                                                                                      right_j2) + 0 * (1.0 * (
                                                                                              -0 * np.sin(
                                                                                          right_j1) - 1.0 * np.cos(
                                                                                          right_j1)) * np.cos(
                                                                                      right_j0) - 0 * np.sin(
                                                                                      right_j0) * np.cos(
                                                                                      right_j1)) * np.cos(
                                                                                      right_j2) + 1.0 * (-1.0 * np.sin(
                                                                                      right_j0) + 0 * np.cos(
                                                                                      right_j0)) * np.cos(
                                                                                      right_j2)) * np.cos(
        right_j3) + 0.400000000000516 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                 right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 0.1363 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                               1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                           right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                              right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) + 0.11 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) + 0.400000000000516 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0.1363 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                          -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j4) + 0.11 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                        1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                    right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                        -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                        -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (
                                        -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(
                                    right_j0) - 0 * np.sin(right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
                                    right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
                                    right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j5) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0)) + (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                              right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1) + 0 * np.sin(right_j0) - 0 * np.cos(
                              right_j0)) * (
                                  0.168499999998653 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j2) - 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * (
                                              -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) + 0.400000000000516 * (
                                              1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 0.1363 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(right_j4) + 0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(right_j4) - 0 * (
                                              -1.0 * (0 * (
                                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                              right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (
                                                                  -1.0 * np.sin(right_j1) + 0 * np.cos(
                                                              right_j1) + 0) * np.sin(right_j3)) * np.sin(
                                          right_j4) + 0 * (0 * (
                                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                              right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (
                                                                       -1.0 * np.sin(right_j1) + 0 * np.cos(
                                                                   right_j1) + 0) * np.cos(right_j3)) * np.cos(
                                          right_j4) + 1.0 * (-1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                                          right_j2) + 0) * np.cos(right_j4)) * np.cos(right_j5) + 0.11 * (1.0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.cos(right_j4) + 0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.sin(right_j4) + 1.0 * (
                                                                                                                      -1.0 * (
                                                                                                                          -0 * np.sin(
                                                                                                                      right_j1) - 1.0 * np.cos(
                                                                                                                      right_j1)) * np.sin(
                                                                                                                  right_j2) + 0 * (
                                                                                                                                  1.0 * np.sin(
                                                                                                                              right_j1) - 0 * np.cos(
                                                                                                                              right_j1)) * np.cos(
                                                                                                                  right_j2) - 0 * np.sin(
                                                                                                                  right_j1) + 0 * np.cos(
                                                                                                                  right_j1) + 0 * np.cos(
                                                                                                                  right_j2) + 0) * np.sin(
                              right_j4)) * np.sin(right_j5) + 0.400000000000516 * (
                                              -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
                              right_j3) + 0.1363 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                              right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                              right_j2) + 0) * np.cos(right_j4) + 0.11 * (
                                              -0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) - 0 * (-1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) + 1.0 * (
                                                          -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
                                          right_j3) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                                          right_j2) + 0) * np.cos(right_j5) - 0.259999999999381 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) - 0 * np.cos(right_j2));
    jaco_b_l6[2, 4] = (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * (0.126499999998653 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) - 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) - 0.126499999998653 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2) + 0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) + 0.400000000000516 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 0.1363 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2) - 0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) + 0.11 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) + 0.400000000000516 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0.1363 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                          -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j4) + 0.11 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                        1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                    right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                        -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                        -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (
                                        -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(
                                    right_j0) - 0 * np.sin(right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
                                    right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
                                    right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j5) - 0 * np.sin(right_j0) * np.cos(right_j1)) + (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) - 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) - 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2) + 0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.cos(right_j1) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0)) * (
                                  0.126499999998653 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j2) - 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * (
                                              -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) + 0.400000000000516 * (
                                              1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 0.1363 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(right_j4) + 0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(right_j4) - 0 * (
                                              -1.0 * (0 * (
                                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                              right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (
                                                                  -1.0 * np.sin(right_j1) + 0 * np.cos(
                                                              right_j1) + 0) * np.sin(right_j3)) * np.sin(
                                          right_j4) + 0 * (0 * (
                                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                              right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (
                                                                       -1.0 * np.sin(right_j1) + 0 * np.cos(
                                                                   right_j1) + 0) * np.cos(right_j3)) * np.cos(
                                          right_j4) + 1.0 * (-1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                                          right_j2) + 0) * np.cos(right_j4)) * np.cos(right_j5) + 0.11 * (1.0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.cos(right_j4) + 0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.sin(right_j4) + 1.0 * (
                                                                                                                      -1.0 * (
                                                                                                                          -0 * np.sin(
                                                                                                                      right_j1) - 1.0 * np.cos(
                                                                                                                      right_j1)) * np.sin(
                                                                                                                  right_j2) + 0 * (
                                                                                                                                  1.0 * np.sin(
                                                                                                                              right_j1) - 0 * np.cos(
                                                                                                                              right_j1)) * np.cos(
                                                                                                                  right_j2) - 0 * np.sin(
                                                                                                                  right_j1) + 0 * np.cos(
                                                                                                                  right_j1) + 0 * np.cos(
                                                                                                                  right_j2) + 0) * np.sin(
                              right_j4)) * np.sin(right_j5) + 0.400000000000516 * (
                                              -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
                              right_j3) + 0.1363 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                              right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                              right_j2) + 0) * np.cos(right_j4) + 0.11 * (
                                              -0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) - 0 * (-1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) + 1.0 * (
                                                          -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
                                          right_j3) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                                          right_j2) + 0) * np.cos(right_j5) + 0 * np.sin(right_j1) - 0 * np.cos(
                              right_j1) - 0 * np.cos(right_j2));
    jaco_b_l6[2, 5] = (-0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * (
                                   -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(
                               right_j2)) * np.cos(right_j3) + 1.0 * (
                                   1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(
                               right_j2)) * np.sin(right_j3) + 1.0 * (
                                   -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3) - 0 * np.sin(
        right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(right_j2) + 0) * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                                                    -1.0 * (1.0 * (-1.0 * np.sin(
                                                                                right_j1) + 0 * np.cos(
                                                                                right_j1)) * np.cos(
                                                                                right_j0) - 0 * np.sin(
                                                                                right_j0) * np.sin(right_j1)) * np.sin(
                                                                                right_j2) + 0 * (1.0 * (-0 * np.sin(
                                                                                right_j1) - 1.0 * np.cos(
                                                                                right_j1)) * np.cos(
                                                                                right_j0) - 0 * np.sin(
                                                                                right_j0) * np.cos(right_j1)) * np.cos(
                                                                                right_j2) + 1.0 * (-1.0 * np.sin(
                                                                                right_j0) + 0 * np.cos(
                                                                                right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) + 0.275000000000516 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                 right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 0.1363 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                               1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                           right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                              right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) + 0.11 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) + 0.275000000000516 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0.1363 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                          -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j4) + 0.11 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                        1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                    right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                        -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                        -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (
                                        -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(
                                    right_j0) - 0 * np.sin(right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
                                    right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
                                    right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j5)) + (0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.sin(right_j1)) * np.sin(right_j2) - 0 * (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                              right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) - 0 * (
                                  -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) - 0 * np.sin(right_j0) * np.cos(right_j1) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0)) * (
                                  -0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                      1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * (
                                              -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) + 0.275000000000516 * (
                                              1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 0.1363 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(right_j4) + 0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(right_j4) - 0 * (
                                              -1.0 * (0 * (
                                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                              right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (
                                                                  -1.0 * np.sin(right_j1) + 0 * np.cos(
                                                              right_j1) + 0) * np.sin(right_j3)) * np.sin(
                                          right_j4) + 0 * (0 * (
                                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                              right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (
                                                                       -1.0 * np.sin(right_j1) + 0 * np.cos(
                                                                   right_j1) + 0) * np.cos(right_j3)) * np.cos(
                                          right_j4) + 1.0 * (-1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                                          right_j2) + 0) * np.cos(right_j4)) * np.cos(right_j5) + 0.11 * (1.0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.cos(right_j4) + 0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.sin(right_j4) + 1.0 * (
                                                                                                                      -1.0 * (
                                                                                                                          -0 * np.sin(
                                                                                                                      right_j1) - 1.0 * np.cos(
                                                                                                                      right_j1)) * np.sin(
                                                                                                                  right_j2) + 0 * (
                                                                                                                                  1.0 * np.sin(
                                                                                                                              right_j1) - 0 * np.cos(
                                                                                                                              right_j1)) * np.cos(
                                                                                                                  right_j2) - 0 * np.sin(
                                                                                                                  right_j1) + 0 * np.cos(
                                                                                                                  right_j1) + 0 * np.cos(
                                                                                                                  right_j2) + 0) * np.sin(
                              right_j4)) * np.sin(right_j5) + 0.275000000000516 * (
                                              -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
                              right_j3) + 0.1363 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                              right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                              right_j2) + 0) * np.cos(right_j4) + 0.11 * (
                                              -0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) - 0 * (-1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) + 1.0 * (
                                                          -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
                                          right_j3) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                                          right_j2) + 0) * np.cos(right_j5) + 0 * np.cos(right_j2));
    jaco_b_l6[2, 6] = (-0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                         1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                     right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                         -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) + 0 * (1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                 right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                     -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 0.1053 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                               1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                           right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                              right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) + 0.11 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) + 0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3) + 0.1053 * (
                                   -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(
                               right_j0) - 0 * np.sin(right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                               1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                           right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
                               right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
                               right_j2) - 0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                               right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(
                               right_j0)) * np.cos(right_j4) + 0.11 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                                                   -0 * np.sin(right_j1) - 1.0 * np.cos(
                                                                               right_j1)) * np.cos(right_j0) - 0 * (
                                                                                   -1.0 * (1.0 * (-1.0 * np.sin(
                                                                               right_j1) + 0 * np.cos(
                                                                               right_j1)) * np.cos(
                                                                               right_j0) - 0 * np.sin(
                                                                               right_j0) * np.sin(right_j1)) * np.sin(
                                                                               right_j2) + 0 * (1.0 * (-0 * np.sin(
                                                                               right_j1) - 1.0 * np.cos(
                                                                               right_j1)) * np.cos(
                                                                               right_j0) - 0 * np.sin(
                                                                               right_j0) * np.cos(right_j1)) * np.cos(
                                                                               right_j2) + 1.0 * (-1.0 * np.sin(
                                                                               right_j0) + 0 * np.cos(
                                                                               right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j5)) * (
                                  -0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                      1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * (
                                              -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) + 0 * (
                                              1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(right_j4) + 0 * (0 * (
                                      -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                  right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                                      -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                                                                                            1.0 * np.sin(
                                                                                                        right_j1) - 0 * np.cos(
                                                                                                        right_j1)) * np.sin(
                              right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(right_j4) + 0 * (
                                              -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
                              right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                              right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                              right_j2) + 0) * np.cos(right_j4) - 0 * np.sin(right_j1) + 0 * np.cos(
                              right_j1) + 0 * np.cos(right_j2) + 0) + (-0 * (
                -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) + 0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 0.1053 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(
        right_j4) - 0 * (-1.0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.cos(right_j4)) * np.cos(right_j5) + 0.11 * (1.0 * (0 * (
                -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.sin(right_j4)) * np.sin(right_j5) + 0 * (
                                                                                   -1.0 * np.sin(right_j1) + 0 * np.cos(
                                                                               right_j1) + 0) * np.cos(
        right_j3) + 0.1053 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.cos(right_j4) + 0.11 * (-0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (1.0 * np.sin(
        right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * (-1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                                                                1.0 * np.sin(right_j1) - 0 * np.cos(
                                                                            right_j1)) * np.cos(right_j2) + 0 * np.cos(
        right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
        right_j3) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(right_j2) + 0) * np.cos(right_j5)) * (0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) - 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) - 0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                                                                                                               -0 * np.sin(
                                                                                                                           right_j1) - 1.0 * np.cos(
                                                                                                                           right_j1)) * np.cos(
        right_j0) + 0 * (-1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                 right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                     -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 0 * (1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                 right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                     -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) + 1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                            1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                        right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) - 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) - 0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3) - 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2) - 0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4) - 0 * np.sin(
        right_j0) * np.cos(right_j1) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0));
    jaco_b_l6[2, 7] = 0;
    jaco_b_l6[3, 1] = 0;
    jaco_b_l6[3, 2] = (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * (0.168499999998653 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) - 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) - 0.168499999998653 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2) - 0.399999999999381 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * (
                                                                                      -1.0 * (1.0 * (-1.0 * np.sin(
                                                                                  right_j1) + 0 * np.cos(
                                                                                  right_j1)) * np.sin(
                                                                                  right_j0) + 0 * np.sin(
                                                                                  right_j1) * np.cos(
                                                                                  right_j0)) * np.sin(right_j2) + 0 * (
                                                                                                  1.0 * (-0 * np.sin(
                                                                                              right_j1) - 1.0 * np.cos(
                                                                                              right_j1)) * np.sin(
                                                                                              right_j0) + 0 * np.cos(
                                                                                              right_j0) * np.cos(
                                                                                              right_j1)) * np.cos(
                                                                                  right_j2) + 1.0 * (0 * np.sin(
                                                                                  right_j0) + 1.0 * np.cos(
                                                                                  right_j0)) * np.cos(
                                                                                  right_j2)) * np.cos(
        right_j3) + 0.400000000000516 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                 right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 0.1363 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                               1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                           right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) + 0.11 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) + 0.400000000000516 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0.1363 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                          -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j4) + 0.11 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                        1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                    right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                        0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                        -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * (
                                        -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(
                                    right_j0) + 0 * np.sin(right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
                                    right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
                                    right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j5) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0.142500000001273 * np.cos(
        right_j0)) + (-0 * np.sin(right_j0) - 1.0 * np.cos(right_j0)) * (0.168499999998653 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) - 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) - 0.168499999998653 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2) - 0.399999999999381 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (
                                                                                     -1.0 * (1.0 * (-1.0 * np.sin(
                                                                                 right_j1) + 0 * np.cos(
                                                                                 right_j1)) * np.cos(
                                                                                 right_j0) - 0 * np.sin(
                                                                                 right_j0) * np.sin(right_j1)) * np.sin(
                                                                                 right_j2) + 0 * (1.0 * (-0 * np.sin(
                                                                                 right_j1) - 1.0 * np.cos(
                                                                                 right_j1)) * np.cos(
                                                                                 right_j0) - 0 * np.sin(
                                                                                 right_j0) * np.cos(right_j1)) * np.cos(
                                                                                 right_j2) + 1.0 * (-1.0 * np.sin(
                                                                                 right_j0) + 0 * np.cos(
                                                                                 right_j0)) * np.cos(
                                                                                 right_j2)) * np.cos(
        right_j3) + 0.400000000000516 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                 right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 0.1363 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                               1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                           right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                              right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) + 0.11 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) + 0.400000000000516 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0.1363 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                          -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j4) + 0.11 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                        1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                    right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                        -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                        -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (
                                        -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(
                                    right_j0) - 0 * np.sin(right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
                                    right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
                                    right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j5) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0.142500000001273 * np.sin(right_j0) + 0 * np.cos(
        right_j0));
    jaco_b_l6[3, 3] = (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * np.sin(
        right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1) - 0 * np.cos(right_j0)) * (0.168499999998653 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) - 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) - 0.168499999998653 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2) - 0.259999999999381 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (
                                                                                                   -1.0 * (1.0 * (
                                                                                                       -1.0 * np.sin(
                                                                                                   right_j1) + 0 * np.cos(
                                                                                                   right_j1)) * np.cos(
                                                                                               right_j0) - 0 * np.sin(
                                                                                               right_j0) * np.sin(
                                                                                               right_j1)) * np.sin(
                                                                                               right_j2) + 0 * (1.0 * (
                                                                                                       -0 * np.sin(
                                                                                                   right_j1) - 1.0 * np.cos(
                                                                                                   right_j1)) * np.cos(
                                                                                               right_j0) - 0 * np.sin(
                                                                                               right_j0) * np.cos(
                                                                                               right_j1)) * np.cos(
                                                                                               right_j2) + 1.0 * (
                                                                                                               -1.0 * np.sin(
                                                                                                           right_j0) + 0 * np.cos(
                                                                                                           right_j0)) * np.cos(
                                                                                               right_j2)) * np.cos(
        right_j3) + 0.400000000000516 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                 right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 0.1363 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                               1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                           right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                              right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) + 0.11 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) + 0.400000000000516 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0.1363 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                          -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j4) + 0.11 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                        1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                    right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                        -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                        -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (
                                        -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(
                                    right_j0) - 0 * np.sin(right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
                                    right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
                                    right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j5) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0)) + (
                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                              right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(
                              right_j0)) * (0.168499999998653 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) - 0 * (
                                                        1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                    right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) - 0.168499999998653 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2) - 0.259999999999381 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * (
                                                        -1.0 * (1.0 * (
                                                            -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(
                                                    right_j0) + 0 * np.sin(right_j1) * np.cos(right_j0)) * np.sin(
                                                    right_j2) + 0 * (1.0 * (
                                                            -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                    right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
                                                    right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(
                                                    right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) + 0.400000000000516 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                 right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 0.1363 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                               1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                           right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) + 0.11 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) + 0.400000000000516 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0.1363 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                          -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j4) + 0.11 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                        1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                    right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                        0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                        -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * (
                                        -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(
                                    right_j0) + 0 * np.sin(right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
                                    right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
                                    right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j5) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0));
    jaco_b_l6[3, 4] = (1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.sin(right_j2) - 0 * (
                                   1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                               right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) - 1.0 * (
                                   0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                   -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * np.sin(
        right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1) - 0 * np.cos(right_j0)) * (0.126499999998653 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) - 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) - 0.126499999998653 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2) + 0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) + 0.400000000000516 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 0.1363 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2) - 0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) + 0.11 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) + 0.400000000000516 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0.1363 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                          -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j4) + 0.11 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                        1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                    right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                        -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                        -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (
                                        -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(
                                    right_j0) - 0 * np.sin(right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
                                    right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
                                    right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j5) - 0 * np.sin(right_j0) * np.cos(right_j1)) + (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2) - 0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * (0.126499999998653 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) - 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) - 0.126499999998653 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2) + 0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) + 0.400000000000516 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 0.1363 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                           -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j4)) * np.cos(right_j5) + 0.11 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.cos(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.sin(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                           -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j4)) * np.sin(right_j5) + 0.400000000000516 * (-1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(
        right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3) + 0.1363 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                                                                        -0 * np.sin(
                                                                                                    right_j1) - 1.0 * np.cos(
                                                                                                    right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j4) + 0.11 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                        1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                    right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                        0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                        -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * (
                                        -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(
                                    right_j0) + 0 * np.sin(right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
                                    right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
                                    right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j5) + 0 * np.cos(right_j0) * np.cos(right_j1));
    jaco_b_l6[3, 5] = (0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.sin(right_j2) - 0 * (
                                   1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                               right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) - 0 * (
                                   0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                   -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1) - 0 * np.cos(right_j0)) * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                                                                                          -1.0 * (
                                                                                                                              1.0 * (
                                                                                                                                  -1.0 * np.sin(
                                                                                                                              right_j1) + 0 * np.cos(
                                                                                                                              right_j1)) * np.cos(
                                                                                                                          right_j0) - 0 * np.sin(
                                                                                                                          right_j0) * np.sin(
                                                                                                                          right_j1)) * np.sin(
                                                                                                                      right_j2) + 0 * (
                                                                                                                                      1.0 * (
                                                                                                                                          -0 * np.sin(
                                                                                                                                      right_j1) - 1.0 * np.cos(
                                                                                                                                      right_j1)) * np.cos(
                                                                                                                                  right_j0) - 0 * np.sin(
                                                                                                                                  right_j0) * np.cos(
                                                                                                                                  right_j1)) * np.cos(
                                                                                                                      right_j2) + 1.0 * (
                                                                                                                                      -1.0 * np.sin(
                                                                                                                                  right_j0) + 0 * np.cos(
                                                                                                                                  right_j0)) * np.cos(
                                                                                                                      right_j2)) * np.cos(
        right_j3) + 0.275000000000516 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                 right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 0.1363 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                               1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                           right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                              right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) + 0.11 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) + 0.275000000000516 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0.1363 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                          -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j4) + 0.11 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                        1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                    right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                        -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                        -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (
                                        -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(
                                    right_j0) - 0 * np.sin(right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
                                    right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
                                    right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j5)) + (-0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                              right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                  -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                                                                                          -1.0 * (
                                                                                                                              1.0 * (
                                                                                                                                  -1.0 * np.sin(
                                                                                                                              right_j1) + 0 * np.cos(
                                                                                                                              right_j1)) * np.sin(
                                                                                                                          right_j0) + 0 * np.sin(
                                                                                                                          right_j1) * np.cos(
                                                                                                                          right_j0)) * np.sin(
                                                                                                                      right_j2) + 0 * (
                                                                                                                                      1.0 * (
                                                                                                                                          -0 * np.sin(
                                                                                                                                      right_j1) - 1.0 * np.cos(
                                                                                                                                      right_j1)) * np.sin(
                                                                                                                                  right_j0) + 0 * np.cos(
                                                                                                                                  right_j0) * np.cos(
                                                                                                                                  right_j1)) * np.cos(
                                                                                                                      right_j2) + 1.0 * (
                                                                                                                                      0 * np.sin(
                                                                                                                                  right_j0) + 1.0 * np.cos(
                                                                                                                                  right_j0)) * np.cos(
                                                                                                                      right_j2)) * np.cos(
        right_j3) + 0.275000000000516 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                 right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 0.1363 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                               1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                           right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) + 0.11 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) + 0.275000000000516 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0.1363 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                          -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j4) + 0.11 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                        1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                    right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                        0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                        -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * (
                                        -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(
                                    right_j0) + 0 * np.sin(right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
                                    right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
                                    right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j5));
    jaco_b_l6[3, 6] = (-0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                         1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                     right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                         0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) + 0 * (1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                 right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                     0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 0.1053 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                               1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                           right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                              right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) + 0.11 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                      right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) + 0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3) + 0.1053 * (
                                   -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(
                               right_j0) + 0 * np.sin(right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                               1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                           right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
                               right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
                               right_j2) - 0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                               right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(
                               right_j0)) * np.cos(right_j4) + 0.11 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                                                   -0 * np.sin(right_j1) - 1.0 * np.cos(
                                                                               right_j1)) * np.sin(right_j0) - 0 * (
                                                                                   -1.0 * (1.0 * (-1.0 * np.sin(
                                                                               right_j1) + 0 * np.cos(
                                                                               right_j1)) * np.sin(
                                                                               right_j0) + 0 * np.sin(
                                                                               right_j1) * np.cos(right_j0)) * np.sin(
                                                                               right_j2) + 0 * (1.0 * (-0 * np.sin(
                                                                               right_j1) - 1.0 * np.cos(
                                                                               right_j1)) * np.sin(
                                                                               right_j0) + 0 * np.cos(
                                                                               right_j0) * np.cos(right_j1)) * np.cos(
                                                                               right_j2) + 1.0 * (0 * np.sin(
                                                                               right_j0) + 1.0 * np.cos(
                                                                               right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j5)) * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                                                                                       -0 * np.sin(
                                                                                                                   right_j1) - 1.0 * np.cos(
                                                                                                                   right_j1)) * np.cos(
        right_j0) - 0 * (-1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                 right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                     -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) + 0 * (1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                 right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                     -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                            1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                        right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2) - 0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) + (-0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) + 0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 0.1053 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2) - 0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) + 0.11 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) + 0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3) + 0.1053 * (
                                                                                                   -1.0 * (1.0 * (
                                                                                                       -1.0 * np.sin(
                                                                                                   right_j1) + 0 * np.cos(
                                                                                                   right_j1)) * np.cos(
                                                                                               right_j0) - 0 * np.sin(
                                                                                               right_j0) * np.sin(
                                                                                               right_j1)) * np.sin(
                                                                                               right_j2) + 0 * (1.0 * (
                                                                                                       -0 * np.sin(
                                                                                                   right_j1) - 1.0 * np.cos(
                                                                                                   right_j1)) * np.cos(
                                                                                               right_j0) - 0 * np.sin(
                                                                                               right_j0) * np.cos(
                                                                                               right_j1)) * np.cos(
                                                                                               right_j2) + 1.0 * (
                                                                                                               -1.0 * np.sin(
                                                                                                           right_j0) + 0 * np.cos(
                                                                                                           right_j0)) * np.cos(
                                                                                               right_j2) - 0 * (
                                                                                                               -0 * np.sin(
                                                                                                           right_j1) - 1.0 * np.cos(
                                                                                                           right_j1)) * np.cos(
                                                                                               right_j0) + 0 * np.sin(
                                                                                               right_j0) * np.cos(
                                                                                               right_j1) - 0 * np.sin(
                                                                                               right_j0) + 0 * np.cos(
                                                                                               right_j0)) * np.cos(
        right_j4) + 0.11 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                        1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                    right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                        -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                        -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (
                                        -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(
                                    right_j0) - 0 * np.sin(right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                    1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
                                    right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
                                    right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j5)) * (0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.sin(right_j2) - 0 * (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) - 0 * (
                                  0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) - 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) - 0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j4) - 0 * np.sin(
        right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1) - 0 * np.cos(right_j0));
    jaco_b_l6[3, 7] = 0;
    jaco_b_l6[4, 1] = 0;
    jaco_b_l6[4, 2] = -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0);
    jaco_b_l6[4, 3] = -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0);
    jaco_b_l6[4, 4] = -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                              right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                  -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0);
    jaco_b_l6[4, 5] = -0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                              right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                  -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0);
    jaco_b_l6[4, 6] = -0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                              right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                  -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) + 0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) + 0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0);
    jaco_b_l6[4, 7] = -0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                              right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                  -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) + 0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) + 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2) - 0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.sin(
        right_j5) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                              right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4)) * np.cos(
        right_j5) + 0.999999999973015 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.cos(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) + 0 * np.sin(right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.sin(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2) - 0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.sin(
        right_j5) + 0 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                 1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                             right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                          1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                      right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                          -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j4)) * np.cos(
        right_j5) + 0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3) + 0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(
        right_j2) - 0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j4) - 0 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                                                                                          -0 * np.sin(
                                                                                                                      right_j1) - 1.0 * np.cos(
                                                                                                                      right_j1)) * np.cos(
        right_j0) - 0 * (-1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                 right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                     -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(
        right_j5) + 0.999999999973015 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                                 right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 0 * (-1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                     -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
        right_j0) - 0 * (-1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
        right_j0) * np.sin(right_j1)) * np.sin(right_j2) + 0 * (
                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                 right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                     -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.cos(right_j0) - 0 * np.sin(
            right_j0) * np.sin(right_j1)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                   right_j0) - 0 * np.sin(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       -1.0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j0) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j3) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0)) * np.cos(right_j5) + 0 * np.sin(
        right_j0) * np.cos(right_j1) - 0 * np.sin(right_j0) + 0 * np.cos(right_j0);
    jaco_b_l6[5, 1] = 0;
    jaco_b_l6[5, 2] = 0 * np.sin(right_j0) + 1.0 * np.cos(right_j0);
    jaco_b_l6[5, 3] = -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0);
    jaco_b_l6[5, 4] = -1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                  0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0);
    jaco_b_l6[5, 5] = -0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                  0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0);
    jaco_b_l6[5, 6] = -0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                  0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) + 0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) + 0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                       -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j4) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0);
    jaco_b_l6[5, 7] = -0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                              right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                  0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                  -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) - 0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) + 0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) + 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                           -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j4)) * np.sin(right_j5) - 0 * (-1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.sin(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.cos(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                           -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j4)) * np.cos(right_j5) + 0.999999999973015 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.cos(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.sin(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                           -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j4)) * np.sin(right_j5) + 0 * (1.0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.sin(right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.cos(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j3)) * np.cos(right_j4) + 0 * (0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(
        right_j2)) * np.cos(right_j3) - 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.sin(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(
        right_j2)) * np.sin(right_j3) - 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3)) * np.sin(right_j4) + 1.0 * (-1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 1.0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                           -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j4)) * np.cos(right_j5) + 0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) + 0 * np.sin(right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(
        right_j3) + 0 * (-1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                 right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                     0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                     -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j4) - 0 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (1.0 * (
                -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.cos(right_j0) * np.cos(
        right_j1)) * np.cos(right_j2) + 0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                                                                                          -0 * np.sin(
                                                                                                                      right_j1) - 1.0 * np.cos(
                                                                                                                      right_j1)) * np.sin(
        right_j0) - 0 * (-1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                 right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                     0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.sin(
        right_j5) + 0.999999999973015 * (-0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                                 right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(
        right_j2) + 0 * (0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2) - 0 * (
                                                     -0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
        right_j0) - 0 * (-1.0 * (1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j1) * np.cos(right_j0)) * np.sin(right_j2) + 0 * (
                                     1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                 right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.cos(right_j2) + 1.0 * (
                                     0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.cos(right_j2)) * np.cos(
        right_j3) + 1.0 * (1.0 * (
                1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
            right_j1) * np.cos(right_j0)) * np.cos(right_j2) + 0 * (
                                       1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                   right_j0) + 0 * np.cos(right_j0) * np.cos(right_j1)) * np.sin(right_j2) + 1.0 * (
                                       0 * np.sin(right_j0) + 1.0 * np.cos(right_j0)) * np.sin(right_j2)) * np.sin(
        right_j3) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j0) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j3) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0)) * np.cos(right_j5) + 0 * np.sin(
        right_j0) - 0 * np.cos(right_j0) * np.cos(right_j1) + 0 * np.cos(right_j0);
    jaco_b_l6[6, 1] = 1.00000000000000;
    jaco_b_l6[6, 2] = 0;
    jaco_b_l6[6, 3] = -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0;
    jaco_b_l6[6, 4] = -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0;
    jaco_b_l6[6, 5] = -0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * (
                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                      1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(
                              right_j2)) * np.cos(right_j3) + 1.0 * (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                      1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(
                              right_j2)) * np.sin(right_j3) + 1.0 * (
                                  -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3) - 0 * np.sin(
        right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(right_j2) + 0;
    jaco_b_l6[6, 6] = -0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * (
                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                      1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(
                              right_j2)) * np.cos(right_j3) + 0 * (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                      1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(
                              right_j2)) * np.sin(right_j3) - 1.0 * (0 * (
                -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(
        right_j4) + 0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3) + 1.0 * (
                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                      1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(right_j2) + 0) * np.cos(
        right_j4) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(right_j2) + 0;
    jaco_b_l6[6, 7] = -0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * (
                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                      1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(
                              right_j2)) * np.cos(right_j3) + 0 * (
                                  1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                                      1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(
                              right_j2)) * np.sin(right_j3) - 0 * (0 * (
                -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(
        right_j4) + 0 * (-1.0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.cos(right_j4)) * np.sin(right_j5) - 0 * (-1.0 * (0 * (
                -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.sin(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.cos(
        right_j4) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.cos(right_j4)) * np.cos(right_j5) + 0.999999999973015 * (1.0 * (0 * (
                -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.sin(right_j4)) * np.sin(right_j5) + 0 * (1.0 * (0 * (
                -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                    1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.sin(
        right_j3) + 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.cos(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.sin(right_j3)) * np.cos(
        right_j4) + 0 * (0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * np.cos(right_j2)) * np.cos(
        right_j3) - 1.0 * (1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * np.sin(right_j2)) * np.sin(
        right_j3) - 1.0 * (-1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3)) * np.sin(
        right_j4) + 1.0 * (-1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0) * np.sin(right_j4)) * np.cos(right_j5) + 0 * (
                                  -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(right_j3) + 0 * (
                                  -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                      1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * np.sin(
                              right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(right_j2) + 0) * np.cos(right_j4) - 0 * (
                                  -0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                      1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * (
                                              -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) + 1.0 * (
                                              1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) + 1.0 * (
                                              -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
                              right_j3) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                              right_j2) + 0) * np.sin(right_j5) + 0.999999999973015 * (
                                  -0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(right_j2) + 0 * (
                                      1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(right_j2) - 0 * (
                                              -1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * np.cos(right_j2)) * np.cos(right_j3) + 1.0 * (
                                              1.0 * (-0 * np.sin(right_j1) - 1.0 * np.cos(right_j1)) * np.cos(
                                          right_j2) + 0 * (1.0 * np.sin(right_j1) - 0 * np.cos(right_j1)) * np.sin(
                                          right_j2) + 0 * np.sin(right_j2)) * np.sin(right_j3) + 1.0 * (
                                              -1.0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0) * np.cos(
                              right_j3) - 0 * np.sin(right_j1) + 0 * np.cos(right_j1) + 0 * np.cos(
                              right_j2) + 0) * np.cos(right_j5) - 0 * np.sin(right_j1) + 0 * np.cos(
        right_j1) + 0 * np.cos(right_j2) + 0;


    # Legacy of some matlab conversions
    jaco_b_l6 = np.delete(jaco_b_l6, 0, 0)
    jaco_b_l6 = np.delete(jaco_b_l6, 0, 1)
    jacobian_sawyer = np. matmul(screw_transform_6n,jaco_b_l6)
    return jacobian_sawyer

def jacobianE0(q):
    end_effector_offset=np.array([0.55,0.0,0.0])
    J=jacobian70(q)
    T=transform_70(q)
    S = screw_transform(np.matmul(T[0:3,0:3],end_effector_offset))
    Je=np.matmul(S,J)
    return Je

def jacobianE0_trans(q):
    end_effector_offset=np.array([0.55,0.0,0.0])
    J=jacobian70(q)
    T=transform_70(q)
    S = screw_transform(np.matmul(T[0:3,0:3],end_effector_offset))
    Je=np.matmul(S,J)
    return Je[0:3,:]