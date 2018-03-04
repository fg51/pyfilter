# -*- coding: utf-8 -*-
import numpy as np
from numpy import cos, sin, matrix

from matplotlib import pyplot as plt


def main():
    m = 4
    J = 0.0475  # moment of pitch axis
    r = 0.25    # distance for the centor of force
    g = 9.8
    c = 0.05    # dumping coef

    xe = [0, 0, 0, 0, 0, 0]
    ue = [0, m * g]

    A = np.matrix([
        [0,    0,    0,    1,    0,    0],
        [0,    0,    0,    0,    1,    0],
        [0,    0,    0,    0,    0,    1],
        [0, 0, (-ue[0] * sin(xe[2]) - ue[1] * cos(xe[2])) / m, -c / m, 0, 0],
        [0, 0, (ue[0] * cos(xe[2]) - ue[1] * sin(xe[2])) / m, 0, -c / m, 0],
        [0,    0,    0,    0,    0,    0],
    ])

    # Input 行列
    B = matrix(
       [[0, 0], [0, 0], [0, 0],
        [cos(xe[2]) / m, -sin(xe[2]) / m],
        [sin(xe[2]) / m,  cos(xe[2]) / m],
        [r / J, 0]])

    # Output 行列
    C = matrix([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
    D = matrix([[0, 0], [0, 0]])

    Qx1 = diag([1, 1, 1, 1, 1, 1])
    Qu1a = diag([1, 1])
    (K, X, E) = np.lqr(A, B, Qx1, Qu1a)  # NOTE: K1a = matrix(K)




if __name__ == '__main__':
    main()
