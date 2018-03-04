"""lqr.py
Linear Quadratic Requlator
"""
import numpy as np
from scipy import linalg



def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.

    dx/dt = A x + B u

    cost = integral x.T*Q*x + u.T*R*u
    """
    #ref Bertsekas, p.151

    #first, try to solve the ricatti equation
    X = np.matrix(linalg.solve_continuous_are(A, B, Q, R))  # pylint: disable=no-member

    #compute the LQR gain
    K = np.matrix(linalg.inv(R) * (B.T * X))

    eigVals, eigVecs = linalg.eig(A - B * K)
    return K, X, eigVals


def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T * Q * x[k] + u[k].T * R * u[k]
    """
    #ref Bertsekas, p.151

    #first, try to solve the ricatti equation
    X = np.matrix(linalg.solve_discrete_are(A, B, Q, R))  # pylint: disable=no-member

    #compute the LQR gain
    K = np.matrix(linalg.inv(B.T * X * B + R) * (B.T * X * A))

    eigVals, eigVecs = linalg.eig(A - B * K)
    return K, X, eigVals
