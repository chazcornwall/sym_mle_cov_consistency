import numpy as np
from typing import Callable, Tuple, List

from mlelib import MaximumLikelihood, MlParams

"""
Run experiment for three range measurements to show the effects of propagating noise in a nonlinear estimation

(z - h(x))^T * S^-1 * (z - h(x))

where S is the covariance of the measurements and
h_i = sqrt((x - si)^T(x - si)) where si is the position of a satellite
"""
S = 0.5 * np.eye(3)
Sinv = np.linalg.inv(S)

Sats = np.zeros((2,3))
Sats[:,[0]] = np.array([[0.0, 5.0]]).T
Sats[:,[1]] = np.array([[-1.0, -3.0]]).T
Sats[:,[2]] = np.array([[3.0, 2.0]]).T

def h(x: np.ndarray) -> np.ndarray:
    h = np.zeros((3,1))
    h[0,0] = np.linalg.norm(x - Sats[:,[0]])
    h[1,0] = np.linalg.norm(x - Sats[:,[1]])
    h[2,0] = np.linalg.norm(x - Sats[:,[2]])
    return h

def residual_generator(z: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Returns the residual r(x) = z - h(x)
    """
    return z - h(x)

def residual_jacobian_generator(x: np.ndarray) -> np.ndarray:
    """
    Let r(x) = z - h(x), think of "residual"
    
    Returns:
        Partial derivative of r(x) with respect to x
    """
    hx = h(x)
    J = np.zeros((hx.shape[0], x.shape[0]))
    for i in range(hx.shape[0]):
        J[[i], :] = -1 / hx[i,0] * (x.T - Sats[:,[i]].T)
    return J

def cost_function_generator(res: np.ndarray) -> float:
    """
    Returns:
        Cost function rho(r(x))
    """
    return (res.T @ Sinv @ res)[0,0]

def cost_function_gradient_generator(res: np.ndarray) -> np.ndarray:
    """
    Returns:
        Partial derivative of rho(r(x)) with respect to r(x)
    """
    return 2 * Sinv @ res

def cost_function_hessian_generator(res: np.ndarray) -> np.ndarray:
    """
    Returns:
        Hessian of cost function: d^2 rho(r) / (dr_i dr_j) Should be PSD and resembles the inverse of a covariance matrix
    """
    return 2 * Sinv


def main():
    ml_params = MlParams(
        x0=np.zeros((2,1)),
        residual_generator=residual_generator,
        residual_jacobian_generator=residual_jacobian_generator,
        cost_function_generator=cost_function_generator,
        cost_function_gradient_generator=cost_function_gradient_generator,
        cost_function_hessian_generator=cost_function_hessian_generator,
        max_its=10
    )

    ml = MaximumLikelihood(ml_params)
    L = np.linalg.cholesky(S)
    z = h(np.array([[1,2]]).T) + L @ np.random.randn(3,1)
    print(z)
    ml.mle(z, verbose=True)


if __name__ == "__main__":
    main()