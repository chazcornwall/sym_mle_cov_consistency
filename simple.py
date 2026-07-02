import numpy as np
from typing import Callable, Tuple, List

from mlelib import MaximumLikelihood, MlParams

"""
Run experiment for an identity, 2d measurement to show mlelib actually works

(z - Ix)^T * I * (z - Ix)

where I is the identity matrix
"""


def residual_generator(z: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Returns the residual r(x) = z - h(x)
    """
    return z - x

def residual_jacobian_generator(x: np.ndarray) -> np.ndarray:
    """
    Let r(x) = z - h(x), think of "residual"
    
    Returns:
        Partial derivative of r(x) with respect to x
    """
    return -np.eye(2)

def cost_function_generator(res: np.ndarray) -> float:
    """
    Returns:
        Cost function rho(r(x))
    """
    return np.linalg.norm(res)**2

def cost_function_gradient_generator(res: np.ndarray) -> np.ndarray:
    """
    Returns:
        Partial derivative of rho(r(x)) with respect to r(x)
    """
    return 2 * res

def cost_function_hessian_generator(res: np.ndarray) -> np.ndarray:
    """
    Returns:
        Hessian of cost function: d^2 rho(r) / (dr_i dr_j) Should be PSD and resembles the inverse of a covariance matrix
    """
    return 2 * np.eye(2)


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

    z = np.random.randn(2,1)
    print(z)
    ml.mle(z, verbose=True)


if __name__ == "__main__":
    main()