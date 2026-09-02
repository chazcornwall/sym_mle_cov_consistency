import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from abc import ABC, abstractmethod

def plot_cov(P : np.ndarray, mu: np.ndarray, label: str = "", color: str = ""):
    # Plot covariance ellipses
    t = np.linspace(0, 2*np.pi, 1000)
    v, U = np.linalg.eig(P)

    v1stddev3 = 3 * np.sqrt(np.real(v[0]) + 1e-6) # Prevent sqrt by negative numbers
    v2stddev3 = 3 * np.sqrt(np.real(v[1]) + 1e-6)

    # Unit circle
    x = np.cos(t)
    y = np.sin(t)

    # Transform and scale using
    x_rot = v1stddev3 * np.real(U[0,0]) * x + v2stddev3 * np.real(U[0,1]) * y + mu[0,0]
    y_rot = v1stddev3 * np.real(U[1,0]) * x + v2stddev3 * np.real(U[1,1]) * y + mu[1,0]

    if len(label) > 0:
        plt.plot(x_rot, y_rot, label=label, linewidth=2, linestyle="-", color=color)
    else:
        plt.plot(x_rot, y_rot, label=label, linewidth=2, linestyle="-", color=color)


def a2pi(angle_rad: float) -> float:
    """
    Same as performing atan2(sin(angle_rad, cos(angle_rad))).
    Places angle in range [-pi, pi]
    """
    TWO_M_PI = 2 * np.pi
    angle_unitless = angle_rad / TWO_M_PI
    return TWO_M_PI * (angle_unitless - np.floor(angle_unitless + 0.5))


class ResidualCalculator(ABC):
    """
    Calculates residuals and associated derivatives for optimization and validity checks
    """
    def __init__(self):
        pass

    def res(self, z: np.ndarray, x: np.ndarray) -> np.ndarray:
        return self.h(x) - z

    @abstractmethod
    def h(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def dhdx(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def d2hdx2(self, x: np.ndarray) -> np.ndarray:
        pass


class RobustCostCalculator(ABC):
    """
    Calculates robust costs (and associated derivatitves) assuming quadratic cost q = x^T(Sinv)x.
    Let rho(q(res)) where res is a vector and q : R^n -> R and rho is a robust cost function.
    We make the substitution q = r^2 where r is the error when defining the input for the robust cost function.
    As delta goes to infinity, the robust cost and the cost (1/2 * q) become identical, i.e. Gaussian.
    """

    def __init__(self, Sinv: np.ndarray):
        self.__Sinv = Sinv
        self.__n = Sinv.shape[0]

    @abstractmethod
    def _rho(self, q: float):
        """
        q is the unscaled quadratic cost r^T @ Sinv @ r
        """
        pass

    @abstractmethod
    def _drhodq(self, q: float):
        """
        q is the unscaled quadratic cost r^T @ Sinv @ r
        """
        pass

    @abstractmethod
    def _d2rhodq2(self, q: float):
        """
        q is the unscaled quadratic cost r^T @ Sinv @ r
        """
        pass

    @abstractmethod
    def _d3rhodq3(self, q: float):
        """
        q is the unscaled quadratic cost r^T @ Sinv @ r
        """
        pass

    def __q(self, res: np.ndarray):
        return (res.T @ self.__Sinv @ res)[0,0]

    def __dqdres(self, res: np.ndarray):
        return 2 * self.__Sinv @ res

    def __d2qdres2(self, res: np.ndarray):
        return 2 * self.__Sinv

    def rho(self, res: np.ndarray) -> float:
        return self._rho(self.__q(res))

    def drhodres(self, res: np.ndarray) -> np.ndarray:
        quadratic_cost = self.__q(res)
        dq_dres = self.__dqdres(res)
        drho_dq = self._drhodq(quadratic_cost)
        return drho_dq * dq_dres

    def d2rhodres2(self, res: np.ndarray) -> np.ndarray:
        '''
        Let rho(q(res)) where res is a vector and q : R^n -> R
        Using chain-rule: d^2rho/dres^2 = dq/dres * d^2rho/dqdq * dq/dres^T + drho/dq * d^2q/dresdres^T
        '''
        H = np.zeros((self.__n, self.__n))
        quadratic_cost = self.__q(res)
        drho_dq = self._drhodq(quadratic_cost)
        d2rho_dqdq = self._d2rhodq2(quadratic_cost)
        dq_dres = self.__dqdres(res)
        d2q_dresdresT = self.__d2qdres2(res)
        for i in range(self.__n):
            H[[i], :] = dq_dres[i,0] * d2rho_dqdq * dq_dres.T + drho_dq * d2q_dresdresT[[i],:]
        return H

    def d3rhodres3(self, res: np.ndarray) -> np.ndarray:
        '''
        Use product rule and chain rule applied to the cost hessian 
        (see `cost_hessian_generator` for details) to derive the following expression
        '''
        Hbold = np.zeros((self.__n, self.__n, self.__n))
        quadratic_cost = self.__q(res)
        d2rho_dqdq = self._d2rhodq2(quadratic_cost)
        d3rho_dqdqdq = self._d3rhodq3(quadratic_cost)
        dq_dres = self.__dqdres(res)
        d2q_dresdresT = self.__d2qdres2(res)
        for i in range(self.__n):
            # Since there are 3 terms multiplied together, we use product rule twice
            derivative_of_quadratic_term = d2rho_dqdq * d2q_dresdresT[[i],:].T @ dq_dres.T + \
                dq_dres[i, 0] * d3rho_dqdqdq * dq_dres @ dq_dres.T + \
                dq_dres[i, 0] * d2rho_dqdq * d2q_dresdresT

            # Since there are 2 terms multiplied together, we use product rule once 
            # (second term of product rule is zero and not included)
            derivative_of_linear_term = d2rho_dqdq * dq_dres @ d2q_dresdresT[[i],:] 

            # Add together
            Hbold[[i], :, :] = derivative_of_quadratic_term + derivative_of_linear_term
        return Hbold
    