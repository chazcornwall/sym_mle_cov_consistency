import numpy as np
from typing import Callable, Tuple, List
import matplotlib.pyplot as plt

from mlelib import LikelihoodModel
from pydantic import BaseModel
import yaml
from argparse import ArgumentParser
from typing import Optional, List


class SatelliteLocation(BaseModel):
    x: float
    y: float


class TwoDRangeModelHuberParams(BaseModel):
    sat_locs: List[SatelliteLocation]
    sat_range_stddev: float = 1.0
    delta: float = 1.0


class TwoDRangeModelHuber(LikelihoodModel):

    def __init__(self, params: TwoDRangeModelHuberParams):
        super().__init__()
        self.__n = len(params.sat_locs)
        self.__delta = params.delta

        # Calculate measurement noise
        self.__S = params.sat_range_stddev**2 * np.eye(self.__n)
        self.__Sinv = np.linalg.inv(self.__S)
        self.__Sinvsqrt = np.linalg.cholesky(self.__Sinv)

        # Store satellite locations
        self.__Sats = np.zeros((2, self.__n))
        for i in range(self.__n):
            self.__Sats[:,[i]] = np.array([[params.sat_locs[i].x, params.sat_locs[i].y]]).T

    @property
    def n(self) -> int:
        return self.__n
    
    @property
    def S(self) -> np.ndarray:
        return self.__S

    @property
    def Sats(self) -> np.ndarray:
        return self.__Sats

    def __s(self, i: int) -> np.ndarray:
        return self.__Sats[:, [i]] 

    def h(self, x: np.ndarray) -> np.ndarray:
        h = np.zeros((self.__n,1))
        for i in range(self.__n):
            h[i,0] = np.linalg.norm(x - self.__s(i))
        return h

    def residual_generator(self, z, x):
        return self.h(x) - z

    def residual_jacobian_generator(self, x):
        hx = self.h(x)
        J = np.zeros((self.__n, x.shape[0]))
        for i in range(self.__n):
            J[[i], :] = 1 / hx[i,0] * (x - self.__s(i)).T
        return J

    def residual_hessian_generator(self, x):
        hx = self.h(x)
        JHess = np.zeros((self.__n, x.shape[0], x.shape[0]))
        I = np.eye(x.shape[0])
        for i in range(self.__n):
            si = self.__s(i)
            JHess[[i], :, :] = 1 / hx[i,0] * (I - (x - si) @ (x - si).T / hx[i,0]**2)
        return JHess

    def __quadratic_cost_generator(self, res) -> float:
        return (res.T @ self.__Sinv @ res)[0,0]

    def __quadratic_cost_gradient_generator(self, res) -> np.ndarray:
        return 2 * self.__Sinv @ res

    def __quadratic_cost_hessian_generator(self, res) -> np.ndarray:
        return 2 * self.__Sinv

    def __pseudo_huber_cost_gradient_generator(self, quad_cost) -> float:
        delta = self.__delta
        return delta / 2 * (delta**2 + quad_cost)**(-1/2)

    def __pseudo_huber_cost_hessian_generator(self, quad_cost) -> float:
        delta = self.__delta
        return -delta / 4 * (delta**2 + quad_cost)**(-3/2)

    def __pseudo_huber_cost_deriv_hessian_generator(self, quad_cost) -> float:
        delta = self.__delta
        return (3 * delta) / 8 * (delta**2 + quad_cost)**(-5/2)

    def cost_generator(self, res):
        '''
        pseudo-huber cost: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pseudo_huber.html
        '''
        delta = self.__delta
        quadratic_cost = self.__quadratic_cost_generator(res)
        return delta**2 * (np.sqrt(1 + quadratic_cost / delta**2) - 1)

    def cost_gradient_generator(self, res):
        quadratic_cost = self.__quadratic_cost_generator(res)
        dq_dres = self.__quadratic_cost_gradient_generator(res)
        drho_dq = self.__pseudo_huber_cost_gradient_generator(quadratic_cost)
        return drho_dq * dq_dres

    def cost_hessian_generator(self, res):
        '''
        Let rho(q(res)) where res is a vector and q : R^n -> R
        Using chain-rule: d^2rho/dres^2 = dq/dres * d^2rho/dqdq * dq/dres^T + drho/dq * d^2q/dresdres^T
        '''
        H = np.zeros((self.__n, self.__n))
        quadratic_cost = self.__quadratic_cost_generator(res)
        drho_dq = self.__pseudo_huber_cost_gradient_generator(quadratic_cost)
        d2rho_dqdq = self.__pseudo_huber_cost_hessian_generator(quadratic_cost)
        dq_dres = self.__quadratic_cost_gradient_generator(res)
        d2q_dresdresT = self.__quadratic_cost_hessian_generator(res)
        for i in range(self.__n):
            H[[i], :] = dq_dres[i,0] * d2rho_dqdq * dq_dres.T + drho_dq * d2q_dresdresT[[i],:]
        return H

    def cost_deriv_hessian_generator(self, res):
        '''
        Use product rule and chain rule applied to the cost hessian 
        (see `cost_hessian_generator` for details) to derive the following expression
        '''
        Hbold = np.zeros((self.__n, self.__n, self.__n))
        quadratic_cost = self.__quadratic_cost_generator(res)
        d2rho_dqdq = self.__pseudo_huber_cost_hessian_generator(quadratic_cost)
        d3rho_dqdqdq = self.__pseudo_huber_cost_deriv_hessian_generator(quadratic_cost)
        dq_dres = self.__quadratic_cost_gradient_generator(res)
        d2q_dresdresT = self.__quadratic_cost_hessian_generator(res)
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