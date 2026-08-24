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


class TwoDRangeModelParams(BaseModel):
    sat_locs: List[SatelliteLocation]
    sat_range_stddev: float = 1.0


class TwoDRangeModel(LikelihoodModel):

    def __init__(self, params: TwoDRangeModelParams):
        super().__init__()
        self.__n = len(params.sat_locs)

        # Calculate measurement noise
        self.__S = params.sat_range_stddev**2 * np.eye(self.__n)
        self.__Sinv = np.linalg.inv(self.__S)
        self.__Sinvsqrt = np.linalg.cholesky(self.__Sinv)

        # Store satellite locations
        self.__Sats = np.zeros((2, self.__n))
        for i in range(self.__n):
            self.__Sats[:,[i]] = np.array([[params.sat_locs[i].x, params.sat_locs[i].y]]).T

    def __s(self, i: int) -> np.ndarray:
        return self.__Sats[:, [i]] 

    @property
    def n(self) -> np.ndarray:
        return self.__n

    @property
    def S(self) -> np.ndarray:
        return self.__S

    @property
    def Sats(self) -> np.ndarray:
        return self.__Sats

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
            JHess[[i], :, :] = 1 / hx[i,0] * (I - (x - si) @ (x - si).T / hx[i,0]**3)
        return JHess

    def cost_generator(self, res):
        return 0.5 * (res.T @ self.__Sinv @ res)[0,0]

    def cost_gradient_generator(self, res):
        return self.__Sinv @ res

    def cost_hessian_generator(self, res):
        return self.__Sinv

    def cost_deriv_hessian_generator(self, res):
        return np.zeros((self.__n, self.__n, self.__n))

# """
# Run experiment for three range measurements to show the effects of propagating noise in a nonlinear estimation

# (z - h(x))^T * S^-1 * (z - h(x))

# where S is the covariance of the measurements and
# h_i = sqrt((x - si)^T(x - si)) where si is the position of a satellite
# """
# S = np.array([
#     [0.01, 0.0, 0.0],
#     [0.0, 0.01, 0.0],
#     [0.0, 0.0, 0.01]
# ])
# Sinv = np.linalg.inv(S)
# Sinvsqrt = np.linalg.cholesky(Sinv)

# Sats = np.zeros((2,3))
# Sats[:,[0]] = np.array([[-1.0, 4.2]]).T
# Sats[:,[1]] = np.array([[-1.0, 4.1]]).T
# Sats[:,[2]] = np.array([[-1.0, 4.0]]).T

# def h(x: np.ndarray) -> np.ndarray:
#     h = np.zeros((3,1))
#     h[0,0] = np.linalg.norm(x - Sats[:,[0]])
#     h[1,0] = np.linalg.norm(x - Sats[:,[1]])
#     h[2,0] = np.linalg.norm(x - Sats[:,[2]])
#     return h

# def residual_generator(z: np.ndarray, x: np.ndarray) -> np.ndarray:
#     """
#     Returns the residual r(x) = h(x) - z
#     """
#     return h(x) - z

# def residual_jacobian_generator(x: np.ndarray) -> np.ndarray:
#     """
#     Let r(x) = h(x) - z, think of "residual"
    
#     Returns:
#         Partial derivative of r(x) with respect to x
#     """
#     hx = h(x)
#     J = np.zeros((hx.shape[0], x.shape[0]))
#     for i in range(hx.shape[0]):
#         si = Sats[:,[i]]
#         J[[i], :] = 1 / hx[i,0] * (x.T - si.T)
#     return J

# def residual_hessian_generator(x: np.ndarray) -> np.ndarray:
#     """
#     Returns a tensor representing the Hessian JHess. 
#     The slice JHess[i,:,:] represents the Hessian for the ith 
#     component of the residual. Summing together all the slices
#     gives the "information" wrt x. 
#     """
#     hx = h(x)
#     JHess = np.zeros((hx.shape[0], x.shape[0], x.shape[0]))
#     I = np.eye(x.shape[0])
#     for i in range(hx.shape[0]):
#         si = Sats[:,[i]]
#         JHess[[i], :, :] = 1 / hx[i,0] * (I - (x - si) @ (x - si).T / hx[i,0]**3)
#     return JHess

# def cost_function_generator(res: np.ndarray) -> float:
#     """
#     Returns:
#         Cost function rho(r(x))
#     """
#     return (res.T @ Sinv @ res)[0,0]

# def huber_cost_function_generator(res: np.ndarray, threshold: float) -> float:
#     """
#     Returns:
#         Huber Cost function rho(r(x))
#     """
#     l2_cost = (res.T @ Sinv @ res)[0,0]

#     cost = l2_cost
#     if np.sqrt(cost) > threshold:
#         linear_cost = threshold * (np.sqrt(l2_cost) - threshold)
#         cost = linear_cost
#     return cost

# def cost_function_gradient_generator(res: np.ndarray) -> np.ndarray:
#     """
#     Returns:
#         Partial derivative of rho(r(x)) with respect to r(x)
#     """
#     return Sinv @ res  # Is 2 * Sinv @ res, but removing 2 due to hessian

# def huber_cost_function_gradient_generator(res: np.ndarray, threshold: float) -> np.ndarray:
#     """
#     Returns:
#         Partial derivative of rho(r(x)) with respect to r(x) where rho() is a 
#         Huber cost function
#     """
#     l2_cost = (res.T @ Sinv @ res)[0,0]

#     gradient = Sinv @ res
#     if np.sqrt(l2_cost) > threshold:
#         gradient = threshold / np.sqrt(l2_cost) * Sinv @ res
#     return gradient

# def cost_function_hessian_generator(res: np.ndarray) -> np.ndarray:
#     """
#     REMEMBER: If the same constant appears in this function as the cost_function_gradient_generator,
#         remove the constants! The constants will cancel out during the optimization. If left, the constants
#         will not affect the optimization, but will mess up the covariance calculations!
#     Returns:
#         Hessian of cost function: d^2 rho(r) / (dr_i dr_j) Should be PSD and resembles the inverse of a covariance matrix
#     """
#     return Sinv # Is 2 * Sinv, but removing 2 due to gradient

# def huber_cost_function_hessian_generator(res: np.ndarray) -> np.ndarray:
#     """
#     REMEMBER: If the same constant appears in this function as the cost_function_gradient_generator,
#         remove the constants! The constants will cancel out during the optimization. If left, the constants
#         will not affect the optimization, but will mess up the covariance calculations!
#     Returns:
#         Hessian of cost function: d^2 rho(r) / (dr_i dr_j) Should be PSD and resembles the inverse of a covariance matrix
#     """
#     return Sinv # Is 2 * Sinv, but removing 2 due to gradient

# def quadratic_measurement_models(M: int, x: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:

#     JHess = -residual_hessian_generator(x)
#     J = -residual_jacobian_generator(x)

#     quadratic_func_defs = []
#     for i in range(M):
#         print(f"JHess {i}:\n {JHess[i,:,:]}")
#         print(np.linalg.eigvals(JHess[i,:,:]))
#         print(f"J {i}:\n {J.T[:,[i]]}")
#         x0 = x + np.linalg.solve(JHess[i,:,:], -J.T[:,[i]])
#         print(x0)
#         quadratic_func_defs.append((x0, JHess[i,:,:]))
#     return quadratic_func_defs


# def plot_measurement_function(x: np.ndarray, M: int, N: int):
#     """
#     Params:
#         M: Number of elements in measurement
#         N: Width of grid evaluation
#     """

#     x0 = np.linspace(-10, 10, N)
#     x1 = np.linspace(-10, 10, N)

#     X0, X1 = np.meshgrid(x0, x1)

#     models = quadratic_measurement_models(M, x)

#     R = np.zeros((M, N, N))
#     Qmodel = np.zeros((M, N, N))
#     for i in range(N):
#         for j in range(N):
#             x_eval = np.array([[x0[i], x1[j]]]).T
#             R[:, [i], [j]] = h(x_eval)
#             for k in range(M):
#                 Qmodel[k, [i], [j]] = (x_eval - models[k][0]).T @ models[k][1] @ (x_eval - models[k][0])


#     fig, axs = plt.subplots(2, M+2, figsize=(15,4))

#     print(R[0,0,0])

#     # Plot range measurements
#     for i in range(M):
#         contour_lines = axs[0,i].contourf(X0, X1, R[i,:,:], levels=20, cmap='viridis')
#         axs[0,i].set_title(f'Range from Satellite {i}')
#         axs[0,i].set_xlabel('X axis')
#         axs[0,i].set_ylabel('Y axis')
#         axs[0,i].set_aspect('equal')

#     # Contour of measurements added together
#     contour_lines = axs[0,M].contourf(X0, X1, np.sum(R, axis=0), levels=20, cmap='viridis')
#     axs[0,M].set_title(f'Sum of Ranges')
#     axs[0,M].set_xlabel('X axis')
#     axs[0,M].set_ylabel('Y axis')
#     axs[0,M].set_aspect('equal')
    
#     # Add legend for measurements
#     axs[0,M+1].set_axis_off()
#     fig.colorbar(contour_lines, ax=axs[0,M+1], fraction=0.05, location='left')

#     # Plot quadratic range measurement models
#     for i in range(M):
#         contour_lines = axs[1,i].contourf(X0, X1, Qmodel[i,:,:], levels=20, cmap='viridis')
#         axs[1,i].set_title(f'Range from Satellite {i}')
#         axs[1,i].set_xlabel('X axis')
#         axs[1,i].set_ylabel('Y axis')
#         axs[1,i].set_aspect('equal')

#     # Contour of quadratic range measurement models added together
#     contour_lines = axs[1,M].contourf(X0, X1, np.sum(Qmodel, axis=0), levels=20, cmap='viridis')
#     axs[1,M].set_title(f'Sum of Ranges')
#     axs[1,M].set_xlabel('X axis')
#     axs[1,M].set_ylabel('Y axis')
#     axs[1,M].set_aspect('equal')
    
#     # Add legend for quadratic range measurement models
#     axs[1,M+1].set_axis_off()
#     fig.colorbar(contour_lines, ax=axs[1,M+1], fraction=0.05, location='left')

    
#     plt.tight_layout()
#     plt.show()


# class SatelliteLocation(BaseModel):
#     x: float
#     y: float


# class ValidityExperimentConfig(BaseModel):
#     x_range: Optional[List[float]] = [-10.0, 10.0]
#     y_range: Optional[List[float]] = [-10.0, 10.0]
#     range_resolution: Optional[int] = 100
#     satellite_locs: List[SatelliteLocation]
#     its_per_step: int = 100


# def main(args):

#     exp_params : ValidityExperimentConfig = None
#     with open(args.config, 'r') as f:
#         exp_params = ValidityExperimentConfig(**yaml.safe_load(f))

    
#     ml_params = MlParams(
#         x0=np.zeros((2,1)),
#         residual_generator=residual_generator,
#         residual_jacobian_generator=residual_jacobian_generator,
#         residual_hessian_generator=residual_hessian_generator,
#         cost_function_generator=cost_function_generator,
#         cost_function_gradient_generator=cost_function_gradient_generator,
#         cost_function_hessian_generator=cost_function_hessian_generator,
#         max_its=10
#     )

#     # plot_measurement_function(np.array([[0, 0]]).T, 3, 100)
#     # quadratic_measurement_models(3, np.array([[0,0]]).T)

#     ml = MaximumLikelihood(ml_params)
#     # L = np.linalg.cholesky(S)
#     # z = h(np.array([[1,2]]).T) + L @ np.random.randn(3,1)
#     # print(z)
#     # ml.mle(z, verbose=True)

#     ml.experiment(np.array([[1,2]]).T, h, S, N=100)

#     # Test residual hessian
#     # x0 = np.zeros((2,1))
#     # JHess = residual_hessian_generator(x0)
#     # Info = -JHess[0,:,:]-JHess[1,:,:]-JHess[2,:,:]
#     # print(Info)
#     # J = residual_jacobian_generator(x0)
#     # Infoapprox = J.T @ J
#     # print(Infoapprox)
#     # print(np.linalg.eig(Info))
#     # print(np.linalg.eigvals(JHess[0,:,:]))


# if __name__ == "__main__":
#     parser = ArgumentParser("twod_range_validity")
#     parser.add_argument("--config", default='params/twod_range_validity.yaml')
#     main(parser.parse_args())