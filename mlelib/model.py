from abc import ABC, abstractmethod
import numpy as np


class LikelihoodModel(ABC):
    """
    Defines a likelihood distribution such that f(z|x) propto exp(-cost(res(z,x)))
    """

    def __init__(self):
        pass

    @abstractmethod
    def residual_generator(self, z: np.ndarray, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def residual_jacobian_generator(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def residual_hessian_generator(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def cost_generator(self, res: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def cost_gradient_generator(self, res: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def cost_hessian_generator(self, res: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def cost_deriv_hessian_generator(self, res: np.ndarray) -> np.ndarray:
        pass