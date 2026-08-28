from abc import ABC, abstractmethod
import numpy as np


class LikelihoodModel(ABC):
    """
    Defines a likelihood distribution such that f(z|x) propto exp(-cost(res(z,x)))
    """

    def __init__(self):
        pass

    def sample(self, x: np.ndarray, zmin: np.ndarray, zmax: np.ndarray) -> np.ndarray:
        '''
        Generates a sample from the likelihood distribution using rejection sampling
        https://en.wikipedia.org/wiki/Rejection_sampling#Algorithmic_Definition
        '''
        # f_prop <= Mg(z|x) where we select g to be a uniform distribution.
        # Max of term proportional to likelihood is 1 (e^0). Therefore, we
        # scale g to envelop f_prop such that Mg(z|x) = 1
        rejected_sample = True
        while rejected_sample:
            zsample = np.random.uniform(zmin, zmax)
            prob_of_accepting_sample = np.exp(-self.cost_generator(self.residual_generator(zsample, x)))
            rejected_sample = np.random.uniform() > prob_of_accepting_sample
        return zsample

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
    def cost_generator(self, res: np.ndarray) -> float:
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