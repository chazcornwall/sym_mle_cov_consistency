from mlelib.utils import ResidualCalculator, RobustCostCalculator
import numpy as np


class LikelihoodModel:
    """
    Defines a likelihood distribution such that f(z|x) propto exp(-cost(res(z,x)))
    """

    def __init__(self, res_calc: ResidualCalculator, cost_calc: RobustCostCalculator):
        self.__res_calc = res_calc
        self.__cost_calc = cost_calc

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
            prob_of_accepting_sample = np.exp(-self.__cost_calc.rho(self.__res_calc.res(zsample, x)))
            rejected_sample = np.random.uniform() > prob_of_accepting_sample
        return zsample

    def h(self, x):
        return self.__res_calc.h(x)

    def residual_generator(self, z, x):
        return self.__res_calc.res(z, x)

    def residual_jacobian_generator(self, x):
        return self.__res_calc.dhdx(x)

    def residual_hessian_generator(self, x):
        return self.__res_calc.d2hdx2(x)

    def cost_generator(self, res):
        return self.__cost_calc.rho(res)

    def cost_gradient_generator(self, res):
        return self.__cost_calc.drhodres(res)

    def cost_hessian_generator(self, res):
        return self.__cost_calc.d2rhodres2(res)

    def cost_deriv_hessian_generator(self, res):
        return self.__cost_calc.d3rhodres3(res)