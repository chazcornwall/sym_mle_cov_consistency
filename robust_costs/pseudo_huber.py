from mlelib.utils import RobustCostCalculator
import numpy as np


class PseudoHuberCost(RobustCostCalculator):

    def __init__(self, Sinv, delta: float):
        super().__init__(Sinv)
        self.__delta = delta

    def _rho(self, q):
        delta = self.__delta
        return delta**2 * (np.sqrt(1 + q / delta**2) - 1)

    def _drhodq(self, q):
        delta = self.__delta
        return delta / 2 * (delta**2 + q)**(-1/2)

    def _d2rhodq2(self, q):
        delta = self.__delta
        return -delta / 4 * (delta**2 + q)**(-3/2)

    def _d3rhodq3(self, q):
        delta = self.__delta
        return (3 * delta) / 8 * (delta**2 + q)**(-5/2)
        