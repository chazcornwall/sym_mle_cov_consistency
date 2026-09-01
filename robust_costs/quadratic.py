from mlelib.utils import RobustCostCalculator


class QuadraticCost(RobustCostCalculator):

    def __init__(self, Sinv):
        super().__init__(Sinv)

    def _rho(self, q):
        return 0.5 * q

    def _drhodq(self, q):
        return 0.5

    def _d2rhodq2(self, q):
        return 0.0

    def _d3rhodq3(self, q):
        return 0.0
    