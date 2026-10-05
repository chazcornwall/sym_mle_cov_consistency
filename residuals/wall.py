from mlelib.utils import ResidualCalculator
import numpy as np

class WallResidual(ResidualCalculator):
    """
    The state is the distance from a wall and the measurement is the range to a marker placed at a known, fixed height above the ground.
    """

    def __init__(self, marker_height: float):
        self.__marker_height = marker_height
        super().__init__()

    def h(self, x: np.ndarray) -> np.ndarray:
        h = np.zeros((1,1))
        h[0,0] = np.sqrt(x[0,0]**2 + self.__marker_height**2)
        return h

    def dhdx(self, x):
        hx = self.h(x)
        J = np.zeros((1, 1))
        J[0,0] = 1 / hx[0,0] * x[0,0]
        return J

    def d2hdx2(self, x):
        hx = self.h(x)
        JHess = np.zeros((1, 1, 1))
        JHess[0, 0, 0] = 1 / hx[0,0] * (1 - x[0,0]**2 / hx[0,0]**2)
        return JHess