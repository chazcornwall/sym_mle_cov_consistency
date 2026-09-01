from mlelib.utils import ResidualCalculator
import numpy as np

class RangeResidual(ResidualCalculator):

    def __init__(self, sats: np.ndarray):
        self.__sats = sats
        self.__n = sats.shape[1]
        super().__init__()

    def __s(self, i: int) -> np.ndarray:
        return self.__sats[:, [i]] 

    def h(self, x: np.ndarray) -> np.ndarray:
        h = np.zeros((self.__n,1))
        for i in range(self.__n):
            h[i,0] = np.linalg.norm(x - self.__s(i))
        return h

    def dhdx(self, x):
        hx = self.h(x)
        J = np.zeros((self.__n, x.shape[0]))
        for i in range(self.__n):
            J[[i], :] = 1 / hx[i,0] * (x - self.__s(i)).T
        return J

    def d2hdx2(self, x):
        hx = self.h(x)
        JHess = np.zeros((self.__n, x.shape[0], x.shape[0]))
        I = np.eye(x.shape[0])
        for i in range(self.__n):
            si = self.__s(i)
            JHess[[i], :, :] = 1 / hx[i,0] * (I - (x - si) @ (x - si).T / hx[i,0]**2)
        return JHess