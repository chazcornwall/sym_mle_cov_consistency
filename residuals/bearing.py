from mlelib.utils import ResidualCalculator, a2pi
import numpy as np

class BearingResidual(ResidualCalculator):

    def __init__(self, sats: np.ndarray):
        self.__sats = sats
        self.__n = sats.shape[1]
        super().__init__()

    def __s(self, i: int) -> np.ndarray:
        return self.__sats[:, [i]] 

    def res(self, z, x):
        return a2pi(super().res(z, x))

    def h(self, x):
        h = np.zeros((self.__n,1))
        for i in range(self.__n):
            diff = self.__s(i) - x
            h[i,0] = np.arctan2(diff[1,0], diff[0,0])
        return h

    def dhdx(self, x):
        # See here for definition of derivative of atan2: https://en.wikipedia.org/wiki/Atan2
        J = np.zeros((self.__n, x.shape[0]))
        P = np.array([[0, -1], [1, 0]]) # Permutation matrix
        for i in range(self.__n):
            diff = self.__s(i) - x
            J[[i], :] = -1 / (diff.T @ diff) * (P @ diff).T 
        return J

    def d2hdx2(self, x):
        JHess = np.zeros((self.__n, x.shape[0], x.shape[0]))
        P = np.array([[0, -1], [1, 0]]) # Permutation matrix
        I = np.eye(x.shape[0])
        for i in range(self.__n):
            diff = self.__s(i) - x
            normsq = diff.T @ diff
            JHess[[i], :, :] = 1 / normsq * (I - 2 * diff @ diff.T / (normsq**2)) @ P.T
        return JHess