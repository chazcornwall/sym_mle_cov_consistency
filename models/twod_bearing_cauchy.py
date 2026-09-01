import numpy as np
from typing import Callable, Tuple, List
import matplotlib.pyplot as plt

from mlelib import LikelihoodModel
from robust_costs.cauchy import CauchyCost
from residuals.bearing import BearingResidual

from pydantic import BaseModel
import yaml
from argparse import ArgumentParser
from typing import Optional, List


class SatelliteLocation(BaseModel):
    x: float
    y: float


class TwoDBearingModelCauchyParams(BaseModel):
    sat_locs: List[SatelliteLocation]
    sat_bearing_stddev_deg: float = 1.0
    delta: float = 1.0


class TwoDBearingModelCauchy(LikelihoodModel):

    def __init__(self, params: TwoDBearingModelCauchyParams):
        self.__n = len(params.sat_locs)
        self.__delta = params.delta

        # Calculate measurement noise
        self.__S = np.deg2rad(params.sat_bearing_stddev_deg)**2 * np.eye(self.__n)
        self.__Sinv = np.linalg.inv(self.__S)
        self.__Sinvsqrt = np.linalg.cholesky(self.__Sinv)

        # Store satellite locations
        self.__Sats = np.zeros((2, self.__n))
        for i in range(self.__n):
            self.__Sats[:,[i]] = np.array([[params.sat_locs[i].x, params.sat_locs[i].y]]).T

        res_calc = BearingResidual(self.__Sats)
        cost_calc = CauchyCost(self.__Sinv, self.__delta)
        super().__init__(res_calc, cost_calc)

    def __s(self, i: int) -> np.ndarray:
        return self.__Sats[:, [i]] 

    @property
    def n(self) -> int:
        return self.__n

    @property
    def S(self) -> np.ndarray:
        return self.__S

    @property
    def Sats(self) -> np.ndarray:
        return self.__Sats