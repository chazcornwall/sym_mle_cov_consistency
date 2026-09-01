import numpy as np
from typing import Callable, Tuple, List
import matplotlib.pyplot as plt

from mlelib import LikelihoodModel
from robust_costs.cauchy import CauchyCost
from residuals.range import RangeResidual

from pydantic import BaseModel
import yaml
from argparse import ArgumentParser
from typing import Optional, List


class SatelliteLocation(BaseModel):
    x: float
    y: float


class TwoDRangeModelCauchyParams(BaseModel):
    sat_locs: List[SatelliteLocation]
    sat_range_stddev: float = 1.0
    delta: float = 1.0


class TwoDRangeModelCauchy(LikelihoodModel):

    def __init__(self, params: TwoDRangeModelCauchyParams):
        
        self.__n = len(params.sat_locs)
        self.__delta = params.delta

        # Calculate measurement noise
        self.__S = params.sat_range_stddev**2 * np.eye(self.__n)
        self.__Sinv = np.linalg.inv(self.__S)
        self.__Sinvsqrt = np.linalg.cholesky(self.__Sinv)

        # Store satellite locations
        self.__Sats = np.zeros((2, self.__n))
        for i in range(self.__n):
            self.__Sats[:,[i]] = np.array([[params.sat_locs[i].x, params.sat_locs[i].y]]).T

        # Create cost and residual definitions
        cost_calc = CauchyCost(self.__Sinv, self.__delta)
        res_calc = RangeResidual(self.__Sats)

        super().__init__(res_calc, cost_calc)

    @property
    def n(self) -> int:
        return self.__n
    
    @property
    def S(self) -> np.ndarray:
        return self.__S

    @property
    def Sats(self) -> np.ndarray:
        return self.__Sats