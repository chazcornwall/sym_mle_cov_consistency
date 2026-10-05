import numpy as np
from typing import Callable, Tuple, List

from mlelib import LikelihoodModel
from robust_costs.quadratic import QuadraticCost
from residuals.wall import WallResidual

from pydantic import BaseModel
import yaml
from argparse import ArgumentParser
from typing import Optional, List


class WallQuadraticModelParams(BaseModel):
    marker_height: float = 5.0
    marker_range_stddev: float = 1.0


class WallQuadraticModel(LikelihoodModel):

    def __init__(self, params: WallQuadraticModelParams):
        # Calculate measurement noise
        self.__S = params.marker_range_stddev**2 * np.eye(1)
        self.__Sinv = np.linalg.inv(self.__S)
        self.__Sinvsqrt = np.linalg.cholesky(self.__Sinv)

        res_calc = WallResidual(params.marker_height)
        cost_calc = QuadraticCost(self.__Sinv)
        super().__init__(res_calc, cost_calc)

    @property
    def S(self) -> np.ndarray:
        return self.__S