import numpy as np
from typing import Callable, Tuple, List, Optional
import matplotlib.pyplot as plt
from pydantic import BaseModel
import yaml
from argparse import ArgumentParser

from mlelib import MaximumLikelihood, MlParams
from mlelib import valid

from models.twod_range import residual_generator, residual_jacobian_generator, residual_hessian_generator, cost_function_generator, cost_function_gradient_generator, cost_function_hessian_generator


class SatelliteLocation(BaseModel):
    x: float
    y: float


class ValidityExperimentConfig(BaseModel):
    x_range: Optional[List[float]] = [-10.0, 10.0]
    y_range: Optional[List[float]] = [-10.0, 10.0]
    range_resolution: Optional[int] = 100
    satellite_locs: List[SatelliteLocation]
    satellite_range_stddev: float = 1.0
    its_per_step: int = 100
    initial_guess_stddev: float = 3


def main(args):

    exp_params : ValidityExperimentConfig = None
    with open(args.config, 'r') as f:
        exp_params = ValidityExperimentConfig(**yaml.safe_load(f))

    for x in exp_params.x_range:
        for y in exp_params.y_range:

            ml_params = MlParams(
                x0=np.zeros((x + exp_params.initial_guess_stddev * np.random.randn(), 
                             y + exp_params.initial_guess_stddev * np.random.randn())), 
                residual_generator=residual_generator,
                residual_jacobian_generator=residual_jacobian_generator,
                residual_hessian_generator=residual_hessian_generator,
                cost_function_generator=cost_function_generator,
                cost_function_gradient_generator=cost_function_gradient_generator,
                cost_function_hessian_generator=cost_function_hessian_generator,
                max_its=10
            )
            ml = MaximumLikelihood(ml_params)
            ml.experiment(np.array([[x, y]]).T, h, S, N=exp_params.its_per_step)

            # Plot: heatmap of validity gap and heatmap of anees. Show how much anees is reduced
            # by staying within the threshold


if __name__ == "__main__":
    parser = ArgumentParser("twod_range_validity")
    parser.add_argument("--config", default='params/twod_range_validity.yaml')
    main(parser.parse_args())