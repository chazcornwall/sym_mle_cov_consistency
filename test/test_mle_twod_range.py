from models.twod_range import TwoDRangeModel, TwoDRangeModelParams, SatelliteLocation
from mlelib import MaximumLikelihood, MlParams, MlSolution

import numpy as np

def test_mle_unbiased():

    sat_locs = [
        SatelliteLocation(x=0.0, y=10.0),
        SatelliteLocation(x=10.0, y=0.0),
        SatelliteLocation(x=-10.0, y=10.0)
    ]
    params = TwoDRangeModelParams(sat_locs=sat_locs, sat_range_stddev=2.0)
    likelihood_model = TwoDRangeModel(params)

    x_gt = np.array([[1.0, 1.0]]).T

    # Create random initial starting point for optimization
    x0 = x_gt + 1.0 * np.random.randn(2,1)

    # Evaluate estmator using same ground truth but with random initial points
    ml_params = MlParams(
        x0=x0, 
        model=likelihood_model,
        max_its=100
    )
    ml = MaximumLikelihood(ml_params)

    L = np.linalg.cholesky(likelihood_model.S)
    z_mean = likelihood_model.h(x_gt)

    error = np.zeros((2, 1000))
    for i in range(1000):
        z = z_mean + L @ np.random.randn(z_mean.shape[0], 10)
        output = ml.mle(z)
        error[:,[i]] = output.x - x_gt
    print(f"E[x - x_gt]: {np.mean(error, axis=1)}")
    assert np.all(np.isclose(np.mean(error, axis=1), 0.0, atol=1e-1)) 
    