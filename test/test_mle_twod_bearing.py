from models.twod_bearing import TwoDBearingModel, TwoDBearingModelParams, SatelliteLocation
from mlelib import MaximumLikelihood, MlParams, MlSolution

import numpy as np

def test_mle_unbiased():

    sat_locs = [
        SatelliteLocation(x=0.0, y=5.0),
        SatelliteLocation(x=4.0, y=2.0),
        SatelliteLocation(x=-5.0, y=2.0)
    ]
    params = TwoDBearingModelParams(sat_locs=sat_locs, sat_bearing_stddev_deg=3.0)
    likelihood_model = TwoDBearingModel(params)

    x_gt = np.array([[-5.0, 1.9]]).T

    # Create random initial starting point for optimization
    x0 = np.array([[-5.0, 2.1]]).T # x_gt + 1.0 * np.random.randn(2,1)

    # Evaluate estmator using same ground truth but with random initial points
    ml_params = MlParams(
        x0=x0, 
        model=likelihood_model,
        max_its=20
    )
    ml = MaximumLikelihood(ml_params)

    L = np.linalg.cholesky(likelihood_model.S)
    z_mean = likelihood_model.h(x_gt)

    N = 1000
    error = np.zeros((2, N))
    for i in range(N):
        z = z_mean + L @ np.random.randn(z_mean.shape[0], 1)
        output = ml.mle(z)
        error[:,[i]] = output.x - x_gt
    print(f"E[x - x_gt]: {np.mean(error, axis=1)}")
    assert np.all(np.isclose(np.mean(error, axis=1), 0.0, atol=1e-1)) 