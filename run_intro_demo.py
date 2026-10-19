import numpy as np
from typing import Callable, Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from argparse import ArgumentParser
from joblib import Parallel, delayed
from datetime import datetime

# Import custom library for maximum likelihood estimation
from mlelib import MaximumLikelihood, MlParams, MlExperimentOutput
from mlelib import valid, calc_rstar
from mlelib.validity import validity

from utils.utils import load_likelihood_model

from models.wall_quadratic import WallQuadraticModel, WallQuadraticModelParams

plt.rcParams.update({'font.size': 20})

WALL_RESIDUAL_PARAMS = WallQuadraticModelParams(marker_height=5.0, marker_range_stddev=0.1)
MIN_WALL_DIST = 2
MAX_WALL_DIST = 7
DIST_STEPS = 10
N_ESTS_PER_STEP = 1000
INITIAL_GUESS_STD = 0.3
Nind = 1

def main(args):

    likelihood_model = WallQuadraticModel(WALL_RESIDUAL_PARAMS)

    avg_rmse_array = np.zeros((N_ESTS_PER_STEP, DIST_STEPS))
    rmse_array = np.zeros(DIST_STEPS)
    x_domain = np.linspace(MIN_WALL_DIST, MAX_WALL_DIST, num=DIST_STEPS)

    # Create list of all possible distances
    coords : List[Tuple] = []
    for j, x in enumerate(x_domain):
        coords.append((j, x))

    # Callback for parallel for loop 
    def exp_cb(coord_tuple):
        j, x = coord_tuple
        x_gt = np.array([[x]])

        # Create random initial starting point for optimization
        x0 = x_gt + INITIAL_GUESS_STD * np.random.randn(1,1)

        # Evaluate estmator using ground truth
        ml_params = MlParams(
            x0=x0, 
            model=likelihood_model,
            max_its=50
        )
        ml = MaximumLikelihood(ml_params)

        L = np.linalg.cholesky(likelihood_model.S)
        z_mean =likelihood_model.h(x0)

        x_est_samples = np.zeros((1, N_ESTS_PER_STEP))
        rmse_est_samples = np.zeros((N_ESTS_PER_STEP, 1))
        for k in range(N_ESTS_PER_STEP):
            z = z_mean + L @ np.random.randn(z_mean.shape[0], 1)
            sol = ml.mle(z)
            x_est_samples[:,[k]] = sol.x
            rmse_est_samples[k,0] = np.sqrt(sol.P)[0,0]

        # All metrics are over the domain [0, infinity) with 0 as the best and infinity as the worst
        return (j, np.sqrt(np.cov(x_est_samples)), rmse_est_samples)

    results = Parallel(n_jobs=-1, verbose=True)(delayed(exp_cb)(i) for i in coords)

    # Store results in array
    for result in results:
        j, rmse, avg_rmse = result
        avg_rmse_array[:, [j]] = avg_rmse
        rmse_array[j] = rmse

    # Plot results
    plt.figure(figsize=(8.5, 6.5))
    plt.plot(x_domain, rmse_array, label='Empirical RMSE')
    vp = plt.violinplot(avg_rmse_array, x_domain, showmeans=True)
    # plt.yscale('log')
    plt.xlabel("True Distance [m]")
    plt.ylabel("RMSE")
    plt.title(f"Estimated Distance RMSE")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles + [vp['bodies'][0]], labels + ['Pred RMSE'])
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser("intro_demo")
    main(parser.parse_args())