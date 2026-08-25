import numpy as np
from typing import Callable, Tuple, List, Optional
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import stats
from pydantic import BaseModel
import yaml
from argparse import ArgumentParser
from joblib import Parallel, delayed

from mlelib import MaximumLikelihood, MlParams
from mlelib import valid

from models.twod_range import TwoDRangeModel, TwoDRangeModelParams


class ValidityExperimentConfig(BaseModel):
    x_range: Optional[List[float]] = [-10.0, 10.0]
    y_range: Optional[List[float]] = [-10.0, 10.0]
    range_resolution: Optional[int] = 100
    N_ind_meas: int = 1
    its_per_step: int = 100
    initial_guess_stddev: float = 3.0
    model_params: TwoDRangeModelParams


def main(args):

    exp_params : ValidityExperimentConfig = None
    with open(args.config, 'r') as f:
        exp_params = ValidityExperimentConfig(**yaml.safe_load(f))

    likelihood_model = TwoDRangeModel(exp_params.model_params)

    nominal_likelihood_model_params : TwoDRangeModelParams = exp_params.model_params
    nominal_likelihood_model_params.sat_range_stddev = 1.0
    nominal_likelihood_model = TwoDRangeModel(nominal_likelihood_model_params)

    anees_array = np.zeros((exp_params.range_resolution, exp_params.range_resolution))
    valid_array = np.zeros((exp_params.range_resolution, exp_params.range_resolution))

    # Create list of all possible coordinates where we wish to evaluate ANEES
    coords : List[Tuple] = []
    for j, x in enumerate(np.linspace(exp_params.x_range[0], exp_params.x_range[1], num=exp_params.range_resolution)):
        for i, y in enumerate(np.linspace(exp_params.y_range[0], exp_params.y_range[1], num=exp_params.range_resolution)):
            coords.append((j, i, x, y))

    # Parallel for loop callback
    def exp_cb(coord_tuple):
        j, i, x, y = coord_tuple
        x_gt = np.array([[x, y]]).T

        # Create random initial starting point for optimization
        x0 = x_gt + exp_params.initial_guess_stddev * np.random.randn(2,1)

        # Evaluate estmator using same ground truth but with random initial points
        ml_anees_params = MlParams(
            x0=x0, 
            model=likelihood_model,
            max_its=10
        )
        ml_anees = MaximumLikelihood(ml_anees_params)
        output_anees = ml_anees.experiment(np.array([[x, y]]).T, likelihood_model.h, likelihood_model.S, N=exp_params.its_per_step, Nind=exp_params.N_ind_meas, plot=False)

        # Evaluate nominal estimator for validity calculations
        # We do two implementations of the MLE because we need a nominal case and the baseline.
        # Under normal conditions, we could just calculate the nominal once and then vary N and sigma_z
        ml_nominal_params = MlParams(
            x0=x0, 
            model=nominal_likelihood_model,
            max_its=10
        )
        ml_nominal = MaximumLikelihood(ml_nominal_params)
        output_nominal = ml_nominal.experiment(np.array([[x, y]]).T, nominal_likelihood_model.h, nominal_likelihood_model.S, N=exp_params.its_per_step, Nind=1, plot=False)

        # Calculate validity using gt and nominal likelihood
        validity_gap = np.inf
        for k in range(nominal_likelihood_model.n):
            J_i = nominal_likelihood_model.residual_jacobian_generator(x_gt)[k,:]
            Jcal_i = nominal_likelihood_model.residual_hessian_generator(x_gt)[k,:,:]
            Pz = nominal_likelihood_model.S
            Nindmeas = nominal_likelihood_model.n / 3 * exp_params.N_ind_meas
            curr_validity_gap = valid(output_nominal.cov_empirical, exp_params.model_params.sat_range_stddev**2, Nindmeas, mode=0, J_i=J_i, Jcal_i=Jcal_i)[1]
            if curr_validity_gap < validity_gap:
                validity_gap = curr_validity_gap

        return (i, j, output_anees.anees, validity_gap)

    results = Parallel(n_jobs=-1)(delayed(exp_cb)(i) for i in coords)

    # Store results in array
    for result in results:
        i, j, anees, gap = result
        anees_array[i, j] = anees
        valid_array[i, j] = gap

    if exp_params.range_resolution > 1:
        step_size = (exp_params.x_range[1] - exp_params.x_range[0]) / (exp_params.range_resolution - 1)
    else:
        step_size = 0.0
    print(f"stepsize: {step_size}")
    extent = [exp_params.x_range[0] - step_size/2, exp_params.x_range[1] + step_size/2, exp_params.y_range[0] - step_size/2, exp_params.y_range[1] + step_size/2]

    fig, ax = plt.subplots(nrows=1, ncols=2)

    # Plot ANEES
    im = ax[0].imshow(anees_array, extent=extent, aspect='auto', origin='lower', norm=colors.LogNorm(vmin=1, vmax=1000))

    # Plot satellites
    ax[0].scatter(likelihood_model.Sats[0,:], likelihood_model.Sats[1,:], c='red', label='Satellites')
    
    # 3. Add a colorbar to see the scale
    fig.colorbar(im)
    ax[0].set_title("ANEES over Operation Domain")
    ax[0].set_xlabel("Meters")
    ax[0].set_ylabel("Meters")
    ax[0].legend()

    # Plot Validity Gap
    im = ax[1].imshow(valid_array, extent=extent, aspect='auto', origin='lower')

    # Plot satellites
    ax[1].scatter(likelihood_model.Sats[0,:], likelihood_model.Sats[1,:], c='red', label='Satellites')
    
    # 3. Add a colorbar to see the scale
    fig.colorbar(im)
    ax[1].set_title("Valdidity Gap over Operation Domain")
    ax[1].set_xlabel("Meters")
    ax[1].set_ylabel("Meters")
    ax[1].legend()

    res_p = stats.pearsonr(anees_array.flatten(), valid_array.flatten())
    print(f"Correlation between ANEES and Validity Gap:\n\tPearson: {res_p.statistic}, p-value: {res_p.pvalue}")

    plt.show()
    # Plot: heatmap of validity gap and heatmap of anees. Show how much anees is reduced
    # by staying within the threshold


if __name__ == "__main__":
    parser = ArgumentParser("twod_range_validity")
    parser.add_argument("--config", default='params/twod_range_validity.yaml')
    main(parser.parse_args())