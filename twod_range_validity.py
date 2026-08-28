import numpy as np
from typing import Callable, Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import stats
from pydantic import BaseModel
import yaml
from argparse import ArgumentParser
from joblib import Parallel, delayed
from datetime import datetime

# Import custom library for maximum likelihood estimation
from mlelib import MaximumLikelihood, MlParams, MlExperimentOutput
from mlelib import valid

# Import possible models
from models.twod_range import TwoDRangeModel, TwoDRangeModelParams
from models.twod_range_huber import TwoDRangeModelHuber, TwoDRangeModelHuberParams

plt.rcParams.update({'font.size': 16})


class ValidityExperimentConfig(BaseModel):
    name: str
    save_plot_dir: Optional[str] = "figs"
    save_plot: Optional[bool] = False
    x_range: Optional[List[float]] = [-10.0, 10.0]
    y_range: Optional[List[float]] = [-10.0, 10.0]
    range_resolution: Optional[int] = 100
    N_ind_meas: int = 1
    its_per_step: int = 100
    initial_guess_stddev: float = 3.0
    validity_mode: Optional[int] = 2 # 0 for meas, 1 for cost, 2 for both
    model_name: str
    model_params: Dict 


def create_heatmap(fig, ax, array: np.ndarray, extent: List[float], title: str, sats: np.ndarray):
    im = ax.imshow(array, extent=extent, aspect='equal', origin='lower')
    fig.colorbar(im)
    
    # Plot satellites
    ax.scatter(sats[0,:], sats[1,:], c='red', label='Satellites')
    
    ax.set_title(title)
    ax.set_xlabel("Meters")
    ax.set_ylabel("Meters")
    ax.legend()


def main(args):

    exp_params : ValidityExperimentConfig = None
    with open(args.config, 'r') as f:
        exp_params = ValidityExperimentConfig(**yaml.safe_load(f))

    # Load in the correct likelihood model depending on parameter files
    if exp_params.model_name == "twod_range":
        likelihood_model_params = TwoDRangeModelParams(**exp_params.model_params)
        likelihood_model = TwoDRangeModel(likelihood_model_params)

        # Create nominal likelihood model for calculating nominal covarinace in validity gap
        nominal_likelihood_model_params = TwoDRangeModelParams(**exp_params.model_params)
        nominal_likelihood_model_params.sat_range_stddev = 1.0
        nominal_likelihood_model = TwoDRangeModel(nominal_likelihood_model_params)
        rstar = np.zeros((nominal_likelihood_model.n, 1))
    elif exp_params.model_name == "twod_range_huber":
        likelihood_model_params = TwoDRangeModelHuberParams(**exp_params.model_params)
        likelihood_model = TwoDRangeModelHuber(likelihood_model_params)

        # Create nominal likelihood model for calculating nominal covarinace in validity gap
        nominal_likelihood_model_params = TwoDRangeModelHuberParams(**exp_params.model_params)
        nominal_likelihood_model_params.sat_range_stddev = 1.0
        nominal_likelihood_model = TwoDRangeModelHuber(nominal_likelihood_model_params)
        rstar = np.sqrt(nominal_likelihood_model_params.delta/nominal_likelihood_model.n) * np.ones((nominal_likelihood_model.n, 1))
    else:
        raise ValueError(f"Invalid model name!")

    anees_array = np.zeros((exp_params.range_resolution, exp_params.range_resolution))
    rmse_array = np.zeros((exp_params.range_resolution, exp_params.range_resolution))
    valid_array = np.zeros((exp_params.range_resolution, exp_params.range_resolution))

    # Create list of all possible coordinates where we wish to evaluate ANEES
    coords : List[Tuple] = []
    for j, x in enumerate(np.linspace(exp_params.x_range[0], exp_params.x_range[1], num=exp_params.range_resolution)):
        for i, y in enumerate(np.linspace(exp_params.y_range[0], exp_params.y_range[1], num=exp_params.range_resolution)):
            coords.append((j, i, x, y))

    # Callback for parallel for loop 
    def exp_cb(coord_tuple):
        j, i, x, y = coord_tuple
        x_gt = np.array([[x, y]]).T

        # Create random initial starting point for optimization
        x0 = x_gt + exp_params.initial_guess_stddev * np.random.randn(2,1)

        # Evaluate estmator using ground truth
        ml_params = MlParams(
            x0=x0, 
            model=likelihood_model,
            max_its=50
        )
        ml = MaximumLikelihood(ml_params)
        output : MlExperimentOutput = ml.experiment(x_gt, likelihood_model.h, likelihood_model.S, N=exp_params.its_per_step, Nind=exp_params.N_ind_meas, plot=False)

        # Get terms used in validity calculation
        J = nominal_likelihood_model.residual_jacobian_generator(x_gt)
        Jcal = nominal_likelihood_model.residual_hessian_generator(x_gt)
        H0 = nominal_likelihood_model.cost_hessian_generator(np.zeros((nominal_likelihood_model.n, 1)))
        Hbold = nominal_likelihood_model.cost_deriv_hessian_generator(rstar)
        Hcal = nominal_likelihood_model.cost_hessian_generator(rstar)
        Fbarinv = np.linalg.inv(J.T @ H0 @ J)
        Pz = nominal_likelihood_model.S
        Nindmeas = nominal_likelihood_model.n / 3 * exp_params.N_ind_meas

        # Calculate validity using gt and nominal likelihood
        validity_gap = np.inf
        for k in range(nominal_likelihood_model.n):
            curr_validity_gap = valid(Fbarinv, likelihood_model_params.sat_range_stddev**2, Nindmeas, mode=exp_params.validity_mode, J_i=J[k,:], J=J, Jcal_i=Jcal[k,:,:], Hbold_i=Hbold[k,:,:], Hcal_i=Hcal[k,:], Pz=Pz, rstar=rstar)[1]
            if curr_validity_gap < validity_gap:
                validity_gap = curr_validity_gap

        return (i, j, np.log(output.anees / 2), validity_gap, output.rmse)

    results = Parallel(n_jobs=-1, verbose=True)(delayed(exp_cb)(i) for i in coords)

    # Store results in array
    for result in results:
        i, j, anees, gap, rmse = result
        anees_array[i, j] = anees
        valid_array[i, j] = gap
        rmse_array[i, j] = rmse

    # Calculate the extent of the plots
    if exp_params.range_resolution > 1:
        step_size = (exp_params.x_range[1] - exp_params.x_range[0]) / (exp_params.range_resolution - 1)
    else:
        step_size = 0.0
    print(f"stepsize: {step_size}")
    extent = [exp_params.x_range[0] - step_size/2, exp_params.x_range[1] + step_size/2, exp_params.y_range[0] - step_size/2, exp_params.y_range[1] + step_size/2]

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    create_heatmap(fig, ax[0], anees_array, extent, "lnANEES over Operation Domain", likelihood_model.Sats)
    create_heatmap(fig, ax[1], rmse_array, extent, "RMSE over Operation Domain", likelihood_model.Sats)
    create_heatmap(fig, ax[2], valid_array, extent, "VG over Operation Domain", likelihood_model.Sats)

    # Print Pearson correlations and corresponding p values
    res_anees_p = stats.pearsonr(anees_array.flatten(), valid_array.flatten())
    print(f"Pearson's Correlation between lnANEES and Validity Gap:\n\tPearson: {res_anees_p.statistic}, p-value: {res_anees_p.pvalue}")

    res_rmse_p = stats.pearsonr(rmse_array.flatten(), valid_array.flatten())
    print(f"Pearson's Correlation between RMSE and Validity Gap:\n\tPearson: {res_rmse_p.statistic}, p-value: {res_rmse_p.pvalue}")

    if exp_params.save_plot:
        now = datetime.now()
        filename = exp_params.name + "_" + now.strftime("%Y%m%d_%H%M%S") + ".png"
        plt.savefig("figs/" + filename, bbox_inches="tight")

    plt.tight_layout() 
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser("twod_range_validity")
    parser.add_argument("--config", default='params/twod_range_validity.yaml')
    main(parser.parse_args())