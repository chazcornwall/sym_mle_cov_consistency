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
from mlelib.validity import validity

from utils.utils import load_likelihood_model

plt.rcParams.update({'font.size': 16})

class Location(BaseModel):
    x: float
    y: float


class CostValidityExperimentConfig(BaseModel):
    name: str
    save_plot_dir: Optional[str] = "figs"
    save_plot: Optional[bool] = False
    save_data_dir: Optional[str] = "data"
    save_data: Optional[bool] = False
    delta_range: Optional[List[float]] = [0, 10.0]
    range_resolution: Optional[int] = 100
    eval_locs: List[Location] = []
    N_ind_meas: int = 1
    its_per_step: int = 100
    initial_guess_stddev: float = 3.0
    validity_mode: Optional[int] = 2 # 0 for meas, 1 for cost, 2 for both
    model_name: str
    model_params: Dict 


def plot_eval_locations(fig, ax, locs: np.ndarray, title: str, sats: np.ndarray):
    ax.scatter(sats[0,:], sats[1,:], c='red', label='Satellites')
    ax.scatter(locs[0,:], locs[1,:], c='blue', marker='*', s=144, label="Eval Locs")

    for i in range(locs.shape[1]):
        ax.text(locs[0,i], locs[1,i] + 0.4, f"{i}", 
                horizontalalignment='center', 
                verticalalignment='center', 
                fontsize=12, 
                color='black', 
                weight='bold')
    
    ax.set_title(title)
    ax.set_xlabel("Meters")
    ax.set_ylabel("Meters")
    ax.legend()


def create_heatmap(fig, ax, array: np.ndarray, extent: List[float], title: str, n_locs: int):
    im = ax.imshow(array, extent=extent, aspect='auto', origin='lower')
    fig.colorbar(im)
    ax.set_title(title)
    ax.set_xlabel("Delta")
    ax.set_ylabel("Locations")
    ax.set_yticks(np.arange(n_locs), labels=[str(i) for i in range(n_locs)])


def main(args):

    exp_params : CostValidityExperimentConfig = None
    with open(args.config, 'r') as f:
        exp_params = CostValidityExperimentConfig(**yaml.safe_load(f))

    likelihood_model_cb, likelihood_model, likelihood_model_params, rstar = load_likelihood_model(exp_params.model_name, exp_params.model_params)

    # Create list of all possible coordinates where we wish to evaluate ANEES
    coords : List[Tuple] = []

    # Store evaluation locations
    n_eval_locs = len(exp_params.eval_locs)
    locs = np.zeros((2, n_eval_locs))
    for i in range(n_eval_locs):
        locs[:,[i]] = np.array([[exp_params.eval_locs[i].x, exp_params.eval_locs[i].y]]).T

    if args.data_prefix is None:
        anees_array = np.zeros((n_eval_locs, exp_params.range_resolution))
        bias_mag_array = np.zeros((n_eval_locs, exp_params.range_resolution))
        valid_array = np.zeros((n_eval_locs, exp_params.range_resolution))
        valid_mask_array = np.zeros((n_eval_locs, exp_params.range_resolution))

        for j, delta in enumerate(np.linspace(exp_params.delta_range[0], exp_params.delta_range[1], num=exp_params.range_resolution)):
            for i in range(n_eval_locs):
                coords.append((j, i, delta, locs[:,[i]]))

        # Callback for parallel for loop 
        def exp_cb(coord_tuple):
            j, i, delta, pos = coord_tuple

            # Create random initial starting point for optimization
            x_gt = pos
            x0 = x_gt + exp_params.initial_guess_stddev * np.random.randn(2,1)

            # Evaluate estmator using ground truth
            likelihood_model_params.delta = delta
            likelihood_model = likelihood_model_cb(likelihood_model_params)
            ml_params = MlParams(
                x0=x0, 
                model=likelihood_model,
                max_its=50
            )
            ml = MaximumLikelihood(ml_params)
            output : MlExperimentOutput = ml.experiment(x_gt, likelihood_model.h, likelihood_model.S, N=exp_params.its_per_step, Nind=exp_params.N_ind_meas, plot=False)
            valid_mask, validity_ratio = validity(likelihood_model, likelihood_model.n, exp_params.N_ind_meas, likelihood_model.S, x_gt, rstar, rstar, mode=exp_params.validity_mode)
            valid_mask = 1.0 if valid_mask else 0.0

            return (j, i, np.abs(np.log(output.anees / 2)), validity_ratio, valid_mask, output.bias_mag)

        results = Parallel(n_jobs=-1, verbose=True)(delayed(exp_cb)(i) for i in coords)

        # Store results in array
        for result in results:
            j, i, anees, ratio, mask, bias_mag = result
            anees_array[i, j] = anees
            valid_array[i, j] = ratio
            valid_mask_array[i, j] = mask
            bias_mag_array[i, j] = bias_mag
    else:
        anees_array = np.load(args.data_prefix + "_anees.npy")
        bias_mag_array = np.load(args.data_prefix + "_bias_mag.npy")
        valid_array = np.load(args.data_prefix + "_valid.npy")
        valid_mask_array = np.load(args.data_prefix + "_valid_mask.npy")

    # Calculate the extent of the plots
    if exp_params.range_resolution > 1:
        step_size = (exp_params.delta_range[1] - exp_params.delta_range[0]) / (exp_params.range_resolution - 1)
    else:
        step_size = 0.0
    print(f"stepsize: {step_size}")
    extent = [exp_params.delta_range[0] - step_size/2, exp_params.delta_range[1] + step_size/2, 0 - step_size/2, n_eval_locs-1 + step_size/2]

    fig, ax = plt.subplots(figsize=(5,5))
    plot_eval_locations(fig, ax, locs, "Evaluation Locations", likelihood_model.Sats)
    plt.tight_layout()

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    create_heatmap(fig, ax[0], anees_array, extent, "lnANEES", n_eval_locs)
    create_heatmap(fig, ax[1], bias_mag_array, extent, "Bias Magnitude", n_eval_locs)
    create_heatmap(fig, ax[2], valid_array, extent, "Validity Ratio", n_eval_locs)
    create_heatmap(fig, ax[3], valid_mask_array, extent, "Validity Mask", n_eval_locs)
    plt.tight_layout() 

    output_lines_to_save : List[str] = []

    # Print Pearson correlations and corresponding p values
    anees_non_nan_mask = np.logical_and(~np.isnan(anees_array), ~np.isnan(valid_array))
    res_anees_p = stats.pearsonr(anees_array[anees_non_nan_mask].flatten(), valid_array[anees_non_nan_mask].flatten())
    res_anees_s = stats.spearmanr(anees_array[anees_non_nan_mask].flatten(), valid_array[anees_non_nan_mask].flatten())
    output_lines_to_save.append(f"Pearson's Correlation between lnANEES and VR:\n\tPearson: {res_anees_p.statistic}, p-value: {res_anees_p.pvalue}")
    output_lines_to_save.append(f"Spearmans's Correlation between lnANEES and VR:\n\tSpearman: {res_anees_s.statistic}, p-value: {res_anees_s.pvalue}")

    bias_mag_non_nan_mask = np.logical_and(~np.isnan(bias_mag_array), ~np.isnan(valid_array))
    res_bias_mag_p = stats.pearsonr(bias_mag_array[bias_mag_non_nan_mask].flatten(), valid_array[bias_mag_non_nan_mask].flatten())
    res_bias_mag_s = stats.spearmanr(bias_mag_array[bias_mag_non_nan_mask].flatten(), valid_array[bias_mag_non_nan_mask].flatten())
    output_lines_to_save.append(f"Pearson's Correlation between Bias Magnitude and VR:\n\tPearson: {res_bias_mag_p.statistic}, p-value: {res_bias_mag_p.pvalue}")
    output_lines_to_save.append(f"Spearman's Correlation between Bias Magnitude and VR:\n\tSpearman: {res_bias_mag_s.statistic}, p-value: {res_bias_mag_s.pvalue}")

    for output in output_lines_to_save:
        print(output)

    if exp_params.save_data and args.data_prefix is None: # Do not want to resave data!
        now = datetime.now()
        dir = "data/"
        if exp_params.save_data_dir:
            dir = exp_params.save_data_dir + "/"
        filename = exp_params.name + "_" + now.strftime("%Y%m%d_%H%M%S")
        np.save(filename + "_anees.npy", anees_array)
        np.save(filename + "_bias_mag.npy", bias_mag_array)
        np.save(filename + "_valid.npy", valid_array)
        np.save(filename + "_valid_mask.npy", valid_mask_array)

    if exp_params.save_plot:
        now = datetime.now()
        dir = "figs/"
        if exp_params.save_plot_dir:
            dir = exp_params.save_plot_dir + "/"
        filename = exp_params.name + "_" + now.strftime("%Y%m%d_%H%M%S")
        plt.savefig(dir + filename + ".png", bbox_inches="tight")

        # Save stats in a text file
        with open(dir + filename + ".txt", "w", encoding="utf-8") as file:
            file.write("\n".join(output_lines_to_save))

    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser("twod_range_validity_cost")
    parser.add_argument("--config", default='params/cost_validity/twod_range_huber_validity.yaml')
    parser.add_argument("--data_prefix", default=None)
    main(parser.parse_args())