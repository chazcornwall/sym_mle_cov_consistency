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
from mlelib import valid, calc_rstar
from mlelib.validity import validity

from utils.utils import load_likelihood_model

plt.rcParams.update({'font.size': 16})


class ValidityExperimentConfig(BaseModel):
    name: str
    save_plot_dir: Optional[str] = "figs"
    save_plot: Optional[bool] = False
    save_data_dir: Optional[str] = "data"
    save_data: Optional[bool] = False
    x_range: Optional[List[float]] = [-10.0, 10.0]
    y_range: Optional[List[float]] = [-10.0, 10.0]
    range_resolution: Optional[int] = 100
    N_ind_meas: int = 1
    its_per_step: int = 100
    initial_guess_stddev: float = 3.0
    validity_mode: Optional[int] = 2 # 0 for meas, 1 for cost, 2 for both
    model_name: str
    model_params: Dict 


def create_heatmap(fig, ax, array: np.ndarray, extent: List[float], title: str, sats: np.ndarray, lognorm: bool = False, cmap: str = 'viridis'):
    if lognorm:
        im = ax.imshow(array, extent=extent, aspect='equal', origin='lower', norm=colors.LogNorm(), cmap=cmap)
    else: 
        im = ax.imshow(array, extent=extent, aspect='equal', origin='lower', cmap=cmap)
    fig.colorbar(im, shrink=0.55)
    
    # Plot satellites
    ax.scatter(sats[0,:], sats[1,:], c='red', label='Satellites')
    
    ax.set_title(title)
    ax.set_xlabel("Meters")
    ax.set_ylabel("Meters")
    # ax.legend()


def main(args):

    exp_params : ValidityExperimentConfig = None
    with open(args.config, 'r') as f:
        exp_params = ValidityExperimentConfig(**yaml.safe_load(f))

    _, likelihood_model, likelihood_model_params, rstar = load_likelihood_model(exp_params.model_name, exp_params.model_params)

    if args.data_prefix is None:
        anees_array = np.zeros((exp_params.range_resolution, exp_params.range_resolution))
        bias_mag_array = np.zeros((exp_params.range_resolution, exp_params.range_resolution))
        valid_array = np.zeros((exp_params.range_resolution, exp_params.range_resolution))
        valid_mask_array = np.zeros((exp_params.range_resolution, exp_params.range_resolution))

        def valid_gt(x_gt: np.ndarray) -> bool:
            for loc in likelihood_model_params.sat_locs:
                if abs(x_gt[0,0] - loc.x) < 1e-8 and abs(x_gt[1,0] - loc.y) < 1e-8:
                    return False
            return True

        # Create list of all possible coordinates where we wish to evaluate ANEES, Bias, and VR
        coords : List[Tuple] = []
        for j, x in enumerate(np.linspace(exp_params.x_range[0], exp_params.x_range[1], num=exp_params.range_resolution)):
            for i, y in enumerate(np.linspace(exp_params.y_range[0], exp_params.y_range[1], num=exp_params.range_resolution)):
                coords.append((j, i, x, y))

        # Callback for parallel for loop 
        def exp_cb(coord_tuple):
            j, i, x, y = coord_tuple
            x_gt = np.array([[x, y]]).T

            if valid_gt(x_gt):
                # Create random initial starting point for optimization
                x0 = x_gt + exp_params.initial_guess_stddev * np.random.randn(2,1)
                reject_biased = lambda sol : np.linalg.norm(sol - x_gt) > 5.0

                # Evaluate estmator using ground truth
                ml_params = MlParams(
                    x0=x0, 
                    model=likelihood_model,
                    max_its=50
                )
                ml = MaximumLikelihood(ml_params)
                output : MlExperimentOutput = ml.experiment(x_gt, likelihood_model.h, likelihood_model.S, N=exp_params.its_per_step, Nind=exp_params.N_ind_meas, plot=False, outlier_reject=reject_biased)

                valid_mask, validity_ratio = validity(likelihood_model, likelihood_model.n, exp_params.N_ind_meas, likelihood_model.S, x_gt, rstar, rstar, mode=exp_params.validity_mode)
            else:
                output : MlExperimentOutput = MlExperimentOutput(anees=np.nan, bias_mag=np.nan, cov_empirical=np.full((x_gt.shape[0], x_gt.shape[0]), np.nan))
                validity_ratio = np.nan
                valid_mask = False

            valid_mask = 1.0 if valid_mask else 0.0

            # All metrics are over the domain [0, infinity) with 0 as the best and infinity as the worst
            return (i, j, np.abs(np.log(output.anees / x_gt.shape[0])), validity_ratio, valid_mask, output.bias_mag)

        results = Parallel(n_jobs=-1, verbose=True)(delayed(exp_cb)(i) for i in coords)

        # Store results in array
        for result in results:
            i, j, anees, ratio, mask, bias_mag = result
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
        step_size = (exp_params.x_range[1] - exp_params.x_range[0]) / (exp_params.range_resolution - 1)
    else:
        step_size = 0.0
    print(f"stepsize: {step_size}")
    extent = [exp_params.x_range[0] - step_size/2, exp_params.x_range[1] + step_size/2, exp_params.y_range[0] - step_size/2, exp_params.y_range[1] + step_size/2]

    # Comment the following code block when creating the introduction image
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    if "twod_range" in exp_params.model_name:
        create_heatmap(fig, ax[0], anees_array, extent, "lnANEES", likelihood_model.Sats)
        create_heatmap(fig, ax[1], bias_mag_array, extent, "Bias Magnitude", likelihood_model.Sats)
        create_heatmap(fig, ax[2], valid_array, extent, "Validity Ratio", likelihood_model.Sats)
    else:
        create_heatmap(fig, ax[0], anees_array, extent, "lnANEES", likelihood_model.Sats, lognorm=True)
        create_heatmap(fig, ax[1], bias_mag_array, extent, "Bias Magnitude", likelihood_model.Sats, lognorm=True)
        create_heatmap(fig, ax[2], valid_array, extent, "Validity Ratio", likelihood_model.Sats, lognorm=True)

    create_heatmap(fig, ax[3], valid_mask_array, extent, "Validity Mask", likelihood_model.Sats, cmap='viridis_r')

    # Place legend beneath all subplots
    handles, labels = [], []
    for a in ax.flatten():
        h, l = a.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    fig.legend(
        handles=[handles[-1]],
        labels=[labels[-1]],
        loc="upper center",          # Alignment point of the legend box
        bbox_to_anchor=(0.5, 0.15), # Coordinates relative to the figure (x, y) # bbox_to_achor=(0.5, 0.05) when creating introduction image
        ncols=2                      # Make the legend layout horizontal
    )
    plt.tight_layout() 

    output_lines_to_save : List[str] = []

    # Print Pearson correlations and corresponding p values
    anees_non_nan_mask = np.logical_and(~np.isnan(anees_array), ~np.isnan(valid_array))
    res_anees_p = stats.pearsonr(anees_array[anees_non_nan_mask].flatten(), valid_array[anees_non_nan_mask].flatten())
    res_anees_s = stats.spearmanr(anees_array[anees_non_nan_mask].flatten(), valid_array[anees_non_nan_mask].flatten())
    res_anees_k = stats.kendalltau(anees_array[anees_non_nan_mask].flatten(), valid_array[anees_non_nan_mask].flatten())
    output_lines_to_save.append(f"Pearson's Correlation between lnANEES and VR:\n\tPearson: {res_anees_p.statistic}, p-value: {res_anees_p.pvalue}")
    output_lines_to_save.append(f"Spearmans's Correlation between lnANEES and VR:\n\tSpearman: {res_anees_s.statistic}, p-value: {res_anees_s.pvalue}")
    output_lines_to_save.append(f"Kendall's tau between lnANEES and VR:\n\ttau: {res_anees_k.statistic}, p-value: {res_anees_k.pvalue}")

    bias_mag_non_nan_mask = np.logical_and(~np.isnan(bias_mag_array), ~np.isnan(valid_array))
    res_bias_mag_p = stats.pearsonr(bias_mag_array[bias_mag_non_nan_mask].flatten(), valid_array[bias_mag_non_nan_mask].flatten())
    res_bias_mag_s = stats.spearmanr(bias_mag_array[bias_mag_non_nan_mask].flatten(), valid_array[bias_mag_non_nan_mask].flatten())
    res_bias_mag_k = stats.kendalltau(bias_mag_array[bias_mag_non_nan_mask].flatten(), valid_array[bias_mag_non_nan_mask].flatten())
    output_lines_to_save.append(f"Pearson's Correlation between Bias Magnitude and VR:\n\tPearson: {res_bias_mag_p.statistic}, p-value: {res_bias_mag_p.pvalue}")
    output_lines_to_save.append(f"Spearman's Correlation between Bias Magnitude and VR:\n\tSpearman: {res_bias_mag_s.statistic}, p-value: {res_bias_mag_s.pvalue}")
    output_lines_to_save.append(f"Kendall's tau between Bias Magnitude and VR:\n\ttau: {res_bias_mag_k.statistic}, p-value: {res_bias_mag_k.pvalue}")

    for output in output_lines_to_save:
        print(output)

    if exp_params.save_data and args.data_prefix is None: # Do not want to resave data!
        now = datetime.now()
        dir = "data/"
        if exp_params.save_data_dir:
            dir = exp_params.save_data_dir + "/"
        filename = exp_params.name + "_" + now.strftime("%Y%m%d_%H%M%S")
        np.save(dir + filename + "_anees.npy", anees_array)
        np.save(dir + filename + "_bias_mag.npy", bias_mag_array)
        np.save(dir + filename + "_valid.npy", valid_array)
        np.save(dir + filename + "_valid_mask.npy", valid_mask_array)

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
    parser = ArgumentParser("twod_range_validity")
    parser.add_argument("--config", default='params/twod_range_validity.yaml')
    parser.add_argument("--data_prefix", default=None)
    main(parser.parse_args())