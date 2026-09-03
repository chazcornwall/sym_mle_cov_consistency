from typing import Dict, Tuple
import numpy as np

from mlelib import MaximumLikelihood, MlParams, MlExperimentOutput
from mlelib import calc_rstar

# Import possible models
from models.twod_range import TwoDRangeModel, TwoDRangeModelParams
from models.twod_range_huber import TwoDRangeModelHuber, TwoDRangeModelHuberParams
from models.twod_range_cauchy import TwoDRangeModelCauchy, TwoDRangeModelCauchyParams
from models.twod_bearing import TwoDBearingModel, TwoDBearingModelParams
from models.twod_bearing_huber import TwoDBearingModelHuber, TwoDBearingModelHuberParams
from models.twod_bearing_cauchy import TwoDBearingModelCauchy, TwoDBearingModelCauchyParams


def load_likelihood_model(model_name: str, model_params: Dict):
        # Load in the correct likelihood model depending on parameter files
    if model_name == "twod_range":
        likelihood_model_params = TwoDRangeModelParams(**model_params)
        likelihood_model = TwoDRangeModel(likelihood_model_params)
        likelihood_model_cb = TwoDRangeModel

        meas_stddev = likelihood_model_params.sat_range_stddev
        rstar = calc_rstar(meas_stddev, 0.0, likelihood_model.n, cost="quadratic")
    elif model_name == "twod_range_huber":
        likelihood_model_params = TwoDRangeModelHuberParams(**model_params)
        likelihood_model = TwoDRangeModelHuber(likelihood_model_params)
        likelihood_model_cb = TwoDRangeModelHuber

        # rstar = np.sqrt(nominal_likelihood_model_params.delta/nominal_likelihood_model.n) * np.ones((nominal_likelihood_model.n, 1))
        meas_stddev = likelihood_model_params.sat_range_stddev
        rstar = calc_rstar(meas_stddev, likelihood_model_params.delta, likelihood_model.n, cost="pseudohuber")
    elif model_name == "twod_range_cauchy":
        likelihood_model_params = TwoDRangeModelCauchyParams(**model_params)
        likelihood_model = TwoDRangeModelCauchy(likelihood_model_params)
        likelihood_model_cb = TwoDRangeModelCauchy

        # rstar = np.sqrt(nominal_likelihood_model_params.delta/nominal_likelihood_model.n) * np.ones((nominal_likelihood_model.n, 1))
        meas_stddev = likelihood_model_params.sat_range_stddev
        rstar = calc_rstar(meas_stddev, likelihood_model_params.delta, likelihood_model.n, cost="cauchy")
    elif model_name == "twod_bearing":
        likelihood_model_params = TwoDBearingModelParams(**model_params)
        likelihood_model = TwoDBearingModel(likelihood_model_params)
        likelihood_model_cb = TwoDBearingModel

        meas_stddev = np.deg2rad(likelihood_model_params.sat_bearing_stddev_deg)
        rstar = calc_rstar(meas_stddev, 0.0, likelihood_model.n, cost="quadratic")
    elif model_name == "twod_bearing_huber":
        likelihood_model_params = TwoDBearingModelHuberParams(**model_params)
        likelihood_model = TwoDBearingModelHuber(likelihood_model_params)
        likelihood_model_cb = TwoDBearingModelHuber

        meas_stddev = np.deg2rad(likelihood_model_params.sat_bearing_stddev_deg)
        rstar = calc_rstar(meas_stddev, likelihood_model_params.delta, likelihood_model.n, cost="pseudohuber")
    elif model_name == "twod_bearing_cauchy":
        likelihood_model_params = TwoDBearingModelCauchyParams(**model_params)
        likelihood_model = TwoDBearingModelCauchy(likelihood_model_params)
        likelihood_model_cb = TwoDBearingModelCauchy

        meas_stddev = np.deg2rad(likelihood_model_params.sat_bearing_stddev_deg)
        rstar = calc_rstar(meas_stddev, likelihood_model_params.delta, likelihood_model.n, cost="cauchy")
    else:
        raise ValueError(f"Invalid model name!")

    return likelihood_model_cb, likelihood_model, likelihood_model_params, rstar