# Covariance Validity

This repository contains code for running simmulations to verify the validity ratio (VR), a new metric for checking estimator
consistency without Monte Carlo simulations. Simulations have only been verified to execute properly when using Python 3.12.2 on Ubuntu 24.04.

See `requirements.txt` for required packages. To download all required packages, run

```
pip3 install -r requirements.txt
```

There are two entrypoint scripts: `run_validity_experiment.py` and `run_validity_experiment_on_rcfs.py`. Both of these scripts
require a configuration file in the `params` folder. For example,

```
python3 run_validity_experiment.py --config params/twod_bearing_validity.yaml
```

Configuration files for `run_validity_experiment.py` are in the first level of the `params` directory. 
Configuration files for `run_validity_experiment_on_rcfs.py` in the `params/rcf_validity` directory.

## Adding Models

A `LikelihoodModel` requires two classes as input: `ResidualCalculator` and `RobustCostCalculator`. 
Examples of `ResidualCalculator` are in the `residuals` directory and examples of `RobustCostCalculator` are in the `robust_costs` directory.

## Tests

The scripts in the `test` directory verify the maximum likelihood estimator in `mlelib` is working properly. To run the tests, use

```
python3 -m pytest test/<name_of_script>.py -sv
```