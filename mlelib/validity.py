import numpy as np
from typing import Tuple
from mlelib.model import LikelihoodModel


def calc_rstar(sigmar: float, delta: float, n: int, cost: str = "quadratic") -> np.array:
    """
    Calculates r^* or the linearization point for cost function validity.
    rstar is selected such that rstar's magnitude occurs at an inflection point of
    the cost function's second derivative.

    Params
    ------
    sigmar : float
        Weight parameter for the residual
    delta : float
        Parameter of robust cost function such that delta -> infty implies a quadratic cost
    n : int
        Number of elements in a measurement vector
    cost : str
        Name of cost function. Options are: "quadratic", "pseudohuber", "cauchy"
    """
    if cost == "quadratic":
        rstar_magnitude = 0.0
    elif cost == "pseudohuber":
        rstar_magnitude = sigmar * delta / 2
    elif cost == "cauchy":
        rstar_magnitude = sigmar * delta * (np.sqrt(2) - 1)
    else:
        raise ValueError(f"cost function {cost} is not available!")

    return np.sqrt(rstar_magnitude / n) * np.ones((n,1))


def valid(
        Fbarinv : np.ndarray, 
        N : int, 
        mode: int = 2, 
        J_i : np.ndarray = np.zeros(1), 
        Jcal_i : np.ndarray = np.zeros(1), 
        Hcal_i : np.ndarray = np.zeros(1), 
        Hbold_i : np.ndarray = np.zeros(1), 
        J : np.ndarray = np.zeros(1),  
        Pz : np.ndarray = np.zeros(1), 
        rstar: np.ndarray = np.zeros(1)
    ) -> Tuple[bool, float]:
    """
    Checks whether a single measurement should generate a valid, linear covariance estimate. 
    This function expects the residual of the form r = h(x) - z where r_i is the
    ith element of the residual.

    We assume J and Jcal are evaluated at the optimal state x*.
    We assume Hcal and Hbold are evaluated at the average residual r*.

    Params
    ------
    J_i : np.ndarray
        Residual gradient (dr_i/dx)
    Jcal_i : np.ndarray
        Residual hessian (d^2r_i/dxdx)
    Hcal_i : np.ndarray
        Cost function derivative gradient (d^2rho/dr_idr)
    Hbold_i : np.ndarray
        Cost function derivative hessian (d^3rho/dr_idrdr)
    J : np.ndarray
        Residual Jacobian (dr/rx)
    Fbarinv : np.ndarray
        Optimal covariance (excluding N)
    N : int
        Number of independent measurements (remember, 1 measurement must ensure the solution is defined)
    Pz : np.ndarray
        Measurement covariance matrix
    rstar : np.ndarray
        Average residual
    mode : int, {0,1,2}
        0 for measurement model validation, 1 for cost function validation, 2 for both

    Returns
    -------
    bool, float
        Validity of linear covariance estimate and the validity ratio (should be greater than 1 for a valid solution)
    """

    meas_thresh : float = None
    cost_thresh : float = None

    if mode != 1:
        J_i_norm = np.linalg.norm(J_i, 2) 
        if abs(J_i_norm) > 1e-8:
            Lambda_i = np.linalg.norm(Jcal_i, 'fro')**2 / (J_i.T @ Fbarinv @ J_i)
            meas_thresh = 1 / N * np.trace(Fbarinv)**2 * Lambda_i
        # If J_i_norm is zero, then Jcal_i will be zero, satisfying sufficient condition for validity!

    if mode != 0:
        Hcal_i_norm = np.linalg.norm(Hcal_i, 2)
        if abs(Hcal_i_norm) > 1e-8:
            Pr = 1 / N * J @ Fbarinv @ J.T + Pz
            print(f"Hbold_i: {Hbold_i}")
            print(f"Hcal_i: {Hcal_i}")
            Psi_i = np.linalg.norm(Hbold_i, 'fro')**2 / (Hcal_i.T @ Pr @ Hcal_i) 
            cost_thresh = np.trace(Pr) * (np.trace(Pr) + np.linalg.norm(rstar, 2)**2) * Psi_i
        # If Hcal_i_norm is zero, then Hbold_i will be zero, satisfying sufficient condition for validity!

    max_thresh = 1 

    print(f"Max thresh: {max_thresh}")
    print(f"Meas thresh: {meas_thresh}")
    print(f"Cost thresh: {cost_thresh}")
   
    if mode == 0:
        if meas_thresh is None:
            return True, np.nan
        else:
            return meas_thresh <= max_thresh, meas_thresh
    elif mode == 1:
        if cost_thresh is None:
            return True, np.nan
        else:
            return cost_thresh <= max_thresh, cost_thresh
    elif mode == 2:
        if meas_thresh is not None and cost_thresh is not None:
            max_meas_cost_thresh = max(meas_thresh, cost_thresh)
        elif meas_thresh is not None:
            max_meas_cost_thresh = meas_thresh
        elif cost_thresh is not None:
            max_meas_cost_thresh = cost_thresh
        else:
            return True, np.nan
        return max_meas_cost_thresh <= max_thresh, max_meas_cost_thresh
    else:
        raise SyntaxError(f"Invalid mode! Mode must be 0,1,2")


def validity(likelihood_model: LikelihoodModel, n: int, N: int, Pz: np.ndarray, x_gt: np.ndarray, rstar: np.ndarray, wstar: np.ndarray, mode: int = 2) -> Tuple[bool, float]:
    """
    Checks whether a solution should have a valid, linear covariance estimate. 

    Params
    ------
    likelihood_model : mlelib.model.LikelihoodModel
    n : int
        Number of elements in a single measurement vector
    N : int
        Number of independent measurements (remember, 1 measurement must ensure the solution is defined)
    Pz : np.ndarray
        Measurement covariance matrix
    x_gt : np.ndarray
        Ground truth state
    rstar : np.ndarray
        Symmetric-aware residual
    wstar : np.ndarray
        Symmetric-aware average residual when we have access to ground truth
    mode : int, {0,1,2}
        0 for measurement model validation, 1 for cost function validation, 2 for both

    Returns
    -------
    bool, float
        Validity of linear covariance estimate and the validity ratio (should be greater than 1 for a valid solution)
    """
    J = likelihood_model.residual_jacobian_generator(x_gt)
    Jcal = likelihood_model.residual_hessian_generator(x_gt)
    H0 = likelihood_model.cost_hessian_generator(wstar)
    Hbold = likelihood_model.cost_deriv_hessian_generator(rstar)
    Hcal = likelihood_model.cost_hessian_generator(rstar)
    Fbarinv = np.linalg.inv(J.T @ H0 @ J)
    Nindmeas = n / x_gt.shape[0] * N

    # Calculate validity using gt and likelihood
    validity_ratio = 0
    validity_decision = True
    for k in range(likelihood_model.n):
        curr_decision, curr_validity_ratio = valid(Fbarinv, Nindmeas, mode=mode, J_i=J[k,:], J=J, Jcal_i=Jcal[k,:,:], Hbold_i=Hbold[k,:,:], Hcal_i=Hcal[k,:], Pz=Pz, rstar=rstar)
        if curr_validity_ratio > validity_ratio:
            validity_ratio = curr_validity_ratio
        validity_decision = curr_decision and validity_decision
    return validity_decision, validity_ratio
    
    





