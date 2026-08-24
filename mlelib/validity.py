import numpy as np
from typing import Tuple

def valid(
        Fbarinv : np.ndarray, 
        sigma2z : float, 
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
    Checks whether a solution should have a valid, linear covariance estimate. 
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
        Nominal covariance matrix (1 measurement with unit covariance)
    sigma2z : float
        Measurement variance
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
        Validity of linear covariance estimate and the validity gap (should be positive for a valid solution)
    """

    meas_thresh : float = None
    cost_thresh : float = None

    if mode != 1:
        J_i_norm = np.linalg.norm(J_i, 2) 
        if abs(J_i_norm) > 1e-8:
            Lambda_i = np.linalg.norm(Jcal_i, 'fro') / np.linalg.norm(J_i, 2)
            meas_thresh = np.trace(Fbarinv) * Lambda_i

    if mode != 0:
        Hcal_i_norm = np.linalg.norm(Hcal_i, 2)
        if abs(Hcal_i_norm) > 1e-8:
            Psi_i = np.linalg.norm(Hbold_i, 'fro') / np.linalg.norm(Hcal_i, 2)
            denom = 1.0 - Psi_i * (np.trace(Pz) - np.linalg.norm(rstar, 2))
            if abs(denom) > 1e-8:
                cost_thresh = np.trace(Fbarinv) * np.linalg.norm(J, 'fro') * Psi_i / denom 

    max_thresh = N / sigma2z
   
    if mode == 0:
        if meas_thresh is None:
            return False, np.nan
        else:
            return meas_thresh <= max_thresh, max_thresh - meas_thresh
    elif mode == 1:
        if cost_thresh is None:
            return False, np.nan
        else:
            return cost_thresh <= max_thresh, max_thresh - cost_thresh
    elif mode == 2:
        if meas_thresh is not None and cost_thresh is not None:
            max_meas_cost_thresh = max(meas_thresh, cost_thresh)
        elif meas_thresh is not None:
            max_meas_cost_thresh = meas_thresh
        elif cost_thresh is not None:
            max_meas_cost_thresh = cost_thresh
        else:
            return False, np.nan
        return max_meas_cost_thresh <= max_thresh, max_thresh - max_meas_cost_thresh
    else:
        raise SyntaxError(f"Invalid mode! Mode must be 0,1,2")

    





