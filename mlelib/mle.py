import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
from pydantic import BaseModel, ConfigDict, Field

from mlelib.utils import plot_cov


class MlParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    x0: np.ndarray = \
        Field(description="Initial guess for the solution")
    residual_generator: Callable[[np.ndarray, np.ndarray], np.ndarray] = \
        Field(description="Returns the residual r(x) = z - h(x)")
    residual_jacobian_generator : Callable[[np.ndarray], np.ndarray] = \
        Field(description="Returns partial derivative of r(x) with respect to x where r(x) = z - h(x)")
    residual_hessian_generator : Callable[[np.ndarray], np.ndarray] = \
        Field(description="Returns Hessian of the residual: a 3D tensor where [i,:,:] is the Hessian of the ith component of the residual")
    cost_function_generator : Callable[[np.ndarray], float] = \
        Field(description="Evaluates the cost function rho(r(x))")
    cost_function_gradient_generator : Callable[[np.ndarray], np.ndarray] = \
        Field(description="Partial derivative of rho(r(x)) with respect to r(x)")
    cost_function_hessian_generator : Callable[[np.ndarray], np.ndarray] = \
        Field(description="Hessian of cost function: d^2 rho(r) / (dr_i dr_j) Should be PSD and resembles the inverse of a covariance matrix")
    max_its: int = 100
    max_ls_its: int = 100
    min_x_diff: float = 1e-8
    alpha0: float = 1.0

class MlSolution(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    it: int
    error: float
    x: np.ndarray
    x_seq : List[np.ndarray] = []
    P: np.ndarray

class MlExperimentOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    anees: np.ndarray

class MaximumLikelihood:

    def __init__(self, params: MlParams):
        self.__params = params

    def mle(self, z: np.ndarray, verbose: bool = False) -> MlSolution:
        '''
        Maximum likelihood estimator of the form:

        $$
        arg max_x ln(f(z|x)) or arg min_x -ln(f(z|x))
        $$

        where x is assumed to not be random (if x were random, we would be doing MAP estimation).
        We also assume f(z|x) is distributed such that

        $$
        f(z|x) propto exp(-rho(z - h(x)))
        $$

        where rho() is a non-negative cost function
        '''

        # Extract functions and parameters for the Gauss-Newton iteration
        x_seq : List[np.ndarray] = []
        x = self.__params.x0
        r = lambda z, x : self.__params.residual_generator(z, x)
        J = lambda x : self.__params.residual_jacobian_generator(x)
        g = lambda res : self.__params.cost_function_gradient_generator(res)
        H = lambda res : self.__params.cost_function_hessian_generator(res)
        Hresx = lambda res : self.__params.residual_hessian_generator(res)
        rho = lambda res : self.__params.cost_function_generator(res)

        x_seq.append(x)
        curr_i = 0
        for i in range(self.__params.max_its):
            
            # Calculate matrices/vectors for Gauss-Newton iteration
            x_prev = np.copy(x)
            res = r(z, x_prev)
            Jxprev = J(x_prev)
            Hres = H(res)
            FI = Jxprev.T @ Hres @ Jxprev # Approximate Fisher Information matrix (approximate due to Gauss-Newton assumptions)
            grad = -Jxprev.T @ g(res) # Calculate gradient using chain rule

            if verbose:
                print(f"Before it {i+1}")
                print(f"\tGradient (flattened): {grad.T}")
                print(f"\tCost: {rho(res)}")

            # Update damping factor using line search
            alpha = self.__params.alpha0
            for j in range(self.__params.max_ls_its):

                # Solve the following equation: J.T @ H @ J @ (x - x_prev) = -J.T @ g(res)
                x = x_prev + alpha * np.linalg.solve(FI, grad)

                # Once we find a good damping factor, keep going!
                if (rho(res) < rho(r(z, x))):
                    alpha *= 0.5
                else:
                    break 

            if j + 1 >= self.__params.max_ls_its:
                print(f"WARNING: Could not find cost-reducing line search!")

            if verbose:
                print(f"After it: {i+1}")
                print(f"\talpha: {alpha}")
                print(f"\tx: {x.T}")
                print(f"\tCost: {rho(r(z, x))}")

            x_seq.append(x)
            if np.linalg.norm(x - x_prev) < self.__params.min_x_diff:
                curr_i = i
                break
        
        if i + 1 >= self.__params.max_its and verbose:
            print(f"WARNING: Did not converge within max iterations!")
        
        # According to the Cramer-Rao Lower Bound, all ML estimators approach a covariance of (FI)^-1

        FI_gn = J(x).T @ H(r(z, x)) @ J(x)
        FI_missing = np.zeros((x.shape[0], x.shape[0]))
        Hresx = Hresx(x)
        for i in range(z.shape[0]):
            FI_missing += Hresx[i,:,:] * g(r(z, x))[i,0]
            print(f"tr(Hessian) {i}:\n {np.trace(Hresx[i,:,:].T @ Hresx[i,:,:])}")
        print(f"Jacobian:\n {J(x)}")
        print(f"Thresh: {3/(0.01*2) * np.linalg.norm(J(x)[0,:])**2 / (np.linalg.norm(x)**2  + np.trace(np.linalg.inv(FI)))}")

        # if np.linalg.inv(FI)[0,0] > 50:
        # print(f"FI_missing: \n{FI_missing}")
        # print(f"FI_gn: \n{FI_gn}")

        # print(f"{min(np.linalg.eigvals(2*H(r(z, x)) - g(r(z, x)) @ g(r(z, x)).T))}")

        # for i in range(x.shape[0]):
        #     for j in range(x.shape[0]):
        #         Proj = np.eye(z.shape[0]) - J(x) @ np.linalg.inv(J(x).T @ J(x)) @ J(x).T
                # print(Proj @ np.array([Hresx[:,i,j]]).T)

        # print(f"cov psd test: {min(np.linalg.eigvals(Jxprev @ Jxprev.T - np.linalg.inv(Hres)))}")

        return MlSolution(it=curr_i + 1, error=rho(res), x=x, x_seq=x_seq, P=np.linalg.inv(FI))
    
    def experiment(self, x_gt: np.ndarray, h: Callable, z_cov: np.ndarray, N: int = 1000, plot : bool = True) -> MlExperimentOutput:
        L = np.linalg.cholesky(z_cov)
        x_est_samples = np.zeros((self.__params.x0.shape[0], N))
        nees = np.zeros(N) # Normalized estimation error squared
        P_est_list : List = []
        P_avg = np.zeros((x_gt.shape[0], x_gt.shape[0]))
        z_mean = h(x_gt)
        for i in range(N):
            z = z_mean + L @ np.random.randn(z_mean.shape[0], 1)
            sol = self.mle(z)
            if i == 0:
                P_avg = sol.P
            else:
                P_avg = 1/(i+1) * sol.P + i/(i+1) * P_avg
            x_est_samples[:, [i]] = sol.x
            nees[i] = ((sol.x - x_gt).T @ np.linalg.inv(sol.P) @ (sol.x - x_gt))[0,0]
            if i % int(N / 10 + 0.5) == 0:
                P_est_list.append(sol.P)

        # Average normalized estimation error squared. Should be number of dimensions in state
        print(f"ANEES: {np.mean(nees)}")
        print(f"E[xhat - xgt]: {np.mean(x_est_samples - x_gt, axis=1)}")
        P_empirical = np.cov(x_est_samples)
        print(f"Var(xhat): \n {P_empirical}")
        
        if plot:
            mu_empirical = np.array([np.mean(x_est_samples, 1)]).T
            plt.scatter(x_est_samples[[0],:], x_est_samples[[1],:], label="Sample Estimates")
            # plt.hist2d(x_est_samples[0,:], x_est_samples[1,:], label="Sample Estimates", bins=50)
            # plt.colorbar(label="Counts")
            plot_cov(P_empirical, mu_empirical, label="Empirical Cov", color="red")
            for i, P_est in enumerate(P_est_list):
                # Metric: P_est - P_empirical should be PSD for a conservative estimator!
                # print(f" min(eigval(P_est - P_emp)): {min(np.linalg.eigvals(P_est - P_empirical))}")
                if i == 0:
                    plot_cov(P_est, mu_empirical, label="Estimated Cov", color="orange")
                else:
                    plot_cov(P_est, mu_empirical, color="orange")
            plot_cov(P_avg, mu_empirical, label="Avg Cov", color="blue")
            plt.legend()
            plt.show()
        return MlExperimentOutput(anees=np.mean(nees))