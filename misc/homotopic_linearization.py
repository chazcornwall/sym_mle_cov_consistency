import numpy as np
import matplotlib.pyplot as plt
from typing import Callable


class TestHomotopicLinearization:

    def __init__(self, f0: Callable[[float], float], fprime: Callable[[float], float], alpha: float = 0.99):
        self.__alpha = alpha
        self.__zpast = 0.0
        self.__f0 = f0
        self.__f0prime = fprime
        self.__fpast = 0.0
        self.__fpastprime = 0.0
        self.__first_iteration = True

    def scale_homotopy(self, z: float) -> float:
        if self.__first_iteration:
            fcurr = self.__f0(z)
            fcurrprime = self.__f0prime(z)
            self.__first_iteration = False
        else:
            fcurr = self.__alpha * (self.__fpastprime * (z - self.__zpast) + self.__fpast) + \
                (1 - self.__alpha) * self.__f0(z)
            fcurrprime = self.__alpha * self.__fpastprime + (1 - self.__alpha) * self.__f0prime(z)

        # Next iteration
        self.__fpast = fcurr
        self.__fpastprime = fcurrprime

        return fcurrprime
    
    def scale_baseline(self, z: float) -> float:
        return self.__f0prime(z)
    


def main():


    N = 10000

    # Define which function will be applied to the Gaussian
    f0 = lambda x : x**2
    fprime = lambda x : 2*x

    # Generate samples and calculate ground truth covariance estimate
    meas_cov = 1
    samples = 5 + np.sqrt(meas_cov)*np.random.randn(N)
    cov_gt = np.cov(f0(samples))
    print(cov_gt)

    # Create other covariance estimates using linearization and propagation
    thl = TestHomotopicLinearization(f0, fprime, alpha = 0.7)
    cov_homotopy_est = np.zeros(N)
    cov_baseline_est = np.zeros(N)
    for i in range(N):
        cov_homotopy_est[i] = meas_cov * thl.scale_homotopy(samples[i])**2
        cov_baseline_est[i] = meas_cov * thl.scale_baseline(samples[i])**2
        
    print(f"Variance of homotopy: {np.cov(cov_homotopy_est)}")
    print(f"Bias of homotopy: {np.mean(cov_homotopy_est) - cov_gt}")
    print(f"Variance of baseline {np.cov(cov_baseline_est)}")
    print(f"Bias of baseline: {np.mean(cov_baseline_est) - cov_gt}")

    plt.hist(samples)
    plt.show()

if __name__ == "__main__":
    main()