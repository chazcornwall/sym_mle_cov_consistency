import numpy as np
import matplotlib.pyplot as plt
import scipy

# Load data
array_invalid = np.load("500_-500_20261102_091956.npy") # Bad data
array_valid = np.load("250_0_20261102_091959.npy") # Good data

# Print stats to commandline
DoF = 2
rv = scipy.stats.chi2(DoF)
mean, var, skew, kurt = scipy.stats.chi2.stats(DoF, moments='mvsk')
print(f"Predicted Chi2 (DoF = 2) Mean: {mean}")
print(f"Empirical Mean of Valid Data : {np.mean(array_invalid)}")
print(f"Empirical Mean of Invalid Data : {np.mean(array_valid)}")
result_valid = scipy.stats.kstest(array_valid, lambda x : scipy.stats.chi2.cdf(x, DoF))
result_invalid = scipy.stats.kstest(array_invalid, lambda x : scipy.stats.chi2.cdf(x, DoF))
print(f"KS Test Valid Data: {result_valid}")
print(f"KS Test Invalid Data: {result_invalid}")

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

x_valid = np.linspace(0, np.max(array_valid), 1000)
axs[0].hist(array_valid, density=True, bins=40, label="Normalized Frequencies")
axs[0].plot(x_valid, rv.pdf(x_valid), label="Chi2 (dof=2) PDF")
axs[0].set_xlabel("Normalized Estimation Error Squared (NEES)")
axs[0].set_ylabel("Probability Density")
axs[0].set_title("NEES in Clustered, Bearing Positioning at $x_{gt} = (2.5, 0)$")
axs[0].text(10, 0.35, f"KS Test\n Statistic: {result_valid.statistic:0.3f}\n Pvalue: {result_valid.pvalue:0.5f}")
axs[0].text(10, 0.25, f"ANEES: {np.mean(array_valid):0.3f}")
axs[0].legend()

x_invalid = np.linspace(0, np.max(array_invalid), 1000)
axs[1].hist(array_invalid, density=True, bins=40, label="Normalized Frequencies")
axs[1].plot(x_invalid, rv.pdf(x_invalid), label="Chi2 (dof=2) PDF")
axs[1].set_xlabel("Normalized Estimation Error Squared (NEES)")
axs[1].set_ylabel("Probability Density")
axs[1].set_title("NEES in Clustered, Bearing Positioning at $x_{gt} = (5,-5)$")
axs[1].text(20, 0.35, f"KS Test\n Statistic: {result_invalid.statistic:0.3f}\n Pvalue: {result_invalid.pvalue:0.5f}")
axs[1].text(20, 0.25, f"ANEES: {np.mean(array_invalid):0.3f}")
axs[1].legend()
plt.show()