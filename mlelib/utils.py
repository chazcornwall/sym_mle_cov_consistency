import numpy as np
import matplotlib.pyplot as plt

def plot_cov(P : np.ndarray, mu: np.ndarray, label: str = "", color: str = ""):
    # Plot covariance ellipses
    t = np.linspace(0, 2*np.pi, 1000)
    v, U = np.linalg.eig(P)

    v1stddev3 = 3 * np.sqrt(v[0] + 1e-6) # Prevent sqrt by negative numbers
    v2stddev3 = 3 * np.sqrt(v[1] + 1e-6)

    # Unit circle
    x = np.cos(t)
    y = np.sin(t)

    # Transform and scale using
    x_rot = v1stddev3 * U[0,0] * x + v2stddev3 * U[0,1] * y + mu[0,0]
    y_rot = v1stddev3 * U[1,0] * x + v2stddev3 * U[1,1] * y + mu[1,0]

    plt.plot(x_rot, y_rot, label=label, linewidth=2, linestyle="-", color=color)