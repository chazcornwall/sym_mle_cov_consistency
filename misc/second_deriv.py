import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Plot Settings
# ============================================================

FIG_WIDTH = 14
FIG_HEIGHT = 10
DPI = 120

TITLE_SIZE = 16
LABEL_SIZE = 13
LEGEND_SIZE = 11

XMAX = 6.0
NPTS = 2000

delta = 1.0      # Robust kernel scale

# ============================================================
# Domain
# ============================================================

r = np.linspace(-XMAX, XMAX, NPTS)

# ============================================================
# Second Derivatives
# ============================================================

def quadratic_second(r):
    return np.ones_like(r)


def huber_second(r, delta):
    y = np.zeros_like(r)
    y[np.abs(r) < delta] = 1.0
    return y


def pseudo_huber_second(r, delta):
    return 1.0 / (1 + (r/delta)**2)**1.5


def cauchy_second(r, delta):
    x = r / delta
    return (1 - x**2) / (1 + x**2)**2


def geman_mcclure_second(r, delta):
    x = r / delta
    return 2 * (1 - 3*x**2) / (1 + x**2)**3


def welsch_second(r, delta):
    x = r / delta
    return np.exp(-0.5*x**2) * (1 - x**2) / delta**0


def tukey_second(r, delta):
    y = np.zeros_like(r)

    mask = np.abs(r) < delta
    x = r[mask] / delta

    # rho''(r)
    y[mask] = (1 - x**2) * (1 - 5*x**2)

    return y

# ============================================================
# Compute
# ============================================================

curves = {
    "Quadratic": quadratic_second(r),
    "Huber": huber_second(r, delta),
    "Pseudo-Huber": pseudo_huber_second(r, delta),
    "Cauchy": cauchy_second(r, delta),
    "Geman-McClure": geman_mcclure_second(r, delta),
    "Welsch": welsch_second(r, delta),
    "Tukey": tukey_second(r, delta),
}

# ============================================================
# Plot
# ============================================================

plt.rcParams.update({'font.size': 16})

fig, axs = plt.subplots(
    4,
    2,
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    dpi=DPI,
    sharex=True,
)

axs = axs.flatten()

for ax, (name, y) in zip(axs, curves.items()):

    ax.plot(r, y, linewidth=2)

    ax.axhline(0, color='black', linestyle='--', linewidth=0.75)
    ax.axvline(0, color='gray', linestyle=':', linewidth=0.75)

    ax.set_title(name, fontsize=LABEL_SIZE)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(min(y.min(), -1.2), max(y.max(), 1.2))

# Remove unused subplot
fig.delaxes(axs[-1])

for ax in axs[:-1]:
    ax.set_ylabel(r"$\rho''(r)$", fontsize=LABEL_SIZE)

axs[-2].set_xlabel("Residual", fontsize=LABEL_SIZE)
axs[-3].set_xlabel("Residual", fontsize=LABEL_SIZE)

fig.suptitle(
    "Second Derivatives of Popular Robust Cost Functions",
    fontsize=TITLE_SIZE,
)

plt.tight_layout()
plt.show()