import sympy as sp
from sympy import DiracDelta
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Symbols
# ============================================================

r = sp.symbols('r', real=True)
delta = sp.symbols('delta', positive=True)
sigma = sp.symbols('sigma', positive=True)
x = sp.symbols('x', real=True)

# r = x # /sigma

# ============================================================
# Robust Cost Functions
# ============================================================

costs = {}

# 1. Quadratic
costs["Quadratic"] = sp.Rational(1, 2) * (r)**2

# 2. Huber
costs["Huber"] = sp.Piecewise(
    (sp.Rational(1, 2)*r**2, sp.Abs(r) <= delta),
    (delta*(sp.Abs(r) - delta/2), True)
)

# 3. Pseudo-Huber
costs["Pseudo-Huber"] = delta**2 * (
    sp.sqrt(1 + (r/delta)**2) - 1
)

# 4. Cauchy
costs["Cauchy"] = sp.Rational(1,2) * delta**2 * sp.log(
    1 + (r/delta)**2
)

# 5. Geman-McClure
costs["Geman-McClure"] = r**2 / (2*(delta**2 + r**2))

# # 6. Welsch (Leclerc)
# costs["Welsch"] = delta**2 * (
#     1 - sp.exp(-r**2/(2*delta**2))
# )

# 7. Tukey Biweight
x = r/delta

costs["Tukey"] = sp.Piecewise(
    (
        delta**2/6 * (1 - (1 - x**2)**3),
        sp.Abs(r) <= delta
    ),
    (delta**2/6, True)
)

# ============================================================
# Compute Symbolic Derivatives
# ============================================================

third_derivatives = {}

print("\nAnalytical Third Derivatives\n")

for name, rho in costs.items():

    d3 = sp.diff(rho, r, 3)
    d3 = sp.simplify(d3)

    third_derivatives[name] = d3

    print("="*60)
    print(name)
    print("="*60)
    sp.pprint(d3)
    print()

# ============================================================
# Plot
# ============================================================

plt.rcParams.update({'font.size': 16})

delta_value = np.ones(1500)

r_vals = np.linspace(-5, 5, 1500)

fig, axs = plt.subplots(
    2,
    3,
    figsize=(20,5),
    sharex=True
)

axs = axs.flatten()

for ax, (name, expr) in zip(axs, third_derivatives.items()):

    f = sp.lambdify((r, delta), expr, modules=["sympy"])

    y = np.zeros(1500)
    for i, res in enumerate(r_vals):
        y[i] = f(res, 1)

    ax.plot(r_vals, y, lw=2)

    if name == "Huber":

        # Dirac delta at x = 0
        ax.annotate(
            "",
            xy=(-1, 1),
            xytext=(-1, 0),
            arrowprops=dict(arrowstyle='->', lw=2, color="tab:blue")
        )
        ax.annotate(
            "",
            xy=(1, -1),
            xytext=(1, 0),
            arrowprops=dict(arrowstyle='->', lw=2, color="tab:blue")
        )
        # ax.text(0.05, 0.95, r'$\delta(x)$', fontsize=14)

        ax.set_xlim(-2,2)
        ax.set_ylim(-1.2,1.2)

    ax.set_title(name)

    ax.grid(True)

    ax.axhline(0, color='k', ls='--', lw=0.75)
    ax.axvline(0, color='gray', ls=':')

# fig.delaxes(axs[-1])

plt.tight_layout()

plt.show()