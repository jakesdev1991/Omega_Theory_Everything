# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: MIT

r"""Simulation 2: phenomenological information-depletion ODE

Status: a toy differential equation, not a covariant cosmological model.

The executable code assumes

    dI/dt = -gamma * I,        I = 1/a,        a = r/r0.

These definitions imply dr/dt = gamma*r and hence a(t)=exp(gamma*t). The
script numerically integrates that ODE, estimates H=(da/dt)/a, and computes
a kinematic effective-w proxy. It does not derive the depletion law, the
inverse relation I=1/a, or a Friedmann equation from Q-regions or general
relativity. It has no black-hole-area input and makes no comparison with
observational data. The constant gamma is selected as a toy parameter; it is
not fitted to or predicted from an independent dataset.

Because the ODE is exponential by construction, the limiting analytic result
is constant H=gamma and w_eff=-1 under the displayed proxy definition. Small
numerical excursions in the plotted w_eff arise from finite differences.

Run with `MPLBACKEND=Agg python Sim2_Cosmology.py` for a headless execution.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def safe_log(x, eps=1e-12):
    return np.log(np.clip(x, eps, None))


# ---------------------------
# 1. Setup Parameters
# ---------------------------
t_max = 10.0
gamma_decay = 0.35  # The intrinsic decay constant (1/Gyr)
r0 = 1.0  # Initial separation (arbitrary units)


# ---------------------------
# 2. The Differential Equation
# ---------------------------
def system(t, y):
    r = y[0]
    # Assumed depletion ODE dI/dt = -gamma * I.
    # Under the model choice I = 1/a and a = r/r0, this implies dr/dt = gamma*r.
    dr_dt = gamma_decay * r
    return [dr_dt]


# ---------------------------
# 3. Run Simulation
# ---------------------------
t_eval = np.linspace(0, t_max, 100)
sol = solve_ivp(system, [0, t_max], [r0], t_eval=t_eval)

t = sol.t
r = sol.y[0]

# ---------------------------
# 4. Compute Cosmological Variables
# ---------------------------
# Scale factor a(t)
a_info = r / r0

# Hubble Rate H(t) = (da/dt) / a
# Since a = e^(gamma*t), H should be constant = gamma
H_info = np.gradient(a_info, t) / a_info

# Equation of State w_eff
# w = -1 - (2/3) * (H_dot / H^2)
# For pure exponential growth, H_dot = 0, so w = -1.
H_smooth = H_info  # No noise in analytic sim
dH_dt = np.gradient(H_smooth, t)
w_eff = -1.0 - (2.0 / 3.0) * (dH_dt / (H_smooth**2 + 1e-9))

# ---------------------------
# 5. Visualization
# ---------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 7))

# Panel 1: Separation / Scale Factor
axes[0, 0].plot(t, a_info, "r-", lw=2, label=f"a(t) ~ e^({gamma_decay}t)")
axes[0, 0].set_title("Scale Factor (De Sitter Growth)")
axes[0, 0].set_ylabel("a(t)")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Panel 2: Information Content
# I = 1/a
I_content = 1.0 / a_info
axes[0, 1].plot(t, I_content, "b-", lw=2)
axes[0, 1].set_title("Global Information I(t)")
axes[0, 1].set_ylabel("I (Normalized)")
axes[0, 1].set_yscale("log")
axes[0, 1].grid(alpha=0.3)

# Panel 3: Hubble Rate
axes[1, 0].plot(t, H_info, "k-", lw=2)
axes[1, 0].axhline(gamma_decay, color="g", ls="--", label=f"H = {gamma_decay}")
axes[1, 0].set_title("Hubble Parameter H(t)")
axes[1, 0].set_ylim(0, gamma_decay * 1.5)
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Panel 4: Equation of State
axes[1, 1].plot(t, w_eff, "m-", lw=2)
axes[1, 1].axhline(-1.0, color="k", ls="--", label="w = -1 (Dark Energy)")
axes[1, 1].set_title("Effective Equation of State")
axes[1, 1].set_ylim(-1.5, -0.5)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("sim2_v3_depletion_results.png")
plt.show()
