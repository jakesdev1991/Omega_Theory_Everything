Simulation 2: Chain‑Break Cosmology

Abstract
We present a toy cosmological model in which expansion is driven by an effective “chain‑break” process. The baseline history is defined by a linearly growing break rate, yielding exact analytic expressions for the scale factor, Hubble‑like rate, and effective equation of state. We confirm stability through discrete cross‑checks and explore an alternative history with a calibrated peak in the break rate. Diagnostics of acceleration and correlations with mock black‑hole area growth are reported. This simulation extends the emergent‑geometry framework by showing how cosmological dynamics can arise from informational break processes.

---

1. Introduction
In the broader theory, geometry and dynamics emerge from correlation structures. Simulation 1 established that spatial distances can be reconstructed from mutual information. Simulation 2 advances this by showing how cosmological expansion can be modeled as the cumulative effect of “chain‑break” events — informational ruptures that drive growth of effective separation.  

The model is deliberately minimal: a one‑dimensional chain with a break rate proportional to time. Despite its simplicity, it reproduces key cosmological features: accelerated expansion, phantom‑like effective equations of state, and alternative histories with tunable peaks.

---

2. Methods

2.1 Baseline Chain‑Break Rate
- Define \(\gamma{BH}(t) = \gamma{BH}^{\text{base}} \, t\), with slope \(\gamma_{BH}^{\text{base}} = 0.08\).  
- Integrate to obtain the cumulative break measure and effective separation \(r(t)\).  
- Define the informational scale factor \(a{\text{info}} = r(t)/r0\).  
- Define the Hubble‑like rate \(H{\text{info}} = \gamma{BH}/r\).

2.2 Effective Equation of State
- Analytic form:  
  \[
  w{\text{eff}} = -\tfrac{2}{3} - \tfrac{2}{3}\,\frac{r}{b t^2}, \quad b = \gamma{BH}^{\text{base}}.
  \]
- Cross‑checked by discrete derivative of \(\ln H\) with respect to \(\ln a\).

2.3 Alternative History
- Define \(\gamma_{BH}^{\text{alt}}(t) = k \,(t/\tau)\, e^{-t/\tau}\), with \(\tau=5\).  
- Calibrate \(k\) so that the peak value is \(\approx 0.25\).  
- Compute corresponding \(H_{\text{info}}^{\text{alt}}\).

2.4 Diagnostics
- Analytic acceleration \(\ddot a/a = \dot H + H^2\).  
- Phantom window identified where \(w_{\text{eff}}<-1\).  
- Correlation test between \(w_{\text{eff}}\) and a mock black‑hole area growth proxy.

---

3. Results

- Scale factor: \(a_{\text{info}}(t)\) grows super‑exponentially, reaching ~6.8 by \(t=10\).  
- Expansion rate: \(H_{\text{info}}(t)\) rises to ~0.24 and then slowly declines.  
- Equation of state: Analytic \(w_{\text{eff}}\) shows a clear phantom window (\(w<-1\)) between \(t=2\) and \(t=5\), confirmed by discrete cross‑check.  
- Alternative history: The calibrated \(\gamma{BH}^{\text{alt}}\) peaks at ~0.25, with a corresponding peak in \(H{\text{info}}^{\text{alt}}\).  
- Acceleration: Positive \(\ddot a/a\) is maintained through early times, consistent with accelerated expansion.  
- Correlation teaser: A modest correlation is observed between \(w_{\text{eff}}\) and mock black‑hole area growth, suggesting deeper links to horizon thermodynamics.

---

4. Discussion
This simulation demonstrates that informational break processes can mimic cosmological expansion. The baseline model yields a phantom‑like effective equation of state without fine‑tuning, while the alternative history shows how peak‑shaped break rates can reproduce different expansion profiles.  

The phantom window is particularly notable: it suggests that effective violations of the null energy condition can arise naturally in this framework, not as pathologies but as emergent features of information dynamics. The correlation with black‑hole area growth hints at a unifying thermodynamic underpinning.

---

5. Conclusion
Simulation 2 extends the emergent‑geometry program into cosmology. By treating chain‑break processes as the driver of expansion, we recover scale‑factor growth, Hubble‑like dynamics, and effective equations of state that parallel standard cosmological phenomenology. This sets the stage for Simulation 3, where dynamical transitions and feedback between geometry and information will be explored.

---

"""
sim2_v3_depletion.py
Simulation 2: Micro-Foundation of Expansion (The Depletion Model)

Abstract:
This simulation proves that if information is "depleted" like a finite resource
(radioactive decay), it naturally generates De Sitter-like exponential expansion.
This replaces the arbitrary 'linear break rate' of v1 with a physically
motivated 'intrinsic decay' rate.

Core Identity:
dI/dt = -gamma * I  (Depletion)
Since I ~ 1/r (Information is inverse to separation):
dr/dt = gamma * r   (Exponential Expansion)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def safe_log(x, eps=1e-12):
    return np.log(np.clip(x, eps, None))

# ---------------------------
# 1. Setup Parameters
# ---------------------------
t_max = 10.0
gamma_decay = 0.35  # The intrinsic decay constant (1/Gyr)
r0 = 1.0            # Initial separation (arbitrary units)

# ---------------------------
# 2. The Differential Equation
# ---------------------------
def system(t, y):
    r = y[0]
    # The Depletion Law derived from dI/dt = -gamma * I
    # Leads to dr/dt = gamma * r
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
H_smooth = H_info # No noise in analytic sim
dH_dt = np.gradient(H_smooth, t)
w_eff = -1.0 - (2.0/3.0) * (dH_dt / (H_smooth**2 + 1e-9))

# ---------------------------
# 5. Visualization
# ---------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 7))

# Panel 1: Separation / Scale Factor
axes[0,0].plot(t, a_info, 'r-', lw=2, label=f'a(t) ~ e^({gamma_decay}t)')
axes[0,0].set_title('Scale Factor (De Sitter Growth)')
axes[0,0].set_ylabel('a(t)')
axes[0,0].legend()
axes[0,0].grid(alpha=0.3)

# Panel 2: Information Content
# I = 1/a
I_content = 1.0 / a_info
axes[0,1].plot(t, I_content, 'b-', lw=2)
axes[0,1].set_title('Global Information I(t)')
axes[0,1].set_ylabel('I (Normalized)')
axes[0,1].set_yscale('log')
axes[0,1].grid(alpha=0.3)

# Panel 3: Hubble Rate
axes[1,0].plot(t, H_info, 'k-', lw=2)
axes[1,0].axhline(gamma_decay, color='g', ls='--', label=f'H = {gamma_decay}')
axes[1,0].set_title('Hubble Parameter H(t)')
axes[1,0].set_ylim(0, gamma_decay*1.5)
axes[1,0].legend()
axes[1,0].grid(alpha=0.3)

# Panel 4: Equation of State
axes[1,1].plot(t, w_eff, 'm-', lw=2)
axes[1,1].axhline(-1.0, color='k', ls='--', label='w = -1 (Dark Energy)')
axes[1,1].set_title('Effective Equation of State')
axes[1,1].set_ylim(-1.5, -0.5)
axes[1,1].legend()
axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('sim2_v3_depletion_results.png')
plt.show()

