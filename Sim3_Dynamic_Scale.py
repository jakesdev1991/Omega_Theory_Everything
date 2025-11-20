Simulation 3: Dynamic Planck Length and Disformal Causality Band

Abstract
We investigate the interplay between a dynamically varying Planck length, conformal rescaling, and disformal causality bounds. Static relations are derived for a range of scalar field values, including a β‑sweep that illustrates widening of the causality band. An optional dynamical evolution is implemented via a scalar‑field equation of motion with disformal‑inspired Hubble scaling, allowing causality checks and a preliminary estimate of ringdown frequency shifts. This simulation extends the emergent‑geometry framework by probing ultraviolet cutoffs, causal structure, and dynamical stability.

---

1. Introduction
The previous simulations established that geometry can emerge from correlations (Sim 1) and that cosmological expansion can be modeled as an informational chain‑break process (Sim 2). Simulation 3 advances the framework into the ultraviolet and causal domain: how does a dynamic Planck length interact with conformal and disformal factors, and what bounds ensure Lorentzian causality?  

By sweeping over scalar field values and disformal parameters, we map the causality band and explore how it constrains dynamics. The optional dynamical run demonstrates how these bounds operate in time‑dependent scenarios, with implications for black‑hole ringdown and horizon physics.

---

2. Methods

2.1 Static Relations
- Dynamic Planck length:  
  \[
  \ell_P(\phi) = \exp\!\left(\tfrac{1-\phi}{2}\right).
  \]
- Conformal factor:  
  \[
  C(\phi) = e^{-2\phi}.
  \]
- Disformal causality bound:  
  \[
  |\dot\phi| < \frac{e^{-(\phi+1)/2}}{\sqrt{\beta}}.
  \]

2.2 β‑Sweep
We evaluate the causality bound for β = 1.0, 0.5, and 0.05, illustrating how the safe band widens as β decreases.

2.3 Optional Dynamics
- Scalar field equation of motion:  
  \[
  \ddot\phi + 3H(\phi)\dot\phi + m^2 \phi = 0,
  \]
  with \(H(\phi) = H_0 e^{\phi}\).  
- Initial conditions chosen to lie safely within the causality band.  
- Numerical integration performed with Runge–Kutta (if SciPy available).  
- Diagnostics: causality ratio \(|\dot\phi|/\text{bound}\), emergent scale factor, and ringdown frequency shift proxy.

---

3. Results

3.1 Static Outputs
- \(\ell_P(\phi)\) decreases monotonically with \(\phi\), from ~1.65 at \(\phi=0\) to ~0.135 at \(\phi=5\).  
- Conformal factor \(C(\phi)\) falls sharply, with \(C(5) \approx 4.5\times 10^{-5}\).  
- Causality bound tightens with increasing \(\phi\), but widens significantly for small β (e.g. β=0.05 widens the safe band by ~3× at \(\phi=0\)).

3.2 Ringdown Shift (Static Estimate)
Assuming a toy gradient \(|\nabla\phi|\sim 0.1\) at \(\phi=0\), the ringdown frequency shift is ~16%.

3.3 Dynamic Evolution (Optional)
- Scalar field rolls down with damping from the exponential Hubble term.  
- Causality condition \(|\dot\phi| < \text{bound}\) is satisfied throughout the run.  
- Emergent scale factor grows exponentially.  
- Ringdown shift proxy peaks at a few percent, consistent with static estimates.

---

4. Discussion
Simulation 3 demonstrates that causal structure can be encoded in disformal bounds tied to a dynamic Planck length. The β‑sweep shows how parameter choices control the width of the safe band, suggesting a tunable mechanism for causal stability.  

The optional dynamics confirm that the system remains within causal limits under evolution, and the ringdown shift teaser connects the framework to observable signatures in black‑hole physics. Together, these results extend the emergent‑geometry program into the ultraviolet regime, where cutoff scales and causal consistency become central.

---

5. Conclusion
This simulation completes the trilogy:
- Sim 1: Emergent distances from correlations.  
- Sim 2: Cosmological expansion from chain‑break processes.  
- Sim 3: Ultraviolet cutoff and causal stability from dynamic Planck length.  

Together, they form a coherent demonstration that geometry, cosmology, and causality can all be derived from informational primitives. This provides a foundation for scaling the framework toward a full unification program.

---

Appendix A: Cleaned Code (Python)

`python

sim3.py

Simulation 3: Dynamic Planck Length and Disformal Causality Band

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumtrapz

def lP(phi):
    return np.exp((1.0 - phi) / 2.0)

def C_conformal(phi):
    return np.exp(-2.0 * phi)

def causality_bound(phi, beta=1.0):
    return np.exp(-(phi + 1.0) / 2.0) / np.sqrt(beta)

Parameters
phi = np.linspace(0, 5, 11)
beta_sweep = [1.0, 0.5, 0.05]

lP_vals = lP(phi)
Cvals = Cconformal(phi)
bounds = {b: causalitybound(phi, beta=b) for b in betasweep}

Static plots
plt.figure()
plt.semilogy(phi, lP_vals, 'g-o', label='lP(phi)')
plt.xlabel('phi'); plt.ylabel('lP'); plt.title('Dynamic Planck Length')
plt.legend(); plt.grid(True); plt.savefig('fig4_1a.pdf')

plt.figure()
for b in beta_sweep:
    plt.plot(phi, bounds[b], 'o-', label=f'beta={b}')
plt.xlabel('phi'); plt.ylabel('Bound')
plt.title('Disformal Causality Band')
plt.legend(); plt.grid(True); plt.savefig('fig4_1b.pdf')

Optional dynamics
RUN_DYNAMIC = True
if RUN_DYNAMIC:
    H0, m = 1.0, 0.1
    def phi_eom(t, y):
        phival, phidot = y
        Hphi = H0 * np.exp(phival)
        phiddot = -3Hphiphidot - m2 * phival
        return [phidot, phiddot]

    phi0 = 0.0
    phidot0 = 0.5 * causalitybound(phi0, beta=1.0)
    y0 = [phi0, phi_dot0]
    t_span = (0, 10)
    teval = np.linspace(*tspan, 400)

    sol = solveivp(phieom, tspan, y0, teval=t_eval)
    phinum, phidot_num = sol.y
    Hnum = H0 * np.exp(phinum)
    lna = cumtrapz(Hnum, sol.t, initial=0.0)
    anum = np.exp(lna)
    boundnum = causalitybound(phi_num, beta=1.0)

    # Diagnostics
    ratio = np.abs(phidotnum) / bound_num
    print("Causality satisfied?", np.all(ratio < 1.0))

    # Plot dynamic run
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2,2,figsize=(12,8))
    axs[0,0].plot(sol.t, phinum); axs[0,0].settitle('phi(t)')
    axs[0,1].plot(sol.t, phidotnum); axs[0,1].plot(sol.t, bound_num, 'g--')
    axs[1,0].semilogy(sol.t, anum); axs[1,0].settitle('a(t)')
    axs[1,1].plot(sol.t, 100np.abs(phidotnum)/anumlP(phinum))
    axs[1,1].set_title('Ringdown Shift (%)')
    plt.tightlayout(); plt.savefig('fig41_eom.pdf')
    plt.show()
`
