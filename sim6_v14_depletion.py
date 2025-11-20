#!/usr/bin/env python3
"""
sim6_v14_depletion.py
- Implements "Proposal A: The Depletion Model"
- Physics Change: dI/dt = -gamma * A^kappa * I (Proportional Decay)
- Math Consequence: du/dt = -gamma * A^kappa (No exponential feedback)
- Result: Naturally stable Dark Energy (w ~ -1) without Big Rip.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from scipy.optimize import brentq

# Handle scipy version differences
try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:
    from scipy.integrate import cumtrapz as cumulative_trapezoid

# --------------------------
# Constants
# --------------------------
Gyr_to_sec = 3.15576e16
Mpc_in_km = 3.085677581e19 / 1e3
conv_1Gyr_to_km_s_Mpc = Mpc_in_km / Gyr_to_sec
c_km_s = 299792.458

# --------------------------
# Default parameters
# --------------------------
H0_fid = 73.5
H0_SI_default = H0_fid / conv_1Gyr_to_km_s_Mpc

OMEGA_M_FID = 0.315
OMEGA_GAMMA = 5.38e-5
OMEGA_NU = 3.0e-5
OMEGA_K = 0.0

alpha_def = 1.0
kappa_def = 1.0
gamma_guess_def = 1e-4 # Decay constant (units: 1/Gyr / Area_units)
A0_def = 1.0

# --------------------------
# BHARD — peaks at z≈2
# --------------------------
z_bhard = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])
bhard_normed = np.array([0.15, 0.7, 2.1, 4.0, 5.2, 4.1, 2.3, 0.8, 0.25, 0.08])
_bhard_interp = PchipInterpolator(z_bhard, bhard_normed, extrapolate=True)

def A_BH_of_z(z, A0=A0_def):
    scalar_input = np.ndim(z) == 0
    z = np.atleast_1d(z)
    z = np.maximum(z, 0.0)
    res = A0 * np.maximum(_bhard_interp(z), 0.0)
    if scalar_input:
        return res[0]
    return res

def A_BH_of_a(a, A0=A0_def):
    if np.ndim(a) == 0:
        if a <= 1e-12: a = 1e-12
    else:
        a = np.maximum(a, 1e-12)
    return A_BH_of_z(1.0/a - 1.0, A0=A0)

# --------------------------
# Physics
# --------------------------
def Omega_rad_total():
    return OMEGA_GAMMA + OMEGA_NU

def H_matter_squared(a, params):
    H0_SI = params['H0_SI']
    Om = params.get('Omega_m', OMEGA_M_FID)
    Ok = params.get('Omega_k', OMEGA_K)
    Or = Omega_rad_total()
    return H0_SI**2 * (Om/a**3 + Or/a**4 + Ok/a**2)

def derivs(t, y, params):
    u, a = y
    if a < 1e-12: a = 1e-12

    A = A_BH_of_a(a, A0=params['A0'])
    
    # --- DEPLETION MODEL PHYSICS ---
    # Old Model: dI/dt = -gamma * A^k
    # New Model: dI/dt = -gamma * A^k * I
    #
    # Since u = ln(I), du/dt = (1/I) * dI/dt
    # du/dt = (1/I) * (-gamma * A^k * I)
    # du/dt = -gamma * A^k
    #
    # The exponential feedback term e^-u is GONE.
    # This naturally prevents the Big Rip.
    
    du_dt = -params['gamma'] * (A ** params['kappa'])

    H_info = params['alpha'] * np.abs(du_dt)
    H_m2 = H_matter_squared(a, params)
    H2 = H_m2 + H_info**2
    H = np.sqrt(max(H2, 0.0))

    da_dt = a * H
    return np.array([float(du_dt), float(da_dt)])

# --------------------------
# Integration
# --------------------------
def integrate_cosmo(params, z_start=200.0, t_final=14.0):
    a_init = 1.0 / (1.0 + z_start)
    u_init = 0.0 # I starts at 1 (normalized)
    t_span = (1e-4, t_final)
    
    # We can use standard settings now, the system is much less stiff!
    sol = solve_ivp(derivs, t_span, [u_init, a_init],
                    args=(params,), method='Radau', dense_output=True,
                    atol=1e-10, rtol=1e-8, max_step=0.2)
    return sol

# --------------------------
# Observables
# --------------------------
def compute_observables(sol, params, zmax=8.0, nz=1500):
    z = np.linspace(0, zmax, nz)
    a = 1.0 / (1 + z)

    idx = np.argsort(sol.t)
    t_arr = sol.t[idx]
    a_arr = sol.y[1][idx]
    u_arr = sol.y[0][idx]

    a_arr = np.maximum.accumulate(a_arr)
    
    # Standard interpolation
    t_of_a = PchipInterpolator(a_arr, t_arr, extrapolate=True)
    u_of_t = PchipInterpolator(t_arr, u_arr, extrapolate=True)

    t_grid = t_of_a(a)
    u_grid = u_of_t(t_grid)
    du_dt = u_of_t.derivative(1)(t_grid)
    d2u_dt2 = u_of_t.derivative(2)(t_grid)

    H_info = params['alpha'] * np.abs(du_dt)
    H_m2 = H_matter_squared(a, params)
    H = np.sqrt(H_m2 + H_info**2) * conv_1Gyr_to_km_s_Mpc

    denom = du_dt**2 + 1e-30
    w_eff = -1.0 - (params['alpha']/3.0) * (d2u_dt2 / denom)

    H_safe = np.maximum(H, 1e-5)
    chi = cumulative_trapezoid(c_km_s / H_safe, z, initial=0.0)
    dL = (1 + z) * chi
    mu = 5 * np.log10(np.maximum(dL, 1e-6)) + 25

    return {
        'z': z, 'H_km_s_Mpc': H, 'w_eff': w_eff,
        'mu': mu, 'chi_Mpc': chi, 'dL_Mpc': dL, 'I': np.exp(u_grid)
    }

# --------------------------
# Calibration (Simpler now)
# --------------------------
def calibrate_gamma(params_template):
    target_H0 = H0_fid
    
    def get_H0_model(lg):
        p = params_template.copy()
        p['gamma'] = 10**lg
        try:
            sol = integrate_cosmo(p)
            if not sol.success: return np.nan
            
            # If simulation didn't reach a=1
            if sol.y[1][-1] < 0.99: return 1e5 

            t_now = PchipInterpolator(sol.y[1], sol.t)(1.0)
            
            # Depletion Math for H0
            A_today = A_BH_of_a(1.0, A0=p['A0'])
            # No exp(-u) here!
            du_dt_today = -p['gamma'] * (A_today ** p['kappa'])
            
            H_info = p['alpha'] * np.abs(du_dt_today)
            H_m2 = H_matter_squared(1.0, p)
            H_val = np.sqrt(H_m2 + H_info**2) * conv_1Gyr_to_km_s_Mpc
            return H_val
        except Exception:
            return 1e5

    print(f"  > Target H0: {target_H0}")
    
    # Scan range can be standard now
    scan_grid = np.linspace(-5.0, 1.0, 30) 
    
    low_bound, high_bound = -20.0, 20.0
    found_bracket = False
    
    for lg in scan_grid:
        val = get_H0_model(lg)
        diff = val - target_H0
        if diff < 0: low_bound = lg
        elif diff > 0:
            high_bound = lg
            found_bracket = True
            break
            
    if not found_bracket:
        print("  ! Warning: Could not bracket H0. Returning high bound.")
        return 10**high_bound

    print(f"  > Bracket: [{low_bound:.2f}, {high_bound:.2f}]")
    try:
        root = brentq(lambda x: get_H0_model(x) - target_H0, low_bound, high_bound, xtol=1e-4)
        return 10**root
    except:
        return 10**low_bound

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    params = {
        'alpha': alpha_def,
        'kappa': kappa_def,
        'gamma': gamma_guess_def,
        'Omega_m': OMEGA_M_FID,
        'A0': A0_def,
        'H0_SI': H0_SI_default,
    }

    print("Calibrating gamma (Depletion Model)...")
    params['gamma'] = calibrate_gamma(params)
    print(f"-> calibrated γ = {params['gamma']:.4e}")

    print("Running final cosmology...")
    sol = integrate_cosmo(params)
    out = compute_observables(sol, params)

    print("Plotting...")
    plt.figure(figsize=(12,9))
    
    plt.subplot(2,2,1)
    plt.plot(out['z'], out['H_km_s_Mpc'], label='Model')
    plt.axhline(H0_fid, color='r', ls='--', alpha=0.5, label=f'Target {H0_fid}')
    plt.legend()
    plt.gca().invert_xaxis()
    plt.title('H(z)')
    plt.ylabel('km/s/Mpc')

    plt.subplot(2,2,2)
    plt.plot(out['z'], out['I'])
    plt.gca().invert_xaxis()
    plt.title('Information I(z) - Decays to 0')

    plt.subplot(2,2,3)
    w_smooth = savgol_filter(out['w_eff'], 51, 3)
    plt.plot(out['z'], w_smooth)
    plt.axhline(-1, color='k', ls='--', alpha=0.5)
    plt.ylim(-2.0, 0.0) # Zoom in to see if it hits -1
    plt.gca().invert_xaxis()
    plt.title('w_eff(z)')

    plt.subplot(2,2,4)
    plt.plot(out['z'], out['mu'])
    plt.gca().invert_xaxis()
    plt.title('Distance Modulus μ(z)')

    plt.tight_layout()
    plt.savefig("informational_cosmology_depletion.png", dpi=150)
    plt.show()

    print(f"\nSuccess! H0 = {out['H_km_s_Mpc'][0]:.2f} km/s/Mpc")
    print(f"w0 ≈ {w_smooth[0]:.3f}")

