# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: Apache-2.0

"""
Simulation 5: Emergent Gravity as Informational Gradient (Omega Theory v4.0)

Implements the Lagrangian stress-energy derivation T_munu = Z(Phi) grad_mu Phi grad_nu Phi - g_munu L_Phi
where mass is energy density stored in gradients of the Chain Overlap Density field Phi.
"""

import numpy as np


def get_emergent_geometry(
    phi_field: np.ndarray,
    lP0: float = 1.0,
    phi_vacuum: float = 1.0,
    phi_c: float = 0.1,
) -> np.ndarray:
    local_l_p = lP0 * np.exp((phi_vacuum - phi_field) / phi_c)
    physical_x = np.cumsum(local_l_p)
    physical_x -= physical_x[0]
    return physical_x


def compute_stress_energy_tensor_1d(
    phi: np.ndarray, dx: float = 0.1, Z_phi: float = 1.0, V_phi: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dphi_dx = np.gradient(phi, dx)
    L_phi = 0.5 * Z_phi * (dphi_dx**2) - V_phi
    T_00 = 0.5 * Z_phi * (dphi_dx**2) + V_phi
    T_11 = 0.5 * Z_phi * (dphi_dx**2) - V_phi
    return L_phi, T_00, T_11


def main() -> None:
    N_REGIONS = 100
    PHI_VACUUM = 1.0

    phi = np.ones(N_REGIONS) * PHI_VACUUM
    phi[40:60] = 0.4  # Informational defect (matter gradient)

    physical_x = get_emergent_geometry(phi)
    L_phi, T_00, T_11 = compute_stress_energy_tensor_1d(phi)

    print("Simulation 5 completed successfully.")
    print(f"Total emergent domain length: {physical_x[-1]:.4f}")
    print(f"Lagrangian density min/max: {np.min(L_phi):.4f}/{np.max(L_phi):.4f}")
    print(f"Peak energy density T_00: {np.max(T_00):.4f}")


if __name__ == "__main__":
    main()
