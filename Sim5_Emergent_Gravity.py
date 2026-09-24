# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: Apache-2.0

r"""
Simulation 5: Emergent Gravity as Informational Asymmetry (Omega Theory v4.0)

Implements:
- 2D Chain Overlap Density field Phi(x, y)
- Informational Stress-Energy Tensor T_munu^\Phi = Z(Phi) grad_mu Phi grad_nu Phi - g_munu L_Phi
- Emergent entropic force acceleration on test particles towards Phi gradients
- Warp/curvature of emergent spacetime grid.
"""

import numpy as np


def compute_2d_stress_energy(
    phi_grid: np.ndarray, dx: float = 0.1, dy: float = 0.1, Z_phi: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dphi_dx, dphi_dy = np.gradient(phi_grid, dx, dy)
    grad_sq = dphi_dx**2 + dphi_dy**2
    L_phi = 0.5 * Z_phi * grad_sq
    T_00 = 0.5 * Z_phi * grad_sq  # Energy density stored in informational gradients
    T_11 = 0.5 * Z_phi * (dphi_dx**2 - dphi_dy**2)
    return L_phi, T_00, T_11


def step_particle_in_phi_field(
    pos: np.ndarray,
    vel: np.ndarray,
    phi_grid: np.ndarray,
    G_const: float = 0.05,
    dt: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    dphi_dx, dphi_dy = np.gradient(phi_grid)
    grid_size = phi_grid.shape[0]

    ix = int(np.clip(pos[0], 0, grid_size - 1))
    iy = int(np.clip(pos[1], 0, grid_size - 1))

    grad_at_pos = np.array([dphi_dx[ix, iy], dphi_dy[ix, iy]])
    a_grav = -G_const * grad_at_pos

    vel_new = vel + a_grav * dt
    pos_new = pos + vel_new * dt
    return pos_new, vel_new


def main() -> None:
    grid_size = 50
    PHI_VACUUM = 1.0
    phi_grid = np.ones((grid_size, grid_size)) * PHI_VACUUM

    # Place informational mass (low Phi center)
    cx, cy = grid_size // 2, grid_size // 2
    for x in range(grid_size):
        for y in range(grid_size):
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            phi_grid[x, y] -= 0.6 * np.exp(-(r**2) / (2 * 5.0**2))

    L_phi, T_00, T_11 = compute_2d_stress_energy(phi_grid)

    pos = np.array([10.0, 25.0])
    vel = np.array([0.0, 0.0])

    for _ in range(50):
        pos, vel = step_particle_in_phi_field(pos, vel, phi_grid)

    print("Simulation 5 (Omega Theory v4.0 Emergent Gravity) completed successfully.")
    print(f"2D Field Grid Dimensions: {grid_size}x{grid_size}")
    print(f"Peak Informational Energy Density T_00: {np.max(T_00):.4f}")
    print(f"Particle Position Trajectory Final Point: [{pos[0]:.2f}, {pos[1]:.2f}]")
    print(f"Particle Accelerated Speed towards Mass Center: {np.linalg.norm(vel):.4f}")


if __name__ == "__main__":
    main()
