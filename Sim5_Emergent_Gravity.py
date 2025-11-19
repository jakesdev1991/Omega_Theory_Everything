import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Use a dark style for the "Computational Universe" aesthetic
plt.style.use('dark_background')

# === VISUAL POLISH ===
plt.rcParams['animation.html'] = 'html5'
plt.rcParams['axes.facecolor'] = '#0a0a0a'
plt.rcParams['figure.facecolor'] = '#0a0a0a'

# ==========================================
# 1. OMEGA PROTOCOL: THE UNIVERSE CONFIG
# ==========================================

# Simulation Resolution
N_REGIONS = 200         
DT = 0.1                
TOTAL_FRAMES = 300      

# Omega Theory Constants
PHI_VACUUM = 1.0        
PHI_CRITICAL = 0.1      
L_PLANCK_BASE = 1.0     
KAPPA = 5.0             
G = 0.05                

# Initialize the scalar field phi
phi = np.ones(N_REGIONS) * PHI_VACUUM

# ==========================================
# 2. CREATE MATTER AND BLACK HOLE
# ==========================================
particle_width = 12
phi_particle_val = 0.4

# Particles: Position 50 (Left), Position 150 (Right)
positions = [float(N_REGIONS // 4), float(3 * N_REGIONS // 4)]
velocities = [0.0, 0.0]

# Black Hole
bh_pos = N_REGIONS - 20
bh_width = 5
phi_bh_val = 0.1
bh_start = bh_pos - bh_width
bh_end = bh_pos + bh_width

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def get_emergent_geometry(phi_field):
    local_l_p = L_PLANCK_BASE * np.exp((PHI_VACUUM - phi_field) / PHI_CRITICAL)
    physical_x = np.cumsum(local_l_p)
    physical_x -= physical_x[0]
    return physical_x

def get_mass_total(phi_field):
    mass_density = np.maximum(0, 1.0 - phi_field)
    return np.sum(mass_density)

def is_shredded(start, end):
    return start < bh_end and end > bh_start

# ==========================================
# 4. MAIN LOOP
# ==========================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=False)
fig.suptitle('Ω PROTOCOL v3.6: EMERGENT GRAVITY', color='#00ffff', fontsize=18, fontweight='bold')

vel_history = []
time_history = []
shred_events = []

def update(frame):
    global phi, positions, velocities, shred_events

    # --- A. RESET FIELD ---
    phi[:] = PHI_VACUUM
    phi[max(0, bh_start):min(N_REGIONS, bh_end)] = phi_bh_val

    # --- B. PHYSICS UPDATE ---
    active_particles = []
    current_shred = False
    
    for i in range(len(positions)):
        current_int_pos = int(positions[i])
        start = current_int_pos - particle_width
        end = current_int_pos + particle_width
        if start < 0: start = 0
        if end >= N_REGIONS: end = N_REGIONS - 1

        # Check Horizon Crossing
        if is_shredded(start, end):
            shred_events.append(frame)
            current_shred = True
            continue 

        # Apply Matter Density
        if start < end:
            phi[start:end] = phi_particle_val
            active_particles.append(i)

    # Clean up lists
    new_positions = []
    new_velocities = []
    for i in active_particles:
        new_positions.append(positions[i])
        new_velocities.append(velocities[i])
    positions = new_positions
    velocities = new_velocities

    # Calc Gradients
    grad_phi = np.gradient(phi)

    # Apply Forces
    num_particles = len(positions)
    if num_particles > 0:
        total_mass = get_mass_total(phi)
        if total_mass < 0.001: total_mass = 0.001
        current_mass_per_particle = total_mass / num_particles
        applied_force = 0.0

        for i in range(num_particles):
            pos = positions[i]
            grad_at_pos = np.interp(pos, np.arange(N_REGIONS), grad_phi)
            
            # Gravity (Gradient) + Inertia (Kappa)
            a_grav = -G * grad_at_pos
            a_ext = applied_force / (current_mass_per_particle + KAPPA)
            
            velocities[i] += (a_grav + a_ext) * DT
            positions[i] += velocities[i] * DT

    # --- C. UPDATE GEOMETRY ---
    physical_x = get_emergent_geometry(phi)

    # --- D. VISUALIZATION ---
    ax1.clear()
    
    # Dynamic Title
    ax1.set_title(f"Frame {frame} │ Particles: {len(positions)} │ Shredded: {len(shred_events)}", 
                  color='#00ffff', fontsize=12, pad=10)
    
    ax1.set_ylabel("Information Density (phi)")
    ax1.set_xlabel("Emergent Distance (Spacetime)")

    # Draw Field
    ax1.plot(physical_x, phi, color='#00ff00', lw=2, label='Φ Field')
    ax1.fill_between(physical_x, phi, 1.0, color='cyan', alpha=0.2, label='Mass (Non-Overlap)')
    
    # Draw Grid Points (Visualizing Expansion)
    ax1.scatter(physical_x[::5], np.ones_like(physical_x[::5])*0.5, color='white', s=5, alpha=0.4)
    
    # Draw Horizon
    if bh_start >= 0 and bh_end < len(physical_x):
        ax1.axvspan(physical_x[int(bh_start)], physical_x[int(bh_end)], color='red', alpha=0.3, label='Event Horizon')

    # Draw Particles (Yellow Orbs)
    for pos in positions:
        # Map logical grid position to physical warped space
        x_phys = np.interp(pos, np.arange(N_REGIONS), physical_x)
        ax1.scatter(x_phys, 0.4, color='yellow', s=150, marker='o', edgecolor='orange', linewidth=2, zorder=10)

    # FLASH "SHREDDED" TEXT
    if current_shred:
         ax1.text(0.5, 0.5, "SHREDDED", transform=ax1.transAxes, color='red', fontsize=30, 
             ha='center', va='center', alpha=1.0, fontweight='bold', zorder=20)

    ax1.set_ylim(0.0, 1.1)
    ax1.legend(loc='upper right', facecolor='#000000')

    # Velocity Graph
    avg_vel = np.mean(velocities) if velocities else 0.0
    vel_history.append(avg_vel)
    time_history.append(frame)

    ax2.clear()
    ax2.set_ylabel("Avg Velocity")
    ax2.set_xlabel("Time Step")
    ax2.plot(time_history, vel_history, color='#ffff00', lw=2, label='Inertial Velocity')
    ax2.legend(loc='upper left', facecolor='#000000')
    ax2.grid(True, color='gray', linestyle='--', alpha=0.2)

ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=30)
plt.tight_layout()
plt.show()
