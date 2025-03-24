import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

#//////CONSTANTS_AND_SETTINGS//////////CONSTANTS_AND_SETTINGS//////////CONSTANTS_AND_SETTINGS//////////CONSTANTS_AND_SETTINGS////

G = 6.67430e-11  # Gravitational constant [m^3 kg^-1 s^-2]
object_mass = 1.898e27 # Mass of the gravity source [kg]

# Object properties
object_position = np.array([0.0, 0.0])  # Stationary at the origin in its own frame
object_velocity = np.array([13000.0, 0.0])  # Velocity in Galaxy frame [m/s]

# Mission waypoints (fixed in Galaxy frame)
point_A = np.array([-2e8, 3e8])  # Launch point
point_B = np.array([1e8, -1e8])  # Target point

file_name = "Simulation_1_mass_1.898e27" # File, where results are stored

# ======================
#////////GRAVITY_AND_EVENTS///////////GRAVITY_AND_EVENTS///////////GRAVITY_AND_EVENTS///////////GRAVITY_AND_EVENTS///////////
# ======================

def gravity(t, state):
    """
    Computes gravitational acceleration due to a central mass at the origin.
    """
    x, y, vx, vy = state
    r = np.hypot(x, y)
    acc = -G * object_mass / r**3
    return [vx, vy, acc * x, acc * y]


def reach_point_B(t, state):
    """
    Event function to stop integration when within 1000 km of point B.
    """
    x, y = state[0], state[1]
    dist = np.hypot(x - point_B[0], y - point_B[1])
    return dist - 1e4  # Threshold = 1000 km


reach_point_B.terminal = True
reach_point_B.direction = 0  # Detect approach from any direction

#////////OPTIMIZATION_FUNCTION///////////OPTIMIZATION_FUNCTION///////////OPTIMIZATION_FUNCTION///////////OPTIMIZATION_FUNCTION///

def closest_approach_to_B(velocity_galaxy_frame):
    """
    Objective function to minimize the closest distance to point B.
    """
    # Convert velocity to Object frame
    velocity_object_frame = velocity_galaxy_frame - object_velocity

    # Initial state [x, y, vx, vy] in Object frame
    state0 = [
        point_A[0], point_A[1],
        velocity_object_frame[0], velocity_object_frame[1]
    ]

    # Integrate trajectory
    sol = solve_ivp(
        gravity, [0, 5e5], state0,
        method='DOP853', rtol=1e-10, atol=1e-10,
        events=reach_point_B, dense_output=True
    )

    # If reached point B, return 0 distance (ideal)
    if sol.status == 1:
        return 0.0

    # Otherwise, compute minimum distance to point B
    trajectory = sol.y[:2].T
    min_dist = np.min(np.linalg.norm(trajectory - point_B, axis=1))
    return min_dist


#/////DIFFERENTIAL_EVOLUTION_OPTIMIZATION///////////DIFFERENTIAL_EVOLUTION_OPTIMIZATION///////////DIFFERENTIAL_EVOLUTION_OPTIMIZATION//////

# Define bounds for velocity vector [vx, vy]
bounds = [(5000.0, 50000.0), (-50000.0, 5000.0)]

# Run global optimizer
result = differential_evolution(
    closest_approach_to_B, bounds,
    strategy='best1bin', tol=1e-4,
    maxiter=100, disp=True
)

# Extract optimal velocity
v_opt_galaxy_frame = result.x
v_opt_object_frame = v_opt_galaxy_frame - object_velocity

#///////SIMULATE_OPTIMAL_TRAJECTORY_USING_OPTIMAL_VELOCITY///////////SIMULATE_OPTIMAL_TRAJECTORY_USING_OPTIMAL_VELOCITY///////////

# Re-run trajectory with optimized velocity
state0 = [
    point_A[0], point_A[1],
    v_opt_object_frame[0], v_opt_object_frame[1]
]

sol = solve_ivp(
    gravity, [0, 5e5], state0,
    method='DOP853', rtol=1e-10, atol=1e-10,
    events=reach_point_B, dense_output=True
)

# Evaluate solution for plotting
t = np.linspace(sol.t[0], sol.t[-1], 5000)
y = sol.sol(t)
trajectory = y[:2]
velocities = y[2:]

#/////COMPUTE_PERIAPSIS////////COMPUTE_PERIAPSIS////////COMPUTE_PERIAPSIS////////COMPUTE_PERIAPSIS////////COMPUTE_PERIAPSIS////////COMPUTE_PERIAPSIS////

# Compute periapsis (closest approach to the object)
distances = np.linalg.norm(trajectory.T, axis=1)
periapsis_idx = np.argmin(distances)
periapsis = trajectory[:, periapsis_idx]

# Final velocity (object and galaxy frame)
v_final_object_frame = velocities[:, -1]
v_final_galaxy_frame = v_final_object_frame + object_velocity

#//////VISUALISATION///////////VISUALISATION///////////VISUALISATION///////////VISUALISATION///////////VISUALISATION///////////VISUALISATION/////

fig, ax = plt.subplots(figsize=(14, 12))

# Plot trajectory
ax.plot(trajectory[0] / 1e6, trajectory[1] / 1e6, 'b-', label='Optimized Trajectory')
ax.plot(object_position[0], object_position[1], 'o', color='orange', label='Gravity Body', markersize=12)
ax.plot(point_A[0] / 1e6, point_A[1] / 1e6, 'go', label='Launch Point (A)', markersize=10)
ax.plot(point_B[0] / 1e6, point_B[1] / 1e6, 'ro', label='Target Point (B)', markersize=10)
ax.plot(periapsis[0] / 1e6, periapsis[1] / 1e6, 'mo', label='Periapsis', markersize=8)

# Plot velocity vector of gravity body
scale = 2e4 / 1e7
ax.quiver(
    object_position[0] / 1e6, object_position[1] / 1e6,
    object_velocity[0] * scale, object_velocity[1] * scale,
    color='black', angles='xy', scale_units='xy', scale=1,
    label='Object Velocity'
)

# Labels and styling
ax.set_xlabel('X Position (10⁶ m)')
ax.set_ylabel('Y Position (10⁶ m)')
ax.set_title('Gravity Assist Trajectory (Hyperbolic Path)')
ax.grid(True)
ax.legend()
ax.axis('equal')
plt.tight_layout()
plt.savefig(file_name + ".png")
plt.close()

#/////////SAVE_RESULTS///////////////PRINT_RESULTS///////////////PRINT_RESULTS///////////////PRINT_RESULTS///////////////PRINT_RESULTS//////

# Save results to a text file
with open(file_name + ".txt", "w") as f:
    f.write("=== Gravity Assist Simulation Results ===\n")
    f.write(f"Initial velocity (Galaxy frame): {v_opt_galaxy_frame} m/s\n")
    f.write(f"Initial speed: {np.linalg.norm(v_opt_galaxy_frame):.2f} m/s\n\n")

    f.write(f"Final velocity (Galaxy frame): {v_final_galaxy_frame} m/s\n")
    f.write(f"Final speed: {np.linalg.norm(v_final_galaxy_frame):.2f} m/s\n\n")

    f.write(f"Velocity gain: {np.linalg.norm(v_final_galaxy_frame) - np.linalg.norm(v_opt_galaxy_frame):.2f} m/s\n")
    f.write(f"Periapsis position: {periapsis} m\n\n")

    if sol.status == 1:
        f.write("Target reached (within 1000 km of point B).\n")
    else:
        f.write("Target NOT reached within the time limit.\n")

