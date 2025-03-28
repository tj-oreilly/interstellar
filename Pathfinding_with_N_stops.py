import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import heapq
from math import sqrt
from itertools import permutations

# Constants
SPEED_OF_LIGHT = 9.715e-9  # pc/s
SECONDS_IN_YEAR = 31_557_600
N_OF_STARS = 200
RADIUS_OF_STARS = 1e-4  # pc
SPACE_SIZE = 10  # pc
MIN_STEP = 5e-9
MAX_STEP = 5e-9
N_STOPS = 5
V_STAR = 7e-12  # pc/s
PROBE_MASS = 478  # kg

# Globals
stars = pd.DataFrame(columns=["x-coordinate [pc]", "y-coordinate [pc]"])
selected_stops = []
STEP_SIZES = np.arange(MIN_STEP, MAX_STEP + 1e-12, 5e-11)


#/////GENERATORS//////////GENERATORS//////////GENERATORS//////////GENERATORS//////////GENERATORS//////////GENERATORS//////////GENERATORS/////
def generate_stars():
    global stars
    objects_list = []
    while len(objects_list) < N_OF_STARS:
        x = round(np.random.uniform(1.0, SPACE_SIZE - 1), 1)
        y = round(np.random.uniform(1.0, SPACE_SIZE - 1), 1)
        if all(check_if_objects_do_not_intersect(
            x, y, RADIUS_OF_STARS, obj[0], obj[1], RADIUS_OF_STARS
        ) for obj in objects_list):
            objects_list.append([x, y])
    stars = pd.DataFrame(objects_list, columns=["x-coordinate [pc]", "y-coordinate [pc]"])
    
def generate_stops():
    global selected_stops
    selected_stops = stars.sample(n=N_STOPS, random_state=25).index.tolist()
    return selected_stops



#/////COMPUTATIONS_AND_CONDITIONS/////#/////COMPUTATIONS_AND_CONDITIONS/////#/////COMPUTATIONS_AND_CONDITIONS/////#/////COMPUTATIONS_AND_CONDITIONS/////

def check_if_objects_do_not_intersect(x1, y1, r1, x2, y2, r2):
    return sqrt((x2 - x1)**2 + (y2 - y1)**2) > (r1 + r2)


def compute_distances_between_objects(x1, y1, x2, y2):
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)


def path_intersects_any_star(x1, y1, x2, y2, star_index):
    for i, (sx, sy) in stars.iterrows():
        if i == star_index or (sx == x1 and sy == y1) or (sx == x2 and sy == y2):
            continue
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        t = ((sx - x1) * dx + (sy - y1) * dy) / (dx**2 + dy**2)
        t = max(0, min(1, t))
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        if compute_distances_between_objects(sx, sy, closest_x, closest_y) < RADIUS_OF_STARS:
            return True
    return False


#//////PATHFINDING_PART///////////PATHFINDING_PART///////////PATHFINDING_PART///////////PATHFINDING_PART///////////PATHFINDING_PART/////


def stars_separation():
    distances = {}
    for i, (x1, y1) in stars.iterrows():
        distances[i] = []
        for j, (x2, y2) in stars.iterrows():
            if i != j:
                if path_intersects_any_star(x1, y1, x2, y2, i):
                    continue
                dist = compute_distances_between_objects(x1, y1, x2, y2)
                travel_time = dist / STEP_SIZE
                distances[i].append((travel_time, dist, j))
    return distances


def Dijkstra_Algorithm(distances, start, end):
    queue = [(0, start, [])]
    min_time = {node: float('inf') for node in distances}
    min_time[start] = 0
    best_path = {node: [] for node in distances}
    best_path[start] = [start]

    while queue:
        time, current, path = heapq.heappop(queue)
        if current == end:
            return time, best_path[current]
        for segment_time, _, neighbor in distances[current]:
            new_time = time + segment_time
            if new_time < min_time[neighbor]:
                min_time[neighbor] = new_time
                best_path[neighbor] = best_path[current] + [neighbor]
                heapq.heappush(queue, (new_time, neighbor, best_path[neighbor]))

    print("No path found between points.")
    return float('inf'), []


def find_best_stop_order(selected_stops):
    distances = stars_separation()
    best_time = float('inf')
    best_order = []
    best_result = None

    for perm in permutations(selected_stops):
        total_time = 0
        total_dist = 0
        full_path = []
        seg_times = []
        seg_velocities = []
        velocity = STEP_SIZE
        valid = True

        for i in range(len(perm) - 1):
            A = perm[i]
            B = perm[i + 1]
            t, path = Dijkstra_Algorithm(distances, A, B)
            if not path:
                valid = False
                break
            x1, y1 = stars.loc[A]
            x2, y2 = stars.loc[B]
            dist = compute_distances_between_objects(x1, y1, x2, y2)
            total_time += t
            total_dist += dist
            if i > 0:
                path = path[1:]
            full_path.extend(path)
            seg_times.append(t)
            seg_velocities.append(velocity)
            if i < len(perm) - 2:
                velocity = relativistic_velocity_addition(velocity, 2 * V_STAR)

        if valid and total_time < best_time:
            best_time = total_time
            best_order = perm
            best_result = (total_time, total_dist, full_path, seg_times, seg_velocities)

    return best_order, *best_result


#////////PHYSICS///////////////PHYSICS///////////////PHYSICS///////////////PHYSICS///////////////PHYSICS///////////////PHYSICS///////

# Time Dilation
def time_dilation_segments(segment_times, segment_velocities):
    proper_time = 0
    for t, v in zip(segment_times, segment_velocities):
        if v >= SPEED_OF_LIGHT:
            raise ValueError("Velocity must be less than the speed of light.")
        gamma = 1 / sqrt(1 - (v**2) / SPEED_OF_LIGHT**2)
        proper_time += t / gamma
    return proper_time


# Energy Function
def relativistic_kinetic_energy(mass, velocity, c=SPEED_OF_LIGHT):
    if velocity >= c:
        raise ValueError("Velocity must be less than the speed of light.")
    gamma = 1 / sqrt(1 - (velocity**2) / c**2)
    return (gamma - 1) * mass * c**2

# Relativistic velocity addition
def relativistic_velocity_addition(u, v, c=SPEED_OF_LIGHT):
    return (u + v) / (1 + (u * v) / (c**2))


#///////SIMULATION_AND_PLOTTING//////////////SIMULATION_AND_PLOTTING//////////////SIMULATION_AND_PLOTTING//////////////SIMULATION_AND_PLOTTING///////
def create_space(step_size):
    global STEP_SIZE
    STEP_SIZE = step_size
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.grid(True, alpha=0.5, linestyle="dashed")

    stops = generate_stops()

    ax.scatter(
        stars["x-coordinate [pc]"],
        stars["y-coordinate [pc]"],
        color="gray", edgecolor="black", label="Stars",
        marker="o", s=10, alpha=0.6, zorder=2
    )

    offset = SPACE_SIZE * 0.02
    for i, stop in enumerate(stops):
        x = stars.loc[stop, "x-coordinate [pc]"]
        y = stars.loc[stop, "y-coordinate [pc]"]
        ax.scatter(x, y, color="orange", s=40, zorder=5)
        ax.text(x + offset, y + offset, str(i + 1),
                color="orange", fontsize=10, weight="bold")

    best_order, travel_time, total_dist, path, seg_times, seg_velocities = \
        find_best_stop_order(stops)

    if travel_time == float('inf'):
        print(f"No valid path found for STEP_SIZE = {STEP_SIZE}.")
        return

    dilated_time = time_dilation_segments(seg_times, seg_velocities)
    delta_time = travel_time - dilated_time

    observer_years = travel_time / SECONDS_IN_YEAR
    probe_years = dilated_time / SECONDS_IN_YEAR
    diff_years = delta_time / SECONDS_IN_YEAR
    final_velocity = seg_velocities[-1]
    energy_gain = relativistic_kinetic_energy(PROBE_MASS, final_velocity)

    print(
        f"STEP_SIZE = {STEP_SIZE}, Travel time: {travel_time:.2f} s "
        f"({observer_years:.2f} yrs, observer), {dilated_time:.2f} s "
        f"({probe_years:.2f} yrs, probe)"
    )

    with open(f"Logs/Simulation_PF_STEP_SIZE={STEP_SIZE}.txt", "w") as f:
        f.write(f"Number of stops: {len(stops)}\n")
        f.write(f"Order of stops: {best_order}\n")
        f.write(f"Start stop: {best_order[0]}\n")
        f.write(f"End stop: {best_order[-1]}\n")
        f.write(f"Velocity of the ship (initial): {STEP_SIZE / SPEED_OF_LIGHT:.2e}c\n")
        f.write(f"Final velocity (after assists): {final_velocity / SPEED_OF_LIGHT:.2e}c\n")
        f.write(f"Total path length: {total_dist:.2f} pc\n")
        f.write(f"Observer frame time: {travel_time:.2f} s ({observer_years:.2f} years)\n")
        f.write(f"Probe frame time: {dilated_time:.2f} s ({probe_years:.2f} years)\n")
        f.write(f"Time dilation: {delta_time:.2f} s ({diff_years:.2f} years)\n")
        f.write(f"Energy gain: {energy_gain:.3e} J\n")

    if path:
        coords = stars.loc[path]
        ax.plot(
            coords["x-coordinate [pc]"],
            coords["y-coordinate [pc]"],
            color='purple', linestyle='-', linewidth=1,
            marker='o', markersize=3, label="Rocket Path"
        )

    ax.set_xlabel("x-coordinate [pc]")
    ax.set_ylabel("y-coordinate [pc]")
    ax.legend()
    ax.set_xlim(0, SPACE_SIZE)
    ax.set_ylim(0, SPACE_SIZE)
    plt.savefig(f"Diagrams/Simulation_PF_STEP_SIZE={STEP_SIZE}.png")
    plt.close()


#/////////MAIN_CODE//////////////MAIN_CODE//////////////MAIN_CODE//////////////MAIN_CODE//////////////MAIN_CODE//////////////MAIN_CODE/////
generate_stars()
for step in STEP_SIZES:
    create_space(step)
