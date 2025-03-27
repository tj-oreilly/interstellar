import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import random
from math import sqrt
import heapq

N_OF_STARS = 200
RADIUS_OF_STARS = 5
SPACE_SIZE = 1000
MIN_STEP = 10
MAX_STEP = 50
N_STOPS = 5  # Number of stops to visit from the stars

stars = pd.DataFrame(columns=["x-coordinate [pc]", "y-coordinate [pc]"])
selected_stops = []  # Global list of selected stop indices
STEP_SIZES = list(range(MIN_STEP, MAX_STEP + 1, 5))  # List of step sizes to simulate

#/////////OBJECTS GENERATORS////////////

def generate_stars():
    global stars
    objects_list = []
    while len(objects_list) < N_OF_STARS:
        x_coord = round(np.random.uniform(1.0, SPACE_SIZE - 1), 1)
        y_coord = round(np.random.uniform(1.0, SPACE_SIZE - 1), 1)
        if all(check_if_objects_do_not_intersect(x_coord, y_coord, RADIUS_OF_STARS, obj[0], obj[1], RADIUS_OF_STARS) for obj in objects_list):
            objects_list.append([x_coord, y_coord])
    stars = pd.DataFrame(objects_list, columns=["x-coordinate [pc]", "y-coordinate [pc]"])

#///////INTERSECTION_AND_SEPARATION_FUNCTIONS/////////

def check_if_objects_do_not_intersect(x1, y1, r1, x2, y2, r2):
    separation = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return separation > r1 + r2

def compute_distances_between_objects(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

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
        distance = compute_distances_between_objects(sx, sy, closest_x, closest_y)
        if distance < RADIUS_OF_STARS:
            return True
    return False

#/////GENERATE_MAIN_STOPS/////////

def generate_stops():
    global selected_stops
    selected_stops = stars.sample(n=N_STOPS, random_state=25).index.tolist()
    return selected_stops

#//////FIND_THE_BEST_ROUTE_ALGORITHM/////////

def stars_separation():
    distances_from_object_i_to_j = {}
    for i, (x1, y1) in stars.iterrows():
        distances_from_object_i_to_j[i] = []
        for j, (x2, y2) in stars.iterrows():
            if i != j:
                if path_intersects_any_star(x1, y1, x2, y2, i):
                    continue
                distance = compute_distances_between_objects(x1, y1, x2, y2)
                travel_time = distance / STEP_SIZE
                distances_from_object_i_to_j[i].append((travel_time, distance, j))
    return distances_from_object_i_to_j

def Dijkstra_Algorithm(distances, start, end):
    priority_queue = [(0, start, [])]
    min_time = {node: float('inf') for node in distances}
    min_time[start] = 0
    best_path = {node: [] for node in distances}
    best_path[start] = [start]
    while priority_queue:
        travel_time, current, path = heapq.heappop(priority_queue)
        if current == end:
            return travel_time, best_path[current]
        for segment_time, _, neighbor in distances[current]:
            new_time = travel_time + segment_time
            if new_time < min_time[neighbor]:
                min_time[neighbor] = new_time
                best_path[neighbor] = best_path[current] + [neighbor]
                heapq.heappush(priority_queue, (new_time, neighbor, best_path[neighbor]))
    print("No path found between points.")
    return float('inf'), []

def find_optimal_path_sequence(stops):
    distances = stars_separation()
    total_travel_time = 0
    full_path = []
    total_distance = 0
    for i in range(len(stops) - 1):
        A = stops[i]
        B = stops[i + 1]
        travel_time, path = Dijkstra_Algorithm(distances, A, B)
        if not path:
            return float('inf'), 0, []
        segment_distance = compute_distances_between_objects(
            stars.loc[A, "x-coordinate [pc]"], stars.loc[A, "y-coordinate [pc]"],
            stars.loc[B, "x-coordinate [pc]"], stars.loc[B, "y-coordinate [pc]"]
        )
        total_distance += segment_distance
        total_travel_time += travel_time
        if i > 0:
            path = path[1:]
        full_path.extend(path)
    return total_travel_time, total_distance, full_path

#/////CREATE_SPACE_FUNCTION/////////

def create_space(step_size):
    global STEP_SIZE
    STEP_SIZE = step_size
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.grid(True, alpha=0.5, linestyle="dashed")
    stops = generate_stops()
    ax.scatter(
        stars["x-coordinate [pc]"],
        stars["y-coordinate [pc]"],
        color="black", label="Stars", marker="o", s=RADIUS_OF_STARS
    )
    for i, stop in enumerate(stops):
        x = stars.loc[stop, "x-coordinate [pc]"]
        y = stars.loc[stop, "y-coordinate [pc]"]
        ax.scatter(x, y, color="orange", s=40, zorder=5)
        ax.text(x + 5, y + 5, str(i + 1), color="orange", fontsize=10, weight="bold")
    travel_time, total_distance, path = find_optimal_path_sequence(stops)
    if travel_time == float('inf'):
        print(f"No valid path found to visit all stops for STEP_SIZE = {STEP_SIZE}.")
    else:
        print(f"STEP_SIZE = {STEP_SIZE}, Travel time: {travel_time:.2f} s")
        with open(f"Logs/Simulation_PF_STEP_SIZE={STEP_SIZE}.txt", "w") as f:
            f.write(f"Number of stops: {len(stops)}\n")
            f.write(f"Order of stops: {stops}\n")
            f.write(f"Start stop: {stops[0]}\n")
            f.write(f"End stop: {stops[-1]}\n")
            f.write(f"Velocity of the ship: {STEP_SIZE} pc/s\n")
            f.write(f"Total path length: {total_distance:.2f} pc\n")
            f.write(f"Total time taken: {travel_time:.2f} s\n")
    if path:
        path_coords = stars.loc[path]
        ax.plot(path_coords["x-coordinate [pc]"], path_coords["y-coordinate [pc]"], color='purple', linestyle='-', linewidth=1, marker='o', markersize=3, label="Rocket Path")
    ax.set_xlabel("x-coordinate [pc]")
    ax.set_ylabel("y-coordinate [pc]")
    ax.legend()
    plt.savefig(f"Diagrams/Simulation_PF_STEP_SIZE={STEP_SIZE}.png")
    plt.close()

#//////MAIN_CODE//////////

generate_stars()
for step in STEP_SIZES:
    create_space(step)
