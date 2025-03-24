import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import random
from math import sqrt, atan2, sin, cos
import heapq

N_OF_TRIVIAL_OBJECTS = 200
N_OF_NONTRIVIAL_OBJECTS = 10
RADII_OF_TRIVIAL_OBJECTS = 5
RADII_OF_NONTRIVIAL_OBJECTS = 1
SPACE_SIZE = 1000
STEP_SIZE = 30 #Velocity of the ship in AU/s
FUEL = 40
FUEL_BURNING_RATE = 0.1 #per 10 AU
G = 1  # Gravitational constant
c = 1  # Speed of light


trivial_objects = pd.DataFrame(columns=["x-coordinate [AU]", "y-coordinate [AU]"])
nontrivial_objects = pd.DataFrame(columns=["x-coordinate [AU]", "y-coordinate [AU]", "mass"])

#/////////OBJECTS GENERATORS////////////OBJECTS GENERATORS////////////OBJECTS GENERATORS////////////OBJECTS GENERATORS////////////

def generate_trivial_objects():
    """Generate trivial objects ensuring they do not overlap."""
    global trivial_objects
    objects_list = []

    while len(objects_list) < N_OF_TRIVIAL_OBJECTS:
        x_coord = round(np.random.uniform(1.0, SPACE_SIZE - 1), 1)
        y_coord = round(np.random.uniform(1.0, SPACE_SIZE - 1), 1)

        if all(
            check_if_objects_do_not_intersect(
                x_coord, y_coord, RADII_OF_TRIVIAL_OBJECTS, obj[0], obj[1], RADII_OF_TRIVIAL_OBJECTS
            ) for obj in objects_list
        ):
            objects_list.append([x_coord, y_coord])

    trivial_objects = pd.DataFrame(objects_list, columns=["x-coordinate [AU]", "y-coordinate [AU]"])

def generate_nontrivial_objects():
    """Generate nontrivial objects ensuring they do not overlap with trivial or other nontrivial objects."""
    global nontrivial_objects
    objects_list = []

    while len(objects_list) < N_OF_NONTRIVIAL_OBJECTS:
        x_coord = round(np.random.uniform(1.0, SPACE_SIZE - 1), 1)
        y_coord = round(np.random.uniform(1.0, SPACE_SIZE - 1), 1)
        mass = np.random.randint(60, 1000) / 1000
        r1 = mass * 1000  # Gravity-based exclusion zone

        # Ensure no intersection with trivial objects
        if not all(
            check_if_objects_do_not_intersect(x_coord, y_coord, r1, row[0], row[1], RADII_OF_TRIVIAL_OBJECTS)
            for row in trivial_objects.to_numpy()
        ):
            continue  # Skip if it intersects with a trivial object

        # Ensure no intersection with other nontrivial objects
        if not all(
            check_if_objects_do_not_intersect(x_coord, y_coord, r1, obj[0], obj[1], obj[2] * 1000)
            for obj in objects_list
        ):
            continue  # Skip if it intersects
        
        objects_list.append([x_coord, y_coord, mass])

    nontrivial_objects = pd.DataFrame(objects_list, columns=["x-coordinate [AU]", "y-coordinate [AU]", "mass"])

#///////INTERSECTION_AND_SEPARATION_FUNCTIONS/////////INTERSECTION_AND_SEPARATION_FUNCTIONS/////////INTERSECTION_AND_SEPARATION_FUNCTIONS//

def check_if_objects_do_not_intersect(x1, y1, r1, x2, y2, r2):
    """Returns True if objects do not intersect, otherwise False."""
    separation = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return separation > r1 + r2

def compute_distances_between_objects(x1, y1, x2, y2):
    separation = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return separation

#//////PROPERTIES_OF_NONTRIVIAL_OBJECTS_FUNCTIONS/////////PROPERTIES_OF_NONTRIVIAL_OBJECTS_FUNCTIONS/////////PROPERTIES_OF_NONTRIVIAL_OBJECTS_FUNCTIONS///

def radii_of_gravity(layer_of_objects):
    """Draw gravity circles around nontrivial objects."""
    for obj in nontrivial_objects.to_numpy():
        x, y, mass = obj
        radius = mass * 1000

        # Create multiple layers for a gradient effect
        for i in range(10, 0, -1):
            alpha = 1 - i / 10  # Transparency effect
            circle = patches.Circle(
                (x, y), radius * (i / 10), edgecolor='blue', facecolor='none', linestyle='-', linewidth=1, alpha=alpha
            )
            
            layer_of_objects.add_patch(circle)

#/////GENERATE_MAIN_STOPS/////////GENERATE_MAIN_STOPS/////////GENERATE_MAIN_STOPS/////////GENERATE_MAIN_STOPS/////////GENERATE_MAIN_STOPS////

def generate_A_and_B():
    """Generate two distinct points A and B from trivial objects."""
    A = trivial_objects.iloc[[np.random.randint(0, len(trivial_objects))]]
    random.seed(25)
    B = trivial_objects.iloc[[np.random.randint(0, len(trivial_objects))]]
    
    while A.index[0] == B.index[0]:
        B = trivial_objects.iloc[[np.random.randint(0, len(trivial_objects))]]
    
    return A, B

#//////PHYSICS//////////PHYSICS//////////PHYSICS//////////PHYSICS//////////PHYSICS//////////PHYSICS//////////PHYSICS//////////PHYSICS////

def gravitational_time_dilation(mass, distance):
    """Calculate time dilation factor due to gravity."""
    if distance == 0:
        return float('inf')  # Prevent division by zero
    
    M = round(np.random.uniform(0.1,1),3)
    R = distance
    event_horizon = (2 * G * M) / (c**2)
    
    if R < event_horizon:
        return float('inf')  # Escape condition: ship cannot move inside event horizon
    
    factor = round(np.random.uniform(0.1,1),4) #sqrt(1 - event_horizon / R)
    return factor  # Time dilation factor

#//////FIND_THE_BEST_ROUTE_ALGORITHM//////////FIND_THE_BEST_ROUTE_ALGORITHM//////////FIND_THE_BEST_ROUTE_ALGORITHM//////////FIND_THE_BEST_ROUTE_ALGORITHM////

def trivial_objects_separation(A_index, B_index):
    
    distances_from_object_i_to_j = {}

    for i, (x1, y1) in trivial_objects.iterrows():
        distances_from_object_i_to_j[i] = []
        for j, (x2, y2) in trivial_objects.iterrows():
            if i != j and not (i == A_index and j == B_index) and not (i == B_index and j == A_index):  #Add this conditions here if you want to forbid straight path A->B: 
                        #  and not (i == A_index and j == B_index) and not (i == B_index and j == A_index):
                distance = compute_distances_between_objects(x1, y1, x2, y2)
                travel_time = distance / STEP_SIZE  # Time required in steps of STEP_SIZE
                distances_from_object_i_to_j[i].append((travel_time, distance, j))  # FIX: Includes distance

    return distances_from_object_i_to_j



def Dijkstra_Algorithm(distances, start, end, nontrivial_objects):
    """Find the most fuel and time-efficient path from A to B using Dijkstra's algorithm with time dilation."""
    priority_queue = [(0, 0, FUEL, start, [])]  # (observer time, proper time, remaining fuel, node, path)
    min_time = {node: float('inf') for node in distances}
    min_time[start] = 0
    best_path = {node: [] for node in distances}
    best_path[start] = [start]
    
    while priority_queue:
        obs_time, prop_time, remaining_fuel, current, path = heapq.heappop(priority_queue)
        
        if current == end:
            return obs_time, prop_time, remaining_fuel, best_path[current]
        
        for travel_time, distance, neighbor in distances[current]:
            fuel_needed = distance * FUEL_BURNING_RATE
            
            if remaining_fuel >= fuel_needed:
                new_obs_time = obs_time + travel_time
                new_prop_time = prop_time

                # Get coordinates of start and end points
                current_x, current_y = trivial_objects.loc[current, ["x-coordinate [AU]", "y-coordinate [AU]"]]
                neighbor_x, neighbor_y = trivial_objects.loc[neighbor, ["x-coordinate [AU]", "y-coordinate [AU]"]]

                # Calculate number of steps based on STEP_SIZE
                num_steps = max(1, int(distance // STEP_SIZE))
                step_x = (neighbor_x - current_x) / num_steps
                step_y = (neighbor_y - current_y) / num_steps

                min_dilation_factor = 1  # Default: no dilation
                in_gravity_zone = False

                # Check for gravity at step intervals
                for step in range(0, num_steps + 1):  # Include the last step
                    step_x_pos = current_x + step * step_x
                    step_y_pos = current_y + step * step_y

                    
                    for gx, gy, mass in nontrivial_objects[["x-coordinate [AU]", "y-coordinate [AU]", "mass"]].to_numpy():
                        gravity_radius = mass * 1000
                        object_distance = compute_distances_between_objects(step_x_pos, step_y_pos, gx, gy)

                        if object_distance < gravity_radius:  # Inside gravity influence
                            
                            dilation_factor = gravitational_time_dilation(mass, object_distance)
                            
                            if dilation_factor == float('inf'):
                                continue  # Skip event horizon cases
                            
                            min_dilation_factor = min(min_dilation_factor, dilation_factor)
                            in_gravity_zone = True

                # Apply time dilation if in gravity zone
                if in_gravity_zone:
                    new_prop_time += travel_time * min_dilation_factor
                else:
                    new_prop_time += travel_time

                new_fuel = remaining_fuel - fuel_needed

                if new_obs_time < min_time[neighbor]:  # Only update if this path is better
                    min_time[neighbor] = new_obs_time
                    best_path[neighbor] = best_path[current] + [neighbor]
                    heapq.heappush(priority_queue, (new_obs_time, new_prop_time, new_fuel, neighbor, best_path[neighbor]))

    print("No path found from A to B within fuel constraints.")
    return float('inf'), float('inf'), 0, []  # No valid path found


def find_optimal_path(A, B, nontrivial_objects):
    """Find the minimum time and fuel-efficient path from A to B."""
    A_index = trivial_objects.index[trivial_objects["x-coordinate [AU]"] == A.iloc[0, 0]][0]
    B_index = trivial_objects.index[trivial_objects["x-coordinate [AU]"] == B.iloc[0, 0]][0]
    
    distances_between_trivial_objects = trivial_objects_separation(A_index, B_index)
    
    observer_time, proper_time, remaining_fuel, path = Dijkstra_Algorithm(
        distances_between_trivial_objects, A_index, B_index, nontrivial_objects
    )
    
    time_delay = observer_time - proper_time
    return observer_time, proper_time, time_delay, remaining_fuel, path

#/////CREATE_SPACE_FUNCTION/////////CREATE_SPACE_FUNCTION/////////CREATE_SPACE_FUNCTION/////////CREATE_SPACE_FUNCTION/////////CREATE_SPACE_FUNCTION////

def create_space():
    """Visualize the space with trivial and nontrivial objects."""
    fig, layer_of_objects = plt.subplots(figsize=(7, 7))
    layer_of_objects.grid(True, alpha=0.5, linestyle="dashed")
    A, B = generate_A_and_B()

    # Plot trivial objects
    layer_of_objects.scatter(
        trivial_objects["x-coordinate [AU]"],
        trivial_objects["y-coordinate [AU]"],
        color="black", label="Trivial Objects", marker="o", s=RADII_OF_TRIVIAL_OBJECTS
    )

    # Draw gravity influence
    radii_of_gravity(layer_of_objects)
    
    # Plot nontrivial objects
    layer_of_objects.scatter(
        nontrivial_objects["x-coordinate [AU]"],
        nontrivial_objects["y-coordinate [AU]"],
        color="orange", label="Nontrivial Objects", marker="o", s=RADII_OF_NONTRIVIAL_OBJECTS, alpha=1
    )
    
    observer_time, proper_time, time_delay, remaining_fuel, path = find_optimal_path(A, B, nontrivial_objects)
    
    if observer_time == float('inf') or  proper_time == float('inf'):
        print("No valid path found from A to B within fuel constraints.")
    else:
        print(f"Observer time required to get from A to B: {observer_time:.2f} s")
        print(f"Proper time required to get from A to B: {proper_time:.2f} s")
        print(f"Time delay: {time_delay:.2f} s")
        print(f"Remaining fuel after travel: {remaining_fuel:.2f}")

    if path:
        path_coords = trivial_objects.loc[path]
        layer_of_objects.plot(path_coords["x-coordinate [AU]"], path_coords["y-coordinate [AU]"], color='purple', linestyle='-', linewidth=1, marker='o', markersize = 3,label="Rocket Path")
    
    layer_of_objects.scatter(
        A["x-coordinate [AU]"], A["y-coordinate [AU]"], color="green", label="Point A", marker="X"
    )
    layer_of_objects.scatter(
        B["x-coordinate [AU]"], B["y-coordinate [AU]"], color="red", label="Point B", marker="X"
    )
    
    
    layer_of_objects.set_xlabel("x-coordinate [AU]")
    layer_of_objects.set_ylabel("y-coordinate [AU]")
    layer_of_objects.legend()

    plt.show()


#//////MAIN_CODE//////////MAIN_CODE//////////MAIN_CODE//////////MAIN_CODE//////////MAIN_CODE//////////MAIN_CODE//////////MAIN_CODE////

generate_trivial_objects()
generate_nontrivial_objects()
create_space()
