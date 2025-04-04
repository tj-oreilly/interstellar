"""
A calculation of the slingshot parameters given a set of stars and a path to traverse these.

This is based upon Sasha's code.
"""

import pandas as pd
import numpy as np
import scipy
from itertools import combinations
from scipy.optimize import minimize
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import cm

PC_LIM = 20.0  # The search radius in parsecs

G = 6.67e-11  # Gravitational constant in m^3kg^-1s^-2
SOLAR_RAD = 6.96e5  # Solar radius in km
SOLAR_MASS = 1.98e30  # Solar mass in kg
AU = 1.50e8  # AU in km


def read_gaia_coords(file_name):
    """
    Takes a GAIA CSV file and returns a DataFrame with the Cartesian coordinate computed from RA
    and DEC values.
    """

    stars_csv = None
    try:
        stars_csv = pd.read_csv(file_name)
    except FileNotFoundError or PermissionError:
        print(f"Could not open the file {file_name}")
        return None

    # Calculate Cartesian coord
    pc_dist = 1000.0 / stars_csv["parallax"]
    dec = stars_csv["dec"]
    ra = stars_csv["ra"]

    stars_csv["pc_dist"] = pc_dist
    stars_csv["x"] = pc_dist * np.cos(dec) * np.cos(ra)
    stars_csv["y"] = pc_dist * np.cos(dec) * np.sin(ra)
    stars_csv["z"] = pc_dist * np.sin(dec)

    return stars_csv


def plotting_function_3d(stars_data, path):
    """
    Plots the stars and the path that will be taken in 3D.
    """

    OFFSET = 0.8

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10), facecolor="black")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")

    # Set axis pane colors to black
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Set the grid and pane edge colors
    ax.xaxis.pane.set_edgecolor("white")
    ax.yaxis.pane.set_edgecolor("white")
    ax.zaxis.pane.set_edgecolor("white")
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)

    # Scale the star radii
    radii_scaled = 20.0 * (
        stars_data["radius_gspphot"] / stars_data["radius_gspphot"].max()
    )

    # Create 3D scatter plot
    scatter = ax.scatter(
        stars_data["x"],
        stars_data["y"],
        stars_data["z"],  # Add z-coordinate
        s=radii_scaled,
        c=stars_data["teff_gspphot"],
        cmap="plasma",
        norm=mcolors.Normalize(
            vmin=stars_data["teff_gspphot"].min(),
            vmax=stars_data["teff_gspphot"].max(),
        ),
        alpha=0.85,
        edgecolors="none",
        zorder=3
    )

    # Draw probe stops in 3D
    for i, stop in enumerate(path):
        x = stars_data.loc[stop, "x"]
        y = stars_data.loc[stop, "y"]
        z = stars_data.loc[stop, "z"]  # Add z-coordinate
        ax.scatter(x, y, z, color="green", s=40)
        ax.text(
            x + OFFSET,
            y + OFFSET,
            z + OFFSET,  # Add offset to z-coordinate
            str(i + 1),
            color="white",
            fontsize=12,
            weight="bold",
        )

    # Draw path line in 3D
    coords = stars_data.loc[path]

    # Create sun coordinate with z=0
    sun_coord = pd.Series({"x": 0.0, "y": 0.0, "z": 0.0})
    coords = pd.concat([pd.DataFrame([sun_coord]), coords], ignore_index=True)

    ax.plot(
        coords["x"],
        coords["y"],
        coords["z"],  # Add z-coordinate
        color="lime",
        linestyle="-",
        linewidth=1,
        marker="o",
        markersize=3,
        label="Path",
    )

    # Plot sun
    ax.scatter(0, 0, 0, color="red", s=80, label="Sun", zorder=5)

    # Set labels for all axes
    ax.set_xlabel("x [pc]", color="white")
    ax.set_ylabel("y [pc]", color="white")
    ax.set_zlabel("z [pc]", color="white")  # Add z-axis label
    ax.set_title(f"3D Star Map (within {PC_LIM} parsecs)", color="white")


    

    # Add legend
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white")

    # Colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Temperature [K]", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
 
    # Set grid
    ax.grid(True, linestyle="--", alpha=0.3)

    # Add some rotation to better see the 3D effect
    ax.view_init(elev=30, azim=45)

    # Remove grid lines but keep the axes
    ax.grid(False)

    plt.savefig("path_plot_3d.svg", format="svg")
    plt.show()


def plotting_function(stars_data, path):
    """
    Plots the stars and the path that will be taken between them.
    """

    OFFSET = 0.5

    plt.subplots(figsize=(10, 8), facecolor="black")

    ax = plt.gca()
    ax.set_facecolor("black")

    radii_scaled = 20.0 * (
        stars_data["radius_gspphot"] / stars_data["radius_gspphot"].max()
    )

    scatter = ax.scatter(
        stars_data["x"],
        stars_data["y"],
        s=radii_scaled,
        c=stars_data["teff_gspphot"],
        cmap="plasma",
        norm=mcolors.Normalize(
            vmin=stars_data["teff_gspphot"].min(),
            vmax=stars_data["teff_gspphot"].max(),
        ),
        alpha=0.85,
        edgecolors="none",
    )

    # Draw probe stops
    for i, stop in enumerate(path):
        x = stars_data.loc[stop, "x"]
        y = stars_data.loc[stop, "y"]
        ax.scatter(x, y, color="green", s=40, zorder=5)
        ax.text(
            x + OFFSET,
            y + OFFSET,
            str(i + 1),
            color="green",
            fontsize=10,
            weight="bold",
        )

    # Draw path line
    coords = stars_data.loc[path]

    sun_coord = pd.Series({"x": 0.0, "y": 0.0})
    coords = pd.concat(
        [sun_coord, pd.DataFrame([sun_coord]), coords], ignore_index=True
    )

    ax.plot(
        coords["x"],
        coords["y"],
        color="green",
        linestyle="-",
        linewidth=1,
        marker="o",
        markersize=3,
        label="Rocket Path",
    )

    ax.legend()
    ax.scatter(0, 0, color="red", s=80, label="Sun", zorder=5)
    ax.set_xlabel("x [pc]", color="white")
    ax.set_ylabel("y [pc]", color="white")
    ax.set_title(f"Star Map (within {PC_LIM} parsecs)", color="white")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white")

    cbar = plt.colorbar(scatter)
    cbar.set_label("Temperature [K]", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")



    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig("path_plot.svg", format="svg")
    plt.show()


def calculate_deflection(r, v, M):
    """
    Calculates the deflection angle given the periapsis, approach velocity and star mass.

    Parameters:
        r (float): The periapsis (AU)
        v (float): The approach velocity (km/s)
        M (float): The mass of the star (solar mass)

    Returns:
        float: The deflection angle (rad)
    """

    r_km = r * AU  # Convert to km
    GM = G * SOLAR_MASS * M * 1e-9  # Get GM in km^3 M_sol^-1 s^-2
    return 2 * np.arcsin(1.0 / (r_km * v**2 / GM + 1))

    # return 2.0 * np.arctan(GM / (b_km * v**2))


def calculate_velocity(r, angle, M):
    """
    Calculates the incoming velocity for a given periapsis, deflection angle, and star mass.

    Parameters:
        r (float): The periapsis (AU)
        angle (float): The deflection angle (rad)
        M (float): The mass of the star (solar mass)

    Returns:
        float: The incoming velocity (km/s)
    """

    if 1.0 / np.cos(angle) <= 1:
        print("Invalid hyperbolic orbit: eccentricity <= 1")
        return None

    r_km = r * AU  # Convert to km
    GM = G * SOLAR_MASS * M * 1e-9  # Get GM in km^3 M_sol^-1 s^-2
    return np.sqrt((GM * (1.0 / np.sin(angle / 2.0) - 1)) / r_km)


def calculate_r(v, angle, M):
    """
    Calculates the periapsis given a fixed velocity, deflection angle, and star mass.

    Parameters:
        v (float): The velocity (km/s)
        angle (float): The deflection angle (rad)
        M (float): The mass of the star (solar mass)
    """

    GM = G * SOLAR_MASS * M * 1e-9
    b = (GM / v**2) * (1.0 / np.sin(angle / 2) - 1)

    return b / AU


# Scipy optimise functions
def objective(x):
    r, v = x
    return -v


def constraint_deflection(x, target_deflection, M):
    r, v = x
    return calculate_deflection(r, v, M) - target_deflection


def constraint_periapsis(x, min_periapsis, M):
    r, v = x
    return r - min_periapsis


def solve_parameters(target_deflection, min_periapsis, M, initial_guess=(1.0, 1.0)):
    constraints = [
        {
            "type": "eq",
            "fun": constraint_deflection,
            "args": (
                target_deflection,
                M,
            ),
        },
        {
            "type": "ineq",
            "fun": constraint_periapsis,
            "args": (
                min_periapsis,
                M,
            ),
        },
    ]

    result = minimize(objective, initial_guess, constraints=constraints, method="SLSQP")

    if result.success:
        return result.x  # Optimized (r, v)
    else:
        raise ValueError("Optimization failed")


def calculate_slingshot_params(stars_data, path, defined_v=None, debug_print=False):
    """
    Calculates the maximum velocity and the impact parameter for each slingshot in the path.
    Using the formulae from the online theory doc.
    """

    max_v = 3e5

    prev_pos = np.array([0.0, 0.0, 0.0])
    for i, node in enumerate(path):
        if i == len(path) - 1:  # Skip final node
            break

        star_data = stars_data.loc[node]

        pos = np.array([star_data["x"], star_data["y"], star_data["z"]])
        star_rad = star_data["radius_gspphot"] * SOLAR_RAD / AU  # In AU
        star_mass = star_data["mass_flame"]  # In solar masses

        # The incoming and outgoing directions
        next_star_data = stars_data.loc[path[i + 1]]
        next_pos = np.array(
            [next_star_data["x"], next_star_data["y"], next_star_data["z"]]
        )

        in_dir = pos - prev_pos
        out_dir = next_pos - pos

        cos_dist = scipy.spatial.distance.cosine(in_dir, out_dir)
        angle = np.arccos(1 - cos_dist)

        if angle > np.deg2rad(179):  # Doesn't handle close to 180 degrees well
            print("Sharp slingshot encountered... skipping")
            prev_pos = pos
            continue

        if defined_v is not None:
            r = calculate_r(defined_v, angle, star_mass)
            if r < star_rad:
                print(f"Radius inside star {r} vs {star_rad}!")

            print(
                f"Slingshot {i + 1}, star_mass = {
                    star_mass:.2f
                } solar masses, star_rad = {star_rad:.4f} AU r_min = {
                    r:.4f
                } AU, angle = {np.degrees(angle):.2f} deg"
            )

            prev_pos = pos
            continue

        # Find fastest velocity such that r_min > 1.5 R
        try:
            r, v = solve_parameters(angle, star_rad * 1.01, star_mass, (0.01, 100.0))

        except ValueError:
            print(f"Failed to find slingshot {i + 1}, angle = {angle}\n")
            # print("Parameters:")
            # print(f"  Angle = {np.degrees(angle):.1f} deg")
            # print(f"   Mass = {star_mass:.1f} solar masses")
            # print(f" Radius = {star_rad} AU")
            # print("")

            return

        deflection_angle = calculate_deflection(r, v, star_mass)

        if debug_print:
            print(f"Slingshot {i + 1}:")
            print(f"Desired angle = {np.degrees(angle):.1f} deg")
            print(f"    Angle = {np.degrees(deflection_angle):.1f} deg")
            print(f"        v = {v:.4f} km/s")
            print(f"Periapsis = {r:.4f} AU")
            print(f"Star mass = {star_mass:.1f} solar masses")
            print(f" Star rad = {star_rad} AU")

            print("")

        max_v = min(max_v, v)

        prev_pos = pos

    return max_v


def star_distance(star1, star2):
    """
    Calculates the distance between two stars.
    """

    if star1 is None:
        star1 = {"x":0.0, "y":0.0, "z":0.0}

    x_diff = star2["x"] - star1["x"]
    y_diff = star2["y"] - star2["y"]
    z_diff = star2["z"] - star2["z"]
    diff = np.array([x_diff, y_diff, z_diff])

    return np.sqrt(np.dot(diff, diff))


def find_best_stop_order(selected_stops, stars_data):
    """
    Calculates the optimal order of stops using the Held-Karp algorithm
    """

    n = len(selected_stops)

    # Build distance matrix
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                star1 = stars_data.loc[selected_stops[i]]
                star2 = stars_data.loc[selected_stops[j]]
                d = star_distance(star1, star2)
                dist[i][j] = d
            else:
                dist[i][j] = float("inf")

    # Held-Karp DP
    C = {}
    for k in range(1, n):
        C[(frozenset([k]), k)] = (dist[0][k], [0, k])

    for s in range(2, n):
        for subset in combinations(range(1, n), s):
            S = frozenset(subset)
            for k in subset:
                prev_set = S - {k}
                min_cost = float("inf")
                best_path = []
                for m in prev_set:
                    cost, path = C[(prev_set, m)]
                    new_cost = cost + dist[m][k]
                    if new_cost < min_cost:
                        min_cost = new_cost
                        best_path = path + [k]
                C[(S, k)] = (min_cost, best_path)

    # Choose shortest ending path (not necessarily returning to start)
    full_set = frozenset(range(1, n))
    min_total = float("inf")
    final_path = []
    for k in range(1, n):
        cost, path = C[(full_set, k)]
        if cost < min_total:
            min_total = cost
            final_path = path

    stop_order = [selected_stops[i] for i in final_path]

    return stop_order


def orientation(p, q, r):
    """Return the orientation of the triplet (p, q, r).
    0 -> p, q and r are collinear
    1 -> Clockwise
    2 -> Counterclockwise
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0  # collinear
    elif val > 0:
        return 1  # clockwise
    else:
        return 2  # counterclockwise


def on_segment(p, q, r):
    """Check if point q lies on line segment pr."""
    if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(
        p[1], r[1]
    ):
        return True
    return False


def do_intersect(p1, q1, p2, q2):
    """Returns True if the line segments 'p1q1' and 'p2q2' intersect."""
    # Find the 4 orientations needed for the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special cases
    # p1, q1 and p2 are collinear and p2 lies on segment p1q1
    if o1 == 0 and on_segment(p1, p2, q1):
        return True

    # p1, q1 and p2 are collinear and q2 lies on segment p1q1
    if o2 == 0 and on_segment(p1, q2, q1):
        return True

    # p2, q2 and p1 are collinear and p1 lies on segment p2q2
    if o3 == 0 and on_segment(p2, p1, q2):
        return True

    # p2, q2 and p1 are collinear and q1 lies on segment p2q2
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    # If none of the cases apply, then the segments do not intersect
    return False


def check_path_intersection_2d(path, stars_data):
    """Checks if a given path contains intersections."""

    # Build list of segments
    segments = []

    for i, node in enumerate(path):
        if i == len(path) - 1:  # Skip final node
            break

        star_data = stars_data.loc[node]
        pos = np.array([star_data["x"], star_data["y"]])

        next_star_data = stars_data.loc[path[i + 1]]
        next_pos = np.array([next_star_data["x"], next_star_data["y"]])

        if i == 0:
            segments.append((np.array([0.0, 0.0]), pos))
        segments.append((pos, next_pos))

    # Check intersections
    for i in range(len(segments)):
        for j in range(len(segments)):
            if do_intersect(
                segments[i][0], segments[i][1], segments[j][0], segments[j][1]
            ):
                return True

    return False


def main():
    # Read in the GAIA file
    stars_data = read_gaia_coords("./gaia_data/100_pc_stars.csv")

    # Filter by distance
    stars_data = stars_data[stars_data["pc_dist"] <= PC_LIM].reset_index(drop=True)

    stops = [126, 65, 91, 50, 43, 139, 28, 96, 52, 53]
    path = stops  # find_best_stop_order(stops, stars_data)

    # for i in range(10):
    #     # Calculate optimal path with Held-Karp
    #     stops = np.random.choice(np.arange(0, len(stars_data)), size=10, replace=False)
    #
    #
    #
    #     if not check_path_intersection_2d(path, stars_data):
    #         break

    print(path)

    max_v = calculate_slingshot_params(stars_data, path)

    if max_v is None:
        return

    print(f"\nFor the velocity {max_v:.5f} km/s we have the following parameters:")
    calculate_slingshot_params(stars_data, path, max_v)

    # Calculate time
    distance = 0.0
    for index, star_index in enumerate(path):
        if index == len(path) - 1:
            break

        next_star_index = path[index+1]
        star_distance(stars_data.loc[star_index], stars_data.loc[next_star_index])

    distance += star_distance(None, stars_data.loc[path[0]]) # First star

    print(distance)
    time = distance / (max_v / 3.0857e+13)  # Parsec unit convert
    print(time / (86400 * 365)) # years

    # Plot stars and path for visualisation
    plotting_function_3d(stars_data, path)


if __name__ == "__main__":
    main()
