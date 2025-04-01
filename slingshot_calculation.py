"""
A calculation of the slingshot parameters given a set of stars and a path to traverse these.

This is based upon Sasha's code.
"""

import pandas as pd
import numpy as np
import scipy
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

    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
    ax.tick_params(colors="white")
    plt.grid(True, linestyle="--", alpha=0.3)
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


def calculate_slingshot_params(stars_data, path, defined_v=None):
    """
    Calculates the maximum velocity and the impact parameter for each slingshot in the path.
    Using the formulae from the online theory doc.
    """

    max_v = 3e5

    prev_pos = np.array([0.0, 0.0])
    for i, node in enumerate(path):
        if i == len(path) - 1:  # Skip final node
            break

        star_data = stars_data.loc[node]

        pos = np.array([star_data["x"], star_data["y"]])
        star_rad = star_data["radius_gspphot"] * SOLAR_RAD / AU  # In AU
        star_mass = star_data["mass_flame"]  # In solar masses

        # The incoming and outgoing directions
        next_star_data = stars_data.loc[path[i + 1]]
        next_pos = np.array([next_star_data["x"], next_star_data["y"]])

        in_dir = pos - prev_pos
        out_dir = next_pos - pos

        cos_dist = scipy.spatial.distance.cosine(in_dir, out_dir)
        angle = np.arccos(1 - cos_dist)

        if angle > np.deg2rad(179):  # Doesn't handle close to 180 degrees well
            print("Sharp slingshot encountered... skipping")
            continue

        if defined_v is not None:
            r = calculate_r(defined_v, angle, star_mass)
            print(f"Slingshot {i + 1}, r_min = {r:.4f} AU")

            continue

        # Find fastest velocity such that r_min > 1.5 R
        try:
            r, v = solve_parameters(angle, star_rad * 1.5, star_mass, (0.01, 100.0))

        except ValueError:
            print(f"Failed to find slingshot {i + 1}, angle = {angle}\n")
            # print("Parameters:")
            # print(f"  Angle = {np.degrees(angle):.1f} deg")
            # print(f"   Mass = {star_mass:.1f} solar masses")
            # print(f" Radius = {star_rad} AU")
            # print("")

            return

        deflection_angle = calculate_deflection(r, v, star_mass)

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


def main():
    # Read in the GAIA file
    stars_data = read_gaia_coords("./gaia_data/100_pc_stars.csv")

    # Filter by distance
    stars_data = stars_data[stars_data["pc_dist"] <= PC_LIM].reset_index(drop=True)

    # This is fixed for testing purposes (we can define it by TSP or similar later)
    # path = [201, 18, 77, 166, 5]
    path = np.random.choice(np.arange(0, len(stars_data)), size=100, replace=False)

    max_v = calculate_slingshot_params(stars_data, path)

    if max_v is None:
        return

    print(
        f"\nFor the velocity {max_v:.5f} km/s we have the following impact parameters:"
    )
    calculate_slingshot_params(stars_data, path, max_v)

    # Plot stars and path for visualisation
    plotting_function(stars_data, path)


if __name__ == "__main__":
    main()
