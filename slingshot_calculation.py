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
SOLAR_RAD = 6.96e8  # Solar radius in metres
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


def calculate_periapsis(b, v, M):
    """
    Calculates the periapsis given the impact parameter, approach velocity, and star mass.

    Parameters:
        b (float): The impact parameter (AU)
        v (float): The approach velocity (km/s)
        M (float): The mass of the star (solar mass)

    Returns:
        float: The periapsis (AU)
    """

    b_km = b * AU  # Convert to km
    GM = G * SOLAR_MASS * M * 1e-9  # Get GM in km^3 M_sol^-1 s^-2

    periapsis = GM / (v**2 + np.sqrt(v**4 + (2 * GM * v**2) / b_km))
    return periapsis / AU  # Convert back to AU


def calculate_deflection(b, v, M):
    """
    Calculates the deflection angle given the impact parameter, approach velocity and star mass.

    Parameters:
        b (float): The impact parameter (AU)
        v (float): The approach velocity (km/s)
        M (float): The mass of the star (solar mass)

    Returns:
        float: The deflection angle (rad)
    """

    b_km = b * AU  # Convert to km
    GM = G * SOLAR_MASS * M * 1e-9  # Get GM in km^3 M_sol^-1 s^-2
    return 2.0 * np.arctan(GM / (b_km * v**2))


def solve_for_b(v, target_angle, M):
    def func(b):
        return calculate_deflection(b, v, M) - target_angle

    b_solution, info, ier, msg = scipy.optimize.fsolve(
        func, x0=10.0, full_output=True
    )  # Initial guess

    print(func(b_solution))

    return b_solution[0]


def objective(x):
    b, v = x
    return -v  # Maximizing v, so we minimize -v


def constraint_deflection(x, target_deflection, M):
    b, v = x
    return calculate_deflection(b, v, M) - target_deflection


def constraint_periapsis(x, min_periapsis, M):
    b, v = x
    return calculate_periapsis(b, v, M) - min_periapsis


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
        return result.x  # Optimized (b, v)
    else:
        raise ValueError("Optimization failed")


def calculate_slingshot_params(stars_data, path):
    """
    Calculates the maximum velocity and the impact parameter for each slingshot in the path.
    Using the formulae from the online theory doc.
    """

    prev_pos = np.array([0.0, 0.0])
    for i, node in enumerate(path):
        if i == len(path) - 1:  # Skip final node
            break

        star_data = stars_data.loc[node]

        pos = np.array([star_data["x"], star_data["y"]])
        star_rad = star_data["radius_gspphot"]  # In solar radii
        star_mass = star_data["mass_flame"]  # In solar masses

        # The incoming and outgoing directions
        next_star_data = stars_data.loc[path[i + 1]]
        next_pos = np.array([next_star_data["x"], next_star_data["y"]])

        in_dir = pos - prev_pos
        out_dir = next_pos - pos

        cos_dist = scipy.spatial.distance.cosine(in_dir, out_dir)
        angle = np.arccos(1 - cos_dist)

        # Find fastest velocity such that r_min > 1.5 R
        try:
            b, v = solve_parameters(angle, star_rad * 1.5, star_mass, (1.0, 1.0))
        except ValueError:
            print(f"Failed to find slingshot {i + 1}\n")
            continue

        periapsis = calculate_periapsis(b, v, star_mass)
        deflection_angle = calculate_deflection(b, v, star_mass)

        print(f"Slingshot {i + 1}:")
        print(f"Desired angle = {np.degrees(angle):.1f} deg")
        print(f"    Angle = {np.degrees(deflection_angle):.1f} deg")
        print(f"        b = {b:.1f} AU")
        print(f"        v = {v:.1f} km/s")
        print(f"Periapsis = {periapsis:.1f} AU")
        print("")

        prev_pos = pos


def example_slingshot():
    """
    Some debug code to test an example slingshot, plotting the parameter space in b and v and try
    to optimize the velocity from this in a crude way.
    """

    star_mass = 1.0  # Solar masses
    star_rad = 0.01  # AU
    target_deflect = np.deg2rad(133.0)

    b_values = np.linspace(0.01, 10.0, 1000)
    v_values = np.linspace(0.001, 100.0, 1000)
    B, V = np.meshgrid(b_values, v_values)

    Z = np.degrees(calculate_deflection(B, V, star_mass))

    plt.figure(figsize=(10, 8))
    heatmap = plt.pcolormesh(B, V, Z, cmap=cm.viridis, shading="auto")
    plt.colorbar(heatmap, label="Deflection angle (degrees)")

    plt.xlabel("Impact parameter (AU)")
    plt.ylabel("Velocity (km/s)")
    plt.title("Deflection angles")

    plt.tight_layout()
    plt.show()

    b_opt = None
    v_max = None
    periapsis_found = None

    for b in b_values:
        found_v = 0.0
        min_value = None

        for v in v_values:
            calc_val = np.abs(target_deflect - calculate_deflection(b, v, star_mass))
            if min_value is None or calc_val < min_value:
                min_value = calc_val
                found_v = v

        if min_value < target_deflect * 0.01:  # Within appropriate error
            periapsis = calculate_periapsis(b, found_v, star_mass)
            if periapsis > star_rad * 1.5 and (v_max is None or found_v > v_max):
                v_max = found_v
                b_opt = b
                periapsis_found = periapsis

    actual_deflect = np.degrees(calculate_deflection(b_opt, v_max, star_mass))

    print("Found the following configuration:")
    print(f"Target defleciton = {np.degrees(target_deflect)} deg")
    print(f"Actual deflection = {actual_deflect:.2f} deg")
    print(f"        Periapsis = {periapsis_found:.2f} AU")
    print(f"     Impact param = {b_opt:.2f} AU")
    print(f"         Velocity = {v_max:.2f} km/s")


def main():
    # Read in the GAIA file
    stars_data = read_gaia_coords("./gaia_data/100_pc_stars.csv")

    # Filter by distance
    stars_data = stars_data[stars_data["pc_dist"] <= PC_LIM].reset_index(drop=True)

    # This is fixed for testing purposes (we can define it by TSP or similar later)
    path = [201, 18, 77, 166, 5]

    calculate_slingshot_params(stars_data, path)

    # Plot stars and path for visualisation
    # plotting_function(stars_data, path)


if __name__ == "__main__":
    main()
