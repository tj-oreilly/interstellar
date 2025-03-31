"""
A calculation of the slingshot parameters given a set of stars and a path to traverse these.
"""

import pandas as pd
import numpy as np

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

PC_LIM = 20.0  # The search radius in parsecs


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


def main():
    # Read in the GAIA file
    stars_data = read_gaia_coords("./gaia_data/100_pc_stars.csv")

    # Filter by distance
    stars_data = stars_data[stars_data["pc_dist"] <= PC_LIM].reset_index(drop=True)

    # This is fixed for testing purposes (we can define it by TSP or similar later)
    path = [201, 18, 77, 166, 5]

    plotting_function(stars_data, path)


if __name__ == "__main__":
    main()
