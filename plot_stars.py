"""
Some basic code to take in the GAIA CSV data and plot the points for the stars in 2D.

The mapping takes the xy plane through the Earth's equator, with the x axis along the vernal
equinox.
"""

import pygame
import numpy as np
import pandas as pd


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
    ly_dist = stars_csv["distance_ly"]
    dec = stars_csv["dec"]
    ra = stars_csv["ra"]

    stars_csv["x"] = ly_dist * np.cos(dec) * np.cos(ra)
    stars_csv["y"] = ly_dist * np.cos(dec) * np.sin(ra)
    stars_csv["z"] = ly_dist * np.sin(dec)

    return stars_csv


def main():
    SIZE = 800

    # Create window
    pygame.init()
    screen = pygame.display.set_mode([SIZE, SIZE])
    pygame.display.set_caption("GAIA Stars")

    # Read star data
    stars = read_gaia_coords("./50_ly_stars.csv")
    if stars is None:
        return

    stars = stars[stars["distance_ly"] < 25.0]  # Filter by distance

    # Scale up the points so they fill the screen
    max_dist = np.max(stars["distance_ly"])
    scale = SIZE / (2 * max_dist)

    # Draw
    CENTER = np.array([SIZE / 2, SIZE / 2])

    for i, star in stars.iterrows():
        pygame.draw.circle(
            screen,
            [255, 255, 255],
            CENTER + scale * np.array([star["x"], star["y"]]),
            2.0,
        )

    pygame.draw.circle(screen, [255, 0, 0], CENTER, 5.0)  # Draw sun

    pygame.display.flip()

    # Event loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return


if __name__ == "__main__":
    main()
