import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Load your Gaia data (CSV file with radius, temperature, parallax, RA/Dec)
csv_path = "GAIA_data/data_GAIA_radius_=_20pc_parallax_error<5percent.csv"
df = pd.read_csv(csv_path)

# Filter out rows missing necessary values
df = df.dropna(subset=['ra', 'dec', 'parallax', 'teff_gspphot', 'radius_gspphot'])

# Convert coordinates: RA/DEC/parallax → x, y (in parsecs)
ra_rad = np.deg2rad(df['ra'])
dec_rad = np.deg2rad(df['dec'])
distance_pc = 1000 / df['parallax']  # convert parallax (mas) to distance (pc)

df['x'] = distance_pc * np.cos(dec_rad) * np.cos(ra_rad)
df['y'] = distance_pc * np.cos(dec_rad) * np.sin(ra_rad)

# Normalize radii for scatter dot sizes
radii_scaled = 20 * df['radius_gspphot'] / df['radius_gspphot'].max()

# Create the star map
plt.figure(figsize=(10, 8), facecolor='black')
ax = plt.gca()
ax.set_facecolor("black")

# Plot stars
sc = plt.scatter(
    df['x'], df['y'],
    s=radii_scaled,
    c=df['teff_gspphot'],
    cmap='plasma',
    norm=mcolors.Normalize(vmin=df['teff_gspphot'].min(), vmax=df['teff_gspphot'].max()),
    alpha=0.85,
    edgecolors='none'
)

# Plot Earth at the origin
plt.scatter(0, 0, color='red', s=80, label='Earth', zorder=5)

# Add labels and colorbar
plt.xlabel('x [pc]', color='white')
plt.ylabel('y [pc]', color='white')
plt.title('Star Map (within 20 parsecs)', color='white')

plt.legend(facecolor='black', edgecolor='white', labelcolor='white')

cbar = plt.colorbar(sc)
cbar.set_label('Temperature [K]', color='white')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

# Style grid and ticks
ax.tick_params(colors='white')
plt.grid(True, linestyle='--', alpha=0.3)

# Save and show the figure
plt.tight_layout()
plt.savefig("Diagrams/Star_maps_GAIA/star_map_GAIA_20pc_parallax_error<5percent.png", dpi=300)
plt.show()
