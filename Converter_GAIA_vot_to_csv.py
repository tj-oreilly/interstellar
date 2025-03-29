from astropy.io.votable import parse_single_table
import pandas as pd

# Path to your .vot file
votable_path = "data_GAIA_radius_=_100pc_parallax_error<5percent.vot"

# Parse the VOTable and convert to Astropy table
table = parse_single_table(votable_path).to_table()

# Convert to pandas DataFrame
df = table.to_pandas()
df_filtered = df[df['parallax'] >= 20]

# Save to CSV
csv_path = "GAIA_data/data_GAIA_radius_=_50pc_parallax_error<5percent.csv"
df_filtered.to_csv(csv_path, index=False)

print(f"Saved CSV to {csv_path}")
