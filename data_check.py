import pandas as pd
import numpy as np

data_path = 'GAIA_data/data_GAIA_radius_=_100pc_parallax_error<5percent.csv'
df = pd.read_csv(data_path)
G = 6.67430 * 10**(-11)
C = 299_792_458
SUN_MASS_to_R_ratio = 1988400 * 10**24 / (695700 * 10**3)
CONSTANT = 2 * G / C**2
epsilon = 1e-5

df["r_s/R_star"] = (df['mass_flame']) * (1/df['radius_gspphot']) * CONSTANT * SUN_MASS_to_R_ratio
print(df["r_s/R_star"][:10])
print(len(df[df["r_s/R_star"] < epsilon]) - len(df))
