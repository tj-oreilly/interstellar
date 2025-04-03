"""
Velocity against closest Approach           03/04/2025

This code:
    - lets you input the values for star mass and radius and the deflection
    angle
    - Plots a function of velocity against rmin for a given mass of a star
    and deflection angle
    - Finds the maximum velocity given the previous constants and the radius
    of the star
""" 

import numpy as np
import matplotlib.pyplot as plt 

# Constants
G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
solar_mass = 1.989e30  # Solar mass in kg
solar_radius = 6.955e8  # Solar radius in meters 

# User inputs
M = float(input("Enter the mass of the star in solar masses: ")) * solar_mass
R = float(input("Enter the radius of the star in solar radii: ")) * solar_radius
theta = np.radians(float(input("Enter the deflection angle (degrees): ")))

# Define function for velocity
def velocity(rmin, G, M, theta):
    return np.sqrt((G * M / rmin) * ((1 / np.sin(theta / 2)) - 1)) 

# Define range of rmin values from 0.5R to 5R
rmin_values = np.linspace(0.5 * R, 5 * R, 100)
v_values = velocity(rmin_values, G, M, theta) 

# Identify maximum velocity (when rmin = R)
v_max = velocity(R, G, M, theta) 

# Plot
plt.figure(figsize=(8, 5))
plt.plot(rmin_values / R, v_values / 1e3, label='Velocity vs. Closest Approach Distance', color='b')
plt.plot([1, 1], [0, v_max / 1e3], 'g--', label='r = R')  # Vertical line fully extended
plt.plot([0.5, 1], [v_max / 1e3, v_max / 1e3], 'r--', label=f'Max Velocity (r = R): {v_max/1e3:.2f} km/s')  # Horizontal line 

# Adjust axes limits
plt.xlabel('Closest Approach Distance (rmin / R)')
plt.ylabel('Velocity (km/s)')
plt.xlim(0.5, 5)  # Ensure x-axis starts at 0.5R
plt.ylim(0, max(v_values) / 1e3 + 0.2 * max(v_values) / 1e3)  # Allow some extra space above the max velocity 

plt.title('Velocity vs. Closest Approach Distance')
plt.legend()
plt.grid()
plt.show()