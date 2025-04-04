"""
Plots a hyperbolic orbit around a mass.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

STAR_RADIUS = 0.5  # Solar radii
STAR_MASS = 0.01  # Solar masses

# Constants
G = 6.67e-11
SOLAR_MASS = 1.99e30  # kg
SOLAR_RADIUS = 1.96e8  # m

def e(b,v):
  return np.sqrt(1 + (b**2 * v**4) / ((G * STAR_MASS * SOLAR_MASS)**2))

def calc_max_theta(b, v):
  return np.arccos(-1/e(b,v)) * 0.99

def trajectory(theta, b, v):
  h = b * v
  mu = G * STAR_MASS * SOLAR_MASS
  return (h**2 / mu) * 1.0 / (1 + e(b,v)*np.cos(theta))

def plot_hyperbolic_orbit(ax, b, v, col):

  theta = np.linspace(-calc_max_theta(b, v), calc_max_theta(b,v), 1000)
  r = trajectory(theta, b, v) / SOLAR_RADIUS  # Scale down to graph size

  x = r * np.cos(theta)
  y = r * np.sin(theta)

  ax.plot(x, y, color=col, linewidth=1, label=f"Eccentricity={e(b,v):.3f}")


def main():

  # Create figure and axes
  fig, ax = plt.subplots(figsize=(10, 6))

  # Set the figure and axes background colors to black
  fig.patch.set_facecolor('black')
  ax.set_facecolor('black')

  ax.xaxis.label.set_color('white')
  ax.yaxis.label.set_color('white')
  ax.title.set_color('white')

  plot_hyperbolic_orbit(ax, SOLAR_RADIUS * 20.0, 10000, "white")
  plot_hyperbolic_orbit(ax, SOLAR_RADIUS * 30.0, 10000, "lime")
  plot_hyperbolic_orbit(ax, SOLAR_RADIUS * 40.0, 10000, "orange")

  circle = Circle((0, 0), radius=STAR_RADIUS, facecolor='yellow', edgecolor='yellow', linewidth=2)
  ax.add_patch(circle)

  ax.set_aspect('equal')

  size = 50.0
  ax.set_xlim(-size, size)
  ax.set_ylim(-size, size)

  legend = ax.legend()
  legend.get_frame().set_facecolor('black')

  # Set legend text color to white
  for text in legend.get_texts():
      text.set_color('white')

  # Set the spines (borders) to white
  for spine in ax.spines.values():
    spine.set_color('white')

  plt.savefig("hyperbolic.svg", format="svg")
  plt.show()


if __name__ == "__main__":
  main()