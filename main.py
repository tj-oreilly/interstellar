import os
import re
from itertools import combinations
from math import sqrt

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Physical constants
SPEED_OF_LIGHT = 9.715e-9  # pc/s
GRAVITATIONAL_CONST = 6.67430e-11  # m^3 kg^-1 s^-2
SOLAR_MASS = 1.989e30  # kg
SOLAR_RADIUS = 695700  # km
PROBE_MASS = 478  # kg

# Convertation
PC_TO_M = 3.0857e16  # parsec to meters
PC_PER_S_TO_KM_PER_H = 3.0857e13 * 3600
KMS_TO_PC_PER_S = 3.24078e-14
PM_CONVERSION = 4.74
SECONDS_IN_YEAR = 31_557_600
MAS_TO_DEG = 1 / 3_600_000  # Convert milliarcseconds to degrees

# Simulation constants
N_STOPS = 6 # number of stops
MIN_STEP = 5e-12  # Minimum velocity pc/s
MAX_STEP = 5e-12  # Maximum velocity pc/s
STEP_SIZES = np.arange(MIN_STEP, MAX_STEP + 1e-12, 5e-11)
SEED_RANGE = 20

# GAIA data.csv file
GAIA_data_path = "GAIA_data/data_GAIA_radius_=_20pc_parallax_error<5percent.csv"

sun_coord = pd.DataFrame({
    'x [pc]': [0.0],
    'y [pc]': [0.0]
})

sun_params = pd.DataFrame({
    'ra': [0.0],
    'dec': [0.0],
    'parallax': [1.0],
    'teff_gspphot': [5778.0],
    'radius_gspphot': [1.0]
})


#/////GENERATORS//////////GENERATORS//////////GENERATORS//////////GENERATORS//////////GENERATORS//////////GENERATORS//////////GENERATORS/////

def take_stars_from_GAIA_data():
    global stars_coordinates, N_OF_STARS, SPACE_SIZE, radius, stars_parameters

    data_filename = os.path.basename(GAIA_data_path)
    match = re.search(
        r'data_GAIA_radius_=_([0-9]+)pc_parallax_error<5percent', data_filename
    )

    if not match:
        raise ValueError("Could not extract radius from filename.")

    radius = int(match.group(1))
    SPACE_SIZE = 2 * radius

    GAIA_data = pd.read_csv(GAIA_data_path)

    ra_rad = np.deg2rad(GAIA_data['ra'])
    dec_rad = np.deg2rad(GAIA_data['dec'])
    distance_pc = 1000 / GAIA_data['parallax']

    GAIA_data['x [pc]'] = distance_pc * np.cos(dec_rad) * np.cos(ra_rad)
    GAIA_data['y [pc]'] = distance_pc * np.cos(dec_rad) * np.sin(ra_rad)

    stars_coordinates = pd.concat([
        sun_coord,
        GAIA_data[['x [pc]', 'y [pc]']]
    ], ignore_index=True)

    stars_parameters = pd.concat([
        sun_params,
        GAIA_data.drop(columns=['x [pc]', 'y [pc]'])
    ], ignore_index=True)

    N_OF_STARS = len(stars_coordinates)
    print(f"{len(GAIA_data)} stars are in GAIA dataset")
    
    vx_list = []
    vy_list = []


    for i, row in GAIA_data.iterrows():
        parallax = row['parallax']
        ra = np.deg2rad(row['ra'])
        dec = np.deg2rad(row['dec'])
        pmra = row['pmra']
        pmdec = row['pmdec']
        rv_kms = row['radial_velocity'] if 'radial_velocity' in row and not np.isnan(row['radial_velocity']) else 0

        distance_pc = 1000 / parallax
        mu_ra_arcsec = pmra / 1000  # arcsec/year
        mu_dec_arcsec = pmdec / 1000

        # Tangential components (km/s)
        vx_tan = PM_CONVERSION * mu_ra_arcsec * distance_pc
        vy_tan = PM_CONVERSION * mu_dec_arcsec * distance_pc

        # Convert proper motion into RA/Dec directions
        v_ra = vx_tan  # along increasing RA (East)
        v_dec = vy_tan  # along increasing Dec (North)

        # Radial velocity projection into x/y (plane of sky)
        # Unit vector of radial direction in Cartesian coords
        x_hat = np.cos(dec) * np.cos(ra)
        y_hat = np.cos(dec) * np.sin(ra)

        vx_radial = rv_kms * x_hat
        vy_radial = rv_kms * y_hat

        # Total projected velocity
        vx_total = v_ra * (-np.sin(ra)) + vx_radial  # RA increases east → x decreases
        vy_total = v_ra * (np.cos(ra)) + vy_radial   # RA increases east → y increases
        vx_total += v_dec * (-np.sin(dec) * np.cos(ra))
        vy_total += v_dec * (-np.sin(dec) * np.sin(ra))

        vx_list.append(vx_total * KMS_TO_PC_PER_S)
        vy_list.append(vy_total * KMS_TO_PC_PER_S)


    sun_velocity = pd.DataFrame({'vx [pc/s]': [0.0], 'vy [pc/s]': [0.0]})
    star_velocities = pd.DataFrame({'vx [pc/s]': vx_list, 'vy [pc/s]': vy_list})
    stars_velocities = pd.concat([sun_velocity, star_velocities], ignore_index=True)

    return radius, SPACE_SIZE, stars_velocities


def generate_stops(N_STOPS, stars_coordinates, seed=None):
    if N_STOPS == 0:
        return [0]  # Only the Sun

    random_stops = stars_coordinates.sample(
        n=N_STOPS, random_state=seed
    ).index.tolist()

    return [0] + random_stops  # 0 is Sun


#/////COMPUTATIONS_AND_CONDITIONS/////#/////COMPUTATIONS_AND_CONDITIONS/////#/////COMPUTATIONS_AND_CONDITIONS/////#/////COMPUTATIONS_AND_CONDITIONS/////

def check_if_objects_do_not_intersect(x1, y1, r1, x2, y2, r2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) > (r1 + r2)


def compute_distances_between_objects(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def check_path_validity(r0_values, r0_uncertainties, parameters_df):
    for i, (idx, r0) in enumerate(r0_values):
        uncertainty_r0 = r0_uncertainties[i][1]
        upper_star_radius_km = (
            parameters_df.loc[idx, 'radius_flame_upper'] * SOLAR_RADIUS
        )
        if r0 - uncertainty_r0 < upper_star_radius_km:
            print(
                f"\U0001f6d1 Invalid path: r0 - σ_r0 = {r0 - uncertainty_r0:.3e} < "
                f"upper radius = {upper_star_radius_km:.3e} for star idx = {idx}"
            )
            return False
    return True

def compute_intercept_direction_and_time(i, j, coordinates_df, velocities_df, probe_speed):
    r0 = coordinates_df.loc[i, ['x [pc]', 'y [pc]']].values  # current probe position
    rT = coordinates_df.loc[j, ['x [pc]', 'y [pc]']].values  # target star now
    vT = velocities_df.loc[j, ['vx [pc/s]', 'vy [pc/s]']].values  # target velocity

    d = rT - r0  # relative position

    # Coefficients of quadratic
    a = - np.dot(vT, vT) + probe_speed ** 2
    b =  - 2 * np.dot(d, vT)
    c =  - np.dot(d, d)

    # Solve for t
    discriminant = b ** 2 - 4 * a * c
    
    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    t = min(filter(lambda x: x > 0, [t1, t2]))  # pick earliest valid time

    # Compute the probe's required velocity vector
    intercept_vector = (d + vT * t) / t
    unit_vector = intercept_vector / np.linalg.norm(intercept_vector)

    return unit_vector, t

def propagate_star_position(idx: int, arrival_time: float, coordinates_df, velocities_df):
    """
    Propagate a star's position forward by 'arrival_time' seconds using its velocity.

    Parameters:
        idx (int): Index of the star in the dataframe.
        arrival_time (float): Time in seconds since t=0.
        coordinates_df (DataFrame): Original star positions ('x [pc]', 'y [pc]').
        velocities_df (DataFrame): Velocities of stars ('vx [pc/s]', 'vy [pc/s]').

    Returns:
        (x, y): Tuple of propagated coordinates in parsecs.
    """
    x0 = coordinates_df.loc[idx, 'x [pc]']
    y0 = coordinates_df.loc[idx, 'y [pc]']
    vx = velocities_df.loc[idx, 'vx [pc/s]']
    vy = velocities_df.loc[idx, 'vy [pc/s]']

    x = x0 + vx * arrival_time
    y = y0 + vy * arrival_time

    return x, y


#//////HELD_KARP_PATHFINDING///////////HELD_KARP_PATHFINDING///////////HELD_KARP_PATHFINDING///////////HELD_KARP_PATHFINDING///////////HELD_KARP_PATHFINDING/////

def find_best_stop_order(
    selected_stops,
    step_size,
    star_coordinates,
    velocities_df=None,
    mode="static"  # or "dynamic"
):
    """
    Finds optimal stop order using Held-Karp algorithm.

    Parameters:
        selected_stops (list): List of star indices (first is Sun)
        step_size (float): Probe speed in pc/s
        star_coordinates (DataFrame): DataFrame with x, y [pc] columns indexed by star IDs
        velocities_df (DataFrame): Required only for dynamic mode
        mode (str): "static" or "dynamic"

    Returns:
        stop_order (list): Ordered list of star indices
        total_time (float): Total path time (static = distance / v, dynamic = intercept time sum)
        dp_table (dict): Held-Karp memoization dictionary
    """
    n = len(selected_stops)
    idx_to_star = {i: selected_stops[i] for i in range(n)}
    star_to_idx = {v: k for k, v in idx_to_star.items()}

    # Build time matrix
    time_matrix = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            star_i = idx_to_star[i]
            star_j = idx_to_star[j]

            if mode == "static":
                x1, y1 = star_coordinates.loc[star_i]
                x2, y2 = star_coordinates.loc[star_j]
                d = compute_distances_between_objects(x1, y1, x2, y2)
                time_matrix[i][j] = d / step_size

            elif mode == "dynamic":
                if velocities_df is None:
                    raise ValueError("velocities_df required for dynamic mode.")
                _, t = compute_intercept_direction_and_time(
                    star_i, star_j,
                    coordinates_df=star_coordinates,
                    velocities_df=velocities_df,
                    probe_speed=step_size
                )
                if t is not None and t > 0:
                    time_matrix[i][j] = t
            else:
                raise ValueError("Mode must be 'static' or 'dynamic'.")

    # Held-Karp DP
    C = {}
    for k in range(1, n):
        C[(frozenset([k]), k)] = (time_matrix[0][k], [0, k])

    for s in range(2, n):
        for subset in combinations(range(1, n), s):
            S = frozenset(subset)
            for k in subset:
                prev_set = S - {k}
                min_cost = float('inf')
                best_path = []
                for m in prev_set:
                    cost, path = C.get((prev_set, m), (float('inf'), []))
                    new_cost = cost + time_matrix[m][k]
                    if new_cost < min_cost:
                        min_cost = new_cost
                        best_path = path + [k]
                C[(S, k)] = (min_cost, best_path)

    # Final best path ending at any stop
    full_set = frozenset(range(1, n))
    min_total = float('inf')
    final_path = []
    for k in range(1, n):
        cost, path = C[(full_set, k)]
        if cost < min_total:
            min_total = cost
            final_path = path

    stop_order = [idx_to_star[i] for i in final_path]
    total_time = min_total
    
    # Compute arrival times only for dynamic mode
    segment_arrival_times = []
    if mode == "dynamic":
        t = 0.0
        segment_arrival_times.append(t)  # Arrival at first star = 0
        for i in range(len(final_path) - 1):
            t += time_matrix[final_path[i]][final_path[i + 1]]
            segment_arrival_times.append(t)
            

    return stop_order, total_time, C, (segment_arrival_times if mode == "dynamic" else None)



#////////PHYSICS///////////////PHYSICS///////////////PHYSICS///////////////PHYSICS///////////////PHYSICS///////////////PHYSICS///////

def time_dilation_segments(segment_times, segment_velocities):
    proper_time = 0
    for t, v in zip(segment_times, segment_velocities):
        if v >= SPEED_OF_LIGHT:
            raise ValueError("Velocity must be less than the speed of light.")
        gamma = 1 / sqrt(1 - (v ** 2) / SPEED_OF_LIGHT ** 2)
        proper_time += t / gamma
    return proper_time


def relativistic_kinetic_energy(mass, velocity, c=SPEED_OF_LIGHT):
    if velocity >= c:
        raise ValueError("Velocity must be less than the speed of light.")
    gamma = 1 / sqrt(1 - (velocity ** 2) / c ** 2)
    return (gamma - 1) * mass * c ** 2


def relativistic_velocity_addition_2d(v_probe, v_star, c):
    vx, vy = v_probe
    ux, uy = v_star

    dot = (vx * ux + vy * uy) / c ** 2
    gamma = 1 / np.sqrt(1 - (ux ** 2 + uy ** 2) / c ** 2) if (ux ** 2 + uy ** 2) != 0 else 1.0

    denominator = 1 + dot
    if ux ** 2 + uy ** 2 == 0:
        vx_prime = vx
        vy_prime = vy
    else:
        vx_prime = (
            vx + ux + (gamma - 1) * (vx * ux + vy * uy) * ux / (ux ** 2 + uy ** 2)
        ) / denominator
        vy_prime = (
            vy + uy + (gamma - 1) * (vx * ux + vy * uy) * uy / (ux ** 2 + uy ** 2)
        ) / denominator

    return np.array([vx_prime, vy_prime])


def rotate_vector(v, angle_rad):
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    return rotation_matrix @ v


def get_star_velocity_vector(idx, parameters_df):
    parallax = parameters_df.loc[idx, 'parallax']
    pmra = parameters_df.loc[idx, 'pmra']
    pmdec = parameters_df.loc[idx, 'pmdec']
    ra = parameters_df.loc[idx, 'ra']
    dec = parameters_df.loc[idx, 'dec']
    rv_kms = parameters_df.loc[idx, 'radial_velocity'] if 'radial_velocity' in parameters_df.columns else np.nan

    if np.isnan(parallax) or parallax == 0:
        return np.array([0.0, 0.0])  # Avoid division by zero or missing data

    distance_pc = 1000 / parallax
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)

    # Unit vectors for 3D coordinate system
    los_unit = np.array([
        np.cos(dec_rad) * np.cos(ra_rad),
        np.cos(dec_rad) * np.sin(ra_rad),
        np.sin(dec_rad)
    ])
    ra_unit = np.array([
        -np.sin(ra_rad),
        np.cos(ra_rad),
        0.0
    ])
    dec_unit = np.array([
        -np.cos(ra_rad) * np.sin(dec_rad),
        -np.sin(ra_rad) * np.sin(dec_rad),
         np.cos(dec_rad)
    ])

    v_alpha = PM_CONVERSION * (pmra / 1000) * distance_pc  # km/s
    v_delta = PM_CONVERSION * (pmdec / 1000) * distance_pc  # km/s
    v_radial = rv_kms if not np.isnan(rv_kms) else 0.0      # km/s

    # Full 3D velocity vector
    v_3d = v_radial * los_unit + v_alpha * ra_unit + v_delta * dec_unit

    # Project to 2D Cartesian frame (x = RA-cosine, y = RA-sine)
    vx_kms = v_3d[0]
    vy_kms = v_3d[1]

    # Convert to pc/s
    vx_pc_s = vx_kms * KMS_TO_PC_PER_S
    vy_pc_s = vy_kms * KMS_TO_PC_PER_S

    return np.array([vx_pc_s, vy_pc_s])



#////////SLINGSHOT_PHYSICS///////////////SLINGSHOT_PHYSICS///////////////SLINGSHOT_PHYSICS///////////////SLINGSHOT_PHYSICS///////////////SLINGSHOT_PHYSICS////

def deflection_angle_from_path(path_indices, coordinates_df, velocities_of_stars, arrival_times=None):
    """
    Compute deflection angles for each slingshot segment.

    Parameters:
        path_indices (list): Indices of the stars in order of visitation.
        coordinates_df (DataFrame): Star positions (original at t=0).
        velocities_of_stars (DataFrame): Star velocity vectors.
        arrival_times (list or None): Arrival times at each star in seconds. If None, use static positions.

    Returns:
        List of tuples (star_idx, deflection_angle_deg)
    """
    results = []

    for i in range(1, len(path_indices) - 1):
        prev_idx = path_indices[i - 1]
        curr_idx = path_indices[i]
        next_idx = path_indices[i + 1]

        if arrival_times is None:
            # Static case: fixed positions
            a = coordinates_df.loc[prev_idx, ['x [pc]', 'y [pc]']].values
            b = coordinates_df.loc[curr_idx, ['x [pc]', 'y [pc]']].values
            c = coordinates_df.loc[next_idx, ['x [pc]', 'y [pc]']].values
        else:
            # Dynamic case: propagate to arrival time
            a = np.array(propagate_star_position(prev_idx, arrival_times[i - 1], coordinates_df, velocities_of_stars))
            b = np.array(propagate_star_position(curr_idx, arrival_times[i], coordinates_df, velocities_of_stars))
            c = np.array(propagate_star_position(next_idx, arrival_times[i + 1], coordinates_df, velocities_of_stars))

        # Get star velocity vector
        v_star = velocities_of_stars.loc[curr_idx, ['vx [pc/s]', 'vy [pc/s]']].values

        # Relative vectors
        vec_in = a - b
        vec_out = c - b

        # Transform into the star frame
        vec_in_star = vec_in - v_star
        vec_out_star = vec_out - v_star

        vec_in_unit = vec_in_star / np.linalg.norm(vec_in_star)
        vec_out_unit = vec_out_star / np.linalg.norm(vec_out_star)

        dot = np.clip(np.dot(vec_in_unit, vec_out_unit), -1.0, 1.0)
        theta_rad = np.arccos(dot)
        deflection_rad = np.pi - theta_rad
        deflection_deg = np.degrees(deflection_rad)

        results.append((curr_idx, deflection_deg))

    return results


def r0_periapsis_function(path_indices, parameters_df, deflection_angles_deg, velocity_magnitudes_list):
    """
    Compute periapsis distances for slingshot maneuvers.

    Returns:
        List of tuples (star_idx, r0_km)
    """
    results = []
    C = SPEED_OF_LIGHT * PC_TO_M  # speed of light in m/s

    for i in range(1, len(path_indices) - 1):
        idx = path_indices[i]
        theta_rad = np.radians(deflection_angles_deg[i - 1][1])
        v = velocity_magnitudes_list[i - 1] * PC_TO_M  # convert to m/s
        M = parameters_df.loc[idx, 'mass_flame'] * SOLAR_MASS

        if theta_rad == 0 or np.isnan(theta_rad) or np.isnan(v) or np.isnan(M):
            r0_km = float('inf')
        else:
            r0_m = (2 * GRAVITATIONAL_CONST * M / theta_rad) * (
                1 / C**2 + 1 / v**2
            )
            r0_km = r0_m * 1e-3

        results.append((idx, r0_km))

    return results



def apply_relativistic_gravity_assists(
    path_indices,
    deflection_data,
    coordinates_df,
    parameters_df,
    initial_velocity
):
    c = SPEED_OF_LIGHT
    velocities = []
    gains = []

    v_probe = initial_velocity * (
        coordinates_df.loc[path_indices[1], ['x [pc]', 'y [pc]']].values -
        coordinates_df.loc[path_indices[0], ['x [pc]', 'y [pc]']].values
    )
    v_probe = v_probe / np.linalg.norm(v_probe) * initial_velocity

    for i, (idx, deflection_deg) in enumerate(deflection_data):
        deflection_rad = np.radians(deflection_deg)

        p_prev = coordinates_df.loc[path_indices[i], ['x [pc]', 'y [pc]']].values
        p_curr = coordinates_df.loc[path_indices[i + 1], ['x [pc]', 'y [pc]']].values
        direction = p_curr - p_prev
        direction /= np.linalg.norm(direction)

        v_before = np.linalg.norm(v_probe)
        v_before_vector = v_before * direction

        v_star = get_star_velocity_vector(idx, parameters_df)

        v_in_star_frame = relativistic_velocity_addition_2d(
            v_before_vector, -v_star, c
        )
        v_rotated = rotate_vector(v_in_star_frame, deflection_rad)
        v_after_vector = relativistic_velocity_addition_2d(
            v_rotated, v_star, c
        )

        v_probe = v_after_vector
        gain = np.linalg.norm(v_after_vector) - np.linalg.norm(v_before_vector)

        velocities.append(v_after_vector.copy())
        gains.append(gain)

    return velocities, gains


#/////////UNCERTAINTIES///////////////UNCERTAINTIES///////////////UNCERTAINTIES///////////////UNCERTAINTIES///////////////UNCERTAINTIES//////

def uncertainties_deflection_angles(path_indices, parameters_df, arrival_times=None, velocities_df=None, n_samples=1000):
    results = []

    for i in range(1, len(path_indices) - 1):
        a_idx = path_indices[i - 1]
        b_idx = path_indices[i]
        c_idx = path_indices[i + 1]

        def sample_position(idx, t=None):
            ra = parameters_df.loc[idx, 'ra']
            dec = parameters_df.loc[idx, 'dec']
            parallax = parameters_df.loc[idx, 'parallax']

            ra_error_deg = parameters_df.loc[idx, 'ra_error'] * MAS_TO_DEG
            dec_error_deg = parameters_df.loc[idx, 'dec_error'] * MAS_TO_DEG
            parallax_error = parameters_df.loc[idx, 'parallax_error']

            ra_samples = np.random.normal(ra, ra_error_deg, size=n_samples)
            dec_samples = np.random.normal(dec, dec_error_deg, size=n_samples)
            parallax_samples = np.random.normal(parallax, parallax_error, size=n_samples)

            ra_rad = np.deg2rad(ra_samples)
            dec_rad = np.deg2rad(dec_samples)
            distances_pc = 1000 / parallax_samples

            x = distances_pc * np.cos(dec_rad) * np.cos(ra_rad)
            y = distances_pc * np.cos(dec_rad) * np.sin(ra_rad)

            if t is not None and velocities_df is not None:
                vx = velocities_df.loc[idx, 'vx [pc/s]']
                vy = velocities_df.loc[idx, 'vy [pc/s]']
                x += vx * t
                y += vy * t

            return np.vstack((x, y)).T

        # Sample positions
        if arrival_times is not None:
            b_samples = sample_position(b_idx, arrival_times[i])
            c_samples = sample_position(c_idx, arrival_times[i + 1])

            if i == 1:
                # First star after Sun: set a = [0, 0] with no uncertainty
                a_samples = np.zeros_like(b_samples)
            else:
                a_samples = sample_position(a_idx, arrival_times[i - 1])
        else:
            b_samples = sample_position(b_idx)
            c_samples = sample_position(c_idx)

            if i == 1:
                a_samples = np.zeros_like(b_samples)
            else:
                a_samples = sample_position(a_idx)

        # Compute deflection angles
        deflection_angles = []
        for j in range(n_samples):
            a = a_samples[j]
            b = b_samples[j]
            c = c_samples[j]

            vec_in = a - b
            vec_out = c - b

            vec_in_unit = vec_in / np.linalg.norm(vec_in)
            vec_out_unit = vec_out / np.linalg.norm(vec_out)

            dot = np.dot(vec_in_unit, vec_out_unit)
            theta = np.arccos(np.clip(dot, -1.0, 1.0))
            deflection = np.degrees(np.pi - theta)
            deflection_angles.append(deflection)

        std_dev = np.std(deflection_angles)
        results.append((b_idx, std_dev))

    return results


def uncertainties_probe_velocities(
    path_indices,
    deflection_data,
    angle_uncertainties,
    parameters_df,
    initial_velocity,
    n_samples=1000
):
    std_devs = []
    c = SPEED_OF_LIGHT
    deflection_rads = [np.radians(d[1]) for d in deflection_data]
    deflection_errors_rad = [np.radians(d[1]) for d in angle_uncertainties]

    def sample_star_position(idx):
        ra = parameters_df.loc[idx, 'ra']
        dec = parameters_df.loc[idx, 'dec']
        parallax = parameters_df.loc[idx, 'parallax']

        ra_error_deg = parameters_df.loc[idx, 'ra_error'] * MAS_TO_DEG
        dec_error_deg = parameters_df.loc[idx, 'dec_error'] * MAS_TO_DEG
        parallax_error = parameters_df.loc[idx, 'parallax_error']

        ra_samples = np.random.normal(ra, ra_error_deg, size=n_samples)
        dec_samples = np.random.normal(dec, dec_error_deg, size=n_samples)
        parallax_samples = np.random.normal(parallax, parallax_error, size=n_samples)

        ra_rad = np.deg2rad(ra_samples)
        dec_rad = np.deg2rad(dec_samples)
        distances_pc = 1000 / parallax_samples

        x = distances_pc * np.cos(dec_rad) * np.cos(ra_rad)
        y = distances_pc * np.cos(dec_rad) * np.sin(ra_rad)
        return np.vstack((x, y)).T

    def sample_star_velocity(idx):
        parallax = parameters_df.loc[idx, 'parallax']
        parallax_error = parameters_df.loc[idx, 'parallax_error']
        pmra = parameters_df.loc[idx, 'pmra']
        pmra_error = parameters_df.loc[idx, 'pmra_error']
        pmdec = parameters_df.loc[idx, 'pmdec']
        pmdec_error = parameters_df.loc[idx, 'pmdec_error']
        rv = parameters_df.loc[idx, 'radial_velocity']
        rv_error = parameters_df.loc[idx, 'radial_velocity_error']

        # Sample observables
        parallax_samples = np.random.normal(parallax, parallax_error, size=n_samples)
        pmra_samples = np.random.normal(pmra, pmra_error, size=n_samples)
        pmdec_samples = np.random.normal(pmdec, pmdec_error, size=n_samples)
        rv_samples = (
            np.random.normal(rv, rv_error, size=n_samples)
            if not np.isnan(rv) else np.zeros(n_samples)
        )

        distance_pc = 1000 / parallax_samples

        # Sampled RA/Dec (from existing sample_star_position)
        ra = parameters_df.loc[idx, 'ra']
        dec = parameters_df.loc[idx, 'dec']
        ra_error_deg = parameters_df.loc[idx, 'ra_error'] * MAS_TO_DEG
        dec_error_deg = parameters_df.loc[idx, 'dec_error'] * MAS_TO_DEG

        ra_samples = np.random.normal(ra, ra_error_deg, size=n_samples)
        dec_samples = np.random.normal(dec, dec_error_deg, size=n_samples)
        ra_rad = np.deg2rad(ra_samples)
        dec_rad = np.deg2rad(dec_samples)

        # Unit vectors
        los_unit = np.column_stack([
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad)
            ])
        ra_unit = np.column_stack([
            -np.sin(ra_rad),
            np.cos(ra_rad),
            np.zeros(n_samples)
            ])
        dec_unit = np.column_stack([
            -np.cos(ra_rad) * np.sin(dec_rad),
            -np.sin(ra_rad) * np.sin(dec_rad),
            np.cos(dec_rad)
            ])

        # Compute tangential velocities
        v_alpha = PM_CONVERSION * (pmra_samples / 1000) * distance_pc
        v_delta = PM_CONVERSION * (pmdec_samples / 1000) * distance_pc

        v_tan_vecs = ra_unit * v_alpha[:, np.newaxis] + dec_unit * v_delta[:, np.newaxis]
        v_radial_vecs = los_unit * rv_samples[:, np.newaxis]

        v_total_3d = v_tan_vecs + v_radial_vecs

        # Project to 2D (X, Y plane)
        vx_kms = v_total_3d[:, 0]
        vy_kms = v_total_3d[:, 1]

        vx_pc_s = vx_kms * KMS_TO_PC_PER_S
        vy_pc_s = vy_kms * KMS_TO_PC_PER_S

        return np.column_stack([vx_pc_s, vy_pc_s])

    samples = {}
    for idx in set(path_indices):
        if idx == 0:
            samples[idx] = np.tile(np.array([[0.0, 0.0]]), (n_samples, 1))
        else:
            samples[idx] = sample_star_position(idx)

    v_probes = np.tile(initial_velocity, (n_samples, 1))

    for i, ((idx, _), deflection_rad, deflection_error_rad) in enumerate(
        zip(deflection_data, deflection_rads, deflection_errors_rad)
    ):
        a_idx = path_indices[i]
        b_idx = path_indices[i + 1]


        star_velocity_samples = sample_star_velocity(idx)
        deflection_samples = np.random.normal(
            deflection_rad, deflection_error_rad, size=n_samples
        )

        new_v_probes = []
        for j in range(n_samples):
            p_prev = samples[a_idx][j]
            p_curr = samples[b_idx][j]
            direction = p_curr - p_prev
            direction /= np.linalg.norm(direction)
            v_before = np.linalg.norm(v_probes[j]) * direction

            v_star = star_velocity_samples[j]

            v_in_star_frame = relativistic_velocity_addition_2d(
                v_before, -v_star, c
            )
            v_rotated = rotate_vector(v_in_star_frame, deflection_samples[j])
            v_after = relativistic_velocity_addition_2d(v_rotated, v_star, c)

            new_v_probes.append(v_after)

        new_v_probes = np.array(new_v_probes)
        mean_v = np.mean(new_v_probes, axis=0)
        std_v = np.std(new_v_probes, axis=0)
        std_speed = np.std(np.linalg.norm(new_v_probes, axis=1))

        std_devs.append((std_v[0], std_v[1], std_speed))
        v_probes = new_v_probes

    return std_devs


def uncertainties_r0_periapsis(
    deflection_data,
    angle_uncertainties,
    velocity_uncertainties,
    velocities,
    parameters_df
):
    """
    Compute uncertainties in periapsis distances r0.

    Returns:
        List of tuples (star_idx, sigma_r0_km)
    """
    results = []

    for i, (idx, deflection_deg) in enumerate(deflection_data):
        θ = np.radians(deflection_deg)
        σ_θ = np.radians(angle_uncertainties[i][1])
        v = np.linalg.norm(velocities[i]) * PC_TO_M
        σ_v = velocity_uncertainties[i][2] * PC_TO_M

        M = parameters_df.loc[idx, 'mass_flame'] * SOLAR_MASS
        M_upper = parameters_df.loc[idx, 'mass_flame_upper'] * SOLAR_MASS
        M_lower = parameters_df.loc[idx, 'mass_flame_lower'] * SOLAR_MASS
        σ_M = (M_upper - M_lower) / 2

        C = SPEED_OF_LIGHT * PC_TO_M
        term = (1 / C**2) + (1 / v**2)

        # Partial derivatives
        dr0_dM = (2 * GRAVITATIONAL_CONST / θ) * term
        dr0_dθ = -(2 * GRAVITATIONAL_CONST * M / θ**2) * term
        dr0_dv = (2 * GRAVITATIONAL_CONST * M / θ) * (-2 / v**3)

        σ_r0_squared = (
            (dr0_dM * σ_M)**2 +
            (dr0_dθ * σ_θ)**2 +
            (dr0_dv * σ_v)**2
        )

        σ_r0_km = np.sqrt(σ_r0_squared) / 1e3
        results.append((idx, σ_r0_km))

    return results


#///////SIMULATION_AND_PLOTTING//////////////SIMULATION_AND_PLOTTING//////////////SIMULATION_AND_PLOTTING//////////////SIMULATION_AND_PLOTTING///////

def setup_star_map(ax, radius: float):
    """
    Set up the starfield plot background and render all stars with scaling and colors.
    """
    ax.set_facecolor("black")

    radii_scaled = radius * stars_parameters['radius_gspphot'] / stars_parameters['radius_gspphot'].max()

    scatter = ax.scatter(
        stars_coordinates['x [pc]'],
        stars_coordinates['y [pc]'],
        s=radii_scaled,
        c=stars_parameters['teff_gspphot'],
        cmap='plasma',
        norm=mcolors.Normalize(
            vmin=stars_parameters['teff_gspphot'].min(),
            vmax=stars_parameters['teff_gspphot'].max()
        ),
        alpha=0.85,
        edgecolors='none'
    )

    return scatter


def simulate_probe_path(best_order, step_size: float, space_size: float, velocities_df, path_type: str, arrival_times=None):

    deflection_angles = deflection_angle_from_path(
        path_indices=best_order,
        coordinates_df=stars_coordinates,
        velocities_of_stars=velocities_df,
        arrival_times=arrival_times
    )

    angle_uncertainties = uncertainties_deflection_angles(
        path_indices=best_order,
        parameters_df=stars_parameters,
        arrival_times=arrival_times,
        velocities_df=velocities_df,
        n_samples=1000
    )

    # Determine correct initial velocity
    if path_type == "dynamic":
        direction_vector, _ = compute_intercept_direction_and_time(
            best_order[0], best_order[1],
            coordinates_df=stars_coordinates,
            velocities_df=velocities_df,
            probe_speed=step_size
        )
        v_probe_init = direction_vector * step_size  # velocity vector for dynamic case
    else:
        v_probe_init = step_size  # scalar for static

    velocities, velocity_gains = apply_relativistic_gravity_assists(
        path_indices=best_order,
        deflection_data=deflection_angles,
        coordinates_df=stars_coordinates,
        parameters_df=stars_parameters,
        initial_velocity=v_probe_init
    )

    velocity_uncertainties = uncertainties_probe_velocities(
        path_indices=best_order,
        deflection_data=deflection_angles,
        angle_uncertainties=angle_uncertainties,
        parameters_df=stars_parameters,
        initial_velocity=v_probe_init,
        n_samples=1000
    )

    velocity_magnitudes_pc_s = [np.linalg.norm(v) for v in velocities]

    r0_periapsis = r0_periapsis_function(
        path_indices=best_order,
        parameters_df=stars_parameters,
        deflection_angles_deg=deflection_angles,
        velocity_magnitudes_list=velocity_magnitudes_pc_s
    )

    r0_uncertainties = uncertainties_r0_periapsis(
        deflection_data=deflection_angles,
        angle_uncertainties=angle_uncertainties,
        velocity_uncertainties=velocity_uncertainties,
        velocities=velocities,
        parameters_df=stars_parameters
    )

    if not check_path_validity(r0_periapsis, r0_uncertainties, stars_parameters):
        return None

    # --- Time dilation and energy ---
    if path_type == "dynamic":
        velocity_magnitudes = [np.linalg.norm(velocities[0])] + [np.linalg.norm(v) for v in velocities]
        segment_distances = []
        observer_time_segments = []

        for i in range(len(best_order) - 1):
            t0 = arrival_times[i]
            t1 = arrival_times[i + 1]

            x0, y0 = propagate_star_position(best_order[i], t0, stars_coordinates, velocities_df)
            x1, y1 = propagate_star_position(best_order[i + 1], t1, stars_coordinates, velocities_df)

            d = np.linalg.norm([x1 - x0, y1 - y0])
            segment_distances.append(d)
            observer_time_segments.append(t1 - t0)

    else:
        velocity_magnitudes = [step_size] + [np.linalg.norm(v) for v in velocities]
        segment_distances = [
            np.linalg.norm(
                stars_coordinates.loc[best_order[i + 1]][['x [pc]', 'y [pc]']].values -
                stars_coordinates.loc[best_order[i]][['x [pc]', 'y [pc]']].values
            )
            for i in range(len(best_order) - 1)
        ]
        observer_time_segments = [
            d / v for d, v in zip(segment_distances, velocity_magnitudes)
        ]

    proper_time_segments = [
        t / (1 / np.sqrt(1 - (v ** 2 / SPEED_OF_LIGHT ** 2)))
        for t, v in zip(observer_time_segments, velocity_magnitudes)
    ]

    observer_time = np.sum(observer_time_segments)
    proper_time = np.sum(proper_time_segments)

    delta_time = observer_time - proper_time
    observer_years = observer_time / SECONDS_IN_YEAR
    probe_years = proper_time / SECONDS_IN_YEAR
    diff_years = delta_time / SECONDS_IN_YEAR
    final_velocity = velocity_magnitudes[-1]
    energy_gain = relativistic_kinetic_energy(PROBE_MASS, final_velocity)

    print(f"\n===== {path_type.upper()} PATH =====")
    print(f"Initial velocity: {step_size:.2e} pc/s")
    print(f"Observer time: {observer_time:.2f} s ({observer_years:.2f} yr)")
    print(f"Probe proper time: {proper_time:.2f} s ({probe_years:.2f} yr)")
    print(f"Time dilation: {delta_time:.2f} s ({diff_years:.2f} yr)")

    return {
        "best_order": best_order,
        "velocities": velocities,
        "observer_time": observer_time,
        "proper_time": proper_time,
        "observer_years": observer_years,
        "probe_years": probe_years,
        "time_dilation_seconds": delta_time,
        "time_dilation_years": diff_years,
        "final_velocity": final_velocity,
        "energy_gain": energy_gain,
        "r0_periapsis": r0_periapsis,
        "r0_uncertainties": r0_uncertainties,
        "deflection_angles": deflection_angles,
        "angle_uncertainties": angle_uncertainties,
        "velocity_uncertainties": velocity_uncertainties,
        "velocity_gains": velocity_gains,
        "segment_distances": segment_distances
    }


def plotting_function(
    ax,
    scatter_obj,
    step_size: float,
    coords_static: pd.DataFrame,
    coords_dynamic: pd.DataFrame,
    space_size: float,
    total_dynamic_time: float,
    velocities_df: pd.DataFrame,
    seed: int  
):

    offset = space_size * 0.02

    # --- Plot stars as dashed open circles at static positions ---
    ax.scatter(
        coords_static["x [pc]"],
        coords_static["y [pc]"],
        facecolors='none',
        edgecolors='white',
        linestyle='--',
        linewidths=0.8,
        s=60,
        label="Stars (Static)"
    )

    # --- Plot static probe path (white solid line) ---
    ax.plot(
        coords_static["x [pc]"],
        coords_static["y [pc]"],
        color='white',
        linestyle='--',
        linewidth=1.5,
        marker='x',
        markersize=4,
        alpha = 0.5,
        label="Static Path"
    )

    # --- Plot dynamic probe path (cyan solid line) ---
    ax.plot(
        coords_dynamic["x [pc]"],
        coords_dynamic["y [pc]"],
        color='cyan',
        linestyle='-',
        linewidth=2,
        marker='o',
        markersize=4,
        label="Dynamic Path"
    )

    # --- Draw green dash-dot arrows from static to dynamic positions ---
    for (x0, y0), (x1, y1) in zip(coords_static.values, coords_dynamic.values):
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="green", linestyle='-.'), zorder=4
        )

    # --- Label dynamic path stops with orange dots and numbers ---
    for i, row in coords_dynamic.iterrows():
        x = row["x [pc]"]
        y = row["y [pc]"]
        ax.scatter(x, y, color="orange", s=40, zorder=5)
        ax.text(x + offset, y + offset, str(i + 1), color="orange", fontsize=9, weight="bold")

    # --- Sun at origin (bigger red point) ---
    ax.scatter(0, 0, color='red', s=80, label='Sun', zorder=5)

    # --- Axes appearance ---
    ax.set_facecolor("black")
    ax.set_xlabel('x [pc]', color='white')
    ax.set_ylabel('y [pc]', color='white')
    ax.set_title(f'Star Map at Arrival Time (t = {total_dynamic_time / SECONDS_IN_YEAR:.1f} yr)', color='white')
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    plt.grid(True, linestyle='--', alpha=0.3)

    # --- Colorbar ---
    cbar = plt.colorbar(scatter_obj)
    cbar.set_label('Temperature [K]', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # --- Save ---
    os.makedirs("Diagrams/Pathfinding_diagrams", exist_ok=True)
    plt.savefig(f"Diagrams/Pathfinding_diagrams/Simulation_STATIC_and_DYNAMIC_STEP_SIZE={step_size}_seed={seed}.png")
    plt.close()



def create_space(step_size, radius, space_size, stars_velocities, stops, seed):
    if N_STOPS == 0:
        print("You stay at the origin (N_STOPS = 0)")
        return None, None

    # Get optimal stop orders
    path_static, time_static, C_static, _ = find_best_stop_order(
        stops, step_size, stars_coordinates, stars_velocities, mode="static"
    )

    path_dynamic, time_dynamic, C_dynamic, dynamic_arrival_times = find_best_stop_order(
        stops, step_size, stars_coordinates, stars_velocities, mode="dynamic"
    )

    # --- STEP 3: Build coordinate paths ---
    coords_static = stars_coordinates.loc[path_static][['x [pc]', 'y [pc]']].copy()
    coords_dynamic = [
        propagate_star_position(idx, t, stars_coordinates, stars_velocities)
        for idx, t in zip(path_dynamic, dynamic_arrival_times)
        ]
    coords_dynamic = pd.DataFrame(coords_dynamic, columns=['x [pc]', 'y [pc]'])


    # Simulate both paths
    result_static = simulate_probe_path(
        best_order=path_static,
        step_size=step_size,
        space_size=space_size,
        velocities_df=stars_velocities,
        path_type="static",
        arrival_times=None
    )


    result_dynamic = simulate_probe_path(
        best_order=path_dynamic,
        step_size=step_size,
        space_size=space_size,
        velocities_df=stars_velocities,
        path_type="dynamic",
        arrival_times=dynamic_arrival_times
    )




    if result_static:
        log_probe_path(
            result_static, step_size, "static",
            coordinates_df=stars_coordinates,
            velocities_df=stars_velocities,
            arrival_times=None,
            dp_table=C_static,
            seed = seed
            )


    if result_dynamic:
        log_probe_path(
            result_dynamic, step_size, "dynamic",
            coordinates_df=stars_coordinates,
            velocities_df=stars_velocities,
            arrival_times=dynamic_arrival_times,
            dp_table=C_dynamic,
            seed = seed
            )


    # Plot if both are valid
    if result_static and result_dynamic:
        fig = plt.subplots(figsize=(10, 8), facecolor='black')
        ax = plt.gca()
        sc = setup_star_map(ax, radius)

        plotting_function(
            ax=ax,
            scatter_obj=sc,
            step_size=step_size,
            coords_static=coords_static,
            coords_dynamic=coords_dynamic,
            space_size=space_size,
            total_dynamic_time=result_dynamic["observer_time"],
            velocities_df=stars_velocities,
            seed=seed  # <-- add this
            )


    return result_static, result_dynamic


def log_probe_path(result: dict, step_size: float, path_type: str, coordinates_df, velocities_df, arrival_times, dp_table, seed):
    """
    Logs all data about a probe path to a text file.
    """
    filename = f"Logs/Simulation_PF_{path_type.upper()}_STEP_SIZE={step_size}_seed={seed}.txt"
    os.makedirs("Logs", exist_ok=True)

    with open(filename, "w") as f:
        best_order = result["best_order"]

        f.write(f"Path Type: {path_type.upper()}\n")
        f.write(f"Number of stops: {len(best_order) - 1}\n")
        f.write(f"Order of stops (indices): {best_order}\n")

        f.write("\n--- Coordinates of Stops ---\n")
        for i, idx in enumerate(best_order):
            if path_type == "static":
                x, y = coordinates_df.loc[idx, ['x [pc]', 'y [pc]']]
                f.write(f"STOP {i}: idx={idx}, x={x:.5f} pc, y={y:.5f} pc\n")
            else:
                t = arrival_times[i]
                vx, vy = velocities_df.loc[idx, ['vx [pc/s]', 'vy [pc/s]']]
                x = coordinates_df.loc[idx, 'x [pc]'] + vx * t
                y = coordinates_df.loc[idx, 'y [pc]'] + vy * t
                f.write(f"STOP {i}: idx={idx}, x={x:.5f} pc, y={y:.5f} pc, arrival_time={t:.2f} s\n")

        f.write(f"\n--- Kinematics ---\n")
        f.write(f"Initial velocity: {step_size * PC_PER_S_TO_KM_PER_H:.3e} km/h\n")
        f.write(f"Final velocity: {result['final_velocity'] * PC_PER_S_TO_KM_PER_H:.3e} km/h\n")
        f.write(f"Energy gain: {result['energy_gain']:.3e} J\n")

        f.write(f"\n--- Time ---\n")
        f.write(f"Observer time: {result['observer_time']:.2f} s ({result['observer_years']:.2f} yr)\n")
        f.write(f"Probe proper time: {result['proper_time']:.2f} s ({result['probe_years']:.2f} yr)\n")
        f.write(f"Time dilation: {result['time_dilation_seconds']:.2f} s ({result['time_dilation_years']:.2f} yr)\n")

        f.write(f"\n--- Segment Details ---\n")
        for i in range(len(result["r0_periapsis"])):
            idx = result["r0_periapsis"][i][0]
            r0 = result["r0_periapsis"][i][1]
            angle = result["deflection_angles"][i][1]
            angle_unc = result["angle_uncertainties"][i][1]
            v_vec = result["velocities"][i]
            gain = result["velocity_gains"][i]
            vx_unc, vy_unc, v_unc = result["velocity_uncertainties"][i]
            r0_unc = result["r0_uncertainties"][i][1]

            v_kmh = np.linalg.norm(v_vec) * PC_PER_S_TO_KM_PER_H
            v_unc_kmh = v_unc * PC_PER_S_TO_KM_PER_H
            gain_kmh = gain * PC_PER_S_TO_KM_PER_H

            # --- Distance and time ---
            if path_type == "dynamic":
                t0 = arrival_times[i]
                t1 = arrival_times[i + 1]

                x0, y0 = coordinates_df.loc[best_order[i], ['x [pc]', 'y [pc]']]
                vx0, vy0 = velocities_df.loc[best_order[i], ['vx [pc/s]', 'vy [pc/s]']]
                x1, y1 = coordinates_df.loc[best_order[i + 1], ['x [pc]', 'y [pc]']]
                vx1, vy1 = velocities_df.loc[best_order[i + 1], ['vx [pc/s]', 'vy [pc/s]']]

                pos0 = np.array([x0 + vx0 * t0, y0 + vy0 * t0])
                pos1 = np.array([x1 + vx1 * t1, y1 + vy1 * t1])
                dist = np.linalg.norm(pos1 - pos0)

                observer_time = t1 - t0
            else:
                dist = result["segment_distances"][i]
                observer_time = dist / np.linalg.norm(v_vec)

            gamma = 1 / np.sqrt(1 - (np.linalg.norm(v_vec) ** 2 / SPEED_OF_LIGHT ** 2))
            proper_time = observer_time / gamma

            f.write(
                f"\nSTOP {i+1} (index {idx}):\n"
                f"  deflection_angle = {angle:.3f} ± {angle_unc:.3f} deg\n"
                f"  periapsis r0 = {r0:.4e} ± {r0_unc:.2e} km\n"
                f"  |v| = {v_kmh:.3e} ± {v_unc_kmh:.3e} km/h\n"
                f"  σ_vx = {vx_unc:.3e} km/h, σ_vy = {vy_unc:.3e} km/h\n"
                f"  Δv (gain) = {gain_kmh:.3e} km/h\n"
                f"  distance (from previous stop) = {dist:.4f} pc\n"
                f"  travel time from previous stop (observer): {observer_time:.2f} s ({observer_time / SECONDS_IN_YEAR:.2f} yr)\n"
                f"  travel time from previous stop (probe): {proper_time:.2f} s ({proper_time / SECONDS_IN_YEAR:.2f} yr)\n"
            )

        # --- Formatted DP Table ---
        f.write(f"\n=== HELD-KARP DP TABLE ===\n")
        for key, value in sorted(dp_table.items(), key=lambda kv: (len(kv[0][0]), sorted(kv[0][0]), kv[0][1])):
            subset, end_idx = key
            g, path = value
            subset_str = "{" + ",".join(str(i) for i in sorted(subset)) + "}"
            path_str = " → ".join(str(i) for i in path)
            f.write(f"S = {subset_str:<10}, end = {end_idx}, g = {g:.2f}, path = [{path_str}]\n")



#/////////MAIN_CODE//////////////MAIN_CODE//////////////MAIN_CODE//////////////MAIN_CODE//////////////MAIN_CODE//////////////MAIN_CODE/////
def main(seed_range=None):
    radius, space_size, stars_velocities = take_stars_from_GAIA_data()

    for step in STEP_SIZES:
        if seed_range is not None:
            for seed in range(seed_range):
                print(f"\nRunning simulation with seed = {seed}")
                stops = generate_stops(N_STOPS, stars_coordinates.iloc[1:], seed)
                create_space(step, radius, space_size, stars_velocities, stops, seed)
        else:
            stops = generate_stops(N_STOPS, stars_coordinates.iloc[1:])
            create_space(step, radius, space_size, stars_velocities, stops, None)

if __name__ == "__main__":
    main(SEED_RANGE)
