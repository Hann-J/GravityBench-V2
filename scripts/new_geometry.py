"""
This file is for future extensions
"""

import numpy as np
import pandas as pd

def star_masses(df, binary_sim, verification=True, return_empirical=False):
    """
    Calculate the masses of star1 (M1) and star2 (M2) using Newton's law of gravitation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing simulation data with position columns for both stars
    binary_sim : object
        Simulation object containing gravitational constant G and reference masses for verification
    verification : bool, optional
        Whether to verify calculated masses against reference values (default True)
    return_empirical : bool, optional
        If True, return empirically calculated values, if False return simulation values (default False)
    
    Returns:
    --------
    tuple
        (M1, M2) masses of star1 and star2 in simulation units
    """
    # Calculate separation between the stars
    separation = np.sqrt((df['star1_x'] - df['star2_x'])**2 + 
                         (df['star1_y'] - df['star2_y'])**2 + 
                         (df['star1_z'] - df['star2_z'])**2)
    
    # Calculate acceleration of star2
    acceleration_star2x = np.gradient(np.gradient(df['star2_x'], df['time']), df['time'])
    acceleration_star2y = np.gradient(np.gradient(df['star2_y'], df['time']), df['time'])
    acceleration_star2z = np.gradient(np.gradient(df['star2_z'], df['time']), df['time'])
    acceleration_star2 = np.sqrt(acceleration_star2x**2 + acceleration_star2y**2 + acceleration_star2z**2)
    
    # Calculate acceleration of star1
    acceleration_star1x = np.gradient(np.gradient(df['star1_x'], df['time']), df['time'])
    acceleration_star1y = np.gradient(np.gradient(df['star1_y'], df['time']), df['time'])
    acceleration_star1z = np.gradient(np.gradient(df['star1_z'], df['time']), df['time'])
    acceleration_star1 = np.sqrt(acceleration_star1x**2 + acceleration_star1y**2 + acceleration_star1z**2)
    
    # Calculate masses
    M1 = np.median(acceleration_star2 * separation**2 / binary_sim.sim.G)
    M2 = np.median(acceleration_star1 * separation**2 / binary_sim.sim.G)
    if verification:
        assert abs(M1 - binary_sim.star1_mass) < 0.02 * binary_sim.star1_mass, f"{M1} and {binary_sim.star1_mass} are not within 2% of each other"
        assert abs(M2 - binary_sim.star2_mass) < 0.02 * binary_sim.star2_mass, f"{M2} and {binary_sim.star2_mass} are not within 2% of each other"
    
    if return_empirical:
        return M1, M2
    else:
        return binary_sim.star1_mass, binary_sim.star2_mass

def calculate_velocities(df, binary_sim, verification=True, return_empirical=False):
    """
    Calculate velocities of both stars using finite differences of positions or return stored values.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing simulation data
    return_empirical : bool, optional
        If True, calculate velocities empirically, if False use stored values (default False)

    Returns:
    --------
    tuple
        Six arrays in order:
        (star1_vx, star1_vy, star1_vz, star2_vx, star2_vy, star2_vz)
        representing velocity components of both stars
    """
    if return_empirical:
        velocities = {}
        for star in ['star1', 'star2']:
            for axis in ['x', 'y', 'z']:
                velocities[f'{star}_v{axis}'] = np.gradient(df[f'{star}_{axis}'], df['time'])
    else:
        velocities = {
            'star1_vx': df['star1_vx'],
            'star1_vy': df['star1_vy'],
            'star1_vz': df['star1_vz'],
            'star2_vx': df['star2_vx'],
            'star2_vy': df['star2_vy'],
            'star2_vz': df['star2_vz']
        }

    if verification and return_empirical:
        for star in ['star1', 'star2']:
            for axis in ['x', 'y', 'z']:
                v_calc = velocities[f'{star}_v{axis}']
                v_stored = df[f'{star}_v{axis}']
                percent_diff = (v_calc - v_stored) / v_stored
                assert np.abs(np.mean(percent_diff)) < 0.02, f"{star} velocity {axis} component differs by more than 2%"

    return (velocities['star1_vx'], velocities['star1_vy'], velocities['star1_vz'],
            velocities['star2_vx'], velocities['star2_vy'], velocities['star2_vz'])

def calculate_accelerations(df, binary_sim, verification=True, return_empirical=False):
    """
    Calculate total acceleration magnitudes or return stored values.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing simulation data
    binary_sim : object
        Simulation object (unused but kept for consistency)
    return_empirical : bool, optional
        If True, calculate accelerations empirically, if False use stored values (default False)

    Returns:
    --------
    tuple
        (acc_star1, acc_star2) containing arrays of acceleration magnitudes
        for star1 and star2 over time
    """
    if return_empirical:
        total_accelerations = {}
        for star in ['star1', 'star2']:
            acc_x = np.gradient(np.gradient(df[f'{star}_x'], df['time']), df['time'])
            acc_y = np.gradient(np.gradient(df[f'{star}_y'], df['time']), df['time'])
            acc_z = np.gradient(np.gradient(df[f'{star}_z'], df['time']), df['time'])
            total_accelerations[star] = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    else:
        total_accelerations = {
            'star1': df['star1_accel'],
            'star2': df['star2_accel']
        }

    if verification and return_empirical:
        for star in ['star1', 'star2']:
            acc_calc = total_accelerations[star]
            acc_stored = df[f'{star}_accel']
            if 'Modified Gravity' not in binary_sim.filename and 'Drag' not in binary_sim.filename:
                percent_diff = (acc_calc - acc_stored) / acc_stored
                assert np.abs(np.mean(percent_diff)) < 0.02, f"{star} acceleration differs by more than 2%"

    return total_accelerations['star1'], total_accelerations['star2']


def calculate_semi_major_axes(df, M1, M2, binary_sim, verification=True, return_empirical=False):
    """
    Calculate semi-major axes for the binary system and individual stars.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing simulation data with position columns
    M1 : float
        Mass of star1
    M2 : float
        Mass of star2
    binary_sim : object
        Simulation object containing reference values for verification
    verification : bool, optional
        Whether to verify against rebound calculations (default True)
    return_empirical : bool, optional
        If True, return empirically calculated values, if False return simulation values (default False)

    Returns:
    --------
    tuple
        (a_total, a_star1, a_star2) containing:
        - a_total: Total semi-major axis of the binary system
        - a_star1: Semi-major axis of star1's orbit around center of mass
        - a_star2: Semi-major axis of star2's orbit around center of mass
    """
    # Calculate the total semi-major axis
    distances = np.sqrt(
        (df['star2_x'] - df['star1_x'])**2 +
        (df['star2_y'] - df['star1_y'])**2 +
        (df['star2_z'] - df['star1_z'])**2
    )
    semi_major_axis_total = (np.max(distances) + np.min(distances)) / 2

    # Calculate semi-major axes of star1 and star2
    semi_major_axis_star1 = (M2 / (M1 + M2)) * semi_major_axis_total
    semi_major_axis_star2 = (M1 / (M1 + M2)) * semi_major_axis_total

    semi_major_axis_star_1_rebound = df['semimajor_axis'].iloc[0] * binary_sim.star2_mass / (binary_sim.star1_mass + binary_sim.star2_mass)
    semi_major_axis_star_2_rebound = df['semimajor_axis'].iloc[0] * binary_sim.star1_mass / (binary_sim.star1_mass + binary_sim.star2_mass)

    if verification:
        assert abs(semi_major_axis_star1 - semi_major_axis_star_1_rebound) < 0.02 * semi_major_axis_star_1_rebound, f"{semi_major_axis_star1} and {semi_major_axis_star_1_rebound} are not within 2% of each other"
        assert abs(semi_major_axis_star2 - semi_major_axis_star_2_rebound) < 0.02 * semi_major_axis_star_2_rebound, f"{semi_major_axis_star2} and {semi_major_axis_star_2_rebound} are not within 2% of each other"
        assert abs(semi_major_axis_total - df['semimajor_axis'].iloc[0]) < 0.02 * df['semimajor_axis'].iloc[0], f"{semi_major_axis_total} and {df['semimajor_axis'].iloc[0]} are not within 2% of each other"
    
    if return_empirical:
        return semi_major_axis_total, semi_major_axis_star1, semi_major_axis_star2
    else:
        return df['semimajor_axis'].iloc[0], semi_major_axis_star_1_rebound, semi_major_axis_star_2_rebound


def calculate_eccentricity(df, binary_sim, verification=True, return_empirical=False):
    """
    Calculate orbital eccentricity using maximum and minimum separations.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing simulation data with position columns
    binary_sim : object
        Simulation object containing initial conditions for verification
    verification : bool, optional
        Whether to verify against rebound calculations (default True)
    return_empirical : bool, optional
        If True, return empirically calculated value, if False return simulation value (default False)

    Returns:
    --------
    float
        Orbital eccentricity of the binary system
    """
    distances = np.sqrt(
        (df['star1_x'] - df['star2_x'])**2 +
        (df['star1_y'] - df['star2_y'])**2 +
        (df['star1_z'] - df['star2_z'])**2
    )
    r_max = np.max(distances)
    r_min = np.min(distances)
    eccentricity = (r_max - r_min) / (r_max + r_min)
    if verification:
        assert abs(eccentricity - df['eccentricity'].iloc[0]) < 0.02 * df['eccentricity'].iloc[0], f"{eccentricity} and {df['eccentricity'].iloc[0]} are not within 2% of each other"

    if return_empirical:
        return eccentricity
    else:
        return df['eccentricity'].iloc[0]

import scipy.signal
def calculate_period(df, binary_sim, verification=True, return_empirical=False):
    """
    Calculate orbital period by analyzing separation distance peaks.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing simulation data with position and time columns
    binary_sim : object
        Simulation object containing initial conditions for verification
    verification : bool, optional
        Whether to verify against rebound calculations (default True)
    return_empirical : bool, optional
        If True, return empirically calculated value, if False return simulation value (default False)

    Returns:
    --------
    float
        Orbital period of the binary system

    Raises:
    -------
    ValueError
        If insufficient peaks are found to calculate period
    """
    separation = np.sqrt(
        (df['star1_x'] - df['star2_x'])**2 +
        (df['star1_y'] - df['star2_y'])**2 +
        (df['star1_z'] - df['star2_z'])**2
    )
    peaks, _ = scipy.signal.find_peaks(separation)
    peak_times = df['time'].iloc[peaks]

    if len(peak_times) < 2:
        raise ValueError("Not enough peaks found to calculate period.")

    periods = peak_times.diff().dropna()
    period = periods.mean()
    
    if verification:
        assert abs(period - df['orbital_period'].iloc[0]) < 0.02 * df['orbital_period'].iloc[0], f"{period} and {df['orbital_period'].iloc[0]} are not within 2% of each other"
    
    if return_empirical:
        return period
    else:
        return df['orbital_period'].iloc[0]

def calculate_time_of_pericenter_passage(df, binary_sim, verification=True, return_empirical=False):
    """
    Calculate time of pericenter passage by finding minimum separation within first orbital period,
    or return stored value.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing simulation data with position and time columns
    binary_sim : object
        Simulation object containing initial conditions
    verification : bool, optional
        Whether to verify results (default True)
    return_empirical : bool, optional
        If True, calculate empirically, if False use stored value (default False)

    Returns:
    --------
    float
        Time of pericenter passage
    """

    
    # Get orbital period
    period = calculate_period(df, binary_sim, verification=verification)
    time = df['time']
    
    # Find index after first period
    idx_after_one_period = np.argmax(time > period)
    
    # Calculate separation distance
    df['distance'] = np.sqrt((df['star1_x'] - df['star2_x'])**2 + 
                            (df['star1_y'] - df['star2_y'])**2 + 
                            (df['star1_z'] - df['star2_z'])**2)
    
    # Find minimum separation (pericenter) within first period
    pericenter_idx = df['distance'][:idx_after_one_period].idxmin()
    time_pericenter_pass = df['time'].iloc[pericenter_idx]

    if verification:
        stored_time = df['time_of_pericenter_passage'].iloc[0] + df['orbital_period'].iloc[0]
        assert abs(time_pericenter_pass - stored_time) < 0.02 * stored_time, \
            f"Calculated time of pericenter passage differs from stored value by more than 2%"
        
    if return_empirical:
        return time_pericenter_pass
    else:
        return df['time_of_pericenter_passage'].iloc[0] + df['orbital_period'].iloc[0]


def random_geometry(df, binary_sim, verification=True):
    """
    Randomly transform the geometry of the binary system from a xy orientation to an xyz orientation. This is done in four steps:
    1. Randomly translate the binary system x,y,z. The range of translation is restricted between (-COM, COM) in each perpendicular direction, 
    where COM is the center of mass of the binary system.
    2. Randomly rotate the binary system about the x-axis by a random inclination angle.
    3. Randomly rotate the binary system about the z-axis by a random longitude of ascending node angle.
    4. Randomly rotate the binary system about the normal axis of the orbital plane by a random periapsis angle.

    Parameters:
    -----------
    df : pandas.DataFrame
       DataFrame containing simulation data with position and time columns (detailed_sims)
    binary_sim : object
        Simulation object containing initial conditions
    verification : bool, optional
        Whether to verify results (default True)
    return_empirical : bool, optional
        If True, calculate empirically, if False use stored value (default False)

    Returns:
    --------
    Pandas DataFrame
        DataFrame with updated positions of stars after random geometry transformation.
    """

    df = df.copy(deep=True)  # Ensure we don't modify the original DataFrame

    # Find the center of mass of the binary system
    # Get masses for COM calculation
    m1, m2 = df['star1_mass'].iloc[0], df['star2_mass'].iloc[0]
    total_mass = m1 + m2

    # Calculate COM coordinates
    df['COMx'] = (m1*df['star1_x'] + m2*df['star2_x'])/total_mass
    df['COMy'] = (m1*df['star1_y'] + m2*df['star2_y'])/total_mass
    df['COMz'] = (m1*df['star1_z'] + m2*df['star2_z'])/total_mass

    COMx = df['COMx'].mean()
    COMy = df['COMy'].mean()
    COMz = df['COMz'].mean()

    # Random translation in x, y, z with range from 
    translation_x = np.random.uniform(-COMx, COMx)
    translation_y = np.random.uniform(-COMy, COMy)
    translation_z = np.random.uniform(-COMz, COMz)

    # Apply translation to positions
    df['star1_x'] += translation_x
    df['star1_y'] += translation_y
    df['star1_z'] += translation_z
    df['star2_x'] += translation_x
    df['star2_y'] += translation_y
    df['star2_z'] += translation_z
    df['COMx'] += translation_x
    df['COMy'] += translation_y
    df['COMz'] += translation_z
        

    # Random inclination about the xy plane, longitude of ascending node about positive x-axis, and argument of pericenter within the orbital plane
    inclination = np.random.uniform(0, np.pi)  # Random inclination between 0 and pi
    longitude_of_ascending_node = np.random.uniform(0, 2 * np.pi)  # Random longitude of ascending node between 0 and 2*pi
    argument_of_periapsis = np.random.uniform(0, 2 * np.pi) # Random argument of periapsis between 0 and 2*pi

    # Update the geometry inclination, longitude of ascending node, and argument of periapsis in the DataFrame
    df['inclination'] = inclination
    df['longitude_of_ascending_node'] = longitude_of_ascending_node
    df['argument_of_periapsis'] = argument_of_periapsis

    new_geometry = f"New geometry, (x,y,z) translation: {translation_x, translation_y, translation_z}, inclination: {inclination}, longitude of ascending node: {longitude_of_ascending_node}, argument of periapsis: {argument_of_periapsis}"

    # Apply random inclination using Rodrigues' rotation matrix
    R = rotate_about_axis([1, 0, 0], inclination)  # Rotate about x-axis by inclination angle

    rel_star1 = np.stack([
        df['star1_x'] - df['COMx'],
        df['star1_y'] - df['COMy'],
        df['star1_z'] - df['COMz']], axis = 1)  # Relative position of star1 from COM (Shape: (N, 3))
    
    rel_star2 = np.stack([
        df['star2_x'] - df['COMx'],
        df['star2_y'] - df['COMy'],
        df['star2_z'] - df['COMz']], axis = 1)  # Relative position of star2 from COM (Shape: (N, 3))

    rotated_rel_star1 = rel_star1 @ R.T # Rotated relative position of star1 (Shape: (3, N))
    rotated_rel_star2 = rel_star2 @ R.T # Rotated relative position of star2 (Shape: (3, N))

    df['star1_x'] = rotated_rel_star1[:, 0] + df['COMx']
    df['star1_y'] = rotated_rel_star1[:, 1] + df['COMy']
    df['star1_z'] = rotated_rel_star1[:, 2] + df['COMz']
    df['star2_x'] = rotated_rel_star2[:, 0] + df['COMx']
    df['star2_y'] = rotated_rel_star2[:, 1] + df['COMy']
    df['star2_z'] = rotated_rel_star2[:, 2] + df['COMz']

    # Apply Rodrigues' rotation formula for the star velocities with inclination
    vel_star1 = np.stack([
        df['star1_vx'],
        df['star1_vy'],
        df['star1_vz']
    ], axis=1)

    vel_star2 = np.stack([
        df['star2_vx'],
        df['star2_vy'],
        df['star2_vz']
    ], axis=1)

    rotated_vel_star1 = vel_star1 @ R.T
    rotated_vel_star2 = vel_star2 @ R.T

    df['star1_vx'] = rotated_vel_star1[:, 0]
    df['star1_vy'] = rotated_vel_star1[:, 1]
    df['star1_vz'] = rotated_vel_star1[:, 2]
    df['star2_vx'] = rotated_vel_star2[:, 0]
    df['star2_vy'] = rotated_vel_star2[:, 1]
    df['star2_vz'] = rotated_vel_star2[:, 2]

    # Apply random longitude of ascending node using Rodrigues' rotation formula
    # Check for current longitude of ascending node
    r_rel = np.array([
        df['star2_x'].iloc[0] - df['star1_x'].iloc[0],
        df['star2_y'].iloc[0] - df['star1_y'].iloc[0],
        df['star2_z'].iloc[0] - df['star1_z'].iloc[0]])
    v_rel = np.array([
        df['star2_vx'].iloc[0] - df['star1_vx'].iloc[0],
        df['star2_vy'].iloc[0] - df['star1_vy'].iloc[0],
        df['star2_vz'].iloc[0] - df['star1_vz'].iloc[0]])
    # Calculate the specific angular momentum vector
    h_vec = np.cross(r_rel, v_rel)[2] # Z-component of the specific angular momentum vector

    if h_vec > 0:
        current_longitude_of_ascending_node = (3/2) * np.pi  # If h_vec is positive, longitude of ascending node is 3/2 pi
    elif h_vec < 0:
        current_longitude_of_ascending_node = (1/2) * np.pi # If h_vec is negative, longitude of ascending node is 1/2 pi
    else:
        current_longitude_of_ascending_node = 0

    R = rotate_about_axis([0, 0, 1], longitude_of_ascending_node - current_longitude_of_ascending_node)  # Rotate about z-axis of COM of the binary system
    
    rel_star1 = np.stack([
        df['star1_x'] - df['COMx'],
        df['star1_y'] - df['COMy'],
        df['star1_z'] - df['COMz']], axis = 1)  # Relative position of star1 from COM (Shape: (N, 3))
    
    rel_star2 = np.stack([
        df['star2_x'] - df['COMx'],
        df['star2_y'] - df['COMy'],
        df['star2_z'] - df['COMz']], axis = 1)  # Relative position of star2 from COM (Shape: (N, 3))

    rotated_rel_star1 = rel_star1 @ R.T # Rotated relative position of star1 (Shape: (3, N))
    rotated_rel_star2 = rel_star2 @ R.T # Rotated relative position of star2 (Shape: (3, N))

    df['star1_x'] = rotated_rel_star1[:, 0] + df['COMx']
    df['star1_y'] = rotated_rel_star1[:, 1] + df['COMy']
    df['star1_z'] = rotated_rel_star1[:, 2] + df['COMz']
    df['star2_x'] = rotated_rel_star2[:, 0] + df['COMx']
    df['star2_y'] = rotated_rel_star2[:, 1] + df['COMy']
    df['star2_z'] = rotated_rel_star2[:, 2] + df['COMz']

    # Apply Rodrigues' rotation formula for the star velocities with random longitude of ascending node
    vel_star1 = np.stack([
        df['star1_vx'],
        df['star1_vy'],
        df['star1_vz']
    ], axis=1)

    vel_star2 = np.stack([
        df['star2_vx'],
        df['star2_vy'],
        df['star2_vz']
    ], axis=1)

    rotated_vel_star1 = vel_star1 @ R.T
    rotated_vel_star2 = vel_star2 @ R.T

    df['star1_vx'] = rotated_vel_star1[:, 0]
    df['star1_vy'] = rotated_vel_star1[:, 1]
    df['star1_vz'] = rotated_vel_star1[:, 2]
    df['star2_vx'] = rotated_vel_star2[:, 0]
    df['star2_vy'] = rotated_vel_star2[:, 1]
    df['star2_vz'] = rotated_vel_star2[:, 2]

    # Apply random argument of periapsis using Rodrigues' rotation formula

    # Calculate the eccentricity vector
    r_rel = np.stack([
        df['star2_x'] - df['star1_x'],
        df['star2_y'] - df['star1_y'],
        df['star2_z'] - df['star1_z']
    ], axis=1)
    
    v_rel = np.stack([
        df['star2_vx'] - df['star1_vx'],
        df['star2_vy'] - df['star1_vy'],
        df['star2_vz'] - df['star1_vz']
    ], axis=1)  

    # Calculate the eccentricity vector
    reduced_mass = (m1 + m2)/total_mass # Reduced mass of the binary system
    r_norm = np.linalg.norm(r_rel, axis=1).reshape(-1, 1)
    eccentricity_vector = np.mean((np.cross(v_rel, r_rel) / reduced_mass) - (r_rel / r_norm), axis=0)

    # Calculate the specific angular momentum vector
    h_vec = np.cross(r_rel, v_rel)
    h_avg = h_vec.mean(axis=0)
    h_unit = h_avg / np.linalg.norm(h_avg)
    longitude_of_ascending_node_vector =  np.cross([0, 0, 1], h_unit)

    # Calculate the argument of periapsis
    current_argument_of_periapsis = np.arccos(np.dot(eccentricity_vector, longitude_of_ascending_node_vector) / (np.linalg.norm(eccentricity_vector) * np.linalg.norm(longitude_of_ascending_node_vector))) % 2*np.pi

    R = rotate_about_axis(h_unit, argument_of_periapsis - current_argument_of_periapsis) # Rotational matrix about the normal axis of the orbital plane

    rel_star1 = np.stack([
        df['star1_x'] - df['COMx'],
        df['star1_y'] - df['COMy'],
        df['star1_z'] - df['COMz']], axis = 1)  # Relative position of star1 from COM (Shape: (N, 3))
    
    rel_star2 = np.stack([
        df['star2_x'] - df['COMx'],
        df['star2_y'] - df['COMy'],
        df['star2_z'] - df['COMz']], axis = 1)  # Relative position of star2 from COM (Shape: (N, 3))

    rotated_rel_star1 = rel_star1 @ R.T # Rotated relative position of star1 (Shape: (3, N))
    rotated_rel_star2 = rel_star2 @ R.T # Rotated relative position of star2 (Shape: (3, N))

    df['star1_x'] = rotated_rel_star1[:, 0] + df['COMx']
    df['star1_y'] = rotated_rel_star1[:, 1] + df['COMy']
    df['star1_z'] = rotated_rel_star1[:, 2] + df['COMz']
    df['star2_x'] = rotated_rel_star2[:, 0] + df['COMx']
    df['star2_y'] = rotated_rel_star2[:, 1] + df['COMy']
    df['star2_z'] = rotated_rel_star2[:, 2] + df['COMz']

    # Apply Rodrigues' rotation formula for the star velocities with random argument of periapsis
    vel_star1 = np.stack([
        df['star1_vx'],
        df['star1_vy'],
        df['star1_vz']
    ], axis=1)

    vel_star2 = np.stack([
        df['star2_vx'],
        df['star2_vy'],
        df['star2_vz']
    ], axis=1)

    rotated_vel_star1 = vel_star1 @ R.T
    rotated_vel_star2 = vel_star2 @ R.T

    df['star1_vx'] = rotated_vel_star1[:, 0]
    df['star1_vy'] = rotated_vel_star1[:, 1]
    df['star1_vz'] = rotated_vel_star1[:, 2]
    df['star2_vx'] = rotated_vel_star2[:, 0]
    df['star2_vy'] = rotated_vel_star2[:, 1]
    df['star2_vz'] = rotated_vel_star2[:, 2]
    
    # Check for verificaiton
    if verification:
        # Rebound verification
        import rebound
        test_df = pd.DataFrame(columns=['star1_x', 'star1_y', 'star1_z', 'star2_x', 'star2_y', 'star2_z', 'Inclination', 'Longitude of ascending node', 'Argument of periapsis'])
        sim = rebound.Simulation()
        sim.units = ('m', 'kg', 's')  # Set units to SI units
    
        # Add stars with initial conditions from the new tranformed DataFrame
        sim.add(m=df['star1_mass'].iloc[0], x=df['star1_x'].iloc[0], y=df['star1_y'].iloc[0], z=df['star1_z'].iloc[0], 
                vx=df['star1_vx'].iloc[0], vy=df['star1_vy'].iloc[0], vz=df['star1_vz'].iloc[0])
        sim.add(m=df['star2_mass'].iloc[0], x=df['star2_x'].iloc[0], y=df['star2_y'].iloc[0], z=df['star2_z'].iloc[0],
                vx=df['star2_vx'].iloc[0], vy=df['star2_vy'].iloc[0], vz=df['star2_vz'].iloc[0])
        
        # Record the simulation data
        for time in df['time']:  # Follow the time in the tranformed DataFrame
            sim.integrate(time)  # Integrate the simulation to the current time
            p1 = sim.particles[0]
            p2 = sim.particles[1]
            orbit = p2.orbit(primary=p1)  # Calculate orbital elements
            detailed_row = [p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, orbit.inc, orbit.Omega, orbit.omega]
            test_df = pd.concat([test_df, pd.DataFrame([detailed_row])])
        
        # Check for a few random rows to ensure the transformation is correct
        for i in np.random.uniform(0, len(df), 10).astype(int):
            df_row = df.iloc[i]
            test_row = test_df.iloc[i]
            assert abs(df_row['star1_x'] - test_row['star1_x']) < 0.02 * test_row['star1_x'], f"{df_row['star1_x']} and {test_row['star1_x']} are not within 2% of each other"
            assert abs(df_row['star1_y'] - test_row['star1_y']) < 0.02 * test_row['star1_y'], f"{df_row['star1_y']} and {test_row['star1_y']} are not within 2% of each other"
            assert abs(df_row['star1_z'] - test_row['star1_z']) < 0.02 * test_row['star1_z'], f"{df_row['star1_z']} and {test_row['star1_z']} are not within 2% of each other"
            assert abs(df_row['star2_x'] - test_row['star2_x']) < 0.02 * test_row['star2_x'], f"{df_row['star2_x']} and {test_row['star2_x']} are not within 2% of each other"
            assert abs(df_row['star2_y'] - test_row['star2_y']) < 0.02 * test_row['star2_y'], f"{df_row['star2_y']} and {test_row['star2_y']} are not within 2% of each other"
            assert abs(df_row['star2_z'] - test_row['star2_z']) < 0.02 * test_row['star2_z'], f"{df_row['star2_z']} and {test_row['star2_z']} are not within 2% of each other"

        
    csv_file_detailed = f"scenarios/detailed_sims/21.3 M, 3.1 M_Inc_{inclination:.3f}_Long_{longitude_of_ascending_node:.3f}_Arg_{argument_of_periapsis:.3f}.csv"
    with open(csv_file_detailed, mode='w', newline='') as file_detailed:
        df.to_csv(file_detailed, index=False)


# Helper function to rotate vectors about an arbitrary axis using Rodrigues' rotation formula
def rotate_about_axis(axis, theta):
    """
    Rotate 3D vectors around an arbitrary axis.

    Parameters:
        axis : list or array of 3 floats
            Rotation axis direction (does not need to be unit length).
        theta : float
            Rotation angle in radians (positive = counterclockwise around axis).

    Returns:
        rotated_vectors : list of [x, y, z]
            Rotated vectors.
    """
    axis = np.array(axis, dtype=float)

    # Normalize the axis vector
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    one_minus_cos = 1 - cos_t

    # Rodrigues' rotation matrix
    R = np.array([
        [cos_t + x*x*one_minus_cos,
         x*y*one_minus_cos - z*sin_t,
         x*z*one_minus_cos + y*sin_t],

        [y*x*one_minus_cos + z*sin_t,
         cos_t + y*y*one_minus_cos,
         y*z*one_minus_cos - x*sin_t],

        [z*x*one_minus_cos - y*sin_t,
         z*y*one_minus_cos + x*sin_t,
         cos_t + z*z*one_minus_cos]
    ])

    return R

# Test
#df = pd.read_csv(f"scenarios/detailed_sims/21.3 M, 3.1 M.csv")
#random_geometry(df, binary_sim=None, verification=True)