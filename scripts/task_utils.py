"""
This file contains utility functions for calculating various quantities from simulation data.
These are used for the human-ref solutions in scenarios/
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
        if (df['star1_z'].mean() == 0.0) and (df['star2_z'].mean() == 0.0):
            for star in ['star1', 'star2']:
                for axis in ['x', 'y']:
                    v_calc = velocities[f'{star}_v{axis}']
                    v_stored = df[f'{star}_v{axis}']
                    percent_diff = (v_calc - v_stored) / v_stored
                    assert np.abs(np.mean(percent_diff)) < 0.02, f"{star} velocity {axis} component differs by more than 2%"
        else:
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
