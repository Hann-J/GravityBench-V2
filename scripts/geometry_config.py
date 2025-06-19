
"""
This script contains a function to randomly transform the geometry of a binary system from an xy orientation to an xyz orientation.

The function writes to two CSV files, one to scenarios/detailed_sims and one to scenarios/sims. A rebound simulation is run with the new geometry,
and the results are saved compared to the original tranformed DataFrame. However, their differences differ by a very small amount, that cascades a larger amount as 
time passes. Nonetheless, the differences are usually small enough to be negligible, but the rebound simulation data is used as a baseline for more accuracy.
"""

import numpy as np
import pandas as pd
import rebound
import os
import json
import sys


def random_geometry(df, file_name:str, verification=False):
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
    file_name : ostr
        Simulation object containing initial conditions
    verification : bool, optional
        Whether to verify results (default True)

    Returns:
    --------
    str
        The name of the file containing the transformed geometry.
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
    longitude_of_ascending_node = np.random.uniform(-np.pi, np.pi)  # Random longitude of ascending node between -pi and pi, with positive x-axis as reference
    argument_of_periapsis = np.random.uniform(0, 2*np.pi) # Random argument of periapsis between -pi and pi

    # Update the geometry inclination, longitude of ascending node, and argument of periapsis in the DataFrame
    df['inclination'] = inclination
    df['longitude_of_ascending_node'] = longitude_of_ascending_node
    df['argument_of_periapsis'] = argument_of_periapsis

    new_geometry = f"New geometry, (x,y,z) translation: {translation_x, translation_y, translation_z}, inclination: {inclination}, longitude of ascending node: {longitude_of_ascending_node}, argument of periapsis: {argument_of_periapsis}"

    # Apply random inclination using Rodrigues' rotation matrix
    rel_star1 = np.stack([
        df['star1_x'] - df['COMx'],
        df['star1_y'] - df['COMy'],
        df['star1_z'] - df['COMz']], axis = 1)  # Relative position of star1 from COM (Shape: (N, 3))
    
    rel_star2 = np.stack([
        df['star2_x'] - df['COMx'],
        df['star2_y'] - df['COMy'],
        df['star2_z'] - df['COMz']], axis = 1)  # Relative position of star2 from COM (Shape: (N, 3))
    
    # Specific angular momentum vector
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

    h_vec = np.cross(r_rel, v_rel)  # Specific angular momentum vector (Shape: (N, 3))
    h_avg = h_vec.mean(axis=0)  # shape: (3,)

    if h_avg[2] < 0:
        R = rotate_about_axis([0, 1, 0], inclination + np.pi)  # Rotate about x-axis by inclination angle
    else:
        R = rotate_about_axis([0, 1, 0], inclination)   

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

    # Calculate the specific angular momentum vector
    h_vec = np.cross(r_rel, v_rel)
    h_avg = h_vec.mean(axis=0)
    h_unit = h_avg / np.linalg.norm(h_avg)  # Normalize the specific angular momentum vector

    if h_avg[0] < 0:
        current_longitude_of_ascending_node = -(1/2) * np.pi  # If h_vec is positive, longitude of ascending node is -1/2 pi
    elif h_avg[0] > 0:
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

    # Apply Rodrigues' rotation formula to the star velocities with random longitude of ascending node
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

    # Calculate the specific angular momentum vector
    h_vec = np.cross(r_rel, v_rel)
    h_avg = h_vec.mean(axis=0)  # shape: (3,)
    h_unit = h_avg / np.linalg.norm(h_avg)
    longitude_of_ascending_node_vector =  np.cross([0, 0, 1], h_unit)

    # Calculate the eccentricity vector
    reduced_mass = (m1 * m2)/total_mass # Reduced mass of the binary system
    r_norm = np.linalg.norm(r_rel, axis=1).reshape(-1, 1)
    eccentricity_vector = np.mean((np.cross(v_rel, h_vec) / reduced_mass) - (r_rel / r_norm), axis=0)

    # Calculate the argument of periapsis
    current_argument_of_periapsis = np.arccos(np.dot(eccentricity_vector, longitude_of_ascending_node_vector) / (np.linalg.norm(eccentricity_vector) * np.linalg.norm(longitude_of_ascending_node_vector)))
    
    # sin(omega) disambiguation
    sin_argp = np.dot(np.cross(longitude_of_ascending_node_vector, eccentricity_vector), h_unit)

    # Flip angle if sin is negative
    if sin_argp < 0:
        current_argument_of_periapsis = 2 * np.pi - current_argument_of_periapsis

    R = rotate_about_axis(h_unit, argument_of_periapsis - current_argument_of_periapsis) # Rotational matrix about the normal axis of the orbital plane
    
    rel_star1 = np.stack([
        df['star1_x'] - df['COMx'],
        df['star1_y'] - df['COMy'],
        df['star1_z'] - df['COMz']], axis = 1)  # Relative position of star1 from COM (Shape: (N, 3))
    
    rel_star2 = np.stack([
        df['star2_x'] - df['COMx'],
        df['star2_y'] - df['COMy'],
        df['star2_z'] - df['COMz']], axis = 1)  # Relative position of star2 from COM (Shape: (N, 3))

    rotated_rel_star1 = rel_star1 @ R.T # Rotated relative position of star1 (Shape: (N, 3))
    rotated_rel_star2 = rel_star2 @ R.T # Rotated relative position of star2 (Shape: (N, 3))

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
    
    
    # Rebound verification
    sim = rebound.Simulation()
    sim.units = ('m', 's', 'kg')  # Set units to SI units
    
    # Add stars with initial conditions from the new tranformed DataFrame
    sim.add(m=df['star1_mass'].iloc[0], x=df['star1_x'].iloc[0], y=df['star1_y'].iloc[0], z=df['star1_z'].iloc[0], 
            vx=df['star1_vx'].iloc[0], vy=df['star1_vy'].iloc[0], vz=df['star1_vz'].iloc[0])
    sim.add(m=df['star2_mass'].iloc[0], x=df['star2_x'].iloc[0], y=df['star2_y'].iloc[0], z=df['star2_z'].iloc[0],
            vx=df['star2_vx'].iloc[0], vy=df['star2_vy'].iloc[0], vz=df['star2_vz'].iloc[0])
        
    # Record the simulation data
    rows = []
    time_passed = 0 # Start time of the simulation
    dt = df['time'].iloc[1] - df['time'].iloc[0]  # Time difference from the start of the simulation
    while time_passed <= df['time'].iloc[-1]:
        sim.integrate(time_passed)  # Integrate the simulation to the current time
        time_passed += dt # Update the time
        p1 = sim.particles[0]
        p2 = sim.particles[1]
        
        # Calculate detailed orbital parameters
        separation = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
        force = sim.G * p1.m * p2.m / separation**2  # Newtonian force
        star1_accel = force / p1.m
        star2_accel = force / p2.m
        orbit = p2.orbit(primary=p1)  # Calculate orbital elements    

        detailed_row = [
                    time_passed,
                    p1.x, p1.y, p1.z, p2.x, p2.y, p2.z,
                    p1.vx, p1.vy, p1.vz, p2.vx, p2.vy, p2.vz,
                    p1.m, p2.m, separation, force, star1_accel, star2_accel,
                    orbit.h, orbit.P, orbit.n, orbit.a, orbit.e,
                    orbit.inc, orbit.Omega, orbit.omega, orbit.f, orbit.M, orbit.T, orbit.d
                ]
        rows.append(detailed_row)
        
    sim_df = pd.DataFrame(rows, columns=[
                'time', 'star1_x', 'star1_y', 'star1_z', 'star2_x', 'star2_y', 'star2_z',
                'star1_vx', 'star1_vy', 'star1_vz', 'star2_vx', 'star2_vy', 'star2_vz', 
                'star1_mass', 'star2_mass', 'separation', 'force', 'star1_accel', 'star2_accel',
                'specific_angular_momentum', 'orbital_period', 'mean_motion', 'semimajor_axis', 
                'eccentricity', 'inclination', 'longitude_of_ascending_node', 'argument_of_periapsis','true_anomaly', 'mean_anomaly', 
                'time_of_pericenter_passage', 'radial_distance_from_reference'
            ])

# Testing the transformation    
#    sim_df = sim_df[['time', 'star1_x', 'star1_y', 'star1_z', 'star2_x', 'star2_y', 'star2_z', 'inclination',
#                     'longitude_of_ascending_node', 'argument_of_periapsis']]

# Manually transformed DataFrame        
#    csv_file_detailed = f"scenarios/detailed_sims/{file_name}_Inc_{inclination:.3f}_Long_{longitude_of_ascending_node:.3f}_Arg_{argument_of_periapsis:.3f}.csv"
#    with open(csv_file_detailed, mode='w', newline='') as file_detailed:
#        df.to_csv(file_detailed, index=False)

    csv_file_detailed_sims = f"scenarios/detailed_sims/{file_name}_Inc_{inclination:.3f}_Long_{longitude_of_ascending_node:.3f}_Arg_{argument_of_periapsis:.3f}.csv"
    with open(csv_file_detailed_sims, mode='w', newline='') as file_detailed_actual:
        sim_df.to_csv(file_detailed_actual, index=False)
        
    csv_file_sims = f"scenarios/sims/{file_name}_Inc_{inclination:.3f}_Long_{longitude_of_ascending_node:.3f}_Arg_{argument_of_periapsis:.3f}.csv"
    with open(csv_file_sims, mode='w', newline='') as file_sims:
        sim_df[['time', 'star1_x', 'star1_y', 'star1_z', 'star2_x', 'star2_y', 'star2_z']].to_csv(file_sims, index=False)

    # Check for verificaiton
    if verification:
        # Check for a few random rows to ensure the transformation is correct
        for i in np.random.uniform(0, len(df), 20).astype(int):
            df_row = df.iloc[i]
            test_row = sim_df.iloc[i]
            assert abs(df_row['star1_x'] - test_row['star1_x']) <= 0.0000001 * abs(test_row['star1_x']), f"{df_row['star1_x']} and {test_row['star1_x']} are not within 0.00001% of each other in {i}"
            assert abs(df_row['star1_y'] - test_row['star1_y']) <= 0.0000001 * abs(test_row['star1_y']), f"{df_row['star1_y']} and {test_row['star1_y']} are not within 0.00001% of each other in {i}"
            assert abs(df_row['star1_z'] - test_row['star1_z']) <= 0.0000001 * abs(test_row['star1_z']), f"{df_row['star1_z']} and {test_row['star1_z']} are not within 0.00001% of each othe in {i}"
            assert abs(df_row['star2_x'] - test_row['star2_x']) <= 0.0000001 * abs(test_row['star2_x']), f"{df_row['star2_x']} and {test_row['star2_x']} are not within 0.00001% of each other in {i}"
            assert abs(df_row['star2_y'] - test_row['star2_y']) <= 0.0000001 * abs(test_row['star2_y']), f"{df_row['star2_y']} and {test_row['star2_y']} are not within 0.00001% of each other in {i}"
            assert abs(df_row['star2_z'] - test_row['star2_z']) <= 0.0000001 * abs(test_row['star2_z']), f"{df_row['star2_z']} and {test_row['star2_z']} are not within 0.00001% of each other in {i}"

    return f"{file_name}_Inc_{inclination:.3f}_Long_{longitude_of_ascending_node:.3f}_Arg_{argument_of_periapsis:.3f}"

# Remove the randomly transformed variations
def remove_variations():
    folder_path_detailed = "scenarios/detailed_sims"
    file_names = os.listdir(folder_path_detailed)
    for file in file_names:
        if "Inc" in file:
            file_path_detailed = f"scenarios/detailed_sims/{file}"
            file_path_sims = f"scenarios/sims/{file}"
            if os.path.exists(file_path_detailed):
                os.remove(file_path_detailed)
            if os.path.exists(file_path_sims):
                os.remove(file_path_sims)

    # Update json file
    json_path = "scripts/scenarios_config.json"

    # Load existing JSON data (if file exists)
    with open(json_path, 'r') as f:
        scenario = json.load(f)

    # Update the json file back to original variations
    for scenario_name, scenario_set_ups in scenario.items():
        scenario_set_ups['variations'] = [var for var in scenario_set_ups['variations'] if "Inc" not in var]

    with open(json_path, 'w') as f:
        json.dump(scenario, f, indent=4)



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

#Test
#df = pd.read_csv(f"scenarios/detailed_sims/21.3 M, 3.1 M.csv")
#random_geometry(df, file_name="21.3 M, 3.1 M", verification=True)
#print("task_utils.py executed successfully.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "remove_variations":
        remove_variations()