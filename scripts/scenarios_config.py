"""
Configuration and setup for binary star system scenarios used in physics simulations.
Handles scenario variations, orbital calculations, and integration with rebound N-body simulator.
"""
import sys
import os
import pandas as pd

# Add project root to sys.path so Python can find generalscenarios/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import numpy as np
from generalscenarios.Binary import Binary
import importlib
import json
import rebound

class BinaryScenario:
    """
    Represents a configuration for a binary star system scenario.
    
    Attributes:
        variation_name (str): Unique identifier for this scenario variation
        star1_mass (float): Mass of primary star in kg (or specified units)
        star2_mass (float): Mass of secondary star in kg (or specified units)
        star1_pos (np.array): 3D position of primary star
        star2_pos (np.array): 3D position of secondary star
        ellipticity (float): Orbital eccentricity (0 = circular, >1 = unbound)
        proper_motion_*: Parameters for system-wide velocity components
        drag_tau (float): Optional atmospheric drag timescale
        mod_gravity_exponent (float): Optional modified gravity parameter
        units (tuple): Unit system used (m/s/kg, AU/yr/Msun, or cm/s/g)
        face_on_projection (bool): Whether to use face-on projection for visualization
    """
    
    def __init__(self, variation_name, star1_mass, star2_mass, star1_pos, star2_pos, 
                 maxtime=None, num_orbits=10, ellipticity=0.0, proper_motion_direction=None, 
                 proper_motion_magnitude=0.0, drag_tau=None, mod_gravity_exponent=None, 
                 units=('m', 's', 'kg'), face_on_projection=False):
        """Initialize binary system with physical parameters and dynamic properties."""
        
        # Basic system parameters
        self.variation_name = variation_name
        self.filename = self.variation_name  # Used for data storage
        self.star1_mass = star1_mass
        self.star2_mass = star2_mass
        self.star1_pos = np.array(star1_pos)
        self.star2_pos = np.array(star2_pos)
        
        # Dynamic parameters
        self.ellipticity = ellipticity
        self.proper_motion_direction = proper_motion_direction
        self.proper_motion_magnitude = proper_motion_magnitude
        self.drag_tau = drag_tau  # Atmospheric drag parameter
        self.mod_gravity_exponent = mod_gravity_exponent  # For modified gravity tests
        
        # Unit system handling (SI, astronomical units, or CGS)
        self.units = units
        
        # Observation parameters (defaults, can be overridden later)
        self.face_on_projection = face_on_projection # Whether to visualize in face-on projection
        self.max_observations = 10  # Total observations allowed
        self.max_observations_per_request = 10  # Observations per API call

        # Calculate initial momenta with orbital mechanics
        self.star1_momentum, self.star2_momentum = calculate_orbital_momentum(
            self.star1_mass, self.star2_mass,
            self.star1_pos, self.star2_pos,
            modified_exponent=self.mod_gravity_exponent or 2.0,
            ellipticity=self.ellipticity,
            proper_motion_direction=self.proper_motion_direction,
            proper_motion_magnitude=self.proper_motion_magnitude,
            units=self.units
        )

        # Simulation duration handling
        if num_orbits is not None and maxtime is None:
            # Validate parameters for orbital period calculation
            if self.mod_gravity_exponent is not None:
                raise ValueError("Orbital period calculation not reliable with modified gravity")
            if self.ellipticity > 1.0:
                raise ValueError("Orbital period undefined for unbound systems")
                
            # Calculate duration based on number of orbits
            orbital_period = self.calculate_orbital_period()
            self.maxtime = orbital_period * num_orbits
        elif maxtime is not None:
            self.maxtime = maxtime
        else:
            raise ValueError("Must specify either maxtime or num_orbits")

    def create_binary(self, prompt, final_answer_units, skip_simulation=False):
        """Factory method to create a Binary simulation instance."""
        return Binary(
            self.star1_mass, self.star2_mass,
            self.star1_pos, self.star2_pos,
            self.star1_momentum, self.star2_momentum,
            self.maxtime, self.max_observations, self.max_observations_per_request,
            self.filename, prompt, final_answer_units, 
            drag_tau=self.drag_tau, 
            mod_gravity_exponent=self.mod_gravity_exponent,
            units=self.units, face_on_projection=self.face_on_projection,
            skip_simulation=skip_simulation
        )
    
    def calculate_orbital_period(self):
        """Calculate orbital period using rebound N-body simulator."""
        sim = rebound.Simulation()
        sim.units = self.units  # Handle unit conversions
        
        # Add stars to simulation
        sim.add(
            m=self.star1_mass,
            x=self.star1_pos[0], y=self.star1_pos[1], z=self.star1_pos[2],
            vx=self.star1_momentum[0]/self.star1_mass,
            vy=self.star1_momentum[1]/self.star1_mass,
            vz=self.star1_momentum[2]/self.star1_mass
        )
        sim.add(
            m=self.star2_mass,
            x=self.star2_pos[0], y=self.star2_pos[1], z=self.star2_pos[2],
            vx=self.star2_momentum[0]/self.star2_mass,
            vy=self.star2_momentum[1]/self.star2_mass,
            vz=self.star2_momentum[2]/self.star2_mass
        )
        
        return sim.particles[1].P  # Return orbital period of second particle

def calculate_orbital_momentum(
    star1_mass,
    star2_mass,
    star1_pos,
    star2_pos,
    modified_exponent=2.0,
    ellipticity=0.0,
    proper_motion_direction=None,
    proper_motion_magnitude=0.0,
    units=('m', 's', 'kg')
):
    """
    Calculate initial momenta for binary system with optional modifications.
    
    Args:
        modified_exponent: Gravity modification (2 = Newtonian)
        ellipticity: Orbital eccentricity (0-1)
        proper_motion_*: System-wide velocity components
        units: Unit system for gravitational constant
        
    Returns:
        tuple: (star1_momentum, star2_momentum) as numpy arrays
    """
    
    total_mass = star1_mass + star2_mass
    relative_position = np.array(star2_pos) - np.array(star1_pos)
    
    # Gravitational constant handling for different unit systems
    if units == ('yr', 'AU', 'Msun'):
        G = 4 * np.pi**2  # AU^3/(Msun*yr^2)
    elif units == ('s', 'cm', 'g'):
        G = 6.67430e-8  # cm^3/(g*s^2)
    else:  # Default SI units
        G = 6.67430e-11  # m^3/(kg*s^2)
    
    # Calculate base orbital velocity
    r = np.linalg.norm(relative_position)
    orbital_velocity = np.sqrt(G * total_mass / r**(modified_exponent - 1))

    # Determine orbital plane direction
    if relative_position[0] != 0 or relative_position[1] != 0:
        velocity_direction = np.cross(relative_position, [0, 0, 1])  # z-axis
    else:
        velocity_direction = np.cross(relative_position, [0, 1, 0])  # y-axis
    velocity_direction = velocity_direction.astype(float)
    velocity_direction /= np.linalg.norm(velocity_direction)

    # Add eccentricity component
    radial_direction = relative_position / r
    velocity_radial = radial_direction * (orbital_velocity * ellipticity)
    velocity_total = velocity_direction * orbital_velocity + velocity_radial

    # Mass-weighted velocity distribution
    star2_velocity = velocity_total * (star1_mass / total_mass)
    star1_velocity = -velocity_total * (star2_mass / total_mass)

    # Add proper motion velocity component
    if proper_motion_magnitude > 0:
        if proper_motion_direction is None:
            proper_motion_dir = np.array([1.0, 1.0, 0.0])
        else:
            proper_motion_dir = np.array(proper_motion_direction, dtype=float)
        proper_motion_dir /= np.linalg.norm(proper_motion_dir)
        proper_motion_velocity = proper_motion_dir * proper_motion_magnitude
        star2_velocity += proper_motion_velocity
        star1_velocity += proper_motion_velocity

    # Calculate final momenta
    return (
        star1_velocity * star1_mass,
        star2_velocity * star2_mass
    )

# Solar mass constant for conversions
Msun = 1.989e30  # kg

# Preconfigured scenario variations
variations = {
    '21.3 M, 3.1 M, FOP': BinaryScenario('21.3 M, 3.1 M, FOP', 21.3*Msun, 3.1*Msun, [-5e12, -7e12, 0], [-3e12, -8e12, 0], ellipticity=0.6, face_on_projection=True)}
#    '9.6 M, 3.1 M': BinaryScenario('9.6 M, 3.1 M', 9.6*Msun, 3.1*Msun, [-1e12, 6e12, 1e9], [-1e12, 3e12, 6.3e9], ellipticity=0.6),
#    '0.18 M, 0.63 M': BinaryScenario('0.18 M, 0.63 M', 0.18*Msun, 0.63*Msun, [7e11, 2e11, 7e11], [2e11, 1e11, -3e12], ellipticity=0.6),
#    '9.6 M, 3.1 M, Proper Motion': BinaryScenario('9.6 M, 3.1 M, Proper Motion', 9.6*Msun, 3.1*Msun, [4e12, 3e12, 5e12], [2e12, 3e12, -1e12], ellipticity=0.8, proper_motion_direction=[1, 1, 0], proper_motion_magnitude=1e3),
#    '9.6 M, 3.1 M, Proper Motion2': BinaryScenario('9.6 M, 3.1 M, Proper Motion2', 9.6*Msun, 3.1*Msun, [4e11, -4e11, -8e11], [7e11, -6e11, -1e11], ellipticity=0.9, proper_motion_direction=[2, -1, 0], proper_motion_magnitude=1e3),
#
#    '9.6 M, 3.1 M, yrAUMsun': BinaryScenario('9.6 M, 3.1 M, yrAUMsun', 
#                                         9.6, 3.1,  # In Msun
#                                         [-1e12/1.496e11, 6e12/1.496e11, -4e12/1.496e11], [-1e12/1.496e11, 3e12/1.496e11, 7e12/1.496e11],  # Convert m to AU
#                                         ellipticity=0.6,
#                                         units=('yr', 'AU', 'Msun')),
#    '9.6 M, 3.1 M, cgs': BinaryScenario('9.6 M, 3.1 M, cgs', 
#                                    9.6*Msun*1000, 3.1*Msun*1000,  # Convert kg to g
#                                    [-1e12*100, 6e12*100, 8e12*100], [-1e12*100, 3e12*100, -2e12*100],  # Convert m to cm
#                                    ellipticity=0.6,
#                                    units=('s', 'cm', 'g')),
#
#    '3.1 M, 0.18 M Elliptical': BinaryScenario('3.1 M, 0.18 M Elliptical', 3.1*Msun, 0.18*Msun, [4e10, -4e10, -2.2e10], [1.5e11, -7e10, 4.5e12], ellipticity=0.93),
#    '3.1 M, 0.18 M, Elliptical, Single Orbit': BinaryScenario('3.1 M, 0.18 M, Elliptical, Single Orbit', 3.1*Msun, 0.18*Msun, [4e10, -4e10, 8e10], [1.5e11, -7e10, -2e10], num_orbits=1, ellipticity=0.93),
#    '7.7 M, 4.9 M, Drag tau = 1.7e9': BinaryScenario('7.7 M, 4.9 M, Drag tau = 1.7e9', 7.7*Msun, 4.9*Msun, [1e11, 1e11, 1e11], [1.5e11, 7e11, 7e11],  ellipticity=0.55, drag_tau=1.7e9, maxtime=7e8),
#    '7.7 M, 4.9 M, Drag tau = 8.3e8': BinaryScenario('7.7 M, 4.9 M, Drag tau = 8.3e8', 7.7*Msun, 4.9*Msun, [1e11, 1e11, 9e11], [1.5e11, 7e11, -9e11],  ellipticity=0.55, drag_tau=8.3e8, maxtime=5e8),
#    '7.7 M, 4.9 M, Drag tau = 8.3e8 Proper Motion': BinaryScenario('7.7 M, 4.9 M, Drag tau = 8.3e8 Proper Motion', 7.7*Msun, 4.9*Msun, [1e11, 1e11, 4e11], [1.5e11, 7e11, 3e11],  ellipticity=0.55, drag_tau=8.3e8, proper_motion_direction=[1, 1, 0], proper_motion_magnitude=1e4, maxtime=5e8),
#    '10.1 M, 5.6 M, Unbound': BinaryScenario('10.1 M, 5.6 M, Unbound', 10.1*Msun, 5.6*Msun, [-1e11, 7e11, 4e11], [1.5e11, 7e10, -2e11], maxtime=1e8, ellipticity=1.5),
#    '10.1M, 5.6 M, Modified Gravity 1.97': BinaryScenario('10.1M, 5.6 M, Modified Gravity 1.97', 10.1*Msun, 5.6*Msun, [5e10, 1e11, 9e11], [7.5e10, 3e10, 6e11], maxtime=9e7, ellipticity=0.8, mod_gravity_exponent=1.97),
#    '10.1M, 5.6 M, Modified Gravity 2.03': BinaryScenario('10.1M, 5.6 M, Modified Gravity 2.03', 10.1*Msun, 5.6*Msun, [5e10, 1e11, 3e10], [7.5e10, 3e10, -6e10], maxtime=2.3e8, ellipticity=0.8, mod_gravity_exponent=2.03),
#    '10.1M, 5.6 M, Modified Gravity 1.97 Proper Motion': BinaryScenario('10.1M, 5.6 M, Modified Gravity 1.97 Proper Motion', 10.1*Msun, 5.6*Msun, [5e10, 1e10, 6e10], [7.5e10, 3e10, 4e9], maxtime=2.4e7, ellipticity=0.8, mod_gravity_exponent=1.97, proper_motion_direction=[1, 1, 0], proper_motion_magnitude=1e4)
#}
# Define scenarios with their variations
#load scenarios_config.json
with open('scripts/scenarios_config.json') as f:
    all_scenarios = json.load(f)

def get_all_scenarios():
    """Return all loaded scenario configurations."""
    return all_scenarios

def get_scenario(scenario_name, variation_name, row_wise=False, 
                max_observations_total=10, max_observations_per_request=10, 
                scenario_folder='scenarios'):
    """
    Load a specific scenario configuration with parameters.
    
    Args:
        scenario_name: Name from scenarios_config.json
        variation_name: Which variation to use
        row_wise: Whether to use row-wise observation mode
        max_observations_*: Control data access limits
    """
    # Determine if simulation should be skipped
    sim_csv_file_path = f"scenarios/sims/{variation_name}.csv"
    detailed_sim_csv_file_path = f"scenarios/detailed_sims/{variation_name}.csv"
    
    skip_simulation = os.path.exists(sim_csv_file_path) and os.path.exists(detailed_sim_csv_file_path)
    
    if skip_simulation:
        print(f"INTERNAL: Both simulation data files found for {variation_name}, skipping simulation.")
    else:
        missing_files = []
        if not os.path.exists(sim_csv_file_path):
            missing_files.append(sim_csv_file_path)
        if not os.path.exists(detailed_sim_csv_file_path):
            missing_files.append(detailed_sim_csv_file_path)
        print(f"INTERNAL: Not all simulation data found for {variation_name} (missing: {', '.join(missing_files)}), generating simulation.")

    # Import scenario module and create instance
    scenario_module = importlib.import_module(f"{scenario_folder}.{scenario_name}")

    if "Inc" in variation_name:
        df = pd.read_csv(f"scenarios/detailed_sims/{variation_name}.csv")
        star1_m = df['star1_mass'].iloc[0]
        star2_m = df['star2_mass'].iloc[0]
        star1_x, star1_y, star1_z = df['star1_x'].iloc[0], df['star1_y'].iloc[0], df['star1_z'].iloc[0]
        star2_x, star2_y, star2_z = df['star2_x'].iloc[0], df['star2_y'].iloc[0], df['star2_z'].iloc[0]
        e = df['eccentricity'].iloc[0]
        if "FOP" in variation_name:
            projection = True
        scenario = scenario_module.Scenario(BinaryScenario(variation_name, star1_mass = star1_m, star2_mass=star2_m,
                                                           star1_pos=[star1_x, star1_y, star1_z], star2_pos=[star2_x, star2_y, star2_z],
                                                           ellipticity=e, face_on_projection=projection),
                                            skip_simulation=skip_simulation) # Then run then BinaryScenario with the variation_name and the masses

        # Or other method to split the variation_name
        #star1_mass, star2_mass = variation_name[:-31].split(',') # Splits the randomly transformed variation to original variation name, and split them to get their masses
        #mass1 = float(star1_mass.strip().replace(' M', ''))
        #mass2 = float(star2_mass.strip().replace(' M', ''))
        #scenario = scenario_module.Scenario(BinaryScenario(variation_name, )) # Then run then BinaryScenario with the variation_name and the masses
    else:
        scenario = scenario_module.Scenario(
            variations[variation_name], 
            skip_simulation=skip_simulation
        )
        
    # Configure observation parameters if needed
    if hasattr(scenario, 'binary_sim') and row_wise:
        scenario.binary_sim.max_observations = max_observations_total
        scenario.binary_sim.max_observations_per_request = max_observations_per_request
        scenario.binary_sim.set_row_wise_prompt()
    
    return scenario

def get_scenario_test_new_variation(scenario_name, new_variation, scenario_folder='scenarios'):
    """Helper function for testing new scenario variations."""
    return importlib.import_module(
        f"{scenario_folder}.{scenario_name}"
    ).Scenario(new_variation)

#for variation in variations.values():
    variation.create_binary(
        "Test scenario creation",
        "('m', 's', 'kg')",
        skip_simulation=False
    )