# Binary.py - Binary star system simulation and observation management

import numpy as np
import rebound  # N-body simulation package
import csv
import pandas as pd
from typing import Union, List
from scipy.interpolate import CubicSpline  # For position interpolation

class RowWiseResults:
    """Stores and manages observational data collected through row-wise access"""
    def __init__(self):
        # DataFrame to store time and positional data for both stars
        self.df = pd.DataFrame(columns=['time', 'star1_x', 'star1_y', 'star1_z', 
                                      'star2_x', 'star2_y', 'star2_z'])

class Binary():
    """Main class handling binary star system simulation and observation management"""
    
    def __init__(self, star1_mass, star2_mass, star1_pos, star2_pos, star1_momentum, 
                 star2_momentum, maxtime, max_observations, max_observations_per_request, 
                 filename, prompt, final_answer_units, drag_tau=None, 
                 mod_gravity_exponent=None, units=('m', 's', 'kg'), face_on_projection=False, skip_simulation=False):
        """
        Initialize binary system with physical parameters and simulation settings
        Args:
            drag_tau: Linear drag coefficient (None = no drag)
            mod_gravity_exponent: Gravity law modification (None = Newtonian)
            units: Unit system for simulation (SI, Astronomical, or CGS)
            face_on_projection: Projects the stars onto the yz plane (x=0). The computer is looking at the system along the positive x-axis.
            skip_simulation: Load existing data instead of running new simulation
        """
        
        # Store system parameters
        self.star1_mass = star1_mass
        self.star2_mass = star2_mass
        self.star1_pos = star1_pos  # Initial position [x,y,z]
        self.star2_pos = star2_pos
        self.star1_momentum = star1_momentum  # Initial momentum
        self.star2_momentum = star2_momentum
        self.maxtime = maxtime  # Maximum simulation time
        self.max_observations = max_observations  # Total allowed observations
        self.max_observations_per_request = max_observations_per_request  # Per-call limit
        self.number_of_observations_requested = 0  # Observation counter
        self.row_wise_results = RowWiseResults()  # Stores agent's observational data
        self.filename = filename  # Base name for data files
        self.final_answer_units = final_answer_units  # Required answer units
        self.drag_tau = drag_tau  # Linear drag coefficient
        self.mod_gravity_exponent = mod_gravity_exponent  # Gravity law modification
        self.units = units  # Unit system tuple
        self.face_on_projection = face_on_projection  # Projection onto the y-z plane

        # Initialize REBOUND simulation
        self.sim = rebound.Simulation()
        self.sim.units = self.units
        
        # Set gravitational constant based on unit system
        if self.units == ('yr', 'AU', 'Msun'):
            # Astronomical units: AU^3 / (Msun * yr^2)
            self.sim.G = 4 * np.pi**2
        elif self.units == ('s', 'cm', 'g'):
            # CGS units: cm^3 / (g * s^2)
            self.sim.G = 6.67430e-8  
        else:  # Default SI units
            # SI units: m^3 / (kg * s^2)
            self.sim.G = 6.67430e-11

        # Run simulation unless loading existing data
        if not skip_simulation:
            self.simulate(drag_tau=drag_tau, mod_gravity_exponent=mod_gravity_exponent, 
                        units=units)

        # Load generated simulation data
        self.df = pd.read_csv(f"scenarios/sims/{self.filename}.csv")
        
        # Configure problem prompts for AI agent
        self.task = prompt  # Physics problem description
        # Create unit description string for prompts
        if 'yr' in self.units and 'AU' in self.units:
            self.units_string = "years and AU"
        elif 'm' in self.units and 's' in self.units:
            self.units_string = "seconds and meters"
        elif 'cm' in self.units and 's' in self.units:
            self.units_string = "seconds and centimeters"
        else:
            self.units_string = f"{self.units[0]} and {self.units[1]}"
        
        # Construct final answer requirement prompt
        self.final_answer_prompt = ""
        if self.final_answer_units is not None:
            self.final_answer_prompt += f"You must provide your answer in units of {self.final_answer_units}."
        
        # Base prompt configuration
        self.row_wise_prompt = ""

        # Prompt without face_on_projection
        if not self.face_on_projection:
            self.full_table_tools_and_data_prompt = f"1. A DataFrame `df` containing columns: {', '.join(self.df.columns)}.\n2. A code interpreter with `df` pre-loaded that can execute Python code."
            self.prompt = f"""You are tasked with solving the following physics problem related to a binary star system. You are provided observations of each star's position over time, (t,x,y,z), in units of {self.units_string}.
        
### Problem Description
{self.task}
{self.final_answer_prompt}

### Additional Instructions
To complete this task, you have access to the following tools and data:"""


        # Prompt with face-on projection
        else:
            self.full_table_tools_and_data_prompt = f"1. A DataFrame `df` containing columns: {', '.join(self.df.columns)}. Remember that columns 'star1_x' and 'star2_x' will always be zero since the stars are projected onto the yz plane.\n2. A code interpreter with `df` pre-loaded that can execute Python code."
            self.prompt = f"""You are tasked with solving the following physics problem related to a binary star system. You are provided observations of each star's position projected onto the y-z plane over time, (t,0,y,z), in units of {self.units_string}. You have to take into account the angle of inclination with respect to the xy-plane, the longitude of ascending node with resecpect to the positive x-axis and the arguemnt of periapsis, which can greatly affect the problem you are tasked to solve.

### Problem Description
{self.task}
{self.final_answer_prompt}

### Additional Instructions
To complete this task, you have access to the following tools and data:"""
        
        # Common prompt ending
        self.end_prompt = f"""

When using the code interpreter:
1. Always use print() to display results.
2. Do not use read_csv or attempt to load the DataFrame, as it is already pre-loaded.

Important reminder: Repeated tool access is enabled until you have found the answer and have submitted it with the `submit_answer` tool."""
        
        # Convert all parameters to SI units for answer validation
        self.convert_back_to_SI()
        # Configure full prompt for table access mode
        self.full_table_prompt = self.prompt + f"""\n{self.full_table_tools_and_data_prompt}""" + self.end_prompt

    def set_row_wise_prompt(self):
        """Configure prompt for row-wise observation mode"""
        self.row_wise_prompt = (self.prompt + f"""
1. An observational tool called `Observe` that allows you observe the system at
specific times of your choosing.
2. A code interpreter that can execute Python code.

When using `Observe`:
1. The `times_requested` parameter should be a list that can contain any values in the time window [0.0, {self.maxtime:.2e}] seconds. You cannot request negative times. The upper limit for the time window was chosen to gurantee that the problem is solvable with an appropriate sampling of observations using the total observational budget.
2. You can observe the system at any time within the time window, even if it is in the past compared to the last observation.
3. You can observe the system up to a total of {self.max_observations} times and you can observe up to {self.max_observations_per_request} times per observational request which is the maximum length of the `times_requested` list.
4. After each observation, the dataframe `row_wise_results.df` will be updated. It contains columns: {', '.join(self.df.columns)}. You can access it using the code interpreter tool. For example, to access the first five rows, print(row_wise_results.df.head(n=5))""" 
+ self.end_prompt)


    def simulate(self, drag_tau=None, mod_gravity_exponent=None, units=('m', 's', 'kg')):
        """Run N-body simulation and save results to CSV files"""
        self.sim = rebound.Simulation()
        self.sim.integrator = "whfast"  # Default symplectic integrator
        self.sim.units = units

        # Calculate initial orbital parameters
        r = np.sqrt(sum((np.array(self.star2_pos) - np.array(self.star1_pos))**2))
        M = self.star1_mass + self.star2_mass
        # Estimate orbital period using Kepler's third law
        T = 2 * np.pi * np.sqrt(r**3 / (self.sim.G * M))

        # Time step
        self.sim.dt = T / 5000  # 5000 steps per orbit
        total_steps = int(self.maxtime / self.sim.dt)
        # Adjust time step if needed for special cases
        if total_steps < 1000 or mod_gravity_exponent is not None:
            self.sim.dt = self.maxtime / 5000

        # Configure drag forces if specified
        if drag_tau is not None:
            self.sim.integrator = "ias15"  # Switch to adaptive integrator for non-conservative forces
            def apply_linear_drag(sim, particles, N=2):
                """Apply velocity-dependent linear drag force"""
                for i in range(N):
                    # Drag force proportional to velocity: F_drag = -v/τ
                    particles[i].ax -= particles[i].vx / drag_tau
                    particles[i].ay -= particles[i].vy / drag_tau
                    particles[i].az -= particles[i].vz / drag_tau
            self.sim.additional_forces = lambda reb_sim: apply_linear_drag(reb_sim, self.sim.particles)
            self.sim.force_is_velocity_dependent = 1  # Required for velocity-dependent forces
        
        # Configure modified gravity if specified
        if mod_gravity_exponent is not None:
            self.sim.integrator = "ias15"  # Adaptive integrator for non-Newtonian forces
            def mod_gravity(reb, particles, N, mod_gravity_exponent):
                """Custom force implementation for modified gravity"""
                for i in range(N):
                    for j in range(i+1, N):
                        # Calculate separation vector
                        dx = particles[j].x - particles[i].x
                        dy = particles[j].y - particles[i].y
                        dz = particles[j].z - particles[i].z
                        r = np.sqrt(dx**2 + dy**2 + dz**2)
                        # Modified gravity force: F ∝ 1/r^mod_gravity_exponent
                        F = self.sim.G * particles[i].m * particles[j].m / r**mod_gravity_exponent
                        # Apply forces to both particles
                        particles[i].ax = F * dx / (particles[i].m * r)
                        particles[i].ay = F * dy / (particles[i].m * r)
                        particles[i].az = F * dz / (particles[i].m * r)
                        particles[j].ax = -F * dx / (particles[j].m * r)
                        particles[j].ay = -F * dy / (particles[j].m * r)
                        particles[j].az = -F * dz / (particles[j].m * r)
            self.sim.additional_forces = lambda reb: mod_gravity(reb, self.sim.particles, 
                                                               N=2, mod_gravity_exponent=self.mod_gravity_exponent)

        # Add stars to simulation with initial conditions
        self.sim.add(m=self.star1_mass, x=self.star1_pos[0], y=self.star1_pos[1], z=self.star1_pos[2], 
                     vx=self.star1_momentum[0] / self.star1_mass, 
                     vy=self.star1_momentum[1] / self.star1_mass, 
                     vz=self.star1_momentum[2] / self.star1_mass)

        self.sim.add(m=self.star2_mass, x=self.star2_pos[0], y=self.star2_pos[1], z=self.star2_pos[2], 
                     vx=self.star2_momentum[0] / self.star2_mass, 
                     vy=self.star2_momentum[1] / self.star2_mass, 
                     vz=self.star2_momentum[2] / self.star2_mass)

        # Set up output files
        csv_file_positions = f"scenarios/sims/{self.filename}.csv"
        csv_file_detailed = f"scenarios/detailed_sims/{self.filename}.csv"

        with open(csv_file_positions, mode='w', newline='') as file_positions, \
             open(csv_file_detailed, mode='w', newline='') as file_detailed:
            
            # Initialize CSV writers
            writer_positions = csv.writer(file_positions)
            writer_detailed = csv.writer(file_detailed)

            # Write CSV headers
            header_positions = ['time', 'star1_x', 'star1_y', 'star1_z', 'star2_x', 'star2_y', 'star2_z']
            header_detailed = [
                'time', 'star1_x', 'star1_y', 'star1_z', 'star2_x', 'star2_y', 'star2_z',
                'star1_vx', 'star1_vy', 'star1_vz', 'star2_vx', 'star2_vy', 'star2_vz', 
                'star1_mass', 'star2_mass', 'separation', 'force', 'star1_accel', 'star2_accel',
                'specific_angular_momentum', 'orbital_period', 'mean_motion', 'semimajor_axis', 
                'eccentricity', 'inclination', 'longitude_of_ascending_node', 'argument_of_periapsis','true_anomaly', 'mean_anomaly', 
                'time_of_pericenter_passage', 'radial_distance_from_reference'
            ]
            writer_positions.writerow(header_positions)
            writer_detailed.writerow(header_detailed)

            # Main simulation loop
            time_passed = 0
            while time_passed < self.maxtime:
                self.sim.integrate(self.sim.t + self.sim.dt)
                time_passed += self.sim.dt

                # Get current particle states
                p1 = self.sim.particles[0]
                p2 = self.sim.particles[1]

                # Write basic position data
                data_positions = [time_passed, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z]
                writer_positions.writerow(data_positions)

                # Calculate detailed orbital parameters
                separation = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
                force = self.sim.G * p1.m * p2.m / separation**2  # Newtonian force
                star1_accel = force / p1.m
                star2_accel = force / p2.m
                orbit = p2.orbit(primary=p1)  # Calculate orbital elements

                # Write detailed simulation data
                data_detailed = [
                    time_passed,
                    p1.x, p1.y, p1.z, p2.x, p2.y, p2.z,
                    p1.vx, p1.vy, p1.vz, p2.vx, p2.vy, p2.vz,
                    p1.m, p2.m, separation, force, star1_accel, star2_accel,
                    orbit.h, orbit.P, orbit.n, orbit.a, orbit.e,
                    orbit.inc, orbit.Omega, orbit.omega, orbit.f, orbit.M, orbit.T, orbit.d
                ]
                writer_detailed.writerow(data_detailed)

    def convert_back_to_SI(self):
        """Convert all parameters to SI units for final answer validation"""
        # Return early if already in SI units
        if self.units == ('m', 's', 'kg'):
            return
        
        # Validate supported unit systems
        if self.units not in [('m', 's', 'kg'), ('yr', 'AU', 'Msun'), ('s', 'cm', 'g')]:
            raise ValueError("Unsupported unit system. Only ('m', 's', 'kg'), ('yr', 'AU', 'Msun'), and ('s', 'cm', 'g') are supported.")
        
        # Clean filename from unit identifiers
        self.filename = self.filename.replace(", cgs", "").replace(", yrAUMsun", "")
        
        # Conversion factors for astronomical units
        if self.units == ('yr', 'AU', 'Msun'):
            # Mass: Solar masses to kilograms
            self.star1_mass *= 1.989e30
            self.star2_mass *= 1.989e30
            
            # Position: AU to meters
            self.star1_pos = [pos * 1.496e11 for pos in self.star1_pos]
            self.star2_pos = [pos * 1.496e11 for pos in self.star2_pos]
            
            # Momentum: Msun AU/yr to kg m/s
            self.star1_momentum = [mom * 1.989e30 * 1.496e11 / 3.154e7 for mom in self.star1_momentum]
            self.star2_momentum = [mom * 1.989e30 * 1.496e11 / 3.154e7 for mom in self.star2_momentum]
            
            # Time: Years to seconds
            self.maxtime *= 3.154e7
        
        # Conversion factors for CGS units
        elif self.units == ('s', 'cm', 'g'):
            # Mass: grams to kilograms
            self.star1_mass *= 0.001
            self.star2_mass *= 0.001
            
            # Position: centimeters to meters
            self.star1_pos = [pos * 0.01 for pos in self.star1_pos]
            self.star2_pos = [pos * 0.01 for pos in self.star2_pos]
            
            # Momentum: g cm/s to kg m/s
            self.star1_momentum = [mom * 1e-3 * 1e-2 for mom in self.star1_momentum]
            self.star2_momentum = [mom * 1e-3 * 1e-2 for mom in self.star2_momentum]

        # Update unit system to SI
        self.units = ('m', 's', 'kg')
        self.sim.G = 6.67430e-11  # SI gravitational constant

    def observe_row(self, times_requested: Union[float, List[float]], maximum_observations_per_request: int) -> str:
        """
        Generate interpolated observations at requested times
        Args:
            times_requested: List of observation times
            maximum_observations_per_request: Max allowed per request
            face_on_projection: Projection of stars onto yz plane
        Returns:
            Status message with observation results and remaining budget
        """
        # Normalize input to list and ensure all values are floats
        if not isinstance(times_requested, list):
            times_requested = [float(times_requested)]
        else:
            times_requested = [float(t) for t in times_requested]

        # Validate request limit
        if len(times_requested) > maximum_observations_per_request:
            return f"You can only request a maximum of {maximum_observations_per_request} observations per request. Try again with fewer observations."
        
        # Load full simulation data
        df = pd.read_csv(f"scenarios/sims/{self.filename}.csv")
        
        # Calculate available observations within budget
        remaining_observations = self.max_observations - self.number_of_observations_requested
        observations_to_process = min(len(times_requested), remaining_observations)
        times_to_process = times_requested[:observations_to_process]
        self.number_of_observations_requested += observations_to_process

        observations = []
        max_time_exceeded = False
        negative_time_exceeded = False
        
        # Process each requested time
        for time in times_to_process:
            # Validate time bounds
            if time > 1.01*self.maxtime:
                max_time_exceeded = True
                observations.append([time, None, None, None, None, None, None])
            elif time < 0:
                negative_time_exceeded = True
                observations.append([time, None, None, None, None, None, None])
            else:
                # Find 4 closest points for cubic interpolation
                closest_rows = df.iloc[(df['time'] - time).abs().argsort()[:4]]
                closest_rows = closest_rows.sort_values('time')
                
                # Get times and positions for interpolation
                times = closest_rows['time'].values
                x1 = closest_rows['star1_x'].values
                y1 = closest_rows['star1_y'].values
                z1 = closest_rows['star1_z'].values
                x2 = closest_rows['star2_x'].values
                y2 = closest_rows['star2_y'].values
                z2 = closest_rows['star2_z'].values

                # Cubic spline interpolation for each coordinate
                cs_x1 = CubicSpline(times, x1)
                cs_y1 = CubicSpline(times, y1)
                cs_z1 = CubicSpline(times, z1)
                cs_x2 = CubicSpline(times, x2)
                cs_y2 = CubicSpline(times, y2)
                cs_z2 = CubicSpline(times, z2)

                # Get interpolated values and check for projection onto the yz plane
                if self.face_on_projection:
                    self.state = np.array([time, 
                                     0, cs_y1(time), cs_z1(time),
                                     0, cs_y2(time), cs_z2(time)])
                else: 
                    self.state = np.array([time, 
                                     cs_x1(time), cs_y1(time), cs_z1(time),
                                     cs_x2(time), cs_y2(time), cs_z2(time)])
            observations.append(self.state)

                

        # Update observational dataframe
        new_rows = pd.DataFrame(observations, columns=['time', 'star1_x', 'star1_y', 'star1_z', 'star2_x', 'star2_y', 'star2_z'])
        self.row_wise_results.df = pd.concat([self.row_wise_results.df, new_rows], ignore_index=True)

        # Build result message
        result = ""
        if observations_to_process == 0:
            result = "\nYou have reached the maximum number of observations and can no longer observe the system."
        else:
            result = f"\nObservations added to row_wise_results.df. "
            if observations_to_process < len(times_requested):
                result += f"Only {observations_to_process} out of {len(times_requested)} requested observations were added due to reaching the maximum observation limit. "
            result += f"You have {self.max_observations - self.number_of_observations_requested} observations remaining in your total budget. "
        
        # Add warnings if needed
        if max_time_exceeded:
            result += f"\nNote: Some requested times exceeded the maximum time of {self.maxtime}. For these times, None values were inserted for positions."
        if negative_time_exceeded:
            result += "\nNote: Some requested times were negative. For these times, None values were inserted for positions."
        
        return result