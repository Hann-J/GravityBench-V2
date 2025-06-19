import rebound
import numpy as np
import pandas as pd
import scripts.task_utils as task_utils

#Longitude of ascending node, as calculataed in rebound, as the angle between orbital plane and the positive x-axis
#Angle will be zero if inclination is zero

class Scenario:
    def __init__(self, scenario_creator, face_on_projection=False, skip_simulation=False):
        self.scenario_creator = scenario_creator
        self.face_on_projection = face_on_projection

        prompt = """Determine the longitude of ascending node of the system's orbit. Take the positive x-axis as the reference direction."""
        final_answer_units = "rad"

        self.binary_sim = self.scenario_creator.create_binary(prompt, final_answer_units, skip_simulation=skip_simulation)

    def true_answer(self, N_obs=None, verification=True, return_empirical=False) -> float:
        """
        Return the true answer for the environment.
        
        Args:
            N_obs: Number of observations to use (if None, use all)
            verification: Whether to verify values match
            return_empirical: If True, return the empirically derived value;
                              if False, return the value inputted into the simulation or using Rebound simulated details typically hidden
        """
        # Load the simulation data
        df = pd.read_csv(f"scenarios/detailed_sims/{self.binary_sim.filename}.csv")
        
        if N_obs is not None:
            indices = np.linspace(0, len(df) - 1, N_obs)
            df = df.iloc[indices].reset_index(drop=True)

        # If face_on_projection is False
        if not self.face_on_projection:
            # If inclination is zero, the longitude of ascending node will also be zero, as described by the rebound package
            if (df['star1_z'] == 0).all() and (df['star2_z'] == 0).all():
                empirical_long = 0

            else: 
                # Calculate the unit vector of the orbital plane with angular momentum vector
                # Calculate relative positions
                df['rel_x'] = df['star2_x'] - df['star1_x']
                df['rel_y'] = df['star2_y'] - df['star1_y']
                df['rel_z'] = df['star2_z'] - df['star1_z']
            
                # Calculate relative velocities using task_utils
                _, _, _, star2_vx, star2_vy, star2_vz = task_utils.calculate_velocities(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
                star1_vx, star1_vy, star1_vz, _, _, _ = task_utils.calculate_velocities(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
                
                df['rel_vx'] = star2_vx - star1_vx
                df['rel_vy'] = star2_vy - star1_vy
                df['rel_vz'] = star2_vz - star1_vz
                
                # Compute the specific angular momentum components
                h_x = (df['rel_y'] * df['rel_vz'] - df['rel_z'] * df['rel_vy']).mean()
                h_y  = (df['rel_z'] * df['rel_vx'] - df['rel_x'] * df['rel_vz']).mean()
                h_z = (df['rel_x'] * df['rel_vy'] - df['rel_y'] * df['rel_vx']).mean()
            
                # Compute the unit vector of specific angular momentum vector
                h_magnitude = np.sqrt(h_x**2 + h_y**2 + h_z**2)
                h = np.array([h_x, h_y, h_z]) / h_magnitude
                #vector pointing to the ascending node is perpendicular to the specific angular momentum vector and lies in the xy-plane
                n = np.cross([0, 0, 1], h)  # Cross product with z-axis unit vector to get the node vector in the xy-plane

                # Calculate the longitude of ascending node
                empirical_long = np.arctan2(n[1], n[0])
        else:
            # x = 0
            pass


        # verification, the angle is usually constant throughout the simulation
        Omega_rebound = df['longitude_of_ascending_node'].mean()

        if verification:
            assert abs(empirical_long - Omega_rebound) < 0.01 * abs(Omega_rebound), f"{empirical_long} and {Omega_rebound} are not within 1% of each other"
        
        if return_empirical:
            return empirical_long  # Return the calculated inclination if empirical value is requested
        else:
            return Omega_rebound  # Return the rebound calculated inclination if not requesting empirical value
