import numpy as np
import pandas as pd
import scripts.task_utils as task_utils

class Scenario:
    def __init__(self, scenario_creator, skip_simulation=False):
        self.scenario_creator = scenario_creator

        prompt = """Calculate the minimum absolute value of angular velocity for star2 over the orbit."""
        final_answer_units = "radian/s"

        self.binary_sim = self.scenario_creator.create_binary(prompt, final_answer_units, skip_simulation=skip_simulation)
    
    def true_answer(self, N_obs=None, verification=True, return_empirical=False) -> float:
        """
        Return the true answer for the environment.
        """
        # Load simulation data
        df = pd.read_csv(f"scenarios/detailed_sims/{self.binary_sim.filename}.csv")
        
        if N_obs is not None:
            indices = np.linspace(0, len(df) - 1, N_obs)
            df = df.iloc[indices].reset_index(drop=True)

        time = df['time'].values
        
        # Calculate angular velocity using cross product method
        star1_pos = df[['star1_x','star1_y','star1_z']].values
        star2_pos = df[['star2_x','star2_y','star2_z']].values
        rel_pos = star1_pos - star2_pos
        
        # Calculate velocities by numerical differentiation
        dt = time[1] - time[0]
        rel_vel = np.gradient(rel_pos, dt, axis=0)
        
        # Calculate angular velocity using cross product
        angular_velocity = np.zeros(len(rel_vel))
        cross_val = np.cross(rel_pos, rel_vel)
        angular_velocity = np.linalg.norm(cross_val, axis=1) / np.linalg.norm(rel_pos, axis=1)**2
            
        # Calculate angular velocity using angular momentum method
        h = df['specific_angular_momentum'].values
        r = df['separation'].values
        omega_h = h / (r**2)
        
        if verification:
            min_omega_h = np.min(omega_h)
            min_omega_cross = np.min(angular_velocity)
            assert abs(min_omega_h - min_omega_cross) < 0.02 * min_omega_h, \
                   f"{min_omega_h} and {min_omega_cross} are not within 2% of each other"
            
        if return_empirical:
            return np.min(angular_velocity)
        else:
            return np.min(omega_h)

