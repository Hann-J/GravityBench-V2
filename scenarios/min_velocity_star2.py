import numpy as np
import pandas as pd
import scripts.task_utils as task_utils

class Scenario:
    def __init__(self, scenario_creator, skip_simulation=False):
        self.scenario_creator = scenario_creator

        prompt = """Calculate the minimum absolute value of velocity for star2 over the orbit."""
        final_answer_units = "m/s"

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

        # Calculate velocities using task_utils
        _, _, _, star2_vx, star2_vy, star2_vz = task_utils.calculate_velocities(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
        
        # Calculate velocity magnitude for star2
        velocity_star2 = np.sqrt(star2_vx**2 + star2_vy**2 + star2_vz**2)
        min_velocity = np.min(velocity_star2)
        
        # Verification and return_empirical are done in calculate_velocities
        return min_velocity
