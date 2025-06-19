import numpy as np
import pandas as pd
import scripts.task_utils as task_utils

class Scenario:
    def __init__(self, scenario_creator, skip_simulation=False):
        self.scenario_creator = scenario_creator

        prompt = """Calculate the minimum absolute value of linear momentum for star1 over the orbit."""
        final_answer_units = "kg*m/s"

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
        star1_vx, star1_vy, star1_vz, _, _, _ = task_utils.calculate_velocities(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
        
        # Calculate velocity magnitude for star1
        velocity_star1 = np.sqrt(star1_vx**2 + star1_vy**2 + star1_vz**2)
        
        # Calculate masses using task_utils
        m1, _ = task_utils.star_masses(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
        
        # Calculate momentum
        momentum_star1 = velocity_star1 * m1
        min_momentum = np.min(momentum_star1)
        
        # Verification and return_empirical are done in calculate_velocities and star_masses
        return min_momentum
