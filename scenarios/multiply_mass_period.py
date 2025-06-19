import numpy as np
import pandas as pd
import scripts.task_utils as task_utils

class Scenario:
    def __init__(self, scenario_creator, skip_simulation=False):
        self.scenario_creator = scenario_creator

        prompt = """Determine the factor X by which the central mass should be multipled for the orbital period of the system to be 21 days. You can assume the central mass is star1 and is much larger than star2."""
        final_answer_units = None

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
        # Load simulation data
        df = pd.read_csv(f"scenarios/detailed_sims/{self.binary_sim.filename}.csv")
        
        if N_obs is not None:
            indices = np.linspace(0, len(df) - 1, N_obs)
            df = df.iloc[indices].reset_index(drop=True)

        # Calculate period using task_utils
        # Verification and return_empirical are done in calculate_period
        period_data = task_utils.calculate_period(df, self.binary_sim, verification=verification, return_empirical=return_empirical)

        # Find the factor X by which the central mass should be multiplied for the period of the system to be 21 days
        # Assuming star1_mass is much larger than star2_mass
        # Then Mf/Mi = (Pi/Pf)^2 by Kepler's Third Law
        # So X = (period/21 days)^2
        target_period = 21 * 24 * 3600  # Convert 21 days to seconds
        
        return round((period_data / target_period)**2, 2)
