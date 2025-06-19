import numpy as np
import pandas as pd
import scripts.task_utils as task_utils

class Scenario:
    def __init__(self, scenario_creator, skip_simulation=False):
        self.scenario_creator = scenario_creator

        prompt = """Determine the total mass of the system."""
        final_answer_units = "kg"

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

        # Calculate masses using task_utils
        m1, m2 = task_utils.star_masses(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
        total_mass = m1 + m2

        if verification:
            # Assert total_mass is within 2% of reference values
            assert abs(total_mass - (self.binary_sim.star1_mass + self.binary_sim.star2_mass)) < 0.02 * (self.binary_sim.star1_mass + self.binary_sim.star2_mass), \
                f"{total_mass} and {self.binary_sim.star1_mass + self.binary_sim.star2_mass} are not within 2% of each other"

        if return_empirical:
            return total_mass
        else:
            return self.binary_sim.star1_mass + self.binary_sim.star2_mass