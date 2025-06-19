import numpy as np
import pandas as pd
import scripts.task_utils as task_utils

class Scenario:
    def __init__(self, scenario_creator, skip_simulation=False):
        self.scenario_creator = scenario_creator

        prompt = """Determine the mass ratio of the system with star1 in the numerator and star2 in the denominator. Express your answer as a float decimal."""
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

        # Calculate masses using task_utils
        M1, M2 = task_utils.star_masses(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
        
        # Calculate mass ratio
        mass_ratio = M1 / M2
        
        if verification:
            # Assert estimated mass ratio is within 2% of the true mass ratio
            true_ratio = self.binary_sim.star1_mass/self.binary_sim.star2_mass
            assert abs(mass_ratio - true_ratio) < 0.02 * true_ratio, \
                   f"{mass_ratio} and {true_ratio} are not within 2% of each other"

        if return_empirical:
            return mass_ratio
        else:
            return self.binary_sim.star1_mass/self.binary_sim.star2_mass
