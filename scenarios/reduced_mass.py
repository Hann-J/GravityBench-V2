import numpy as np
import pandas as pd
import scripts.task_utils as task_utils

class Scenario:
    def __init__(self, scenario_creator, skip_simulation=False):
        self.scenario_creator = scenario_creator

        prompt = """Determine the reduced mass of the equivalent one-body problem of the system."""
        final_answer_units = "kg"

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

        # Calculate masses using task_utils
        m1, m2 = task_utils.star_masses(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
        
        # Calculate reduced mass
        reduced_mass = m1 * m2 / (m1 + m2)
        reduced_mass_ref = self.binary_sim.star1_mass * self.binary_sim.star2_mass / (self.binary_sim.star1_mass + self.binary_sim.star2_mass)
        
        if verification:
            # Verification
            assert abs(reduced_mass - reduced_mass_ref) < 0.02 * reduced_mass_ref, f"{reduced_mass} and {reduced_mass_ref} are not within 2% of each other"

        if return_empirical:
            return reduced_mass
        else:
            return reduced_mass_ref
