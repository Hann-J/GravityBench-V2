import numpy as np
import pandas as pd
import scripts.task_utils as task_utils

class Scenario:
    def __init__(self, scenario_creator, skip_simulation=False):
        self.scenario_creator = scenario_creator

        prompt = """Determine if Kepler's third law is satisfied. Answer: True if Kepler's third law is satisfied, and Answer: False if it is not."""
        final_answer_units = None

        self.binary_sim = self.scenario_creator.create_binary(prompt, final_answer_units, skip_simulation=skip_simulation)

    def true_answer(self, N_obs=None, verification=True, return_empirical=False) -> bool:
        """
        Return the true answer for the environment.
        """
        # Load simulation data
        df = pd.read_csv(f"scenarios/detailed_sims/{self.binary_sim.filename}.csv")
        
        if N_obs is not None:
            indices = np.linspace(0, len(df) - 1, N_obs)
            df = df.iloc[indices].reset_index(drop=True)

        # Calculate masses using task_utils
        if 'Modified Gravity' in self.binary_sim.filename:
            m1, m2 = task_utils.star_masses(df, self.binary_sim, verification=False, return_empirical=True)
            period = task_utils.calculate_period(df, self.binary_sim, verification=False, return_empirical=True)
            semi_major_axis, _, _ = task_utils.calculate_semi_major_axes(df, m1, m2, self.binary_sim, verification=False, return_empirical=True)
        else:
            m1, m2 = task_utils.star_masses(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
            period = task_utils.calculate_period(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
            semi_major_axis, _, _ = task_utils.calculate_semi_major_axes(df, m1, m2, self.binary_sim, verification=verification, return_empirical=return_empirical)

        # Calculate P^2/a^3 and 4π^2/(G(M1+M2))
        P2_a3 = period**2 / semi_major_axis**3
        four_pi2_G_m1m2 = 4*np.pi**2 / (self.binary_sim.sim.G*(m1 + m2))
        percent_diff = abs(P2_a3 - four_pi2_G_m1m2) / four_pi2_G_m1m2 * 100

        if verification:
            assert 'Modified Gravity' not in self.binary_sim.filename or not bool(percent_diff < 0.1), \
                f"Kepler's third law determined to be satisfied (percent diff = {percent_diff:.2f}%), " \
                f"but modified gravity means it should not be satisfied."
        return bool(percent_diff < 0.1)
