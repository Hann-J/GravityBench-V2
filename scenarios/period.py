import numpy as np
import pandas as pd
import scripts.task_utils as task_utils

class Scenario:
    def __init__(self, scenario_creator, face_on_projection=False, skip_simulation=False):
        self.scenario_creator = scenario_creator
        self.face_on_projection = face_on_projection

        prompt = """Determine the orbital period of the system."""
        final_answer_units = "s"

        self.binary_sim = self.scenario_creator.create_binary(prompt, final_answer_units, skip_simulation=skip_simulation)

    def true_answer(self, N_obs=None, verification=True, return_empirical=False) -> float:
        """
        Calculate the orbital period of the binary system.
        """
        # Load simulation data
        df = pd.read_csv(f"scenarios/detailed_sims/{self.binary_sim.filename}.csv")
        
        if N_obs is not None:
            indices = np.linspace(0, len(df) - 1, N_obs)
            df = df.iloc[indices].reset_index(drop=True)

        # If face_on_projection is False, we have the full data to calculate the period
        if not self.face_on_projection:
            # Calculate period using task_utils
            # Verification and return_empirical are done in calculate_period
            period = task_utils.calculate_period(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
            return period
        else:
            # We can get a period from the projection similarly with task_utils but with z=0, period is invariant under projection
            df['star1_x'] = 0
            df['star2_x'] = 0
            period = task_utils.calculate_period(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
            return period
            