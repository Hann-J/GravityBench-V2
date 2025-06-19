import numpy as np
import pandas as pd
import scripts.task_utils as task_utils
from scipy.optimize import curve_fit

class Scenario:
    def __init__(self, scenario_creator, skip_simulation=False):
        self.scenario_creator = scenario_creator

        prompt = """This system is governed by an alternative law of gravitation where the r dependence is r^(-(2 + alpha)) where alpha represents the deviation from Newton's inverse square law. Calculate alpha."""
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

        # Calculate separation distance
        df['r'] = np.sqrt((df['star2_x'] - df['star1_x'])**2 +
                         (df['star2_y'] - df['star1_y'])**2 +
                         (df['star2_z'] - df['star1_z'])**2)

        # Calculate accelerations using task_utils
        _, acc_star2 = task_utils.calculate_accelerations(df, self.binary_sim, verification=True, return_empirical=return_empirical)
        acc_star2 = np.nan_to_num(acc_star2, nan=np.nanmean(acc_star2))

        def f(x, m, b):
            return m*x + b
        # log(r) = mod_gravity_exponent * log(a) + constants (including masses and G)

        # Create mask for separations above median
        median_separation = np.median(df['r'])
        separation_mask = df['r'] > median_separation

        # Initial fit to compute residuals
        x_data = np.log(df['r'])  # Define x_data for fitting
        y_data = np.log(acc_star2)  # Define y_data for fitting
        popt_initial, _ = curve_fit(f, x_data, y_data)
        residuals = y_data - f(x_data, *popt_initial)

        # Outlier removal using MAD
        median_residual = np.median(residuals)
        mad = np.median(np.abs(residuals - median_residual))
        outlier_mask = np.abs(residuals - median_residual) < (mad)

        # Combine masks
        final_mask = outlier_mask & separation_mask

        # Final fit using cleaned data
        popt, _ = curve_fit(f, x_data[final_mask], y_data[final_mask])
        predicted_mod_gravity_exponent = abs(popt[0])

        if verification:
            assert abs(predicted_mod_gravity_exponent - self.binary_sim.mod_gravity_exponent) < 0.03 * self.binary_sim.mod_gravity_exponent, f'{predicted_mod_gravity_exponent} and {self.binary_sim.mod_gravity_exponent} are not within 3% of each other'
        if return_empirical:
            return predicted_mod_gravity_exponent - 2.0
        else:
            return self.binary_sim.mod_gravity_exponent - 2.0

