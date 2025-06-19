import numpy as np
import pandas as pd
import scripts.task_utils as task_utils

class Scenario:
    def __init__(self, scenario_creator, skip_simulation=False):
        self.scenario_creator = scenario_creator

        prompt = """Calculate the fraction of time in a single orbit during which the acceleration of star1 is below the mean acceleration."""
        final_answer_units = None

        self.binary_sim = self.scenario_creator.create_binary(prompt, final_answer_units, skip_simulation=skip_simulation)
    def true_answer(self, N_obs=None, verification=True, return_empirical=False) -> float:
        """
        Calculate the fraction of time in a single orbit where the acceleration of star1 is below the mean acceleration.
        """
        # Load simulation data
        df = pd.read_csv(f"scenarios/detailed_sims/{self.binary_sim.filename}.csv")
        
        if N_obs is not None:
            indices = np.linspace(0, len(df) - 1, N_obs)
            df = df.iloc[indices].reset_index(drop=True)

        # Get period and time of pericenter passage
        period = task_utils.calculate_period(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
        time_pericenter_pass = task_utils.calculate_time_of_pericenter_passage(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
        
        # Isolate a single orbit
        df_orbit = df[(df['time'] > time_pericenter_pass) & (df['time'] < time_pericenter_pass + period)]
        
        # Calculate acceleration of star1
        acc_star1, _ = task_utils.calculate_accelerations(df_orbit, self.binary_sim, verification=verification, return_empirical=return_empirical)
        mean_acceleration = np.mean(acc_star1)
        
        # Find times when acceleration is below mean
        below_mean = acc_star1 < mean_acceleration
        df_below_mean = df_orbit[below_mean]
        time_below_mean = np.sum(df_below_mean['time'].diff().dropna().values)
        
        fraction = time_below_mean / period
        
        # Load detailed simulation data for verification
        df_detailed = pd.read_csv(f"scenarios/detailed_sims/{self.binary_sim.filename}.csv")
        period_detailed = df_detailed['orbital_period'].iloc[0]
        time = df_detailed['time']
        idx_after_one_period = np.argmax(time > period_detailed)
        time_pericenter_pass_detailed = df_detailed['time_of_pericenter_passage'].iloc[idx_after_one_period+1]
        
        df_orbit_detailed = df_detailed[(df_detailed['time'] > time_pericenter_pass_detailed) & 
                                        (df_detailed['time'] < time_pericenter_pass_detailed + period_detailed)]
        
        acceleration_detailed = df_orbit_detailed['star1_accel'].values
        mean_acceleration_detailed = acceleration_detailed.mean()
        below_mean_detailed = acceleration_detailed < mean_acceleration_detailed
        df_below_mean_detailed = df_orbit_detailed[below_mean_detailed]
        time_below_mean_detailed = np.nansum(df_below_mean_detailed['time'].diff().values)
        fraction_detailed = time_below_mean_detailed / period_detailed
        if verification:
            assert abs(fraction - fraction_detailed) < 0.02 * fraction_detailed, \
                   f"{fraction} and {fraction_detailed} are not within 2% of each other"

        if return_empirical:
            return fraction
        else:
            return fraction_detailed
