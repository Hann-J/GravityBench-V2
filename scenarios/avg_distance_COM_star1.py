import numpy as np
import pandas as pd
import scripts.task_utils as task_utils

class Scenario:
    def __init__(self, scenario_creator, skip_simulation=False):
        self.scenario_creator = scenario_creator

        prompt = """Calculate the time-averaged distance between star1 and the Center of Mass over a single orbit."""
        final_answer_units = "m"

        self.binary_sim = self.scenario_creator.create_binary(prompt, final_answer_units, skip_simulation=skip_simulation)

    def true_answer(self, N_obs=None, verification=True, return_empirical=False) -> float:
        """
        Calculate the time-averaged distance between star1 and the COM over a single orbit.
        NOTE: Can also be found from a(1 + 1/2 e^2)
        
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

        # Get masses for COM calculation
        m1, m2 = task_utils.star_masses(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
        total_mass = m1 + m2

        # Calculate COM coordinates
        df['COMx'] = (m1*df['star1_x'] + m2*df['star2_x'])/total_mass
        df['COMy'] = (m1*df['star1_y'] + m2*df['star2_y'])/total_mass
        df['COMz'] = (m1*df['star1_z'] + m2*df['star2_z'])/total_mass

        # Get period and time of pericenter passage
        period = task_utils.calculate_period(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
        time_pericenter_pass = task_utils.calculate_time_of_pericenter_passage(df, self.binary_sim, verification=verification, return_empirical=return_empirical)

        # Isolate a single orbit
        mask = (df['time'] > time_pericenter_pass) & (df['time'] < time_pericenter_pass + period)

        # Calculate distance between star1 and COM
        df.loc[mask, 'star1_distance'] = np.sqrt(
            (df.loc[mask, 'star1_x'] - df.loc[mask, 'COMx'])**2 + 
            (df.loc[mask, 'star1_y'] - df.loc[mask, 'COMy'])**2 + 
            (df.loc[mask, 'star1_z'] - df.loc[mask, 'COMz'])**2
        )

        df_orbit = df[mask].copy()
        star1_distance = df_orbit['star1_distance'].values
        time = df_orbit['time'].values

        # Integrating, just in case dt is not constant
        avg_distance = np.trapz(star1_distance, time)/(time[-1] - time[0])

        # Calculate theoretical value using semi-major axis and eccentricity
        a, a1, _ = task_utils.calculate_semi_major_axes(df, m1, m2, self.binary_sim, verification=verification, return_empirical=return_empirical)
        e = task_utils.calculate_eccentricity(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
        theoretical_avg = a1 * (1 + e**2/2)

        if verification:
            assert abs(avg_distance - theoretical_avg) < 0.02 * theoretical_avg, \
                f"{avg_distance} and {theoretical_avg} are not within 2% of each other"

        if return_empirical:
            return avg_distance
        else:
            return theoretical_avg
