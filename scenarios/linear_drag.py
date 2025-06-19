import numpy as np
import pandas as pd
import scripts.task_utils as task_utils

class Scenario:
    def __init__(self, scenario_creator, skip_simulation=False):
        self.scenario_creator = scenario_creator

        prompt = """This system experiences an isotropic drag given by a_i = -v_i/tau, applied in all i-direction, where a is the acceleration. Calculate the value of the coefficient of linear drag, tau, for the system."""
        final_answer_units = "s"

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
        # Empirical Solution
        # Using the fact that:
        # v' = v/tau -> v propto exp(t/tau)
        # Also, v_circ = sqrt(GMM/a) -> a propto 1/(v_circ)^2
        # So: a propto 1/(exp(t/tau))^2 = exp(-2t/tau) where a is the semi-major axis
        
        df = pd.read_csv(f'scenarios/detailed_sims/{self.binary_sim.filename}.csv')
        
        if N_obs is not None:
            indices = np.linspace(0, len(df) - 1, N_obs)
            df = df.iloc[indices].reset_index(drop=True)
            
        import scipy.signal
        separation = np.sqrt((df['star1_x'] - df['star2_x'])**2 + 
                           (df['star1_y'] - df['star2_y'])**2 + 
                           (df['star1_z'] - df['star2_z'])**2)
        peaks, _ = scipy.signal.find_peaks(separation)
        troughs, _ = scipy.signal.find_peaks(-separation)

        if len(peaks) > len(troughs):
            peaks = peaks[:len(troughs)]
        elif len(troughs) > len(peaks):
            troughs = troughs[:len(peaks)]

        peak_distances = separation[peaks].values
        trough_distances = separation[troughs].values
        semi_major_axis = (peak_distances + trough_distances) / 2
        times_peaks = df['time'].values[peaks]
        times_troughs = df['time'].values[troughs]
        times = (times_peaks + times_troughs) / 2

        from scipy.optimize import curve_fit
        def exp_fit(t, a, tau):
            return a * np.exp(-2*t/tau)
        
        popt, pcov = curve_fit(exp_fit, times, semi_major_axis, 
                              p0=[np.nanmean(semi_major_axis), np.nanmean(times)], 
                              maxfev=10000)
        tau_empirical = popt[1]

        if verification:
            assert abs(tau_empirical - self.binary_sim.drag_tau) < 0.04 * self.binary_sim.drag_tau, \
                   f"{tau_empirical} and {self.binary_sim.drag_tau} are not within 3% of each other"
            
        if return_empirical:
            return tau_empirical
        else:
            return self.binary_sim.drag_tau
