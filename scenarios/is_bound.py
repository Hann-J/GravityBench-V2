import numpy as np
import pandas as pd
import scripts.task_utils as task_utils

class Scenario:
    def __init__(self, scenario_creator, skip_simulation=False):
        self.scenario_creator = scenario_creator

        prompt = """Determine whether the system is a bound (True) or unbound (False)."""
        final_answer_units = None

        self.binary_sim = self.scenario_creator.create_binary(prompt, final_answer_units, skip_simulation=skip_simulation)
    def true_answer(self, N_obs=None, verification=True, return_empirical=False) -> bool:
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

        # Calculate velocities using task_utils
        star1_vx, star1_vy, star1_vz, star2_vx, star2_vy, star2_vz = task_utils.calculate_velocities(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
        velocity_sq_star1 = star1_vx**2 + star1_vy**2 + star1_vz**2
        velocity_sq_star2 = star2_vx**2 + star2_vy**2 + star2_vz**2

        # Get masses using task_utils
        m1, m2 = task_utils.star_masses(df, self.binary_sim, verification=verification, return_empirical=return_empirical)

        # Calculate distances, potential energy, velocities, and kinetic energy
        df['distance'] = np.sqrt((df['star1_x'] - df['star2_x'])**2 + (df['star1_y'] - df['star2_y'])**2 + (df['star1_z'] - df['star2_z'])**2)
        df['potential_energy'] = -self.binary_sim.sim.G * m1 * m2 / df['distance']

        df['kinetic_energy'] = 0.5 * m1 * velocity_sq_star1 + \
                            0.5 * m2 * velocity_sq_star2

        # Calculate means and virial theorem check
        K, U = df['kinetic_energy'].mean(), df['potential_energy'].mean()
        
        if verification:
            assert 'Unbound' not in self.binary_sim.filename or not bool(K+U<0), f"Kepler's third law determined to be satisfied, but modified gravity means it should not be satisfied."

        if return_empirical:
            return bool(K+U<0)
        else:
            return 'Unbound' not in self.binary_sim.filename
