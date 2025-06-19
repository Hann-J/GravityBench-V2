import numpy as np
import pandas as pd
import scripts.task_utils as task_utils

class Scenario:
    def __init__(self, scenario_creator, skip_simulation=False):
        self.scenario_creator = scenario_creator

        prompt = """Determine the Roche lobe radius of star1."""
        final_answer_units = 'm'

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
        mass_ratio = m1/m2

        # Calculate semi-major axis using task_utils
        a_total, _, _ = task_utils.calculate_semi_major_axes(df, m1, m2, self.binary_sim, verification=verification, return_empirical=return_empirical)

        # Calculate Roche lobe radius using Eggleton's formula
        # https://en.wikipedia.org/wiki/Roche_lobe
        roche_lobe_radius = a_total * (0.49*mass_ratio**(2/3)) / (0.6*mass_ratio**(2/3) + np.log(1 + mass_ratio**(1/3)))

        # Rebound verification
        import rebound
        sim = rebound.Simulation()
        sim.units = self.binary_sim.units
        sim.add(m=self.binary_sim.star1_mass, x=self.binary_sim.star1_pos[0], y=self.binary_sim.star1_pos[1], z=self.binary_sim.star1_pos[2], 
                vx=self.binary_sim.star1_momentum[0] / self.binary_sim.star1_mass, vy=self.binary_sim.star1_momentum[1] / self.binary_sim.star1_mass, vz=self.binary_sim.star1_momentum[2] / self.binary_sim.star1_mass)
        sim.add(m=self.binary_sim.star2_mass, x=self.binary_sim.star2_pos[0], y=self.binary_sim.star2_pos[1], z=self.binary_sim.star2_pos[2], 
                vx=self.binary_sim.star2_momentum[0] / self.binary_sim.star2_mass, vy=self.binary_sim.star2_momentum[1] / self.binary_sim.star2_mass, vz=self.binary_sim.star2_momentum[2] / self.binary_sim.star2_mass)
        orb = sim.particles[1].orbit(primary=sim.particles[0])
        true_mass_ratio = self.binary_sim.star1_mass/self.binary_sim.star2_mass
        true_roche_lobe_radius = orb.a * (0.49*true_mass_ratio**(2/3)) / (0.6*true_mass_ratio**(2/3) + np.log(1 + true_mass_ratio**(1/3)))

        if verification:
            assert abs(roche_lobe_radius - true_roche_lobe_radius) < 0.02 * true_roche_lobe_radius, f"{roche_lobe_radius} and {true_roche_lobe_radius} are not within 2% of each other"

        if return_empirical:
            return roche_lobe_radius
        else:
            return true_roche_lobe_radius