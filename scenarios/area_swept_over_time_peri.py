import rebound
import numpy as np
import pandas as pd

class Scenario:
    def __init__(self, scenario_creator, skip_simulation=False):
        self.scenario_creator = scenario_creator

        prompt = """Calculate, at periastron, the rate of area swept per unit time by the imaginary line joining star1 to star2."""
        final_answer_units = "m^2/s"

        self.binary_sim = self.scenario_creator.create_binary(prompt, final_answer_units, skip_simulation=skip_simulation)

    def true_answer(self, N_obs=None, verification=True, return_empirical=False) -> float:
        import scripts.task_utils as task_utils
        """
        Calculate the rate of area swept using Kepler's 2nd law: half the specific angular momentum of the system.
        # Note: Apo/Periastron does not matter because rate swept is constant along the orbit.
        Source: https://en.wikipedia.org/wiki/Specific_angular_momentum#Second_law
        Alternative source: http://burro.case.edu/Academics/Astr221/Gravity/kep2rev.htm#:~:text=%22Equal%20areas%20in%20equal%20times,Angular%20momentum%20is%20conserved.
        """
        # Load the simulation data
        df = pd.read_csv(f"scenarios/detailed_sims/{self.binary_sim.filename}.csv")

        if N_obs is not None:
            indices = np.linspace(0, len(df) - 1, N_obs)
            df = df.iloc[indices].reset_index(drop=True)

        df['rel_x'] = df['star2_x'] - df['star1_x']
        df['rel_y'] = df['star2_y'] - df['star1_y']
        df['rel_z'] = df['star2_z'] - df['star1_z']
        
        # Calculate velocities using task_utils
        star1_vx, star1_vy, star1_vz, star2_vx, star2_vy, star2_vz = task_utils.calculate_velocities(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
        
        # Compute relative velocities
        df['rel_vx'] = star2_vx - star1_vx
        df['rel_vy'] = star2_vy - star1_vy
        df['rel_vz'] = star2_vz - star1_vz
        
        # Compute the specific angular momentum (h)
        df['h_x'] = df['rel_y'] * df['rel_vz'] - df['rel_z'] * df['rel_vy']
        df['h_y'] = df['rel_z'] * df['rel_vx'] - df['rel_x'] * df['rel_vz']
        df['h_z'] = df['rel_x'] * df['rel_vy'] - df['rel_y'] * df['rel_vx']
        
        # Compute the magnitude of the specific angular momentum vector
        df['h'] = np.sqrt(df['h_x']**2 + df['h_y']**2 + df['h_z']**2)
        empirical_h = df['h'].mean()

        # Rebound verification
        sim = rebound.Simulation()
        sim.units = self.binary_sim.units
        sim.add(m=self.binary_sim.star1_mass, x=self.binary_sim.star1_pos[0], y=self.binary_sim.star1_pos[1], z=self.binary_sim.star1_pos[2], 
                vx=self.binary_sim.star1_momentum[0] / self.binary_sim.star1_mass, vy=self.binary_sim.star1_momentum[1] / self.binary_sim.star1_mass, vz=self.binary_sim.star1_momentum[2] / self.binary_sim.star1_mass)
        sim.add(m=self.binary_sim.star2_mass, x=self.binary_sim.star2_pos[0], y=self.binary_sim.star2_pos[1], z=self.binary_sim.star2_pos[2], 
                vx=self.binary_sim.star2_momentum[0] / self.binary_sim.star2_mass, vy=self.binary_sim.star2_momentum[1] / self.binary_sim.star2_mass, vz=self.binary_sim.star2_momentum[2] / self.binary_sim.star2_mass)
        orb = sim.particles[1].orbit(primary=sim.particles[0])
        
        if verification:
            assert abs(empirical_h - orb.h) < 0.01 * empirical_h, f"{empirical_h} and {orb.h} are not within 1% of each other"
        
        if return_empirical:
            return empirical_h/2
        else:
            return orb.h/2
