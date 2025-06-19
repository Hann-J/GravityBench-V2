import numpy as np
import pandas as pd
import scripts.task_utils as task_utils

class Scenario:
    def __init__(self, scenario_creator, skip_simulation=False):
        self.scenario_creator = scenario_creator

        prompt = """Determine the absolute value of the specific angular momentum of the system."""
        final_answer_units = "m^2/s"

        self.binary_sim = self.scenario_creator.create_binary(prompt, final_answer_units, skip_simulation=skip_simulation)
    def true_answer(self, N_obs=None, verification=True, return_empirical=False) -> float:
        """
        Calculate the specific angular momentum of the system.
        """
        # Load simulation data
        df = pd.read_csv(f"scenarios/detailed_sims/{self.binary_sim.filename}.csv")
        
        if N_obs is not None:
            indices = np.linspace(0, len(df) - 1, N_obs)
            df = df.iloc[indices].reset_index(drop=True)

        # Calculate relative positions
        df['rel_x'] = df['star2_x'] - df['star1_x']
        df['rel_y'] = df['star2_y'] - df['star1_y']
        df['rel_z'] = df['star2_z'] - df['star1_z']
        
        # Calculate relative velocities using task_utils
        _, _, _, star2_vx, star2_vy, star2_vz = task_utils.calculate_velocities(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
        star1_vx, star1_vy, star1_vz, _, _, _ = task_utils.calculate_velocities(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
        
        df['rel_vx'] = star2_vx - star1_vx
        df['rel_vy'] = star2_vy - star1_vy
        df['rel_vz'] = star2_vz - star1_vz
        
        # Compute the specific angular momentum components
        df['h_x'] = df['rel_y'] * df['rel_vz'] - df['rel_z'] * df['rel_vy']
        df['h_y'] = df['rel_z'] * df['rel_vx'] - df['rel_x'] * df['rel_vz']
        df['h_z'] = df['rel_x'] * df['rel_vy'] - df['rel_y'] * df['rel_vx']
        
        # Compute the magnitude of the specific angular momentum vector
        df['h'] = np.sqrt(df['h_x']**2 + df['h_y']**2 + df['h_z']**2)
        specific_angular_momentum = df['h'].mean()

        # Rebound verification
        import rebound
        sim = rebound.Simulation()
        sim.units = self.binary_sim.units
        sim.add(m=self.binary_sim.star1_mass, x=self.binary_sim.star1_pos[0], y=self.binary_sim.star1_pos[1], z=self.binary_sim.star1_pos[2], 
                vx=self.binary_sim.star1_momentum[0] / self.binary_sim.star1_mass, vy=self.binary_sim.star1_momentum[1] / self.binary_sim.star1_mass, vz=self.binary_sim.star1_momentum[2] / self.binary_sim.star1_mass)
        sim.add(m=self.binary_sim.star2_mass, x=self.binary_sim.star2_pos[0], y=self.binary_sim.star2_pos[1], z=self.binary_sim.star2_pos[2], 
                vx=self.binary_sim.star2_momentum[0] / self.binary_sim.star2_mass, vy=self.binary_sim.star2_momentum[1] / self.binary_sim.star2_mass, vz=self.binary_sim.star2_momentum[2] / self.binary_sim.star2_mass)
        orb = sim.particles[1].orbit(primary=sim.particles[0])
        if verification:
            assert abs(specific_angular_momentum - orb.h) < 0.02 * orb.h, f"{specific_angular_momentum} and {orb.h} are not within 2% of each other"

        if return_empirical:
            return specific_angular_momentum
        else:
            return orb.h
