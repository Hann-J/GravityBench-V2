import numpy as np
import pandas as pd
import scripts.task_utils as task_utils

class Scenario:
    def __init__(self, scenario_creator, skip_simulation=False):
        self.scenario_creator = scenario_creator

        prompt = """Starting from the moment star1 passes its pericenter, determine how long it takes for star1 to cover 20% of the total orbital path along its orbit."""
        final_answer_units = "s"

        self.binary_sim = self.scenario_creator.create_binary(prompt, final_answer_units, skip_simulation=skip_simulation)

    def true_answer(self, N_obs=None, verification=True, return_empirical=False) -> float:
        """
        Calculate the time needed for star1 to travel 20% of the distance along its orbital path.
        Arc length is radial_distance * angle.
        """
        # Load simulation data
        df = pd.read_csv(f"scenarios/detailed_sims/{self.binary_sim.filename}.csv")
        
        if N_obs is not None:
            indices = np.linspace(0, len(df) - 1, N_obs)
            df = df.iloc[indices].reset_index(drop=True)

        # Get period and time of pericenter passage
        period = task_utils.calculate_period(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
        time_pericenter_pass = task_utils.calculate_time_of_pericenter_passage(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
        
        # Isolate one orbit
        df_orbit = df[(df['time'] > time_pericenter_pass) & (df['time'] < time_pericenter_pass + period)]
        
        # Calculate velocities
        star1_vx, star1_vy, star1_vz, _, _, _ = task_utils.calculate_velocities(df_orbit, self.binary_sim, verification=verification, return_empirical=return_empirical)
        velocities = np.sqrt(star1_vx**2 + star1_vy**2 + star1_vz**2)
        
        # Calculate path distance using velocity * dt
        dt = np.diff(df_orbit['time'])
        ds = velocities[:-1] * dt
        path_distance = np.cumsum(ds)
        path_distance = np.cumsum(ds.values if isinstance(ds, pd.Series) else ds)
        
        # Find time at target distance
        perimeter = path_distance[-1]
        target_percent = 0.2
        target_distance = perimeter * target_percent
        idx = np.argmin(np.abs(path_distance - target_distance))
        travel_time = df_orbit['time'].iloc[idx+1] - time_pericenter_pass
        
        # Load detailed simulation data for verification
        df_detailed = pd.read_csv(f"scenarios/detailed_sims/{self.binary_sim.filename}.csv")
        period_detailed = df_detailed['orbital_period'].iloc[0]
        time = df_detailed['time']
        idx_after_one_period = np.argmax(time > period_detailed)
        time_pericenter_pass_detailed = df_detailed['time_of_pericenter_passage'].iloc[idx_after_one_period+1]
        df_orbit_detailed = df_detailed[(df_detailed['time'] > time_pericenter_pass_detailed) & 
                                        (df_detailed['time'] < time_pericenter_pass_detailed + period_detailed)]
        
        dnu = df_orbit_detailed['true_anomaly'].diff().values[1:]
        r = df_orbit_detailed['radial_distance_from_reference'].values[1:]
        dr = df_orbit_detailed['radial_distance_from_reference'].diff().values[1:]
        ds_detailed = np.sqrt(r**2 + (dr/dnu)**2) * dnu
        path_distance_detailed = ds_detailed.cumsum()
        perimeter_detailed = path_distance_detailed[-1]
        target_distance_detailed = perimeter_detailed * target_percent
        idx_detailed = np.argmin(np.abs(path_distance_detailed - target_distance_detailed))
        travel_time_detailed = df_orbit_detailed['time'].values[idx_detailed] - time_pericenter_pass_detailed
        if verification:
            assert abs(travel_time - travel_time_detailed) < 0.02 * travel_time_detailed, \
                f"{travel_time} and {travel_time_detailed} are not within 2% of each other"
        if return_empirical:
            return travel_time
        else:
            return travel_time_detailed