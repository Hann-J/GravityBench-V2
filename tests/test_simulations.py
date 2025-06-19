import unittest
import os
import sys
import pandas as pd
import tqdm
import shutil

# Add the parent directory to the path so we can import scripts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.scenarios_config import variations

class TestSimulations(unittest.TestCase):
    def setUp(self):
        # Delete files inside outputs/sims and detailed_sims folders if they exist
        for folder in ['scenarios/sims', 'scenarios/detailed_sims']:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')

    def test_all_simulations(self):
        for scenario_name, scenario in tqdm.tqdm(variations.items(), desc="Testing simulations", unit="scenario"):
            with self.subTest(scenario_name=scenario_name):
                passed, message = self._run_simulation(scenario_name, scenario)
                self.assertTrue(passed, msg=message)

    def _run_simulation(self, scenario_name, scenario):
        """
        Run the simulation and check if it generates a valid CSV file.
        """
        try:
            binary = scenario.create_binary(prompt="", final_answer_units=None)
            csv_path = f"scenarios/sims/{binary.filename}.csv"

            # Check if the CSV file is generated
            if not os.path.exists(csv_path):
                return False, f"CSV file {csv_path} was not generated for scenario {scenario_name}."

            # Read the CSV file
            df = pd.read_csv(csv_path)

            # Check if the required columns are present
            required_columns = ['time', 'star1_x', 'star1_y', 'star1_z', 'star2_x', 'star2_y', 'star2_z']
            if not all(column in df.columns for column in required_columns):
                return False, f"CSV file {csv_path} is missing required columns for scenario {scenario_name}."

            # Check if the dataframe is not empty
            if df.empty:
                return False, f"CSV file {csv_path} is empty for scenario {scenario_name}."

            return True, f"Simulation for scenario {scenario_name} ran successfully."

        except Exception as e:
            return False, f"Simulation for scenario {scenario_name} failed with error: {e}"

if __name__ == "__main__":
    unittest.main()