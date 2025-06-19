import unittest
import os
import sys
import pandas as pd
import numpy as np
import tqdm

# Add the parent directory to the path so we can import scripts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.scenarios_config import get_all_scenarios, get_scenario

# python -m unittest discover -s tests
# python -m xmlrunner tests/test_scenarios.py -o test-reports/
class TestScenarios(unittest.TestCase):
    def setUp(self):
        # Get all scenario names using the new method
        self.scenarios_to_run = get_all_scenarios()

    def test_scenario_initialization_and_true_answer(self):
        for scenario_name in tqdm.tqdm(self.scenarios_to_run, desc="Testing scenarios", unit="scenario"):
            for variation_name in self.scenarios_to_run[scenario_name]['variations']:
                with self.subTest(scenario_name=scenario_name, variation_name=variation_name):
                    passed, message = self._test_scenario(scenario_name, variation_name)
                    self.assertTrue(passed, msg=message)


    def _test_scenario(self, scenario_name, variation_name):
        """
        Test that the scenario can be initialized and that it generates a valid csv.
        """
        # Initialize the scenario
        scenario = get_scenario(scenario_name, variation_name)

        # Check if the csv file is generated
        csv_path = f"scenarios/sims/{scenario.binary_sim.filename}.csv"
        if not os.path.exists(csv_path):
            return False, f"CSV file {csv_path} was not generated."
        
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Check if the required columns are present
        required_columns = ['time', 'star1_x', 'star1_y', 'star1_z', 'star2_x', 'star2_y', 'star2_z']
        if not all(column in df.columns for column in required_columns):
            return False, f"CSV file {csv_path} is missing required columns."

        # Test true_answer with different configurations
        try:
            # Test all combinations of verification and return_empirical
            combinations = [
                (True, True),    # verification=True, return_empirical=True
                (True, False),   # verification=True, return_empirical=False
                (False, True),   # verification=False, return_empirical=True
                (False, False),  # verification=False, return_empirical=False
            ]
            
            results = {}
            for verify, empirical in combinations:
                answer = scenario.true_answer(verification=verify, return_empirical=empirical)
                if np.isnan(answer) or answer is None or np.isinf(answer):
                    return False, f"scenario.true_answer(verification={verify}, return_empirical={empirical}) returned an invalid value: {answer}"
                results[(verify, empirical)] = answer

            # Test with N_obs specified
            n_obs_answer = scenario.true_answer(N_obs=50, verification=False, return_empirical=True)
            if np.isnan(n_obs_answer) or n_obs_answer is None or np.isinf(n_obs_answer):
                return False, f"scenario.true_answer(N_obs=50, verification=False, return_empirical=True) returned an invalid value: {n_obs_answer}"

            # Verify that empirical and true values are reasonably close
            # Compare both verified and unverified pairs
            for verify in [True, False]:
                empirical_val = results[(verify, True)]
                true_val = results[(verify, False)]
                # Check they are within 5% of each other
                if abs(empirical_val - true_val) > 0.05 * abs(true_val):
                    return False, f"Empirical answer {empirical_val} differs significantly from true answer {true_val} (verification={verify})"

        except Exception as e:
            return False, f"scenario.true_answer() failed with error: {e}"

        return True, "Test passed."

if __name__ == "__main__":
    unittest.main()