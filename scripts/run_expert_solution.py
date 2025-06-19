
from scenarios_config import get_scenario, get_all_scenarios
import pandas as pd
import json
from tqdm import tqdm

# Load scenarios config
with open('scripts/scenarios_config.json') as f:
    scenarios_config = json.load(f)

# Get all scenarios for thresholds
scenarios = get_all_scenarios()

# Store results for each scenario/variation pair
results = []
# Iterate through all scenario/variation pairs
for scenario_name, scenario_data in tqdm(scenarios_config.items()):
    for variation in scenario_data['variations']:
        scenario = get_scenario(scenario_name, variation)
        
        # Get empirical answers with different observation counts
        empirical_answer = scenario.true_answer(verification=True, return_empirical=True)
        empirical_answer_100_obs = scenario.true_answer(N_obs=100, verification=False, return_empirical=True)
        empirical_answer_5000_obs = scenario.true_answer(N_obs=5000, verification=False, return_empirical=True)
        
        # Get extra_info answer
        extra_info_answer = scenario.true_answer(verification=True, return_empirical=False)
        
        # Get scenario-specific threshold for 100 obs
        threshold_100 = scenarios[scenario_name]['correct_threshold_percentage_based_on_100_observations']
        
        # Calculate percent differences and correctness
        if isinstance(empirical_answer, bool) or isinstance(extra_info_answer, bool):
            percent_diff = float('nan')
            correct = empirical_answer == extra_info_answer
        else:
            percent_diff = abs(empirical_answer - extra_info_answer) / abs(extra_info_answer) * 100
            correct = percent_diff <= 5  # 5% threshold for full table
            
        if isinstance(empirical_answer_100_obs, bool) or isinstance(extra_info_answer, bool):
            percent_diff_100_obs = float('nan')
            correct_100_obs = empirical_answer_100_obs == extra_info_answer
        else:
            percent_diff_100_obs = abs(empirical_answer_100_obs - extra_info_answer) / abs(extra_info_answer) * 100
            correct_100_obs = percent_diff_100_obs <= threshold_100
            
        if isinstance(empirical_answer_5000_obs, bool) or isinstance(extra_info_answer, bool):
            percent_diff_5000_obs = float('nan')
            correct_5000_obs = empirical_answer_5000_obs == extra_info_answer
        else:
            percent_diff_5000_obs = abs(empirical_answer_5000_obs - extra_info_answer) / abs(extra_info_answer) * 100
            correct_5000_obs = percent_diff_5000_obs <= 5  # 5% threshold for 5000 obs
        
        print(correct)
        # Store all results at once
        results.append({
            'scenario': scenario_name,
            'variation': variation,
            'empirical': empirical_answer,
            'empirical_100_obs': empirical_answer_100_obs,
            'empirical_5000_obs': empirical_answer_5000_obs,
            'extra_info': extra_info_answer,
            'percent_diff': percent_diff,
            'percent_diff_100_obs': percent_diff_100_obs,
            'percent_diff_5000_obs': percent_diff_5000_obs,
            'correct': correct,
            'correct_100_obs': correct_100_obs,
            'correct_5000_obs': correct_5000_obs
        })
        
        print(f"Scenario: {scenario_name}")
        print(f"Variation: {variation}")
        print(f"Empirical answer (all obs): {empirical_answer}")
        print(f"Empirical answer (100 obs): {empirical_answer_100_obs}") 
        print(f"Empirical answer (5000 obs): {empirical_answer_5000_obs}")
        print(f"Extra info answer: {extra_info_answer}")

# Save to CSV
results_df = pd.DataFrame(results)[['scenario', 'variation', 'empirical', 'empirical_100_obs', 'empirical_5000_obs', 'extra_info', 'percent_diff', 'correct', 'percent_diff_100_obs', 'correct_100_obs', 'percent_diff_5000_obs', 'correct_5000_obs']]
results_df.to_csv('outputs/expert_baseline_results.csv', index=False)