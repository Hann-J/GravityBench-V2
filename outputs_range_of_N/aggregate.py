import json
import os
import pandas as pd

def aggregate_data():
    # Get all folders in current directory
    folders = [f for f in os.listdir('.') if os.path.isdir(f)]
    
    # Initialize list to store all data
    all_data = []
    
    # Process each folder
    for folder in folders:
        json_file = os.path.join(folder, f'{folder}.json')
        
        # Skip if json file doesn't exist
        if not os.path.exists(json_file):
            continue
            
        # Load the json data
        with open(json_file, 'r') as f:
            data = json.loads(f.read())['scenarios']
            
        # Extract data from each run
        for run in data:
            if run['max_observations_total'] is None or run['percent_error'] is None:
                continue
                
            row = {
                'model': folder,
                'scenario_name': run['scenario_name'],
                'variation_name': run['variation_name'],
                'max_observations_total': run['max_observations_total'],
                'observations_attempted': run['observations_attempted'],
                'percent_error': run['percent_error'],
                'human_percent_diff': run['human_percent_diff'],
                'true_answer': run['true_answer'],
                'correct': run['correct']
            }
            all_data.append(row)
    
    # Convert to dataframe and save
    df = pd.DataFrame(all_data)
    df.to_csv('outputs_range_of_N/aggregated_results.csv', index=False)
    
if __name__ == '__main__':
    aggregate_data()