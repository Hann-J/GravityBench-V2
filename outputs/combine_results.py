import numpy as np
import pandas as pd
import json
import os
import glob
import re

def main():
    # --- Part 1: Aggregate results and output combined_results.csv ---

    # Find all json files recursively in outputs/
    json_files = glob.glob('outputs/**/*.json', recursive=True)
    print(f"Found {len(json_files)} JSON files")

    # Initialize empty list to store all dataframes
    dfs = []

    # Process each json file to aggregate results
    for json_file in json_files:
        print(f"Processing: {json_file}")
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if 'scenarios' in data:
                    print(f"  Found {len(data['scenarios'])} scenarios")
                    valid_scenarios = []
                    for scenario in data['scenarios']:
                        if scenario['correct'] is not None:
                            valid_scenarios.append({
                                'scenario_variation': f"{scenario['scenario_name']}-{scenario['variation_name']}", 
                                'model': scenario['model'],
                                'row_wise': scenario['row_wise'],
                                'observations_attempted': scenario['observations_attempted'],
                                'correct': scenario['correct'],
                                'percent_error': scenario['percent_error'],
                                'run_time': scenario['run_time'],
                                'input_tokens_used': scenario['input_tokens_used'],
                                'output_tokens_used': scenario['output_tokens_used'],
                                'cost': scenario['cost'],
                                'source_file': json_file
                            })
                        else:
                            print(f"  Skipping scenario with correct=None: {scenario.get('scenario_name', 'unknown')}")
                    
                    print(f"  Valid scenarios: {len(valid_scenarios)}")
                    if valid_scenarios:
                        df = pd.DataFrame(valid_scenarios)
                        dfs.append(df)
                else:
                    print(f"  No 'scenarios' key found")
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")

    # Optionally load human results if present
    expert_baseline_results_path = 'expert_baseline_results.csv'
    if os.path.exists(expert_baseline_results_path):
        human_df = pd.read_csv(expert_baseline_results_path)
        human_transformed = pd.DataFrame([
            {
                'scenario_variation': f"{row['scenario']}-{row['variation']}", 
                'model': 'human',
                'row_wise': False,
                'observations_attempted': None,
                'correct': row['correct'],
                'percent_error': row['percent_diff'],
                'run_time': None,
                'input_tokens_used': None,
                'output_tokens_used': None,
                'cost': None,
                'source_file': expert_baseline_results_path
            }
            for _, row in human_df.iterrows()
        ])

        # Add row-wise results (100 observations)
        human_transformed_row_wise = pd.DataFrame([
            {
                'scenario_variation': f"{row['scenario']}-{row['variation']}", 
                'model': 'human',
                'row_wise': True,
                'observations_attempted': 100,
                'correct': row['correct_100_obs'],
                'percent_error': row['percent_diff_100_obs'],
                'run_time': None,
                'input_tokens_used': None,
                'output_tokens_used': None,
                'cost': None,
                'source_file': expert_baseline_results_path
            }
            for _, row in human_df.iterrows()
        ])

        dfs.append(human_transformed)
        dfs.append(human_transformed_row_wise)

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    if not combined_df.empty:
        # Remove missing scenarios logic
        combined_df.to_csv('outputs/combined_results.csv', index=False)
        print(f"Saved aggregated results to combined_results.csv")
    else:
        print("No valid scenarios found. Skipping outputs/combined_results.csv generation.")

    # --- Part 2: Aggregate chat histories and output chat_histories.csv ---

    def clean_chat_history(chat_history_str):
        if not isinstance(chat_history_str, str):
            chat_history_str = json.dumps(chat_history_str)
        cleaned = chat_history_str
        chars_to_remove = ['"', "'", ',', '\n', '\r', '\t', '\\']
        for char in chars_to_remove:
            cleaned = cleaned.replace(char, ' ')
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    # Reuse the same json_files list
    dfs_chat = []

    # Process each json file to aggregate chat histories
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if 'scenarios' in data:
                    df_chat = pd.DataFrame([
                        {
                            'scenario_variation': f"{scenario['scenario_name']}-{scenario['variation_name']}", 
                            'model': scenario['model'],
                            'row_wise': scenario['row_wise'],
                            'correct': scenario['correct'],
                            'chat_history': clean_chat_history(scenario.get('chat_history', {})),
                            'source_file': json_file
                        }
                        for scenario in data['scenarios']
                        if scenario.get('chat_history') is not None
                    ])
                    dfs_chat.append(df_chat)
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")

    if dfs_chat:
        combined_chat_df = pd.concat(dfs_chat, ignore_index=True)
        combined_chat_df.to_csv('outputs/chat_histories.csv', index=False)
        print(f"Saved {len(combined_chat_df)} chat histories to chat_histories.csv")
    else:
        print("No chat histories found. Skipping outputs/chat_histories.csv generation.")

    # Display a small sample if data is present
    if dfs_chat:
        print("\nSample of the chat history data:")
        print(combined_chat_df[['scenario_variation', 'model', 'row_wise', 'source_file']].head())

if __name__ == '__main__':
    main()