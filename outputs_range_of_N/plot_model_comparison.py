import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_performance_comparison():
    # Load the aggregated data
    os.makedirs('outputs_range_of_N/plots', exist_ok=True)
    df = pd.read_csv('outputs_range_of_N/aggregated_results.csv')

    # Get unique scenario-variation pairs
    scenario_variations = df[['scenario_name', 'variation_name']].drop_duplicates()
    n_variations = len(scenario_variations)

    # Create figure with 3 columns and enough rows for all variations
    fig, axs = plt.subplots(n_variations, 3, figsize=(18, 5*n_variations))

    # Get all unique models across ALL scenarios and create color mapping once
    all_models = sorted(df['model'].unique())
    
    # Create fixed color mapping with specific colors for model families
    model_colors = {}
    for model in all_models:
        if 'gpt-4' in model:
            model_colors[model] = 'orange'
        elif 'claude' in model:
            model_colors[model] = 'blue'
        else:
            # Fallback color for any other models
            model_colors[model] = 'gray'

    # Calculate global y-axis limits based on all data
    global_min_error = min(
        df['percent_error'].min() * 100,
        df['human_percent_diff'].min() * 100
    )
    global_max_error = max(
        df['percent_error'].max() * 100,
        df['human_percent_diff'].max() * 100
    )
    # Add some padding (10% on log scale)
    global_y_min = global_min_error / 1.1
    global_y_max = global_max_error * 1.1

    # For each unique scenario-variation
    for idx, (_, row) in enumerate(scenario_variations.iterrows()):
        scenario = row['scenario_name']
        variation = row['variation_name']
        
        # Get data for this scenario-variation
        mask = (df['scenario_name'] == scenario) & (df['variation_name'] == variation)
        scenario_df = df[mask]
        
        # Get unique models for this scenario-variation
        models = scenario_df['model'].unique()
        
        # First subplot: max_obs
        for model in models:
            model_data = scenario_df[scenario_df['model'] == model]
            model_color = model_colors[model]
            
            # Plot agent data
            axs[idx,0].plot(model_data['max_observations_total'], 
                          model_data['percent_error']*100, 'o',
                          color=model_color, label=f'{model}')
            
            # Plot human empirical data
            axs[idx,0].plot(model_data['max_observations_total'], 
                          model_data['human_percent_diff']*100, '-',
                          color='red', label='Human')

        axs[idx,0].set_xlabel('Maximum Total Observations')
        axs[idx,0].set_ylabel('Percent Error')
        axs[idx,0].set_title(f'{scenario}\n{variation}\nvs Max Observations')
        axs[idx,0].set_yscale('log')
        axs[idx,0].set_ylim(global_y_min, global_y_max)
        axs[idx,0].set_xlim(0, 120)
        axs[idx,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}%'.format(y)))
        # axs[idx,0].legend()

        # Second subplot: obs_attempted
        for model in models:
            model_data = scenario_df[scenario_df['model'] == model]
            model_color = model_colors[model]
            
            # Plot agent data
            axs[idx,1].plot(model_data['observations_attempted'], 
                          model_data['percent_error']*100, 'o',
                          color=model_color, label=f'{model}')
            
            # Plot human empirical data
            axs[idx,1].plot(model_data['max_observations_total'], 
                          model_data['human_percent_diff']*100, '-',
                          color='red', label='Human')

        axs[idx,1].set_xlabel('Observations Attempted')
        axs[idx,1].set_ylabel('Percent Error')
        axs[idx,1].set_title('vs Observations Attempted')
        axs[idx,1].set_yscale('log')
        axs[idx,1].set_ylim(global_y_min, global_y_max)
        axs[idx,1].set_xlim(0, 120)
        axs[idx,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}%'.format(y)))
        # axs[idx,1].legend()

        # Third subplot: observations attempted vs max observations allowed
        for model in models:
            model_data = scenario_df[scenario_df['model'] == model]
            model_color = model_colors[model]
            
            # Group by max_observations_total
            grouped = model_data.groupby('max_observations_total')
            max_obs = sorted(grouped.groups.keys())
            avg_attempts = grouped['observations_attempted'].mean()
            se_attempts = grouped['observations_attempted'].std() / np.sqrt(grouped.size())

            # Plot individual points
            axs[idx,2].plot(model_data['max_observations_total'], 
                          model_data['observations_attempted'], 'o',
                          color=model_color, alpha=0.2)

            # Plot average with error bars
            axs[idx,2].errorbar(max_obs, avg_attempts, yerr=se_attempts, 
                              color=model_color, alpha=0.8,
                              fmt='.-', label=f'{model} Average ± SE')

        axs[idx,2].plot([0, 120], [0, 120], 'k--', label='y=x')
        axs[idx,2].set_xlabel('Maximum Total Observations')
        axs[idx,2].set_ylabel('Observations Attempted')
        axs[idx,2].set_title('Observations Used')
        axs[idx,2].legend()

    plt.tight_layout()
    plt.savefig('outputs_range_of_N/plots/all_scenarios_combined.png')

if __name__ == '__main__':
    plot_performance_comparison()