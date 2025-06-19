import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
plt.rcParams.update({
    'mathtext.fontset': 'dejavusans',
    'font.family': 'sans-serif',
    'font.size': 18,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.major.size': 7,
    'xtick.minor.size': 3.5,
    'ytick.major.size': 7,
    'ytick.minor.size': 3.5,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.minor.width': 1.5,
    'ytick.minor.width': 1.5,
    'legend.fontsize': 14,
    'xtick.direction': 'in',
    'ytick.direction': 'in'
})
def plot_performance_comparison():
    # Load the aggregated data
    os.makedirs('outputs_range_of_N/plots', exist_ok=True)
    df = pd.read_csv('outputs_range_of_N/aggregated_results.csv')

    # Filter for just max_velocity_star1 scenario and specific variation
    df = df[(df['scenario_name'] == 'max_velocity_star1') & 
            (df['variation_name'] == '9.6 M, 3.1 M')]

    # Create figure with 2 subplots with extra space for legend
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))  # Made figure wider to accommodate legend
    
    # Get all unique models and create color mapping
    all_models = sorted(df['model'].unique())
    model_colors = {}
    model_labels = {}  # New mapping for display labels
    for model in all_models:
        if 'gpt-4' in model:
            model_colors[model] = 'orange'
            model_labels[model] = 'GPT-4o'  # Simplified label
        elif 'claude' in model:
            model_colors[model] = 'blue'
            model_labels[model] = 'Claude 3.5\nSonnet'  # Simplified label
        else:
            model_colors[model] = 'gray'
            model_labels[model] = model

    # Get unique models
    models = df['model'].unique()
    
    # Original second subplot (now first)
    human_line = None
    model_lines = {}  # Use dict to track one line per model
    for model in models:
        model_data = df[df['model'] == model]
        model_color = model_colors[model]
        
        # Plot agent data
        line = axs[0].plot(model_data['observations_attempted'], 
                    model_data['percent_error']*100, 'o',
                    color=model_color)[0]
        model_lines[model] = line
        
        # Plot human empirical data (only once)
        if human_line is None:
            human_line = axs[0].plot(model_data['max_observations_total'], 
                    model_data['human_percent_diff']*100, '-',
                    color='red')[0]

    # Remove individual legends from subplots
    axs[0].get_legend().remove() if axs[0].get_legend() else None

    # Original third subplot (now second)
    for model in models:
        model_data = df[df['model'] == model]
        model_color = model_colors[model]
        
        # Group by max_observations_total
        grouped = model_data.groupby('max_observations_total')
        max_obs = sorted(grouped.groups.keys())
        avg_attempts = grouped['observations_attempted'].mean()
        se_attempts = grouped['observations_attempted'].std() / np.sqrt(grouped.size())

        # Plot individual points (without adding to legend)
        axs[1].plot(model_data['max_observations_total'], 
                    model_data['observations_attempted'], 'o',
                    color=model_color, alpha=0.2)

        # Plot average with error bars (without adding to legend)
        axs[1].errorbar(max_obs, avg_attempts, yerr=se_attempts, 
                        color=model_color, alpha=0.8,
                        fmt='.-')

    
    # Remove individual legend from second subplot
    axs[1].get_legend().remove() if axs[1].get_legend() else None

    # Add axis labels and titles
    axs[0].set_xlabel('Observations Attempted')
    axs[0].set_ylabel('Percent Error')
    
    axs[1].set_xlabel('Observation Budget')
    axs[1].set_ylabel('Observations\nAttempted')

    # Create combined legend outside plots
    legend_elements = [human_line]
    legend_labels = ['Human']
    
    for model in models:
        legend_elements.append(model_lines[model])
        legend_labels.append(model_labels[model])

    fig.legend(legend_elements, legend_labels,
              loc='center right',
              bbox_to_anchor=(1.19, 0.52))

    # Add suptitle
    fig.suptitle('Find the max velocity of $\\mathtt{star1}$', y=0.88)

    plt.tight_layout()
    plt.savefig('outputs_range_of_N/plots/max_velocity_star1.png', bbox_inches='tight')

if __name__ == '__main__':
    plot_performance_comparison()