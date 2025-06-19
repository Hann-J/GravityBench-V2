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

    # Create figure with 4 subplots with extra space for legend
    fig, axs = plt.subplots(1, 4, figsize=(16, 2.5))  # Reduced width from 18 to 16
    plt.subplots_adjust(wspace=0.45)  # Add this line to control horizontal spacing
    
    # Add subplot labels (a), (b), (c), (d) to the top right of each subplot
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']
    for i, ax in enumerate(axs):
        ax.text(0.95, 0.95, subplot_labels[i], transform=ax.transAxes,
                fontsize=12, fontweight='bold', ha='right', va='top')
    
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

    scenarios = [
        ('max_velocity_star1', '9.6 M, 3.1 M'),
        ('periastron', '3.1 M, 0.18 M, Elliptical, Single Orbit')
    ]

    for scenario_idx, (scenario, variation) in enumerate(scenarios):
        # Filter for specific scenario and variation
        scenario_df = df[(df['scenario_name'] == scenario) & 
                        (df['variation_name'] == variation)]

        # Get unique models
        models = scenario_df['model'].unique()
        
        # Error vs Observations subplot
        human_line = None
        model_lines = {}  # Use dict to track one line per model
        ax_idx = scenario_idx * 2  # 0 or 2
        
        # Set y-axis limits from 0 to 100 for left plots
        axs[ax_idx].set_ylim(0, 100)
        
        # Add threshold line based on scenario
        threshold = 20 if scenario == 'max_velocity_star1' else 5
        threshold_line = axs[ax_idx].axhline(y=threshold, color='grey', linestyle='--', alpha=0.3,
                                           label='Threshold' if scenario_idx == 0 else '')
        
        for model in models:
            model_data = scenario_df[scenario_df['model'] == model]
            model_color = model_colors[model]
            
            # Plot agent data
            line = axs[ax_idx].plot(model_data['observations_attempted'], 
                        model_data['percent_error']*100, 'o',
                        color=model_color, rasterized=True)[0]
            model_lines[model] = line
            
            # Plot human empirical data (only once)
            if human_line is None:
                human_line = axs[ax_idx].plot(model_data['max_observations_total'], 
                        model_data['human_percent_diff']*100, '-',
                        color='red')[0]

        # Remove individual legends
        axs[ax_idx].get_legend().remove() if axs[ax_idx].get_legend() else None

        # Observations Attempted vs Budget subplot
        for model in models:
            model_data = scenario_df[scenario_df['model'] == model]
            model_color = model_colors[model]
            
            # Group by max_observations_total
            grouped = model_data.groupby('max_observations_total')
            max_obs = sorted(grouped.groups.keys())
            avg_attempts = grouped['observations_attempted'].mean()
            se_attempts = grouped['observations_attempted'].std() / np.sqrt(grouped.size())

            # Plot individual points
            scatter = axs[ax_idx + 1].plot(model_data['max_observations_total'], 
                        model_data['observations_attempted'], 'o',
                        color=model_color, alpha=0.7, rasterized=True)

            # Plot average with error bars
            axs[ax_idx + 1].errorbar(max_obs, avg_attempts, yerr=se_attempts, 
                            color=model_color, alpha=0.3,
                            fmt='.-')

        
        # Add y=x line (dashed black line)
        x_min, x_max = axs[ax_idx + 1].get_xlim()
        shortened_x_max = x_max - (x_max - x_min) * 0.17
        axs[ax_idx + 1].plot([x_min, shortened_x_max], [x_min, shortened_x_max], 'k--', alpha=0.7)
        axs[ax_idx + 1].set_ylim(0, 100)
        
        # Remove individual legend
        axs[ax_idx + 1].get_legend().remove() if axs[ax_idx + 1].get_legend() else None

        # Add axis labels
        axs[ax_idx].set_xlabel('Observations Attempted')
        axs[ax_idx].set_ylabel('Percent Error', labelpad=-10)
        
        axs[ax_idx + 1].set_xlabel('Observation Budget')
        axs[ax_idx + 1].set_ylabel('Observations\nAttempted',labelpad=-10)

        # Add subplot titles centered between plots
        if scenario == 'max_velocity_star1':
            fig.text((axs[ax_idx].get_position().x1 + axs[ax_idx + 1].get_position().x0)/3+0.01, 
                    axs[ax_idx].get_position().y1 + 0.04, 
                    'Max velocity of $\\mathtt{star1}$',
                    ha='left', va='bottom')
        else:
            fig.text((axs[ax_idx].get_position().x1 + axs[ax_idx + 1].get_position().x0)/2-0.1, 
                    axs[ax_idx].get_position().y1 + 0.04, 
                    'Closest Approach (Periastron)',
                    ha='left', va='bottom')

    # Create combined legend outside plots
    legend_elements = [human_line, threshold_line]
    legend_labels = ['Uniform\nSampling', 'Threshold']
    
    for model in models:
        legend_elements.append(model_lines[model])
        legend_labels.append(model_labels[model])

    fig.legend(legend_elements, legend_labels,
              loc='center right',
              bbox_to_anchor=(1.02, 0.5))  # Moved legend closer (from 1.1 to 1.05)

    plt.savefig('outputs_range_of_N/plots/max_velocity_and_periastron.pdf', 
                bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    plot_performance_comparison()