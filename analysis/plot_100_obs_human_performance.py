import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import json
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

def main():
    # Load config and results
    with open('scripts/scenarios_config.json', 'r') as f:
        config = json.load(f)
    
    df = pd.read_csv('outputs/expert_baseline_results.csv')
    
    # Calculate absolute percent differences for N=100
    df['abs_percent_diff_100_obs'] = abs(df['percent_diff_100_obs'])
    
    # Remove any rows with NaN values
    df = df.dropna(subset=['abs_percent_diff_100_obs'])
    
    # Calculate scenario order based on thresholds instead of mean differences
    scenario_thresholds = []
    for scenario in df['scenario'].unique():
        threshold = None
        if scenario in config:
            threshold = config[scenario].get('correct_threshold_percentage_based_on_100_observations', None)
        scenario_thresholds.append((scenario, threshold))
    
    # Sort scenarios: those with thresholds first (by threshold value), then others
    scenario_order = [s[0] for s in sorted(scenario_thresholds, 
                                         key=lambda x: (-x[1] is None, -x[1] if x[1] is not None else float('inf')))]

    # Before creating the plot, identify outliers
    outlier_threshold = 105  # Since your current ylim is -5 to 105
    df['is_outlier'] = df['abs_percent_diff_100_obs'] > outlier_threshold
    
    # Create plot
    fig, ax = plt.subplots(figsize=(20,6))
    plt.subplots_adjust(left=0.15, bottom=0.2)
    ax.tick_params(labelsize=24)
    ax.axhline(y=0, color='#0000ff', linestyle='-')
    ax.axhline(y=5, color='#4444ff', linestyle=':')
    ax.axhline(y=10, color='#8888ff', linestyle=':')
    ax.axhline(y=15, color='#aaaaff', linestyle=':')
    ax.axhline(y=20, color='#ccccff', linestyle=':')
    ax.set_ylim(-5, 105)

    # Plot non-outliers
    scatter = sns.stripplot(data=df[~df['is_outlier']], 
                 x='scenario', y='abs_percent_diff_100_obs',
                 color='black', size=10, alpha=1, order=scenario_order,
                 jitter=False, ax=ax)
    
    # Rasterize the scatter plot points
    for collection in scatter.collections:
        collection.set_rasterized(True)
    
    # Plot outliers at the top of the chart with triangular markers
    outlier_data = df[df['is_outlier']]
    if not outlier_data.empty:
        # Group outliers by scenario to handle multiple per scenario
        for scenario in outlier_data['scenario'].unique():
            scenario_outliers = outlier_data[outlier_data['scenario'] == scenario]
            scenario_idx = scenario_order.index(scenario)
            
            # Plot markers for each outlier in this scenario
            for i, (_, row) in enumerate(scenario_outliers.iterrows()):
                outlier_marker = ax.scatter(scenario_idx, 102, marker='^', color='black', s=150, zorder=10)
                outlier_marker.set_rasterized(True)
                # Stagger annotations vertically
                ax.annotate(f"{row['abs_percent_diff_100_obs']:.0f}%", 
                           (scenario_idx, 103),
                           xytext=(5, -55 + i*15), textcoords='offset points', 
                           ha='center', va='bottom',
                           fontsize=12,
                           weight='bold')

    # Create threshold markers
    threshold_markers = []
    for idx, scenario_type in enumerate(scenario_order):
        # Add required threshold if exists
        if scenario_type in config:
            threshold = config[scenario_type].get('correct_threshold_percentage_based_on_100_observations', None)
            if threshold is not None:
                ax.hlines(y=threshold, xmin=idx-0.4, xmax=idx+0.4, 
                         color='red', linewidth=3, zorder=5)
                if not threshold_markers:
                    threshold_markers.append(plt.Line2D([0], [0], color='red', linewidth=3))

    # Update legend with outlier marker and performance dot
    lines = [plt.Line2D([0], [0], color=c, linestyle=s) for c, s in 
             zip(['#0000ff', '#4444ff', '#8888ff', '#aaaaff', '#ccccff'],
                 ['-', ':', ':', ':', ':'])]
    legend_labels = ['0%', '5%', '10%', '15%', '20%']
    
    # Add black dot for Performance on Simulation
    lines.append(plt.Line2D([0], [0], marker='o', color='black', linestyle='none', markersize=10))
    legend_labels.append('Performance on simulation')
    
    if df['is_outlier'].any():
        lines.append(plt.Line2D([0], [0], marker='^', color='black', linestyle='none', markersize=10))
        legend_labels.append('>100%')
    
    if threshold_markers:
        lines.extend(threshold_markers)
        legend_labels.append('Required threshold')
    
    ax.legend(lines, legend_labels, fontsize=18, loc='upper right')
    #remove minor ticks on x axis
    ax.xaxis.set_minor_locator(plt.NullLocator())
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=14)
    ax.set_ylabel('Absolute percent difference\nfrom full-obs baseline (%)', fontsize=24)
    ax.set_xlabel('')
    plt.title('Expert solution performance: 100 uniformly spaced observations', fontsize=32)
    
    plt.savefig(os.path.join('analysis', 'plots', 'defining_threshold_100_obs.pdf'), bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    main()
