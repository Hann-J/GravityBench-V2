import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({
    'mathtext.fontset': 'dejavusans',
    'font.family': 'sans-serif', 
    'font.size': 14,
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
df = pd.read_csv('scenarios/detailed_sims/3.1 M, 0.18 M, Elliptical, Single Orbit.csv')

# From outputs_range_of_N/claude-3-5-sonnet-20241022_11-01_21_44_53/claude-3-5-sonnet-20241022_11-01_21_44_53.json
run1 = [
    [0, 23400000.0, 46800000.0, 70200000.0, 93600000.0, 117000000.0, 140400000.0, 163800000.0, 187200000.0, 210600000.0],
    [0, 100000, 500000, 1000000, 2000000, 3000000, 4000000],
    [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000],
    [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
]

max_time = 2.34e+08
# Expert observation strategy (iteratively chosen by an expert with the same setup and information as the AI agent)
run2 = [
    list(np.linspace(0, max_time, 10)),
    list(np.linspace(0, max_time/10, 10)),
    list(np.linspace(max_time-max_time/10, max_time, 10)),
    list(np.linspace(2.31e+08-max_time/50, 2.31e+08+max_time/50, 10)),
    list(np.linspace(2.33e+08-max_time/50, 2.33e+08+max_time/50, 10))
]


#find separation between stars
df['separation'] = np.sqrt((df['star1_x'] - df['star2_x'])**2 + 
                         (df['star1_y'] - df['star2_y'])**2 + 
                         (df['star1_z'] - df['star2_z'])**2)

# Create figure with two subplots and space for colorbar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 1.6), gridspec_kw={'wspace': 0.25})

# Define colormap and normalization consistently
cmap = plt.cm.viridis  # or whatever colormap you prefer
bounds = np.arange(len(run1) + 1)
norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

# Plot for run1
ax1.plot(df['time'], df['separation'], 'k-', alpha=0.5)

# Add points for each observation in run1
for i, times in enumerate(run1):
    separations = []
    for t in times:
        idx = (np.abs(df['time'] - t)).argmin()
        separations.append(df['separation'].iloc[idx])
    # Create array of same color index for all points in this set
    colors = np.full(len(times), i)
    sc = ax1.scatter(times, separations, c=colors, cmap=cmap, norm=norm)
    sc.set_rasterized(True)

ax1.set_xlabel('Time [s]', fontsize=14)
ax1.set_ylabel('Separation\n[m]', fontsize=14)
ax1.set_ylim(np.min(df['separation'])*0.4, np.max(df['separation'])*1.5)
ax1.set_yscale('log')

# Plot for run2
ax2.plot(df['time'], df['separation'], 'k-', alpha=0.5)

# Add points for each observation in run2
for i, times in enumerate(run2):
    separations = []
    for t in times:
        idx = (np.abs(df['time'] - t)).argmin()
        separations.append(df['separation'].iloc[idx])
    # Create array of same color index for all points in this set
    colors = np.full(len(times), i)
    sc = ax2.scatter(times, separations, c=colors, cmap=cmap, norm=norm)
    sc.set_rasterized(True)
#remove y tick labels from ax2
ax2.set_xlabel('Time [s]', fontsize=14)
ax2.set_ylabel('Separation\n[m]', fontsize=14)
ax2.set_ylim(np.min(df['separation'])*0.4, np.max(df['separation'])*1.5)

# Add discrete colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                   cax=cbar_ax,
                   boundaries=bounds,
                   ticks=np.arange(len(run1)) + 0.5)
cbar.ax.set_yticklabels([str(i+1) for i in range(len(run1))])
cbar.ax.tick_params(length=0)
cbar.set_label('Observation\nSet', fontsize=14)

# Add periastron lines and text
actual_min_seperation = np.min(df['separation'])
ax1.axhline(actual_min_seperation, color='red', alpha=0.5, linestyle='--')
ax2.axhline(actual_min_seperation, color='red', alpha=0.5, linestyle='--')

# Add periastron text annotation only to first plot
text = f'Closest approach (periastron)'
ax2.text(0.25e8, actual_min_seperation, text, transform=ax2.transData,
         verticalalignment='bottom',
         fontsize=14)

ax1.set_title('Claude 3.5 Sonnet', fontsize=16)
ax2.set_title('Expert Solution', fontsize=16)
ax2.set_yscale('log')
ax1.set_yscale('log')
plt.savefig('outputs_range_of_N/plots/periastron_claude_v_expert.pdf', 
            bbox_inches='tight', dpi=300, transparent=False)