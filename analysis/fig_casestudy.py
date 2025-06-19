import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
#load scenarios/detailed_sims/9.6 M, 3.1 M.csv
df = pd.read_csv('scenarios/detailed_sims/9.6 M, 3.1 M.csv')


run1 = [
    [0.0, 1550000000.0, 3100000000.0, 4650000000.0, 6200000000.0, 7750000000.0, 9300000000.0, 10850000000.0, 12400000000.0, 13950000000.0],
    [13000000000.0, 13200000000.0, 13400000000.0, 13600000000.0, 13800000000.0, 14000000000.0, 14200000000.0, 14400000000.0, 14600000000.0, 14800000000.0],
    [13900000000.0, 13920000000.0, 13940000000.0, 13960000000.0, 13980000000.0, 14000000000.0, 14020000000.0, 14040000000.0, 14060000000.0, 14080000000.0],
    [13890000000.0, 13892000000.0, 13894000000.0, 13896000000.0, 13898000000.0, 13900000000.0, 13902000000.0, 13904000000.0, 13906000000.0, 13908000000.0]
]
run2 = [
    [0.0, 1550000000.0, 3100000000.0, 4650000000.0, 6200000000.0, 7750000000.0, 9300000000.0, 10850000000.0, 12400000000.0, 13950000000.0],
    [775000000.0, 2325000000.0, 3875000000.0, 5425000000.0, 6975000000.0, 8525000000.0, 10075000000.0, 11625000000.0, 13175000000.0, 14725000000.0],
    [6500000000.0, 6600000000.0, 6700000000.0, 6800000000.0, 6900000000.0, 7000000000.0, 7100000000.0, 7200000000.0, 7300000000.0, 7400000000.0],
    [6650000000.0, 6660000000.0, 6670000000.0, 6680000000.0, 6690000000.0, 6710000000.0, 6720000000.0, 6730000000.0, 6740000000.0, 6750000000.0]
]
#find star1_v using star1_vx, star1_vy, star1_vz
df['star1_v'] = np.sqrt(df['star1_vx']**2 + df['star1_vy']**2 + df['star1_vz']**2)
# Create figure with two subplots
plt.rcParams.update({'font.size': 14})

# Define colormap and normalization consistently
cmap = plt.cm.viridis
bounds = np.arange(len(run1) + 1)
norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

# Create figure with two subplots and space for colorbar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 1.2), gridspec_kw={'wspace': 0.4})

#set font.size to 16

# Plot for run1
ax1.plot(df['time'], df['star1_v'], 'k-', alpha=0.5)

# Add points for each observation in run1
for i, times in enumerate(run1):
    velocities = []
    for t in times:
        idx = (np.abs(df['time'] - t)).argmin()
        velocities.append(df['star1_v'].iloc[idx])
    # Create array of same color index for all points in this set
    colors = np.full(len(times), i)
    ax1.scatter(times, velocities, c=colors, cmap=cmap, norm=norm)

ax1.set_xlabel('Time [s]')
ax1.set_ylabel('$\\mathtt{star1}$' + '\nVelocity\n[m/s]')
ax1.set_ylim(np.min(df['star1_v'])*0.3, np.max(df['star1_v'])*1.1)

# Plot for run2
ax2.plot(df['time'], df['star1_v'], 'k-', alpha=0.5)

# Add points for each observation in run2
for i, times in enumerate(run2):
    velocities = []
    for t in times:
        idx = (np.abs(df['time'] - t)).argmin()
        velocities.append(df['star1_v'].iloc[idx])
    # Create array of same color index for all points in this set
    colors = np.full(len(times), i)
    ax2.scatter(times, velocities, c=colors, cmap=cmap, norm=norm)

#remove y tick labels from ax2
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('$\\mathtt{star1}$' + '\nVelocity\n[m/s]')
ax2.set_ylim(np.min(df['star1_v'])*0.3, np.max(df['star1_v'])*1.1)

# Add discrete colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                   cax=cbar_ax,
                   boundaries=bounds,
                   ticks=np.arange(len(run1)) + 0.5)
cbar.ax.set_yticklabels([str(i+1) for i in range(len(run1))])
cbar.ax.tick_params(length=0)
cbar.set_label('Observation\nSet')

#save to plots/casestudy.png
plt.savefig('analysis/plots/casestudy.png', bbox_inches='tight', dpi=300, transparent=True)