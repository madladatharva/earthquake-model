import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the full dataset
df = pd.read_csv('cluster_directions_1980_2005.csv')

# Setup the figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# ==========================================
# PLOT 1: Directional Stress Heatmap (Time vs Angle)
# ==========================================
# We bin the data: X-axis = Year, Y-axis = Angle (0-360)
# This cleans up the "messy line plot" by showing density instead of jumping lines.

# Create bins for Years and Angles
year_bins = np.arange(1980, 2007, 1)
angle_bins = np.arange(0, 361, 45)  # 45-degree sectors (N, NE, E, etc.)
angle_labels = ['N (0-45)', 'NE (45-90)', 'E (90-135)', 'SE (135-180)', 
                'S (180-225)', 'SW (225-270)', 'W (270-315)', 'NW (315-360)']

# Create a 2D histogram
hist_data, x_edges, y_edges = np.histogram2d(
    df['Year'], 
    df['Angle_wrt_X_Axis'], 
    bins=[year_bins, angle_bins]
)

# Plot heatmap
sns.heatmap(hist_data.T, ax=ax1, cmap="YlOrRd", linewidths=0.5, linecolor='gray',
            xticklabels=year_bins[:-1], yticklabels=angle_labels, cbar_kws={'label': 'Frequency of Stress Events'})

ax1.invert_yaxis() # Origin at bottom
ax1.set_title('Evolution of Stress Direction (1980-2005)\n(Darker = More dominant direction in that year)', fontsize=14)
ax1.set_xlabel('Year')
ax1.set_ylabel('Direction Sector')


# ==========================================
# PLOT 2: Latitudinal Migration Heatmap (Time vs Latitude)
# ==========================================
# This answers: "Is the stress moving North or South over time?"
# We construct "paths" again to fill the space between clusters.

path_years = []
path_lats = []

# Create phantom points along the path for every single entry
for index, row in df.iterrows():
    # Create 20 interpolated points for the path
    lats = np.linspace(row['From_Lat'], row['To_Lat'], 20)
    # Repeat the year for these points so they align on X-axis
    years = [row['Year']] * 20
    
    path_lats.extend(lats)
    path_years.extend(years)

# Plot KDE Heatmap
sns.kdeplot(x=path_years, y=path_lats, ax=ax2, cmap="magma", fill=True, thresh=0.05, levels=15)
# Overlay actual data points for clarity
ax2.scatter(df['Year'], df['From_Lat'], color='cyan', s=15, alpha=0.6, label='Start Cluster')
ax2.scatter(df['Year'], df['To_Lat'], color='white', s=15, alpha=0.6, marker='x', label='End Cluster')

ax2.set_title('North-South Stress Migration (1980-2005)\n(Bright spots = Active Latitudinal Zones)', fontsize=14)
ax2.set_xlabel('Year')
ax2.set_ylabel('Latitude (Degrees)')
ax2.set_xlim(1980, 2006)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()