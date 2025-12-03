import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Setup Output Folder
output_folder = "yearly_heatmaps"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 2. Load Data
try:
    df = pd.read_csv("cluster_directions_final.csv")
    print(f"Loaded {len(df)} stress paths.")
except FileNotFoundError:
    print("Error: 'cluster_directions_final.csv' not found.")
    exit()

# 3. Function to generate path points (for the heatmap "glow")
def interpolate_path(row, num_points=50):
    lats = np.linspace(row['From_Lat'], row['To_Lat'], num_points)
    lons = np.linspace(row['From_Lon'], row['To_Lon'], num_points)
    return lats, lons

# 4. Loop through every Year present in the file
unique_years = df['Year'].unique()

print(f"Generating maps for {len(unique_years)} years...")

for year in unique_years:
    year_data = df[df['Year'] == year]
    
    if year_data.empty:
        continue

    # -- Prepare Heatmap Data --
    path_lats = []
    path_lons = []
    
    for _, row in year_data.iterrows():
        lats, lons = interpolate_path(row)
        path_lats.extend(lats)
        path_lons.extend(lons)

    # -- Setup Plot --
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # A. The Heatmap (The "Stress Highway")
    # We use KDE (Kernel Density Estimate) to make the paths look like glowing energy
    try:
        sns.kdeplot(x=path_lons, y=path_lats, cmap="inferno", fill=True, 
                    thresh=0.05, alpha=0.6, ax=ax, zorder=2)
    except ValueError:
        # Happens if points are perfectly straight line or too few
        print(f"  Skipping heatmap density for {year} (not enough variance), plotting lines only.")

    # B. The Lines (Direction indicators)
    for _, row in year_data.iterrows():
        ax.plot([row['From_Lon'], row['To_Lon']], 
                [row['From_Lat'], row['To_Lat']], 
                color='white', linewidth=1, alpha=0.5, zorder=3)
        
        # Arrow to show direction
        mid_lat = (row['From_Lat'] + row['To_Lat']) / 2
        mid_lon = (row['From_Lon'] + row['To_Lon']) / 2
        ax.annotate('', xy=(row['To_Lon'], row['To_Lat']), xytext=(row['From_Lon'], row['From_Lat']),
                    arrowprops=dict(arrowstyle="->", color='white', lw=1), zorder=3)

    # C. The Clusters (Start/End points)
    ax.scatter(year_data['From_Lon'], year_data['From_Lat'], c='cyan', s=50, edgecolors='black', zorder=4, label='Start')
    ax.scatter(year_data['To_Lon'], year_data['To_Lat'], c='red', s=50, edgecolors='black', zorder=4, label='End')
    
    # Label Clusters
    for _, row in year_data.iterrows():
        ax.text(row['From_Lon'], row['From_Lat'], str(int(row['From_Cluster'])), fontsize=9, color='white', fontweight='bold', zorder=5)
        ax.text(row['To_Lon'], row['To_Lat'], str(int(row['To_Cluster'])), fontsize=9, color='white', fontweight='bold', zorder=5)

    # D. Formatting
    ax.set_title(f"Seismic Stress Migration - {int(year)}", fontsize=14, color='black')
    ax.set_facecolor('#1a1a1a') # Dark background makes the heatmap pop
    ax.grid(True, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    # Set consistent bounds if you want all maps to look the same size
    # (Optional: remove if you want auto-zoom)
    # ax.set_xlim(78, 95) 
    # ax.set_ylim(26, 33)

    # -- Save and Close --
    filename = f"{output_folder}/stress_map_{int(year)}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filename}")

print("\nDone! Check the 'yearly_heatmaps' folder.")