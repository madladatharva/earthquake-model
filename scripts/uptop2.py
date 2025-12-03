import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os

# 1. Setup Output Folder
output_folder = "yearly_heatmaps_geojson"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 2. Load Cluster Data
try:
    df = pd.read_csv("cluster_directions_final.csv")
    print(f"Loaded {len(df)} stress paths.")
except FileNotFoundError:
    print("Error: 'cluster_directions_final.csv' not found.")
    exit()

# 3. Load Background Map (GeoJSON) and ensure CRS matches data (lat/lon EPSG:4326)
try:
    geo_df = gpd.read_file("eu_in_plates.geojson")
    # If GeoJSON has a CRS, convert to EPSG:4326 (lon/lat). If not, assume it's already lon/lat.
    if hasattr(geo_df, 'crs') and geo_df.crs is not None:
        try:
            geo_df = geo_df.to_crs(epsg=4326)
        except Exception:
            # If conversion fails, continue with original (best-effort)
            pass
except Exception as e:
    print(f"Warning: could not read 'eu_in_plates.geojson' ({e}). Continuing without background map.")
    geo_df = None
# 4. Interpolation Function
def interpolate_path(row, num_points=50):
    lats = np.linspace(row['From_Lat'], row['To_Lat'], num_points)
    lons = np.linspace(row['From_Lon'], row['To_Lon'], num_points)
    return lats, lons

# 5. Loop through Years
unique_years = sorted(df['Year'].unique())

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
    fig, ax = plt.subplots(figsize=(12, 10))

    # A. Plot Background Map (GeoJSON) if available
    if geo_df is not None:
        geo_df.plot(ax=ax, color="#ba2525", edgecolor='#999999', linewidth=0.5, zorder=1)
    else:
        ax.grid(True, color='gray', linestyle='--', alpha=0.3)

    # B. The Heatmap (The "Stress Highway")
    # Using 'magma' or 'Reds' looks better on white than 'inferno'
    try:
        if len(path_lons) > 3:
            sns.kdeplot(x=path_lons, y=path_lats, cmap="magma_r", fill=True,
                        thresh=0.05, alpha=0.5, ax=ax, zorder=2)
    except Exception:
        # If KDE fails for any reason, skip it but continue plotting lines/points
        pass

    # C. The Lines (Direction indicators) - CHANGED TO BLACK for visibility
    for _, row in year_data.iterrows():
        ax.plot([row['From_Lon'], row['To_Lon']], 
                [row['From_Lat'], row['To_Lat']], 
                color='black', linewidth=1, alpha=0.6, zorder=3)
        
        # Arrow
        ax.annotate('', xy=(row['To_Lon'], row['To_Lat']), xytext=(row['From_Lon'], row['From_Lat']),
                    arrowprops=dict(arrowstyle="->", color='black', lw=1.5), zorder=3)

    # D. The Clusters
    ax.scatter(year_data['From_Lon'], year_data['From_Lat'], c='cyan', s=60, edgecolors='black', zorder=4, label='Start')
    ax.scatter(year_data['To_Lon'], year_data['To_Lat'], c='red', s=60, edgecolors='black', zorder=4, label='End')
    
    # Label Clusters
    for _, row in year_data.iterrows():
        # Offset text slightly so it doesn't overlap the dot
        ax.text(row['From_Lon'], row['From_Lat'], str(int(row['From_Cluster'])), fontsize=10, color='black', fontweight='bold', zorder=5)
        ax.text(row['To_Lon'], row['To_Lat'], str(int(row['To_Cluster'])), fontsize=10, color='black', fontweight='bold', zorder=5)

    # E. Formatting
    ax.set_title(f"Seismic Stress Migration - {int(year)}", fontsize=16, color='black')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_facecolor('white') # Force white background

    # Ensure axis limits cover both the GeoJSON and the data paths
    # Compute combined bounds (lon/lat)
    all_lons = [] if len(path_lons) == 0 else path_lons
    all_lats = [] if len(path_lats) == 0 else path_lats
    if geo_df is not None:
        minx, miny, maxx, maxy = geo_df.total_bounds
        all_lons.extend([minx, maxx])
        all_lats.extend([miny, maxy])

    if all_lons and all_lats:
        pad_lon = (max(all_lons) - min(all_lons)) * 0.05 if max(all_lons) != min(all_lons) else 0.1
        pad_lat = (max(all_lats) - min(all_lats)) * 0.05 if max(all_lats) != min(all_lats) else 0.1
        ax.set_xlim(min(all_lons) - pad_lon, max(all_lons) + pad_lon)
        ax.set_ylim(min(all_lats) - pad_lat, max(all_lats) + pad_lat)
    
    # Keep equal aspect for geographic accuracy
    ax.set_aspect('equal', adjustable='box')

    # -- Save --
    filename = f"{output_folder}/stress_map_{int(year)}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filename}")

print(f"\nDone! Images saved in folder: '{output_folder}'")