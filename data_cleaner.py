import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# ---------------------------------------------
# 1. Load and Clean Data (Using Extra Columns)
# ---------------------------------------------
print("Loading data...")
csv_df = pd.read_csv("coordinates.csv")
csv_df['time'] = pd.to_datetime(csv_df['time'])

# FILTER 1: Only real earthquakes
if 'type' in csv_df.columns:
    csv_df = csv_df[csv_df['type'] == 'earthquake']

# FILTER 2: Quality Control (Using 'gap' and 'horizontalError')
# If gap > 180, the location is mathematically unstable.
if 'gap' in csv_df.columns:
    csv_df = csv_df[(csv_df['gap'] <= 180) | (csv_df['gap'].isna())] # Keep NaNs if you want, or drop them

# ---------------------------------------------
# 2. Define 3D Metric
# ---------------------------------------------
def haversine_surface(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def custom_metric_3d(p1, p2):
    """
    p1, p2 = [lat, lon, depth, mag]
    """
    lat1, lon1, depth1, mag1 = p1
    lat2, lon2, depth2, mag2 = p2
    
    # 1. Check Magnitude FIRST (Fastest fail)
    # Relaxed constraint: If mag is huge (>7), we allow larger differences
    if abs(mag1 - mag2) > 0.5: # Increased from 0.3 to 0.5 to catch more relations
        return 1e9
        
    # 2. Calculate Surface Distance
    dist_surf = haversine_surface(lat1, lon1, lat2, lon2)
    
    # 3. Calculate Depth Difference
    dist_depth = abs(depth1 - depth2)
    
    # 4. Pythagorean theorem for 3D distance (Hypocentral distance)
    dist_3d = sqrt(dist_surf**2 + dist_depth**2)
    
    # 5. Threshold (e.g., 50km radius in 3D space)
    if dist_3d <= 50:
        return dist_3d
    
    return 1e9

# ---------------------------------------------
# 3. Loop through Years 1980-2005
# ---------------------------------------------
all_summaries = []

for year in range(2020, 2025):
    print(f"\nProcessing Year: {year}")
    year_df = csv_df[csv_df['time'].dt.year == year].copy()

    if year_df.empty:
        print(f"  No valid data for {year}.")
        continue

    # We MUST include depth now
    # Ensure depth is present, fill with 0 if missing for calculation safety
    if 'depth' not in year_df.columns:
        year_df['depth'] = 0
    else:
        year_df['depth'] = year_df['depth'].fillna(0)
        
    data_matrix = year_df[['lat', 'lon', 'depth', 'mag']].to_numpy()

    # Run DBSCAN
    # print(f"  Clustering {len(data_matrix)} events...")
    try:
        db = DBSCAN(
            eps=50, 
            min_samples=2, 
            metric=custom_metric_3d
        ).fit(data_matrix)
        
        year_df['cluster'] = db.labels_
        
        # Separate Noise (-1) from Clusters
        clusters_df = year_df[year_df['cluster'] != -1].copy()
        
        if not clusters_df.empty:
            # Rank clusters by time (optional, but good for consistency)
            cluster_order = (
                clusters_df.groupby('cluster')['time']
                .min()
                .sort_values()
                .reset_index()
            )
            rank_map = {row['cluster']: i+1 for i, row in cluster_order.iterrows()}
            clusters_df['cluster_rank'] = clusters_df['cluster'].map(rank_map)

            summary = clusters_df.groupby('cluster_rank').agg(
                centroid_lat=('lat', 'mean'),
                centroid_lon=('lon', 'mean'),
                centroid_depth=('depth', 'mean'),
                avg_mag=('mag', 'mean'),
                start_time=('time', 'min'),
                end_time=('time', 'max'),
                event_count=('time', 'count')
            ).reset_index()
            
            summary['year'] = year
            all_summaries.append(summary)
            print(f"  Found {len(summary)} clusters.")
        else:
            print("  No clusters found.")
            
    except Exception as e:
        print(f"  Error processing {year}: {e}")

# ---------------------------------------------
# 4. Export Consolidated Summary
# ---------------------------------------------
if all_summaries:
    final_df = pd.concat(all_summaries, ignore_index=True)
    
    # Reorder columns for better readability
    cols = ['year', 'cluster_rank', 'centroid_lat', 'centroid_lon', 'centroid_depth', 'avg_mag', 'start_time', 'end_time', 'event_count']
    final_df = final_df[cols]
    
    output_filename = "test0_set.csv"
    final_df.to_csv(output_filename, index=False)
    print(f"\nSuccess! Consolidated cluster summary saved to {output_filename}")
    print(final_df.head())
else:
    print("\nNo clusters found in the entire period.")