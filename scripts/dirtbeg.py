import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2, degrees

# 1. Load the summary file
try:
    df = pd.read_csv("cluster_summary_1980_2005.csv")
    
    # --- THE FIX IS HERE ---
    # format='mixed' allows pandas to handle rows with AND without microseconds
    df['start_time'] = pd.to_datetime(df['start_time'], format='mixed')
    df['end_time'] = pd.to_datetime(df['end_time'], format='mixed')
    
    print("Loaded cluster summary successfully.")
except FileNotFoundError:
    print("Error: Please run your clustering script first to generate 'cluster_summary_1980_2005.csv'")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# 2. Function to calculate Bearing (Direction)
def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dLon = lon2 - lon1
    x = sin(dLon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - (sin(lat1) * cos(lat2) * cos(dLon))
    
    initial_bearing = atan2(x, y)
    deg = degrees(initial_bearing)
    compass_bearing = (deg + 360) % 360
    return compass_bearing

def get_compass_direction(bearing):
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    ix = round(bearing / 45)
    return dirs[ix % 8]

# 3. Process the data YEAR BY YEAR
results = []

if 'year' in df.columns:
    # Group by year to find paths WITHIN that year
    groups = df.groupby('year')
else:
    # If no year column, just treat whole file as one sequence
    groups = [('All', df)]

for year, group in groups:
    # Sort by time
    group = group.sort_values('start_time')
    clusters = group.to_dict('records')
    
    if len(clusters) < 2:
        continue 
        
    for i in range(len(clusters) - 1):
        c1 = clusters[i]
        c2 = clusters[i+1]
        
        bearing = calculate_bearing(c1['centroid_lat'], c1['centroid_lon'], 
                                    c2['centroid_lat'], c2['centroid_lon'])
        
        math_angle = (90 - bearing) % 360
        direction_label = get_compass_direction(bearing)
        
        row = {
            'Year': year,
            'From_Cluster': c1['cluster_rank'],
            'To_Cluster': c2['cluster_rank'],
            'From_Time': c1['end_time'],
            'To_Time': c2['start_time'],
            'From_Lat': c1['centroid_lat'],
            'From_Lon': c1['centroid_lon'],
            'To_Lat': c2['centroid_lat'],
            'To_Lon': c2['centroid_lon'],
            'Bearing_Degrees': round(bearing, 2),
            'Angle_wrt_X_Axis': round(math_angle, 2),
            'Direction': direction_label
        }
        results.append(row)

# 4. Save Output
if results:
    directions_df = pd.DataFrame(results)
    output_file = "cluster_directions_final.csv"
    directions_df.to_csv(output_file, index=False)
    print(f"\nSuccess! Calculated {len(directions_df)} stress migration paths.")
    print(f"Saved to: {output_file}")
else:
    print("No consecutive clusters found to calculate directions.")