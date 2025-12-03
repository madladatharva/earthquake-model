import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from math import radians, sin, cos, sqrt, atan2, degrees

# ---------------------------------------------
# 1. Helper Functions
# ---------------------------------------------

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    a = max(0.0, min(1.0, a)) # Clamp to [0, 1]
    
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

def custom_distance(p1, p2):
    """
    Custom metric for DBSCAN:
    distance valid ONLY IF:
      distance <= 50 km AND |mag diff| <= 0.3
    otherwise return large distance
    """
    lat1, lon1, mag1 = p1
    lat2, lon2, mag2 = p2

    dist_km = haversine(lat1, lon1, lat2, lon2)
    mag_diff = abs(mag1 - mag2)

    if dist_km <= 50 and mag_diff <= 0.3:
        return dist_km

    return 1e9

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculates the initial bearing (forward azimuth) between two points.
    """
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    dlon_rad = radians(lon2 - lon1)

    y = sin(dlon_rad) * cos(lat2_rad)
    x = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(dlon_rad)
    
    bearing_rad = atan2(y, x)
    bearing_deg = degrees(bearing_rad)
    
    # Normalize to 0-360
    return (bearing_deg + 360) % 360

def get_cardinal_direction(bearing):
    """
    Converts a bearing in degrees to a cardinal direction string.
    """
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    ix = round(bearing / 45)
    return dirs[ix % 8]

# ---------------------------------------------
# 2. Main Processing
# ---------------------------------------------

def main():
    # Load Data
    print("Loading coordinates.csv...")
    try:
        csv_df = pd.read_csv("coordinates.csv")
    except FileNotFoundError:
        print("Error: coordinates.csv not found.")
        return

    csv_df['time'] = pd.to_datetime(csv_df['time'])
    
    results = []
    
    # Loop through years 1980 to 2005
    start_year = 1980
    end_year = 2005
    
    print(f"Processing years {start_year} to {end_year}...")

    for year in range(start_year, end_year + 1):
        # Filter for current year
        year_df = csv_df[csv_df['time'].dt.year == year].copy()
        
        if year_df.empty:
            print(f"Year {year}: No data.")
            continue

        # Prepare clustering data: (lat, lon, mag)
        data = np.vstack((year_df['lat'], year_df['lon'], year_df['mag'])).T

        # DBSCAN
        # Note: metric='precomputed' is usually faster if we compute a matrix, 
        # but for custom logic per-point, we pass the function. 
        # This can be slow for large datasets.
        db = DBSCAN(
            eps=50, 
            min_samples=2, 
            metric=custom_distance
        ).fit(data)

        year_df['cluster'] = db.labels_

        # Remove noise
        clusters_df = year_df[year_df['cluster'] != -1].copy()
        
        if clusters_df.empty:
            print(f"Year {year}: No clusters found.")
            continue

        # Rank clusters by earliest event time
        cluster_order = (
            clusters_df.groupby('cluster')['time']
            .min()
            .sort_values()
            .reset_index()
        )
        
        # Create a mapping from original cluster ID to Rank (1, 2, 3...)
        rank_map = {row['cluster']: i+1 for i, row in cluster_order.iterrows()}
        clusters_df['cluster_rank'] = clusters_df['cluster'].map(rank_map)

        # Compute centroids for each ranked cluster
        centroids = (
            clusters_df.groupby('cluster_rank')
            .agg({
                'lat': 'mean', 
                'lon': 'mean',
                'time': 'min' # Keep track of start time for reference
            })
            .reset_index()
            .sort_values('cluster_rank')
        )

        # Calculate directions between consecutive clusters (1->2, 2->3, etc.)
        num_clusters = len(centroids)
        if num_clusters < 2:
            print(f"Year {year}: Only 1 cluster found, cannot compute directions.")
            continue

        print(f"Year {year}: Found {num_clusters} clusters. Calculating directions...")

        for i in range(num_clusters - 1):
            c1 = centroids.iloc[i]
            c2 = centroids.iloc[i+1]
            
            lat1, lon1 = c1['lat'], c1['lon']
            lat2, lon2 = c2['lat'], c2['lon']
            
            bearing = calculate_bearing(lat1, lon1, lat2, lon2)
            direction = get_cardinal_direction(bearing)
            
            # Calculate angle w.r.t X-axis (East = 0 degrees, Counter-Clockwise)
            # Standard Math Angle = 90 - Bearing
            angle_wrt_x = (90 - bearing) % 360

            results.append({
                'Year': year,
                'From_Cluster': int(c1['cluster_rank']),
                'To_Cluster': int(c2['cluster_rank']),
                'From_Time': c1['time'],
                'To_Time': c2['time'],
                'From_Lat': lat1,
                'From_Lon': lon1,
                'To_Lat': lat2,
                'To_Lon': lon2,
                'Bearing_Degrees': round(bearing, 2),
                'Angle_wrt_X_Axis': round(angle_wrt_x, 2),
                'Direction': direction
            })

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        output_filename = "cluster_directions_1980_2005.csv"
        results_df.to_csv(output_filename, index=False)
        print(f"\nSuccess! Results saved to {output_filename}")
        print(results_df.head())
    else:
        print("\nNo results generated.")

if __name__ == "__main__":
    main()
