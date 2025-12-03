import pandas as pd
import numpy as np
import json
import os
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
from sklearn.cluster import DBSCAN

# Adjustable Parameters
MIN_POINTS = 2  # Minimum number of points required to form a cluster
RADIUS_KM = 50  # Radius in kilometers for clustering

def load_csv_data(csv_file):
    """Load CSV data with lat/lon coordinates, magnitude, and time"""
    try:
        print(f"📂 Loading CSV data from {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # Check for required columns
        required_columns = ['lat', 'lon', 'time']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Missing required columns in CSV: {missing_columns}")
            return None
        
        # Convert 'time' column to datetime and drop invalid rows
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time'])
        
        # Extract the year from the time column
        df['year'] = df['time'].dt.year
        
        # Convert time to months since the earliest date
        earliest_date = df['time'].min()
        df['time_in_months'] = df['time'].apply(lambda x: (x.year - earliest_date.year) * 12 + (x.month - earliest_date.month))
        
        if 'mag' not in df.columns:
            print("⚠️  No magnitude column found. Magnitude analysis will be skipped.")
        else:
            df['mag'] = df['mag'].fillna(0)
        
        # Drop rows with invalid lat/lon values
        df = df.dropna(subset=['lat', 'lon'])
        df = df[(df['lat'] >= -90) & (df['lat'] <= 90)]
        df = df[(df['lon'] >= -180) & (df['lon'] <= 180)]
        
        print(f"✅ Successfully loaded {len(df)} valid data points")
        return df
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return None

def load_plate_boundaries(geojson_file):
    """Load plate boundary data from GeoJSON"""
    try:
        print(f"📂 Loading plate boundaries from {geojson_file}...")
        with open(geojson_file, 'r') as f:
            data = json.load(f)
        
        boundaries = []
        for feature in data['features']:
            plate_name = feature['properties']['Name']
            coords = feature['geometry']['coordinates']
            plate_coords = [(coord[1], coord[0]) for coord in coords]
            boundaries.append({'name': plate_name, 'coordinates': plate_coords})
        print(f"✅ Successfully loaded {len(boundaries)} plate boundaries")
        return boundaries
    except Exception as e:
        print(f"❌ Error loading GeoJSON file: {e}")
        return None

def identify_largest_cluster(df, radius_km=RADIUS_KM, min_points=MIN_POINTS):
    """Identify the largest cluster of earthquakes within a certain radius"""
    # Ensure df is not a view but a copy
    df = df.copy()
    
    coords = df[['lat', 'lon']].values
    
    # Convert radius from km to degrees (approximation)
    radius_deg = radius_km / 111.0  # 1 degree = ~111 km
    
    # Use DBSCAN clustering algorithm
    clustering = DBSCAN(eps=radius_deg, min_samples=min_points, metric='euclidean').fit(coords)
    df.loc[:, 'cluster'] = clustering.labels_  # Use .loc to avoid the warning
    
    # Check if there are clusters
    cluster_sizes = df['cluster'].value_counts()
    if cluster_sizes.empty or cluster_sizes.idxmax() == -1:  # Handle noise points
        return None, None
    
    # Find the largest cluster
    largest_cluster_id = cluster_sizes.idxmax()
    largest_cluster_points = df[df['cluster'] == largest_cluster_id]
    
    # Calculate centroid
    centroid_lat = largest_cluster_points['lat'].mean()
    centroid_lon = largest_cluster_points['lon'].mean()
    return centroid_lat, centroid_lon

def plot_yearly_largest_clusters(df, boundaries, folder_name, start_year=1990):
    """Plot plate boundaries and largest clusters for each year with different colors"""
    plt.figure(figsize=(12, 8))
    
    # Plot plate boundaries
    for boundary in boundaries:
        coords = boundary['coordinates']
        lats, lons = zip(*coords)
        plt.plot(lons, lats, label=f"Plate: {boundary['name']}", linewidth=2)
    
    # Get the range of years in the data
    min_year = max(df['year'].min(), start_year)
    max_year = df['year'].max()
    
    # Generate distinct colors for each year
    cmap = get_cmap('tab20')  # Use a colormap with many distinct colors
    num_years = max_year - min_year + 1
    colors = ListedColormap(cmap.colors[:num_years])
    
    for i, year in enumerate(range(min_year, max_year + 1)):
        # Filter data for the current year
        year_df = df[df['year'] == year]
        
        # Identify the largest cluster for the year
        centroid_lat, centroid_lon = identify_largest_cluster(year_df, radius_km=RADIUS_KM, min_points=MIN_POINTS)
        
        # Plot the largest cluster as a big dot with a distinct color
        if centroid_lat is not None and centroid_lon is not None:
            plt.scatter(centroid_lon, centroid_lat, s=300, c=[colors(i)], alpha=0.8, label=f'{year}')
        else:
            print(f"No clusters found for {year}")
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Plate Boundaries and Largest Earthquake Clusters (1980-2009)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f'{folder_name}/yearly_colored_clusters_1980_2009.png', dpi=300)
    plt.close()

def main():
    # Load data
    df = load_csv_data('coordinates.csv')
    boundaries = load_plate_boundaries('eu_in_plates.geojson')
    
    # Create folder for output
    folder_name = 'yearly_earthquake_clusters_1980_2009'
    os.makedirs(folder_name, exist_ok=True)
    
    # Generate graph with largest clusters for each year
    plot_yearly_largest_clusters(df, boundaries, folder_name, start_year=1980)

if __name__ == '__main__':
    main()