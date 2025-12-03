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
RADIUS_KM = 200  # Radius in kilometers for clustering
last_coords_x = -6942
last_coords_y = -6942


def adjust_shade(color, factor):
    # color = (R, G, B, A), factor <1 = darker, >1 = lighter
    r, g, b, a = color
    r = min(r * factor, 1)
    g = min(g * factor, 1)
    b = min(b * factor, 1)
    return (r, g, b, a)

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
        df['year_month'] = df['time'].apply(lambda x: (x.year - earliest_date.year) * 12 + x.month)

        
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

def plot_yearly_largest_clusters(df, boundaries, folder_name, start_year=1990, end_year=2013):  # Updated default end year
    global last_coords_x
    global last_coords_y
    """Plot plate boundaries and largest clusters for each year with different colors"""
    plt.figure(figsize=(12, 8))
    
    # Plot plate boundaries
    for boundary in boundaries:
        coords = boundary['coordinates']
        lats, lons = zip(*coords)
        plt.plot(lons, lats, label=f"Plate: {boundary['name']}", linewidth=2)
    
    # Get the range of years in the data
    min_year = max(df['year'].min(), start_year)
    max_year = min(df['year'].max(), end_year)
    
    # Generate distinct colors for each year *THIS*
    cmap = plt.get_cmap('tab20')  # Use a colormap with many distinct colors
    num_years = max_year - min_year + 1
    colors = ListedColormap(cmap.colors[:num_years])
    
    for i, year in enumerate(range(min_year, max_year + 1)): #THIS
        # Filter data for the current year
        year_df = df[df['year'] == year]
        
        # Identify the largest cluster for the year
        centroid_lat, centroid_lon = identify_largest_cluster(year_df, radius_km=RADIUS_KM, min_points=MIN_POINTS) #THIS
        
        # Plot the largest cluster as a big dot with a distinct color
        if centroid_lat is not None and centroid_lon is not None:
            plt.scatter(centroid_lon, centroid_lat, s=300, c=[colors(i)], alpha=0.8, label=f'{year}')
        else:
            print(f"No clusters found for {year}")
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Plate Boundaries and Largest Earthquake Clusters ({min_year}-{max_year})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f'{folder_name}/yearly_clusters_{min_year}_{max_year}.png', dpi=300)
    plt.close()
    







def plot_monthly_largest_clusters(df, boundaries, folder_name, start_month=0, end_month=12):
    
    global last_coords_x
    global last_coords_y
    
    # List to store centroid data for CSV
    centroid_data = []
    previous_centroid = None
    
    num_shades = 12
# Generate factors from light (1.5) to dark (0.5)
    factors = [1.5 - i*(1.5-0.5)/(num_shades-1) for i in range(num_shades)]
    
    month_name = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
    
    year_name = list(range(1980, 2014))  # Updated to include 2013
    
    
    """Plot plate boundaries and largest clusters for each year with different colors"""
    plt.figure(figsize=(12, 8))
    
    # Plot plate boundaries
    for boundary in boundaries:
        coords = boundary['coordinates']
        lats, lons = zip(*coords)
        plt.plot(lons, lats, label=f"Plate: {boundary['name']}", linewidth=2)
    
    # Get the range of years in the data
    min_month = max(df['year_month'].min(), start_month)
    max_month = min(df['year_month'].max(), end_month)
    
    # Generate distinct colors for each year *THIS*
    cmap = plt.get_cmap('tab20')  # Use a colormap with many distinct colors
    num_months = max_month//12 - min_month//12 + 1
    colors = ListedColormap(cmap.colors[:num_months])
    
    for i, month in enumerate(range(min_month, max_month + 1)): #THIS
        # Filter data for the current year
        month_df = df[df['year_month'] == month]
        
        if month_df.empty:
            print(f"No data for month {month}")
            continue  # Skip this iteration
        
        # Identify the largest cluster for the year
        centroid_lat, centroid_lon = identify_largest_cluster(month_df, radius_km=RADIUS_KM, min_points=MIN_POINTS) #THIS
        
        # Plot the largest cluster as a big dot with a distinct color
        if centroid_lat is not None and centroid_lon is not None:
            # Calculate distance from previous centroid
            distance_km = 0.0
            if previous_centroid is not None:
                distance_km = geodesic(previous_centroid, (centroid_lat, centroid_lon)).kilometers
            
            # Re-run clustering to get cluster information for statistics
            month_df_copy = month_df.copy()
            coords = month_df_copy[['lat', 'lon']].values
            radius_deg = RADIUS_KM / 111.0
            clustering = DBSCAN(eps=radius_deg, min_samples=MIN_POINTS, metric='euclidean').fit(coords)
            month_df_copy['cluster'] = clustering.labels_
            
            # Get the largest cluster data for additional statistics
            cluster_sizes = month_df_copy['cluster'].value_counts()
            if not cluster_sizes.empty and cluster_sizes.idxmax() != -1:
                largest_cluster_id = cluster_sizes.idxmax()
                largest_cluster_points = month_df_copy[month_df_copy['cluster'] == largest_cluster_id]
            else:
                largest_cluster_points = month_df_copy  # fallback to all points if no clusters
            
            # Calculate median depth and magnitude for the cluster
            median_depth = largest_cluster_points['depth'].median() if 'depth' in largest_cluster_points.columns else 0
            median_magnitude = largest_cluster_points['mag'].median() if 'mag' in largest_cluster_points.columns else 0
            
            # Create event_id and time format
            event_id = f"cluster_{year_name[month//12]}_{month_name[month%12]}"
            time_formatted = f"{year_name[month//12]}-{(month%12 + 1):02d}-15 12:00:00"
            
            # Save centroid data for CSV
            centroid_data.append({
                'event_id': event_id,
                'time': time_formatted,
                'latitude': centroid_lat,
                'longitude': centroid_lon,
                'depth_km': median_depth,
                'magnitude': median_magnitude,
                'distance_from_previous_km': distance_km
            })
            
            # Update previous centroid for next iteration
            previous_centroid = (centroid_lat, centroid_lon)
            
            plt.scatter(centroid_lon, centroid_lat, s=300, c=[adjust_shade(colors(i//12), factors[i%12])], alpha=0.8, label=f'{month_name[month%12]} {year_name[month//12]}',zorder=1) #alpha=(((i%12)/12) + 0.5)/1.5
            if last_coords_x != -6942 and  last_coords_y != -6942:
                plt.arrow(last_coords_x, last_coords_y, centroid_lon - last_coords_x, centroid_lat - last_coords_y,head_width=0.4, head_length=0.4, fc='blue', ec='blue', alpha=0.7, zorder=2)
            last_coords_x = centroid_lon
            last_coords_y = centroid_lat

            print(f'{i} ({month_name[month%12]}) % 12 / 12 = {(month%12)/12}')
        else:
            print(f"No clusters found for {month}")
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Plate Boundaries and Largest Earthquake Clusters ({month_name[min_month%12]} {year_name[min_month//12]}-{month_name[max_month%12]} {year_name[max_month//12]})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f'{folder_name}/monthly_clusters_{month_name[min_month%12]}_{year_name[min_month//12]}_{month_name[max_month%12]}_{year_name[max_month//12]}.png', dpi=300)
    plt.close()
    
    # Save centroid data to CSV
    if centroid_data:
        centroid_df = pd.DataFrame(centroid_data)
        csv_filename = f'{folder_name}/cluster_centroids_{month_name[min_month%12]}_{year_name[min_month//12]}_{month_name[max_month%12]}_{year_name[max_month//12]}.csv'
        centroid_df.to_csv(csv_filename, index=False)
        print(f"✅ Centroid coordinates saved to {csv_filename}")

        
    
    
    
    
    
    
    

def main():
    
    start_month = 1
    end_month = 12
    start_year = 1980
    end_year = 1980
    
    print("Years: 1980-2013")  # Updated to show 2013
    print("Months: 1: January, 2: February, 3: March, 4: April, 5: May, 6: June, 7: July, 8: August, 9: September, 10: October, 11: November, 12: December")
    
    inp = int(input("choose starting year: "))
    if(inp >= 1980 and inp <=2013):  # Updated upper limit to 2013
        start_year = inp
    inp = int(input("choose starting month(1-12): "))
    if(inp >= 1 and inp <= 12):
        start_month = inp
    
    if(start_year < 2013):  # Updated to allow up to 2013
        inp = int(input("choose ending year: "))
        if(inp >= start_year and inp <= 2013):  # Updated upper limit to 2013
            end_year = inp
        else:
            end_year = start_year
    else:
        end_year = 2013  # Updated default end year to 2013
    
    inp = int(input("month: "))
    if start_year == end_year:
        if inp > start_month and inp <= 12:
            end_month = inp
        else:
            end_month = start_month
    elif inp >= 1 and inp <= 12:
        end_month = inp
    
    
    start_month = (start_year-1980)*12 + start_month - 1
    end_month = (end_year-1980)*12 + end_month - 1
    
    print(f'from: {start_month} to: {end_month}')
            
    
    
    
    # Load data
    df = load_csv_data('coordinates.csv')
    boundaries = load_plate_boundaries('eu_in_plates.geojson')
    
    # Create folder for output
    folder_name = 'yearly_earthquake_clusters_1980_2013'  # Updated folder name
    os.makedirs(folder_name, exist_ok=True)
    
    folder2_name = 'monthly_earthquake_clusters_arrows'
    os.makedirs(folder2_name, exist_ok=True)
    
    # Generate graph with largest clusters for each year
    plot_monthly_largest_clusters(df, boundaries, folder2_name, start_month=start_month, end_month=end_month)

if __name__ == '__main__':
    main()