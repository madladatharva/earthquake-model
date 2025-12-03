import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# ---------------------------------------------
# Custom Haversine distance
# ---------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    # Corrected formula: use **2 for squaring, not *2
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    
    # Clamp a to [0, 1] to prevent math domain error
    a = max(0.0, min(1.0, a))
    
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


# ---------------------------------------------
# Custom metric for DBSCAN:
# distance valid ONLY IF:
#   distance <= 50 km AND |mag diff| <= 0.3
# otherwise return large distance
# ---------------------------------------------
def custom_distance(p1, p2):
    lat1, lon1, mag1 = p1
    lat2, lon2, mag2 = p2

    dist_km = haversine(lat1, lon1, lat2, lon2)
    mag_diff = abs(mag1 - mag2)

    if dist_km <= 50 and mag_diff <= 0.3:
        return dist_km

    return 1e9  # treat as extremely far


# ---------------------------------------------
# Load map
# ---------------------------------------------
geo_df = gpd.read_file("eu_in_plates.geojson")

# ---------------------------------------------
# Load CSV
# ---------------------------------------------
csv_df = pd.read_csv("coordinates.csv")
csv_df['time'] = pd.to_datetime(csv_df['time'])

# ---------------------------------------------
# Year filter
# ---------------------------------------------
year = int(input("Enter year: "))
year_df = csv_df[csv_df['time'].dt.year == year].copy()

if year_df.empty:
    print("No earthquakes in that year.")
    exit()

# ---------------------------------------------
# GeoDataFrame for filtered year
# ---------------------------------------------
geometry = [Point(xy) for xy in zip(year_df['lon'], year_df['lat'])]
geo_csv = gpd.GeoDataFrame(year_df, geometry=geometry)
geo_csv.set_crs(geo_df.crs, inplace=True)

# ---------------------------------------------
# Prepare clustering data: (lat, lon, mag)
# ---------------------------------------------
data = np.vstack((geo_csv['lat'], geo_csv['lon'], geo_csv['mag'])).T

# ---------------------------------------------
# DBSCAN with custom metric
# ---------------------------------------------
db = DBSCAN(
    eps=50,        # ignored because metric returns real distances
    min_samples=2,
    metric=custom_distance
).fit(data)

geo_csv['cluster'] = db.labels_

# Remove noise
geo_csv = geo_csv[geo_csv['cluster'] != -1].copy()
if geo_csv.empty:
    print("No clusters found under given rules.")
    exit()

# ---------------------------------------------
# Rank clusters by earliest event time
# ---------------------------------------------
cluster_order = (
    geo_csv.groupby('cluster')['time']
    .min()
    .sort_values()
    .reset_index()
)

rank_map = {row['cluster']: i+1 for i, row in cluster_order.iterrows()}
geo_csv['cluster_rank'] = geo_csv['cluster'].map(rank_map)

# ---------------------------------------------
# Compute cluster centroids
# ---------------------------------------------
centroids = (
    geo_csv.groupby('cluster_rank')
    .agg({'lat': 'mean', 'lon': 'mean'})
    .reset_index()
)

centroids['geometry'] = centroids.apply(
    lambda r: Point(r['lon'], r['lat']),
    axis=1
)

centroids_gdf = gpd.GeoDataFrame(centroids, geometry='geometry', crs=geo_df.crs)

# ---------------------------------------------
# Plot
# ---------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10))

geo_df.plot(ax=ax, color='lightgrey', edgecolor='black')

centroids_gdf.plot(
    ax=ax,
    markersize=80,
    color='red'
)

for _, row in centroids_gdf.iterrows():
    ax.text(row['lon'], row['lat'], str(row['cluster_rank']),
            fontsize=12, ha='center', va='center', color='black')

plt.title(f"Earthquake Clusters (50 km, mag≤0.3, one dot per cluster) – {year}")
plt.show()