import pandas as pd
import numpy as np

# -----------------------------
# 1. Load earthquake catalog
# -----------------------------
catalog_file = "quake.csv"   # <-- replace with your file
df = pd.read_csv(catalog_file)

# Parse times
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

# Reference center for coordinates
lat0, lon0 = df['latitude'].mean(), df['longitude'].mean()

# Convert lat/lon to km (approx)
df['x_km'] = (df['longitude'] - lon0) * 111 * np.cos(np.radians(lat0))
df['y_km'] = (df['latitude'] - lat0) * 111

# First event
t0 = df['time'].iloc[0]

# Convert time to months since first event
df['t_months'] = ((df['time'].dt.year - t0.year) * 12 +
                  (df['time'].dt.month - t0.month))

print("Catalog loaded:")
print(df.head())

# -----------------------------
# 2. Hawkes-like parameters (per month)
# -----------------------------
mu = 1       # background rate ~0.2 events/month (tune!)
alpha = 0.3    # branching ratio (each event spawns ~0.3 aftershocks)
beta = 1/3     # temporal decay per month (~3 month half-life)

spatial_sigma_km = 5.0  # Gaussian spread in km

# -----------------------------
# 3. Simulation
# -----------------------------
forecast_horizon = 24   # predict 24 months ahead
n_sims = 20             # number of simulated catalogs

results = []

for sim_id in range(1, n_sims+1):
    current_time = df['t_months'].max()
    end_time = current_time + forecast_horizon
    
    sim_events = []
    
    # Background events: Poisson(mu * horizon)
    n_background = np.random.poisson(mu * forecast_horizon)
    background_times = np.sort(np.random.uniform(current_time, end_time, n_background))
    
    for t_event in background_times:
        # Pick random parent for coordinates
        parent_idx = np.random.randint(len(df))
        x_parent, y_parent = df.loc[parent_idx, ['x_km','y_km']]
        
        # Add Gaussian spatial noise
        x_pred = x_parent + np.random.normal(0, spatial_sigma_km)
        y_pred = y_parent + np.random.normal(0, spatial_sigma_km)
        
        lon_pred = lon0 + x_pred / (111 * np.cos(np.radians(lat0)))
        lat_pred = lat0 + y_pred / 111
        
        sim_events.append((t_event, lat_pred, lon_pred))
        
        # Aftershocks
        n_after = np.random.poisson(alpha)
        for _ in range(n_after):
            dt = np.random.exponential(1/beta)  # months
            if t_event + dt < end_time:
                x_child = x_parent + np.random.normal(0, spatial_sigma_km)
                y_child = y_parent + np.random.normal(0, spatial_sigma_km)
                lon_child = lon0 + x_child / (111 * np.cos(np.radians(lat0)))
                lat_child = lat0 + y_child / 111
                sim_events.append((t_event+dt, lat_child, lon_child))
    
    # Save results
    for t_event, lat_pred, lon_pred in sim_events:
        # Convert months back to calendar time
        future_time = t0 + pd.DateOffset(months=int(round(t_event)))
        results.append({
            "sim_id": sim_id,
            "time_pred": future_time.strftime("%Y-%m"),
            "latitude_pred": lat_pred,
            "longitude_pred": lon_pred
        })

# -----------------------------
# 4. Save to CSV
# -----------------------------
out_df = pd.DataFrame(results)
out_df.to_csv("predicted_events.csv", index=False)

print("\nSimulation complete!")
print("Saved predictions to predicted_events.csv")
print(out_df.head())
