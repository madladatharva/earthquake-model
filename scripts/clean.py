import pandas as pd
import numpy as np

# 1. Load Data
try:
    df = pd.read_csv("cluster_directions_final.csv")
except FileNotFoundError:
    print("Error: 'cluster_directions_final.csv' not found. Please run the clustering/directions scripts first.")
    exit()

# 2. Define the Grid System (e.g., 20x20 grid over the region)
LAT_MIN, LAT_MAX = 15, 35
LON_MIN, LON_MAX = 65, 100
GRID_SIZE = 20 

def get_sector_id(lat, lon):
    # Normalize lat/lon to 0-1 range
    if lat < LAT_MIN or lat > LAT_MAX or lon < LON_MIN or lon > LON_MAX:
        return -1 # Out of bounds
    
    y_idx = int((lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * GRID_SIZE)
    x_idx = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * GRID_SIZE)
    
    # Clamp indices
    y_idx = min(y_idx, GRID_SIZE - 1)
    x_idx = min(x_idx, GRID_SIZE - 1)
    
    return y_idx * GRID_SIZE + x_idx

# 3. Feature Engineering
model_data = []

# --- THE FIX IS HERE: Added format='mixed' ---
try:
    df['To_Time'] = pd.to_datetime(df['To_Time'], format='mixed')
except Exception as e:
    print(f"Error parsing dates: {e}")
    print("Try ensuring your CSV date columns are clean.")
    exit()

# Sort by time globally to capture the true sequence
df = df.sort_values('To_Time')

# Create a sliding window (Lookback)
LOOKBACK_STEPS = 3 

# Convert all rows to a sequential list
sequence = df.to_dict('records')

print("Generating ML features...")

for i in range(LOOKBACK_STEPS, len(sequence)):
    current_event = sequence[i]
    
    # TARGET: Where did the stress go NEXT?
    target_sector = get_sector_id(current_event['To_Lat'], current_event['To_Lon'])
    if target_sector == -1: continue

    # FEATURES: History of previous steps
    features = {}
    
    # Time features
    features['Month'] = current_event['To_Time'].month
    features['Year'] = current_event['To_Time'].year
    
    # Current State (Where did this specific event start?)
    features['Current_Source_Sector'] = get_sector_id(current_event['From_Lat'], current_event['From_Lon'])
    features['Current_Angle'] = current_event['Angle_wrt_X_Axis']
    features['Current_Bearing'] = current_event['Bearing_Degrees']
    
    # Add Lag Features (Previous N events)
    valid_history = True
    for step in range(1, LOOKBACK_STEPS + 1):
        past_event = sequence[i - step]
        prefix = f"Lag{step}_"
        
        past_target_sector = get_sector_id(past_event['To_Lat'], past_event['To_Lon'])
        if past_target_sector == -1: 
            valid_history = False
            break
            
        features[prefix + 'Sector'] = past_target_sector
        features[prefix + 'Angle'] = past_event['Angle_wrt_X_Axis']
        
        # Time delta between events (in days)
        delta_days = (current_event['To_Time'] - past_event['To_Time']).days
        features[prefix + 'DaysAgo'] = delta_days
    
    if valid_history:
        features['Target_Sector'] = target_sector
        model_data.append(features)

# 4. Save ML Ready Dataset
if model_data:
    ml_df = pd.DataFrame(model_data)
    output_file = "earthquake_ml_dataset.csv"
    ml_df.to_csv(output_file, index=False)

    print(f"Successfully created ML dataset with {len(ml_df)} samples.")
    print(f"Features: {list(ml_df.columns)}")
    print(f"Saved to: {output_file}")
else:
    print("No valid sequences found (check if grid bounds match your data).")