"""
Generate CSV file for Heatmap Visualization
This script creates a comprehensive CSV with predicted earthquake stress zones
that can be used for creating heatmaps on the dashboard.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import os

print("="*70)
print("GENERATING HEATMAP CSV DATA")
print("="*70)

# ==========================================
# GRID SYSTEM (same as stress_predictor_model.py)
# ==========================================
GRID_DIM = 8  # 8x8 grid
LAT_MIN, LAT_MAX = 15, 35
LON_MIN, LON_MAX = 65, 100

def get_sector_id(lat, lon):
    """Convert centroid (lat, lon) to sector ID (0-63)"""
    if lat < LAT_MIN or lat > LAT_MAX or lon < LON_MIN or lon > LON_MAX:
        return -1
    y_idx = int((lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * GRID_DIM)
    x_idx = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * GRID_DIM)
    y_idx = min(y_idx, GRID_DIM - 1)
    x_idx = min(x_idx, GRID_DIM - 1)
    return y_idx * GRID_DIM + x_idx

def get_sector_center(sector_id):
    """Convert sector ID back to (lat, lon)"""
    y_idx = sector_id // GRID_DIM
    x_idx = sector_id % GRID_DIM
    lat_step = (LAT_MAX - LAT_MIN) / GRID_DIM
    lon_step = (LON_MAX - LON_MIN) / GRID_DIM
    lat = LAT_MIN + (y_idx + 0.5) * lat_step
    lon = LON_MIN + (x_idx + 0.5) * lon_step
    return lat, lon

def get_sector_bounds(sector_id):
    """Get the boundary coordinates of a sector"""
    y_idx = sector_id // GRID_DIM
    x_idx = sector_id % GRID_DIM
    lat_step = (LAT_MAX - LAT_MIN) / GRID_DIM
    lon_step = (LON_MAX - LON_MIN) / GRID_DIM
    
    lat_min = LAT_MIN + y_idx * lat_step
    lat_max = LAT_MIN + (y_idx + 1) * lat_step
    lon_min = LON_MIN + x_idx * lon_step
    lon_max = LON_MIN + (x_idx + 1) * lon_step
    
    return lat_min, lat_max, lon_min, lon_max

# ==========================================
# LOAD DATA
# ==========================================
# Set working directory to project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

train_df = pd.read_csv("data/cluster_directions_final.csv")
train_df['From_Time'] = pd.to_datetime(train_df['From_Time'], format='mixed')
train_df = train_df[train_df['Year'] <= 2005].sort_values('From_Time')
print(f"✓ Loaded training data: {len(train_df)} events (1980-2005)")

test_df = pd.read_csv("data/test_set.csv")
test_df['From_Time'] = pd.to_datetime(test_df['From_Time'], format='mixed')
test_df = test_df.sort_values('From_Time')
print(f"✓ Loaded test data: {len(test_df)} events (2006-2011)")

# ==========================================
# FEATURE ENGINEERING
# ==========================================
def create_features(df, lookback=3):
    """Extract features from cluster directions"""
    features_list = []
    df_sorted = df.sort_values('From_Time').reset_index(drop=True)
    
    for i in range(lookback, len(df_sorted)):
        current = df_sorted.iloc[i]
        
        target_sector = get_sector_id(current['To_Lat'], current['To_Lon'])
        if target_sector == -1:
            continue
        
        current_sector = get_sector_id(current['From_Lat'], current['From_Lon'])
        if current_sector == -1:
            continue
        
        features = {}
        features['Year'] = current['Year']
        features['Month'] = current['From_Time'].month
        features['From_Time'] = current['From_Time']
        features['Current_Sector'] = current_sector
        features['Current_Bearing'] = current['Bearing_Degrees']
        features['Current_Angle'] = current['Angle_wrt_X_Axis']
        features['From_Lat'] = current['From_Lat']
        features['From_Lon'] = current['From_Lon']
        features['To_Lat'] = current['To_Lat']
        features['To_Lon'] = current['To_Lon']
        
        valid = True
        for lag in range(1, lookback + 1):
            past = df_sorted.iloc[i - lag]
            past_sector = get_sector_id(past['To_Lat'], past['To_Lon'])
            
            if past_sector == -1:
                valid = False
                break
            
            prefix = f"Lag{lag}_"
            features[prefix + 'Sector'] = past_sector
            features[prefix + 'Bearing'] = past['Bearing_Degrees']
            features[prefix + 'Angle'] = past['Angle_wrt_X_Axis']
            
            delta_days = (current['From_Time'] - past['From_Time']).days
            features[prefix + 'DaysAgo'] = max(delta_days, 1)
        
        if valid:
            features['Target_Sector'] = target_sector
            features_list.append(features)
    
    return pd.DataFrame(features_list)

train_features = create_features(train_df, lookback=3)
test_features = create_features(test_df, lookback=3)

# ==========================================
# PREPARE AND TRAIN MODEL
# ==========================================
feature_cols = ['Month', 'Current_Sector', 'Current_Bearing', 'Current_Angle',
                'Lag1_Sector', 'Lag1_Bearing', 'Lag1_Angle', 'Lag1_DaysAgo',
                'Lag2_Sector', 'Lag2_Bearing', 'Lag2_Angle', 'Lag2_DaysAgo',
                'Lag3_Sector', 'Lag3_Bearing', 'Lag3_Angle', 'Lag3_DaysAgo']

X_train = train_features[feature_cols]
y_train = train_features['Target_Sector']

X_test = test_features[feature_cols]
y_test = test_features['Target_Sector']

le = LabelEncoder()
le.fit(pd.concat([y_train, y_test]))
y_train_enc = le.transform(y_train)
y_test_enc = le.transform(y_test)

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    objective='multi:softprob',
    random_state=42,
    verbosity=0
)

print("Training model...")
model.fit(X_train, y_train_enc)
print("✓ Model trained")

# ==========================================
# GENERATE PREDICTIONS WITH PROBABILITIES
# ==========================================
probs = model.predict_proba(X_test)

results = []

for i in range(len(X_test)):
    real_sector = y_test.iloc[i]
    year = test_features.iloc[i]['Year']
    month = test_features.iloc[i]['Month']
    from_time = test_features.iloc[i]['From_Time']
    
    event_probs = probs[i]
    ranked_indices = np.argsort(event_probs)[::-1]
    
    # Get top predicted sector
    pred_idx = ranked_indices[0]
    pred_sector = le.inverse_transform([pred_idx])[0]
    pred_prob = event_probs[pred_idx]
    
    # Get coordinates
    pred_lat, pred_lon = get_sector_center(pred_sector)
    real_lat, real_lon = get_sector_center(real_sector)
    
    # Get sector bounds for heatmap rectangles
    pred_lat_min, pred_lat_max, pred_lon_min, pred_lon_max = get_sector_bounds(pred_sector)
    real_lat_min, real_lat_max, real_lon_min, real_lon_max = get_sector_bounds(real_sector)
    
    # Get top 3 predictions
    top3_sectors = [le.inverse_transform([idx])[0] for idx in ranked_indices[:3]]
    top3_probs = [event_probs[idx] for idx in ranked_indices[:3]]
    
    # Calculate rank of actual
    real_idx = y_test_enc[i]
    rank_position = np.where(ranked_indices == real_idx)[0]
    rank = rank_position[0] + 1 if len(rank_position) > 0 else 99
    
    results.append({
        'time': from_time,
        'year': int(year),
        'month': int(month),
        # Actual values
        'actual_sector': int(real_sector),
        'actual_lat': real_lat,
        'actual_lon': real_lon,
        'actual_lat_min': real_lat_min,
        'actual_lat_max': real_lat_max,
        'actual_lon_min': real_lon_min,
        'actual_lon_max': real_lon_max,
        # Predicted values
        'pred_sector': int(pred_sector),
        'pred_lat': pred_lat,
        'pred_lon': pred_lon,
        'pred_lat_min': pred_lat_min,
        'pred_lat_max': pred_lat_max,
        'pred_lon_min': pred_lon_min,
        'pred_lon_max': pred_lon_max,
        'pred_probability': round(pred_prob, 4),
        # Top 3 predictions
        'top2_sector': int(top3_sectors[1]) if len(top3_sectors) > 1 else None,
        'top2_prob': round(top3_probs[1], 4) if len(top3_probs) > 1 else None,
        'top3_sector': int(top3_sectors[2]) if len(top3_sectors) > 2 else None,
        'top3_prob': round(top3_probs[2], 4) if len(top3_probs) > 2 else None,
        # Accuracy info
        'actual_rank': int(rank),
        'is_correct': rank == 1,
        'in_top3': rank <= 3,
        'in_top5': rank <= 5
    })

# ==========================================
# SAVE CSV FILES
# ==========================================
results_df = pd.DataFrame(results)

# Main predictions file
results_df.to_csv('data/predicted_heatmap_data.csv', index=False)
print(f"✓ Saved: data/predicted_heatmap_data.csv ({len(results_df)} rows)")

# ==========================================
# GENERATE GRID DENSITY DATA (for heatmap intensity)
# ==========================================
grid_data = []

for year in sorted(results_df['year'].unique()):
    year_data = results_df[results_df['year'] == year]
    
    # Count predictions and actuals per sector
    pred_counts = year_data['pred_sector'].value_counts().to_dict()
    actual_counts = year_data['actual_sector'].value_counts().to_dict()
    
    for sector_id in range(GRID_DIM * GRID_DIM):
        lat, lon = get_sector_center(sector_id)
        lat_min, lat_max, lon_min, lon_max = get_sector_bounds(sector_id)
        
        pred_count = pred_counts.get(sector_id, 0)
        actual_count = actual_counts.get(sector_id, 0)
        
        if pred_count > 0 or actual_count > 0:
            grid_data.append({
                'year': int(year),
                'sector_id': sector_id,
                'center_lat': lat,
                'center_lon': lon,
                'lat_min': lat_min,
                'lat_max': lat_max,
                'lon_min': lon_min,
                'lon_max': lon_max,
                'pred_count': pred_count,
                'actual_count': actual_count,
                'difference': pred_count - actual_count
            })

grid_df = pd.DataFrame(grid_data)
grid_df.to_csv('data/heatmap_grid_density.csv', index=False)
print(f"✓ Saved: data/heatmap_grid_density.csv ({len(grid_df)} rows)")

# ==========================================
# SUMMARY STATISTICS
# ==========================================
print("\n" + "="*70)
print("CSV FILES GENERATED:")
print("="*70)
print("\n1. predicted_heatmap_data.csv - Individual predictions with:")
print("   - time, year, month")
print("   - actual_sector, actual_lat, actual_lon (+ bounds)")
print("   - pred_sector, pred_lat, pred_lon (+ bounds)")
print("   - pred_probability, top2/top3 predictions")
print("   - accuracy metrics (is_correct, in_top3, in_top5)")

print("\n2. heatmap_grid_density.csv - Grid cell densities with:")
print("   - year, sector_id")
print("   - center_lat, center_lon (+ bounds)")
print("   - pred_count, actual_count, difference")

print("\n" + "="*70)
print("YEARLY SUMMARY:")
print("="*70)
for year in sorted(results_df['year'].unique()):
    year_data = results_df[results_df['year'] == year]
    accuracy = year_data['is_correct'].mean() * 100
    top3 = year_data['in_top3'].mean() * 100
    print(f"  {int(year)}: {len(year_data)} events | Accuracy: {accuracy:.1f}% | Top-3: {top3:.1f}%")

print("\n✅ CSV generation complete!")
