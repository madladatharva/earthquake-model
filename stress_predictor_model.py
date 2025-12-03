import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from shapely.ops import transform
import os

print("="*70)
print("STRESS ZONE PREDICTION MODEL")
print("="*70)

# ==========================================
# 1. GRID SYSTEM (same as uptop.py)
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

# ==========================================
# 2. LOAD GEOJSON (Tectonic Plates)
# ==========================================
geo_df = None
try:
    geo_df = gpd.read_file("eu_in_plates.geojson")
    if hasattr(geo_df, 'crs') and geo_df.crs is not None:
        geo_df = geo_df.to_crs(epsg=4326)
    
    def drop_z(geom):
        if geom.has_z:
            return transform(lambda x, y, z: (x, y), geom)
        return geom
    geo_df['geometry'] = geo_df['geometry'].apply(drop_z)
    print("✓ Loaded tectonic plate boundaries")
except Exception as e:
    print(f"⚠ Warning: Could not load GeoJSON ({e})")

# ==========================================
# 3. LOAD TRAINING DATA (1980-2019)
# ==========================================
try:
    train_df = pd.read_csv("cluster_directions_final.csv")
    train_df['From_Time'] = pd.to_datetime(train_df['From_Time'], format='mixed')
    train_df = train_df[train_df['Year'] <= 2019].sort_values('From_Time')
    print(f"✓ Loaded training data: {len(train_df)} events (1980-2019)")
except FileNotFoundError:
    print("Error: 'cluster_directions_final.csv' not found.")
    exit()

# ==========================================
# 4. LOAD TEST DATA (2020-2025)
# ==========================================
try:
    test_df = pd.read_csv("test_set.csv")
    test_df['From_Time'] = pd.to_datetime(test_df['From_Time'], format='mixed')
    test_df = test_df[(test_df['Year'] >= 2020) & (test_df['Year'] <= 2025)].sort_values('From_Time')
    print(f"✓ Loaded test data: {len(test_df)} events (2020-2025)")
except FileNotFoundError:
    print("Error: 'test_set.csv' not found.")
    exit()

# ==========================================
# 5. FEATURE ENGINEERING
# ==========================================
print("\n" + "="*70)
print("FEATURE ENGINEERING")
print("="*70)

def create_features(df, lookback=3):
    """Extract features from cluster directions"""
    features_list = []
    df_sorted = df.sort_values('From_Time').reset_index(drop=True)
    
    for i in range(lookback, len(df_sorted)):
        current = df_sorted.iloc[i]
        
        # TARGET: Next zone
        target_sector = get_sector_id(current['To_Lat'], current['To_Lon'])
        if target_sector == -1:
            continue
        
        # SOURCE: Current zone
        current_sector = get_sector_id(current['From_Lat'], current['From_Lon'])
        if current_sector == -1:
            continue
        
        features = {}
        features['Year'] = current['Year']
        features['Month'] = current['From_Time'].month
        features['Current_Sector'] = current_sector
        features['Current_Bearing'] = current['Bearing_Degrees']
        features['Current_Angle'] = current['Angle_wrt_X_Axis']
        
        # Lag features
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
print(f"✓ Generated {len(train_features)} training feature vectors")

test_features = create_features(test_df, lookback=3)
print(f"✓ Generated {len(test_features)} test feature vectors")

# ==========================================
# 6. PREPARE DATA
# ==========================================
X_train = train_features.drop(columns=['Target_Sector', 'Year'])
y_train = train_features['Target_Sector']

X_test = test_features.drop(columns=['Target_Sector', 'Year'])
y_test = test_features['Target_Sector']

print(f"\nTraining: {len(X_train)} samples, {len(X_train.columns)} features")
print(f"Test: {len(X_test)} samples")

# ==========================================
# 7. ENCODE LABELS
# ==========================================
le = LabelEncoder()
le.fit(pd.concat([y_train, y_test]))
y_train_enc = le.transform(y_train)
y_test_enc = le.transform(y_test)

print(f"✓ Encoded {len(le.classes_)} unique sectors")

# ==========================================
# 8. TRAIN MODEL
# ==========================================
print("\n" + "="*70)
print("MODEL TRAINING")
print("="*70)

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    objective='multi:softprob',
    random_state=42,
    verbosity=0
)

print("Training XGBoost classifier...")
model.fit(X_train, y_train_enc)
print("✓ Model trained successfully")

# ==========================================
# 9. TESTING & EVALUATION
# ==========================================
print("\n" + "="*70)
print("MODEL EVALUATION (TEST SET: 2006-2011)")
print("="*70)

probs = model.predict_proba(X_test)

hits_top1 = 0
hits_top3 = 0
hits_top5 = 0
total = 0

results_log = []

for i in range(len(X_test)):
    real_idx = y_test_enc[i]
    real_sector = y_test.iloc[i]
    year = test_features.iloc[i]['Year']
    
    event_probs = probs[i]
    ranked_indices = np.argsort(event_probs)[::-1]
    
    rank_position = np.where(ranked_indices == real_idx)[0]
    rank = rank_position[0] + 1 if len(rank_position) > 0 else 99
    
    total += 1
    if rank == 1: hits_top1 += 1
    if rank <= 3: hits_top3 += 1
    if rank <= 5: hits_top5 += 1
    
    pred_idx = ranked_indices[0]
    pred_sector = le.inverse_transform([pred_idx])[0]
    
    pred_lat, pred_lon = get_sector_center(pred_sector)
    real_lat, real_lon = get_sector_center(real_sector)
    
    symbol = "🎯" if rank == 1 else "✅" if rank <= 3 else "⚠️" if rank <= 5 else "❌"
    
    print(f"Year {int(year)} | Real: S{real_sector:2d} ({real_lat:.1f}°, {real_lon:.1f}°) | "
          f"Pred: S{pred_sector:2d} ({pred_lat:.1f}°, {pred_lon:.1f}°) | Rank: {rank:2d} {symbol}")
    
    results_log.append({
        'Year': year,
        'Real_Sector': real_sector,
        'Pred_Sector': pred_sector,
        'Rank': rank,
        'Real_Lat': real_lat,
        'Real_Lon': real_lon,
        'Pred_Lat': pred_lat,
        'Pred_Lon': pred_lon
    })

# ==========================================
# 10. SCORECARD
# ==========================================
print("\n" + "="*70)
print("FINAL SCORECARD")
print("="*70)
print(f"Total Events Tested: {total}")
print(f"Exact Matches (Rank 1):  {hits_top1}/{total} ({hits_top1/total*100:.1f}%)")
print(f"Top 3 Matches:           {hits_top3}/{total} ({hits_top3/total*100:.1f}%)")
print(f"Top 5 Matches:           {hits_top5}/{total} ({hits_top5/total*100:.1f}%)")
print("="*70)

# ==========================================
# 11. VISUALIZATION - Heatmaps
# ==========================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

results_df = pd.DataFrame(results_log)

actual_grid = np.zeros((GRID_DIM, GRID_DIM))
predicted_grid = np.zeros((GRID_DIM, GRID_DIM))

for _, row in results_df.iterrows():
    real_sec = int(row['Real_Sector'])
    pred_sec = int(row['Pred_Sector'])
    
    real_y, real_x = real_sec // GRID_DIM, real_sec % GRID_DIM
    pred_y, pred_x = pred_sec // GRID_DIM, pred_sec % GRID_DIM
    
    if 0 <= real_y < GRID_DIM and 0 <= real_x < GRID_DIM:
        actual_grid[real_y, real_x] += 1
    if 0 <= pred_y < GRID_DIM and 0 <= pred_x < GRID_DIM:
        predicted_grid[pred_y, pred_x] += 1

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

im1 = axes[0].imshow(actual_grid, extent=extent, origin='lower', cmap='YlOrRd', aspect='auto', alpha=0.7)
if geo_df is not None:
    geo_df.plot(ax=axes[0], color='black', linewidth=1.5, alpha=0.5, zorder=2)
axes[0].set_title("Actual Next Zones (2006-2011)\nwith Tectonic Plate Boundaries", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
axes[0].grid(True, linestyle='--', alpha=0.3)
plt.colorbar(im1, ax=axes[0], label='Count')

im2 = axes[1].imshow(predicted_grid, extent=extent, origin='lower', cmap='Blues', aspect='auto', alpha=0.7)
if geo_df is not None:
    geo_df.plot(ax=axes[1], color='black', linewidth=1.5, alpha=0.5, zorder=2)
axes[1].set_title("Predicted Next Zones (2006-2011)\nwith Tectonic Plate Boundaries", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")
axes[1].grid(True, linestyle='--', alpha=0.3)
plt.colorbar(im2, ax=axes[1], label='Count')

plt.tight_layout()
plt.savefig('stress_zone_heatmap_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: stress_zone_heatmap_comparison.png")
plt.close()

# Error map
error_grid = predicted_grid - actual_grid

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(error_grid, extent=extent, origin='lower', cmap='RdBu_r', aspect='auto')
if geo_df is not None:
    geo_df.plot(ax=ax, color='black', linewidth=1.5, alpha=0.6, zorder=2)

ax.set_title("Prediction Error Map\n(Red=Over-predicted, Blue=Under-predicted)\nwith Tectonic Plate Boundaries", 
             fontsize=12, fontweight='bold')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, linestyle='--', alpha=0.3)
plt.colorbar(im, ax=ax, label='Prediction Bias')

plt.tight_layout()
plt.savefig('stress_zone_error_map.png', dpi=150, bbox_inches='tight')
print("✓ Saved: stress_zone_error_map.png")
plt.close()

# Scatter plot
fig, ax = plt.subplots(figsize=(12, 10))

for _, row in results_df.iterrows():
    ax.plot([row['Real_Lon'], row['Pred_Lon']], 
            [row['Real_Lat'], row['Pred_Lat']], 
            'gray', alpha=0.2, linewidth=0.5)

ax.scatter(results_df['Real_Lon'], results_df['Real_Lat'], 
          c='blue', s=60, alpha=0.7, label='Actual', edgecolors='black', zorder=3)

ax.scatter(results_df['Pred_Lon'], results_df['Pred_Lat'], 
          c='red', s=60, alpha=0.7, label='Predicted', marker='x', edgecolors='black', zorder=3, linewidths=2)

if geo_df is not None:
    geo_df.plot(ax=ax, color='black', linewidth=1.5, alpha=0.6, zorder=2)

ax.set_title("Actual vs Predicted Stress Zones\nwith Tectonic Plate Boundaries", fontsize=12, fontweight='bold')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('stress_zone_scatter.png', dpi=150, bbox_inches='tight')
print("✓ Saved: stress_zone_scatter.png")
plt.close()

# Feature importance
feature_importance = model.feature_importances_
feature_names = X_train.columns
sorted_idx = np.argsort(feature_importance)[-15:]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(sorted_idx)), feature_importance[sorted_idx], color='steelblue')
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels(feature_names[sorted_idx])
ax.set_xlabel("Importance")
ax.set_title("Top 15 Most Important Features for Zone Prediction", fontsize=12, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
print("✓ Saved: feature_importance.png")
plt.close()

# ==========================================
# 12. HEATMAP VISUALIZATIONS BY YEAR (like uptop.py)
# ==========================================
print("\n" + "="*70)
print("GENERATING YEARLY HEATMAPS")
print("="*70)

output_folder = "yearly_stress_predictions"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

unique_years = results_df['Year'].unique()

for year in sorted(unique_years):
    year_data = results_df[results_df['Year'] == year]
    
    if len(year_data) < 2:
        continue
    
    # Extract paths
    actual_lats = []
    actual_lons = []
    predicted_lats = []
    predicted_lons = []
    
    for _, row in year_data.iterrows():
        # Interpolate paths for heatmap density
        for i in np.linspace(0, 1, 50):
            actual_lats.append(row['Real_Lat'] * (1 - i) + row['Real_Lat'] * i)
            actual_lons.append(row['Real_Lon'] * (1 - i) + row['Real_Lon'] * i)
            predicted_lats.append(row['Pred_Lat'] * (1 - i) + row['Pred_Lat'] * i)
            predicted_lons.append(row['Pred_Lon'] * (1 - i) + row['Pred_Lon'] * i)
    
    # Create side-by-side heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]
    
    # ACTUAL
    try:
        sns.kdeplot(x=actual_lons, y=actual_lats, cmap="YlOrRd", fill=True, 
                    thresh=0.05, alpha=0.7, ax=axes[0], zorder=2)
    except:
        pass
    
    for _, row in year_data.iterrows():
        axes[0].plot([row['Real_Lon'], row['Real_Lon']], 
                     [row['Real_Lat'], row['Real_Lat']], 
                     'o', color='cyan', markersize=8, zorder=4)
    
    if geo_df is not None:
        geo_df.plot(ax=axes[0], color='black', linewidth=1.5, alpha=0.6, zorder=3)
    
    axes[0].set_facecolor('#1a1a1a')
    axes[0].set_title(f"Actual Stress Zones - {int(year)}", fontsize=12, fontweight='bold', color='white')
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].grid(True, color='gray', linestyle='--', alpha=0.3)
    axes[0].set_xlim(LON_MIN, LON_MAX)
    axes[0].set_ylim(LAT_MIN, LAT_MAX)
    
    # PREDICTED
    try:
        sns.kdeplot(x=predicted_lons, y=predicted_lats, cmap="Blues", fill=True, 
                    thresh=0.05, alpha=0.7, ax=axes[1], zorder=2)
    except:
        pass
    
    for _, row in year_data.iterrows():
        axes[1].plot([row['Pred_Lon'], row['Pred_Lon']], 
                     [row['Pred_Lat'], row['Pred_Lat']], 
                     'x', color='red', markersize=10, zorder=4, markeredgewidth=2)
    
    if geo_df is not None:
        geo_df.plot(ax=axes[1], color='black', linewidth=1.5, alpha=0.6, zorder=3)
    
    axes[1].set_facecolor('#1a1a1a')
    axes[1].set_title(f"Predicted Stress Zones - {int(year)}", fontsize=12, fontweight='bold', color='white')
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].grid(True, color='gray', linestyle='--', alpha=0.3)
    axes[1].set_xlim(LON_MIN, LON_MAX)
    axes[1].set_ylim(LAT_MIN, LAT_MAX)
    
    plt.tight_layout()
    filename = f"{output_folder}/stress_comparison_{int(year)}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    
    print(f"  ✓ {int(year)}: {len(year_data)} events")

print(f"✓ Generated {len(sorted(unique_years))} yearly heatmaps in '{output_folder}/'")

print("\n" + "="*70)
print("✅ ALL VISUALIZATIONS COMPLETE")
print("="*70)
print("\nGenerated files:")
print("  1. stress_zone_heatmap_comparison.png")
print("  2. stress_zone_error_map.png")
print("  3. stress_zone_scatter.png")
print("  4. feature_importance.png")
print(f"  5. {len(sorted(unique_years))} yearly heatmaps in '{output_folder}/'")
print("\n" + "="*70)
