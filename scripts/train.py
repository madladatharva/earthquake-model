import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD DATASETS
# ==========================================
# A. Training Data (1980-2005)
try:
    train_df = pd.read_csv("cluster_directions_final.csv")
    train_df['To_Time'] = pd.to_datetime(train_df['To_Time'], format='mixed')
    train_df = train_df.sort_values('To_Time')
except FileNotFoundError:
    print("Error: 'cluster_directions_final.csv' (Training Data) not found.")
    exit()

# B. Test Data (2006-2011) - Loading from the file you provided
try:
    test_df = pd.read_csv("test_set.csv")
    test_df['To_Time'] = pd.to_datetime(test_df['To_Time'], format='mixed')
    test_df = test_df.sort_values('To_Time')
except FileNotFoundError:
    print("Error: 'test_set.csv' not found. Please save your provided data to this file.")
    exit()

# Combine them temporarily just to build the full sequence for feature engineering
full_history = pd.concat([train_df, test_df]).sort_values('To_Time').reset_index(drop=True)

# ==========================================
# 2. TRAIN THE MODEL (1980-2005 ONLY)
# ==========================================
LOOKBACK_STEPS = 3

def create_features(df_sequence):
    data = []
    # Need enough history for lookback
    start_idx = LOOKBACK_STEPS
    
    sequence = df_sequence.to_dict('records')
    
    for i in range(start_idx, len(sequence)):
        current_event = sequence[i]
        
        features = {}
        features['Month'] = current_event['To_Time'].month
        features['Cur_Lat'] = current_event['From_Lat']
        features['Cur_Lon'] = current_event['From_Lon']
        features['Cur_Angle'] = current_event['Angle_wrt_X_Axis']
        
        for step in range(1, LOOKBACK_STEPS + 1):
            past_event = sequence[i - step]
            prefix = f"Lag{step}_"
            features[prefix + 'Lat'] = past_event['To_Lat']
            features[prefix + 'Lon'] = past_event['To_Lon']
            features[prefix + 'Angle'] = past_event['Angle_wrt_X_Axis']
            features[prefix + 'DaysAgo'] = (current_event['To_Time'] - past_event['To_Time']).days
            
        features['Target_Lat'] = current_event['To_Lat']
        features['Target_Lon'] = current_event['To_Lon']
        features['Original_Index'] = i # Keep track to split later
        
        data.append(features)
    return pd.DataFrame(data)

# Generate features for the WHOLE timeline so continuity is perfect
print("Building historical features...")
all_features_df = create_features(full_history)

# SPLIT back into Train (up to 2005) and Test (2006+)
# We find the cut-off index based on the size of the training set
train_size = len(train_df) - LOOKBACK_STEPS 
X_train_df = all_features_df.iloc[:train_size]
X_test_df = all_features_df.iloc[train_size:]

X_train = X_train_df.drop(columns=['Target_Lat', 'Target_Lon', 'Original_Index'])
y_train = X_train_df[['Target_Lat', 'Target_Lon']]

print(f"Training on {len(X_train)} events (1980-2005)...")
model = MultiOutputRegressor(XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5))
model.fit(X_train, y_train)

# ==========================================
# 3. ROLLING PREDICTION ON TEST SET
# ==========================================
print(f"Testing on {len(X_test_df)} events (2006-2011)...")

predictions = []
actuals = []

# Iterate through the test set row by row
for idx, row in X_test_df.iterrows():
    # Prepare input features for this specific event
    # We use the ACTUAL history (Ground Truth) to predict the next step
    # This simulates "Real-time forecasting" where you know what just happened.
    
    input_features = row.drop(['Target_Lat', 'Target_Lon', 'Original_Index']).to_frame().T
    
    # Predict
    pred = model.predict(input_features)[0]
    
    predictions.append(pred)
    actuals.append([row['Target_Lat'], row['Target_Lon']])

predictions = np.array(predictions)
actuals = np.array(actuals)

# ==========================================
# 4. VISUALIZE RESULTS
# ==========================================
plt.figure(figsize=(12, 8))
plt.title("Continuity Test: Predicting 2006-2011\n(Model Trained ONLY on 1980-2005)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)

# Plot Actual Path
plt.plot(actuals[:, 1], actuals[:, 0], 'bo-', label='Actual Path (Ground Truth)', alpha=0.6)

# Plot Predicted Path
plt.plot(predictions[:, 1], predictions[:, 0], 'rx--', label='Model Prediction', markersize=8)

# Connect points to show error magnitude
for i in range(len(predictions)):
    plt.plot([actuals[i, 1], predictions[i, 1]], 
             [actuals[i, 0], predictions[i, 0]], 
             color='gray', linestyle=':', alpha=0.5)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()

# Calculate Error
dist_sq = np.sum((actuals - predictions)**2, axis=1)
rmse_deg = np.sqrt(np.mean(dist_sq))
print(f"\nAverage Prediction Error: ~{rmse_deg * 111:.2f} km") # Approx conversion

# Show Table
results = pd.DataFrame({
    'Actual_Lat': actuals[:, 0],
    'Pred_Lat': predictions[:, 0],
    'Actual_Lon': actuals[:, 1],
    'Pred_Lon': predictions[:, 1]
})
print("\n--- Prediction vs Actual (First 5 Rows) ---")
print(results.head())

plt.show()