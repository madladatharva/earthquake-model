# 🌍 Earthquake Cluster Analysis Dashboard

An interactive dashboard for visualizing earthquake clustering patterns, stress zone migrations, and seismic activity predictions in the Himalayan region.

## 📁 Project Structure

```
earthquak-model/
├── dashboard.py              # Main Streamlit dashboard
├── stress_predictor_model.py # XGBoost prediction model
├── data_cleaner.py           # Data preprocessing utilities
├── requirements.txt          # Python dependencies
│
├── data/                     # Data files
│   ├── coordinates.csv       # Earthquake coordinates
│   ├── cluster_summary_1980_2005.csv
│   ├── cluster_directions_final.csv
│   ├── test_set.csv
│   └── eu_in_plates.geojson  # Tectonic plate boundaries
│
├── globe/                    # 3D Globe visualization assets
│   ├── cords.csv             # Actual vs predicted coordinates
│   ├── textbox.py            # PyGame/OpenGL globe (standalone)
│   └── objloader.py          # 3D model loader
│
├── yearly_heatmaps/          # Generated stress heatmaps
├── yearly_heatmaps_geojson/  # GeoJSON heatmaps
├── stress_migration_maps/    # Migration visualizations
├── energy_transfer_plots/    # Energy transfer visualizations
├── yearly_stress_predictions/# Prediction comparison maps
│
├── scripts/                  # Utility scripts
│   ├── train.py              # Model training
│   ├── cluster_directions.py # Cluster analysis
│   └── ...                   # Other analysis scripts
│
└── docs/                     # Documentation
    ├── FEATURE_ENGINEERING_EXPLAINED.txt
    └── MODELING_STRATEGY.txt
```

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the dashboard:**
   ```bash
   streamlit run dashboard.py
   ```

3. **Open in browser:** http://localhost:8501

## 📊 Dashboard Features

| Tab | Description |
|-----|-------------|
| 🗺️ **Map View** | Interactive earthquake map with plate boundaries |
| 📊 **Statistics** | Temporal analysis and magnitude distributions |
| 🧭 **Migration** | Cluster movement patterns and directions |
| 🔥 **Clusters** | DBSCAN clustering analysis |
| 🌐 **3D View** | 3D scatter plot of earthquakes by depth |
| 🌍 **Globe** | Interactive globe with actual vs predicted locations |
| 🎯 **Predictions** | XGBoost stress zone predictions (2006-2011) |

## 🔧 Tech Stack

- **Frontend:** Streamlit, Plotly
- **ML Model:** XGBoost
- **Data:** Pandas, NumPy
- **Visualization:** Plotly, PyGame/OpenGL (globe)

## 📈 Model Performance

- **Training Period:** 1980-2005
- **Test Period:** 2006-2011
- **Grid System:** 8×8 sectors (LAT: 5-40°N, LON: 65-100°E)

## 🎨 Theme

Fire/Lava color scheme with dark mode UI for optimal visualization of seismic data.
