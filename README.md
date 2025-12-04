# Earthquake Stress Migration Analysis

A professional dashboard for visualizing earthquake clustering patterns, stress zone migrations, and seismic activity predictions in the Himalayan region.

## Project Structure

```
earthquake-model/
├── dashboard.py              # Main Streamlit dashboard (professional UI)
├── dashboard_professional.py # Alternative dashboard version
├── stress_predictor_model.py # XGBoost stress zone prediction model
├── data_cleaner.py           # Data preprocessing utilities
├── requirements.txt          # Python dependencies
│
├── data/                     # Data files
│   ├── coordinates.csv       # Earthquake coordinates (1980-2011)
│   ├── cluster_summary_1980_2005.csv
│   ├── cluster_directions_final.csv
│   ├── test_set.csv          # Test data for predictions
│   ├── predicted_heatmap_data.csv
│   ├── heatmap_grid_density.csv
│   └── eu_in_plates.geojson  # Tectonic plate boundaries
│
├── globe/                    # Globe visualization assets
│   ├── cords.csv             # Actual vs predicted coordinates
│   └── objloader.py          # 3D model loader
│
├── yearly_heatmaps/          # Generated stress heatmaps (PNG)
├── yearly_heatmaps_geojson/  # GeoJSON heatmaps
├── stress_migration_maps/    # Migration visualizations
├── energy_transfer_plots/    # Energy transfer visualizations
├── yearly_stress_predictions/# Prediction vs actual comparison maps
├── images/                   # Static images and assets
│
├── scripts/                  # Utility scripts
│   ├── train.py              # Model training
│   ├── cluster_directions.py # Cluster movement analysis
│   ├── generate_heatmap_csv.py
│   ├── distance_calculator.py
│   └── ...                   # Other analysis scripts
│
└── docs/                     # Documentation
    ├── FEATURE_ENGINEERING_EXPLAINED.txt
    └── MODELING_STRATEGY.txt
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the dashboard:**
   ```bash
   streamlit run dashboard.py
   ```

3. **Open in browser:** http://localhost:8501

## Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **Overview** | Seismic activity map with plate boundaries and magnitude distribution |
| **Globe** | Interactive 3D orthographic projection with multiple views and color schemes |
| **Migration** | Cluster movement patterns and stress migration directions |
| **Heatmaps** | Yearly stress density heatmap gallery |
| **Predictions** | XGBoost stress zone predictions with actual vs predicted comparisons |
| **Data** | Raw data exploration with 3D visualization (lat/lon/depth) |

## Tech Stack

- **Frontend:** Streamlit
- **Visualization:** Plotly (interactive maps, 3D scatter, globe projections)
- **ML Model:** XGBoost (stress zone classification)
- **Data Processing:** Pandas, NumPy

## Model Details

- **Training Period:** 1980-2005
- **Test Period:** 2006-2011
- **Grid System:** 8x8 sectors covering LAT: 5-40°N, LON: 65-100°E
- **Features:** Temporal patterns, spatial clustering, magnitude distributions

## Color Scheme

Professional dark theme with seismic-inspired warm color palette:
- Primary: Burnt orange (#c2410c, #ea580c)
- Accent: Teal (#0d9488, #14b8a6)
- Background: Zinc dark (#18181b, #1c1c22)
