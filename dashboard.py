"""
Earthquake Cluster Analysis Dashboard
A comprehensive dashboard for visualizing earthquake clustering patterns
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Earthquake Cluster Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Dark Mode CSS with smooth animations
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    /* Root Variables for Dark Theme - Fire/Lava Palette */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: #1a1a2e;
        --bg-hover: #252542;
        --accent-primary: #ef4444;
        --accent-secondary: #f97316;
        --accent-glow: rgba(239, 68, 68, 0.4);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --border-color: rgba(255, 255, 255, 0.1);
        --success: #22c55e;
        --warning: #f59e0b;
        --danger: #dc2626;
        --gradient-1: linear-gradient(135deg, #ef4444 0%, #f97316 50%, #f59e0b 100%);
        --gradient-2: linear-gradient(135deg, #dc2626 0%, #ea580c 100%);
        --gradient-3: linear-gradient(135deg, #f97316 0%, #eab308 100%);
    }
    
    /* Global Styles */
    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        padding: 1rem 2rem 2rem 2rem;
        max-width: 100%;
    }
    
    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Custom Header */
    .dashboard-header {
        background: var(--gradient-1);
        padding: 2.5rem 3rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.3);
        animation: fadeInDown 0.6s ease-out;
    }
    
    .dashboard-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    .dashboard-header h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .dashboard-header p {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.9);
        position: relative;
        z-index: 1;
        font-weight: 300;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.05); opacity: 0.8; }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px var(--accent-glow); }
        50% { box-shadow: 0 0 40px var(--accent-glow), 0 0 60px var(--accent-glow); }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Metric Cards */
    .metric-card {
        background: var(--bg-card);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid var(--border-color);
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.5s ease-out;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-1);
        transform: scaleX(0);
        transition: transform 0.4s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: var(--accent-primary);
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.2);
    }
    
    .metric-card:hover::before {
        transform: scaleX(1);
    }
    
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Stat Cards */
    .stat-card {
        background: linear-gradient(145deg, var(--bg-card), var(--bg-secondary));
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.3);
        border-color: var(--accent-primary);
    }
    
    .stat-number {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .stat-label {
        color: var(--text-secondary);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.25rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: var(--text-primary);
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Slider Styling */
    .stSlider > div > div {
        background: var(--bg-card) !important;
    }
    
    .stSlider > div > div > div > div {
        background: var(--gradient-1) !important;
    }
    
    /* Tab Styling */
    .stTabs {
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--bg-secondary);
        padding: 8px;
        border-radius: 16px;
        border: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        padding: 12px 24px;
        color: var(--text-secondary);
        font-weight: 500;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--bg-hover);
        color: var(--text-primary);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--gradient-1) !important;
        color: white !important;
        box-shadow: 0 4px 15px var(--accent-glow);
    }
    
    /* Chart Container */
    .chart-container {
        background: var(--bg-card);
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid var(--border-color);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        animation: fadeInUp 0.5s ease-out;
    }
    
    .chart-container:hover {
        border-color: var(--accent-primary);
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.15);
    }
    
    /* Data Tables */
    .stDataFrame {
        background: var(--bg-card);
        border-radius: 16px;
        border: 1px solid var(--border-color);
        overflow: hidden;
    }
    
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background: var(--bg-card);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    /* Info/Warning boxes */
    .stAlert {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        color: var(--text-primary);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: var(--accent-primary) !important;
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--accent-primary);
        display: inline-block;
    }
    
    /* Glowing border effect */
    .glow-box {
        position: relative;
        background: var(--bg-card);
        border-radius: 20px;
        padding: 2rem;
        overflow: hidden;
    }
    
    .glow-box::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: var(--gradient-1);
        border-radius: 22px;
        z-index: -1;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .glow-box:hover::before {
        opacity: 1;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: var(--text-muted);
        border-top: 1px solid var(--border-color);
        margin-top: 2rem;
    }
    
    .footer a {
        color: var(--accent-primary);
        text-decoration: none;
        transition: color 0.3s ease;
    }
    
    .footer a:hover {
        color: var(--accent-secondary);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--accent-primary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-secondary);
    }
    
    /* Loading Animation */
    .loading-pulse {
        animation: glow 2s ease-in-out infinite;
    }
    
    /* Legend Card */
    .legend-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 15px;
        background: var(--bg-secondary);
        border-radius: 10px;
        margin-bottom: 8px;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .legend-item:hover {
        background: var(--bg-hover);
        transform: translateX(5px);
    }
    
    .legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }
    
    .legend-text {
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Data Loading Functions
@st.cache_data
def load_earthquake_data():
    """Load and preprocess earthquake coordinates data"""
    try:
        df = pd.read_csv('coordinates.csv')
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time', 'lat', 'lon'])
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        df['mag'] = df['mag'].fillna(0)
        return df
    except Exception as e:
        st.error(f"Error loading earthquake data: {e}")
        return None

@st.cache_data
def load_cluster_summary():
    """Load cluster summary data"""
    try:
        df = pd.read_csv('cluster_summary_1980_2005.csv')
        return df
    except Exception as e:
        st.error(f"Error loading cluster summary: {e}")
        return None

@st.cache_data
def load_cluster_directions():
    """Load cluster directions data"""
    try:
        df = pd.read_csv('cluster_directions_final.csv')
        return df
    except Exception as e:
        st.error(f"Error loading cluster directions: {e}")
        return None

@st.cache_data
def load_plate_boundaries():
    """Load plate boundary data from GeoJSON"""
    try:
        with open('eu_in_plates.geojson', 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading plate boundaries: {e}")
        return None

@st.cache_data
def load_test_data():
    """Load test set data for predictions"""
    try:
        df = pd.read_csv('test_set.csv')
        df['From_Time'] = pd.to_datetime(df['From_Time'], format='mixed')
        return df
    except Exception as e:
        return None

@st.cache_data
def run_stress_prediction():
    """Run stress zone prediction model and return results"""
    try:
        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder
        
        # Grid system constants
        GRID_DIM = 8
        LAT_MIN, LAT_MAX = 15, 35
        LON_MIN, LON_MAX = 65, 100
        
        def get_sector_id(lat, lon):
            if lat < LAT_MIN or lat > LAT_MAX or lon < LON_MIN or lon > LON_MAX:
                return -1
            y_idx = int((lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * GRID_DIM)
            x_idx = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * GRID_DIM)
            y_idx = min(y_idx, GRID_DIM - 1)
            x_idx = min(x_idx, GRID_DIM - 1)
            return y_idx * GRID_DIM + x_idx
        
        def get_sector_center(sector_id):
            y_idx = sector_id // GRID_DIM
            x_idx = sector_id % GRID_DIM
            lat_step = (LAT_MAX - LAT_MIN) / GRID_DIM
            lon_step = (LON_MAX - LON_MIN) / GRID_DIM
            lat = LAT_MIN + (y_idx + 0.5) * lat_step
            lon = LON_MIN + (x_idx + 0.5) * lon_step
            return lat, lon
        
        def create_features(df, lookback=3):
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
                features['Current_Sector'] = current_sector
                features['Current_Bearing'] = current['Bearing_Degrees']
                features['Current_Angle'] = current['Angle_wrt_X_Axis']
                
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
        
        # Load data
        train_df = pd.read_csv('cluster_directions_final.csv')
        train_df['From_Time'] = pd.to_datetime(train_df['From_Time'], format='mixed')
        train_df = train_df[train_df['Year'] <= 2005].sort_values('From_Time')
        
        test_df = pd.read_csv('test_set.csv')
        test_df['From_Time'] = pd.to_datetime(test_df['From_Time'], format='mixed')
        test_df = test_df.sort_values('From_Time')
        
        # Create features
        train_features = create_features(train_df, lookback=3)
        test_features = create_features(test_df, lookback=3)
        
        X_train = train_features.drop(columns=['Target_Sector', 'Year'])
        y_train = train_features['Target_Sector']
        X_test = test_features.drop(columns=['Target_Sector', 'Year'])
        y_test = test_features['Target_Sector']
        
        # Encode labels
        le = LabelEncoder()
        le.fit(pd.concat([y_train, y_test]))
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)
        
        # Train model
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            objective='multi:softprob',
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train_enc)
        
        # Predict
        probs = model.predict_proba(X_test)
        
        results_log = []
        hits_top1, hits_top3, hits_top5, total = 0, 0, 0, 0
        
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
            
            results_log.append({
                'Year': year,
                'Real_Sector': real_sector,
                'Pred_Sector': pred_sector,
                'Rank': rank,
                'Real_Lat': real_lat,
                'Real_Lon': real_lon,
                'Pred_Lat': pred_lat,
                'Pred_Lon': pred_lon,
                'Confidence': event_probs[pred_idx] * 100
            })
        
        results_df = pd.DataFrame(results_log)
        
        # Feature importance
        feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        
        metrics = {
            'total': total,
            'top1': hits_top1,
            'top3': hits_top3,
            'top5': hits_top5,
            'top1_pct': hits_top1/total*100 if total > 0 else 0,
            'top3_pct': hits_top3/total*100 if total > 0 else 0,
            'top5_pct': hits_top5/total*100 if total > 0 else 0
        }
        
        return results_df, metrics, feature_importance, GRID_DIM, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
        
    except Exception as e:
        st.error(f"Error running prediction model: {e}")
        return None, None, None, None, None, None, None, None

# Visualization Functions
# Dark theme color palette - Fire/Lava
DARK_THEME = {
    'bg_primary': '#0a0a0f',
    'bg_secondary': '#12121a',
    'bg_card': '#1a1a2e',
    'text_primary': '#f8fafc',
    'text_secondary': '#94a3b8',
    'accent_primary': '#ef4444',
    'accent_secondary': '#f97316',
    'grid_color': 'rgba(255, 255, 255, 0.05)',
    'border_color': 'rgba(255, 255, 255, 0.1)'
}

# Plotly chart configuration with all toolbar options
PLOTLY_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'scrollZoom': True,
    'modeBarButtonsToAdd': [
        'drawline',
        'drawopenpath',
        'drawclosedpath',
        'drawcircle',
        'drawrect',
        'eraseshape'
    ],
    'modeBarButtonsToRemove': [],
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'earthquake_chart',
        'height': 800,
        'width': 1200,
        'scale': 2
    }
}

def get_dark_layout():
    """Return common dark theme layout settings"""
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26, 26, 46, 0.8)',
        font=dict(family='Inter, sans-serif', color=DARK_THEME['text_primary']),
        title_font=dict(family='Space Grotesk, sans-serif', size=20, color=DARK_THEME['text_primary']),
        legend=dict(
            bgcolor='rgba(26, 26, 46, 0.9)',
            bordercolor=DARK_THEME['border_color'],
            borderwidth=1,
            font=dict(color=DARK_THEME['text_secondary'])
        ),
        hoverlabel=dict(
            bgcolor=DARK_THEME['bg_card'],
            font_size=13,
            font_family='Inter, sans-serif',
            bordercolor=DARK_THEME['accent_primary']
        )
    )

def create_main_map(df, cluster_df, plate_data, year_range, mag_range):
    """Create the main interactive map with earthquakes and clusters"""
    
    # Filter data
    filtered_df = df[
        (df['year'] >= year_range[0]) & 
        (df['year'] <= year_range[1]) &
        (df['mag'] >= mag_range[0]) &
        (df['mag'] <= mag_range[1])
    ]
    
    fig = go.Figure()
    
    # Add plate boundaries with glow effect
    if plate_data:
        for feature in plate_data['features']:
            coords = feature['geometry']['coordinates']
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            # Glow effect (wider, transparent line behind)
            fig.add_trace(go.Scattergeo(
                lon=lons,
                lat=lats,
                mode='lines',
                line=dict(width=8, color='rgba(239, 68, 68, 0.3)'),
                hoverinfo='skip',
                showlegend=False
            ))
            # Main line
            fig.add_trace(go.Scattergeo(
                lon=lons,
                lat=lats,
                mode='lines',
                line=dict(width=3, color='#ef4444'),
                name=f"Plate: {feature['properties']['Name']}",
                hoverinfo='name',
                showlegend=True
            ))
    
    # Add earthquake points with better styling
    fig.add_trace(go.Scattergeo(
        lon=filtered_df['lon'],
        lat=filtered_df['lat'],
        mode='markers',
        marker=dict(
            size=filtered_df['mag'] * 2.5 + 4,
            color=filtered_df['mag'],
            colorscale='Turbo',
            colorbar=dict(
                title=dict(text="Magnitude", font=dict(color='#f8fafc')),
                tickfont=dict(color='#94a3b8'),
                bgcolor='rgba(26, 26, 46, 0.9)',
                bordercolor='rgba(255,255,255,0.1)',
                borderwidth=1
            ),
            opacity=0.75,
            line=dict(width=1, color='rgba(255,255,255,0.3)')
        ),
        text=filtered_df.apply(lambda x: f"<b>Magnitude:</b> {x['mag']:.1f}<br><b>Date:</b> {x['time'].strftime('%Y-%m-%d')}<br><b>Depth:</b> {x['depth']:.1f} km<br><b>Location:</b> ({x['lat']:.2f}°, {x['lon']:.2f}°)", axis=1),
        hoverinfo='text',
        name='Earthquakes',
        showlegend=True
    ))
    
    # Add cluster centroids if available
    if cluster_df is not None:
        cluster_filtered = cluster_df[
            (cluster_df['year'] >= year_range[0]) & 
            (cluster_df['year'] <= year_range[1])
        ]
        
        if not cluster_filtered.empty:
            # Glow effect for clusters
            fig.add_trace(go.Scattergeo(
                lon=cluster_filtered['centroid_lon'],
                lat=cluster_filtered['centroid_lat'],
                mode='markers',
                marker=dict(
                    size=25,
                    color='rgba(99, 102, 241, 0.3)',
                    symbol='circle'
                ),
                hoverinfo='skip',
                showlegend=False
            ))
            # Main cluster markers
            fig.add_trace(go.Scattergeo(
                lon=cluster_filtered['centroid_lon'],
                lat=cluster_filtered['centroid_lat'],
                mode='markers',
                marker=dict(
                    size=14,
                    color='#ef4444',
                    symbol='star',
                    line=dict(width=2, color='white')
                ),
                text=cluster_filtered.apply(lambda x: f"<b>Year:</b> {x['year']}<br><b>Cluster:</b> {x['cluster_rank']}<br><b>Events:</b> {x['event_count']}<br><b>Avg Mag:</b> {x['avg_mag']:.2f}", axis=1),
                hoverinfo='text',
                name='Cluster Centroids',
                showlegend=True
            ))
    
    # Dark theme geo styling
    fig.update_geos(
        projection_type="mercator",
        showland=True,
        landcolor='#1e293b',
        countrycolor='#334155',
        coastlinecolor='#475569',
        showocean=True,
        oceancolor='#0f172a',
        showlakes=True,
        lakecolor='#1e3a5f',
        center=dict(lat=28, lon=85),
        projection_scale=8,
        showcountries=True,
        bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_layout(
        title=dict(
            text=f'🌍 Earthquake Distribution ({year_range[0]}-{year_range[1]})',
            font=dict(size=22, color='#f8fafc', family='Space Grotesk, sans-serif'),
            x=0.5
        ),
        height=650,
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(26, 26, 46, 0.95)',
            bordercolor='rgba(255,255,255,0.1)',
            borderwidth=1,
            font=dict(color='#f8fafc')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        geo=dict(bgcolor='rgba(0,0,0,0)')
    )
    
    return fig

def create_cluster_migration_map(directions_df, year_range):
    """Create animated cluster migration visualization"""
    
    filtered = directions_df[
        (directions_df['Year'] >= year_range[0]) & 
        (directions_df['Year'] <= year_range[1])
    ]
    
    fig = go.Figure()
    
    # Fire/Lava color palette for dark theme (with rgba versions for glow)
    colors = [
        ('#ef4444', 'rgba(239, 68, 68, 0.3)'),
        ('#f97316', 'rgba(249, 115, 22, 0.3)'),
        ('#f59e0b', 'rgba(245, 158, 11, 0.3)'),
        ('#eab308', 'rgba(234, 179, 8, 0.3)'),
        ('#dc2626', 'rgba(220, 38, 38, 0.3)'),
        ('#ea580c', 'rgba(234, 88, 12, 0.3)'),
        ('#d97706', 'rgba(217, 119, 6, 0.3)'),
        ('#ca8a04', 'rgba(202, 138, 4, 0.3)'),
        ('#b91c1c', 'rgba(185, 28, 28, 0.3)'),
        ('#c2410c', 'rgba(194, 65, 12, 0.3)')
    ]
    
    years = sorted(filtered['Year'].unique())
    
    for i, year in enumerate(years):
        year_data = filtered[filtered['Year'] == year]
        color, glow_color = colors[i % len(colors)]
        
        for idx, (_, row) in enumerate(year_data.iterrows()):
            # Glow effect line
            fig.add_trace(go.Scattergeo(
                lon=[row['From_Lon'], row['To_Lon']],
                lat=[row['From_Lat'], row['To_Lat']],
                mode='lines',
                line=dict(width=8, color=glow_color),
                hoverinfo='skip',
                showlegend=False
            ))
            # Main migration line
            fig.add_trace(go.Scattergeo(
                lon=[row['From_Lon'], row['To_Lon']],
                lat=[row['From_Lat'], row['To_Lat']],
                mode='lines+markers',
                line=dict(width=3, color=color),
                marker=dict(size=[10, 16], symbol=['circle', 'triangle-up'], color=color, line=dict(width=1, color='white')),
                text=f"<b>Year:</b> {year}<br><b>Direction:</b> {row['Direction']}<br><b>Bearing:</b> {row['Bearing_Degrees']:.1f}°",
                hoverinfo='text',
                name=f'{year}',
                legendgroup=str(year),
                showlegend=bool(idx == 0)
            ))
    
    fig.update_geos(
        projection_type="mercator",
        showland=True,
        landcolor='#1e293b',
        countrycolor='#334155',
        coastlinecolor='#475569',
        showocean=True,
        oceancolor='#0f172a',
        center=dict(lat=29, lon=85),
        projection_scale=6,
        bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_layout(
        title=dict(
            text='🧭 Cluster Migration Patterns',
            font=dict(size=20, color='#f8fafc', family='Space Grotesk, sans-serif'),
            x=0.5
        ),
        height=500,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            title=dict(text="Year", font=dict(color='#f8fafc')),
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(26, 26, 46, 0.95)',
            bordercolor='rgba(255,255,255,0.1)',
            borderwidth=1,
            font=dict(color='#94a3b8')
        )
    )
    
    return fig

def create_magnitude_distribution(df, year_range):
    """Create magnitude distribution visualization"""
    
    filtered = df[
        (df['year'] >= year_range[0]) & 
        (df['year'] <= year_range[1])
    ]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('<span style="color:#f8fafc">📊 Magnitude Distribution</span>', '<span style="color:#f8fafc">📈 Magnitude by Year</span>'),
        specs=[[{"type": "histogram"}, {"type": "box"}]]
    )
    
    # Histogram with gradient effect
    fig.add_trace(
        go.Histogram(
            x=filtered['mag'],
            nbinsx=30,
            marker=dict(
                color='#ef4444',
                line=dict(color='#f97316', width=1)
            ),
            opacity=0.85,
            name='Count'
        ),
        row=1, col=1
    )
    
    # Box plot by year with vibrant colors
    colors = px.colors.sequential.Plasma
    years_list = sorted(filtered['year'].unique())
    for i, year in enumerate(years_list):
        year_data = filtered[filtered['year'] == year]
        fig.add_trace(
            go.Box(
                y=year_data['mag'],
                name=str(year),
                marker_color=colors[i % len(colors)],
                line_color=colors[i % len(colors)],
                showlegend=False
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        height=420,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26, 26, 46, 0.6)',
        font=dict(color='#f8fafc')
    )
    
    # Update axes with dark theme
    fig.update_xaxes(title_text="Magnitude", row=1, col=1, gridcolor='rgba(255,255,255,0.05)', color='#94a3b8', title_font=dict(color='#f8fafc'))
    fig.update_xaxes(title_text="Year", row=1, col=2, gridcolor='rgba(255,255,255,0.05)', color='#94a3b8', title_font=dict(color='#f8fafc'))
    fig.update_yaxes(title_text="Count", row=1, col=1, gridcolor='rgba(255,255,255,0.05)', color='#94a3b8', title_font=dict(color='#f8fafc'))
    fig.update_yaxes(title_text="Magnitude", row=1, col=2, gridcolor='rgba(255,255,255,0.05)', color='#94a3b8', title_font=dict(color='#f8fafc'))
    
    return fig

def create_temporal_analysis(df, year_range):
    """Create temporal analysis charts"""
    
    filtered = df[
        (df['year'] >= year_range[0]) & 
        (df['year'] <= year_range[1])
    ]
    
    # Yearly counts
    yearly_counts = filtered.groupby('year').agg({
        'mag': ['count', 'mean', 'max']
    }).reset_index()
    yearly_counts.columns = ['year', 'count', 'avg_mag', 'max_mag']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '<span style="color:#f8fafc">📅 Earthquakes per Year</span>', 
            '<span style="color:#f8fafc">📈 Average Magnitude Trend</span>',
            '<span style="color:#f8fafc">📆 Monthly Distribution</span>',
            '<span style="color:#f8fafc">🔬 Depth vs Magnitude</span>'
        ),
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # Earthquakes per year with gradient
    fig.add_trace(
        go.Bar(
            x=yearly_counts['year'],
            y=yearly_counts['count'],
            marker=dict(
                color=yearly_counts['count'],
                colorscale='YlOrRd',
                line=dict(color='rgba(255,255,255,0.2)', width=1)
            ),
            name='Count',
            hovertemplate='<b>Year:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Average magnitude trend with glow
    fig.add_trace(
        go.Scatter(
            x=yearly_counts['year'],
            y=yearly_counts['avg_mag'],
            mode='lines+markers',
            line=dict(color='#f97316', width=3, shape='spline'),
            marker=dict(size=10, color='#f97316', line=dict(color='white', width=2)),
            name='Avg Magnitude',
            hovertemplate='<b>Year:</b> %{x}<br><b>Avg Mag:</b> %{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Monthly distribution
    monthly = filtered.groupby('month').size().reset_index(name='count')
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    fig.add_trace(
        go.Bar(
            x=month_names,
            y=monthly['count'],
            marker=dict(
                color=monthly['count'],
                colorscale='OrRd',
                line=dict(color='rgba(255,255,255,0.2)', width=1)
            ),
            name='Monthly',
            hovertemplate='<b>Month:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Depth vs Magnitude scatter with better visibility
    sample = filtered.sample(min(1000, len(filtered))) if len(filtered) > 1000 else filtered
    fig.add_trace(
        go.Scatter(
            x=sample['depth'],
            y=sample['mag'],
            mode='markers',
            marker=dict(
                size=7,
                color=sample['mag'],
                colorscale='Turbo',
                opacity=0.7,
                line=dict(color='rgba(255,255,255,0.3)', width=0.5)
            ),
            name='Depth vs Mag',
            hovertemplate='<b>Depth:</b> %{x:.1f} km<br><b>Magnitude:</b> %{y:.1f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=650,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26, 26, 46, 0.6)',
        font=dict(color='#f8fafc', family='Inter, sans-serif')
    )
    
    # Update all axes with dark theme
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)', color='#94a3b8', row=i, col=j)
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)', color='#94a3b8', row=i, col=j)
    
    return fig

def create_direction_analysis(directions_df):
    """Create direction analysis visualization"""
    
    # Direction frequency
    direction_counts = directions_df['Direction'].value_counts()
    
    # Neon color palette for dark theme
    colors = ['#ef4444', '#f97316', '#f59e0b', '#eab308', '#dc2626', '#ea580c', '#d97706', '#ca8a04']
    
    # Create two separate figures instead of subplots for better control
    # Figure 1: Pie Chart for Direction Frequency
    pie_fig = go.Figure(data=[
        go.Pie(
            labels=direction_counts.index,
            values=direction_counts.values,
            hole=0.5,
            marker=dict(
                colors=colors,
                line=dict(color='#1a1a2e', width=2)
            ),
            textinfo='label+percent',
            textposition='outside',
            textfont=dict(color='#f8fafc', size=11),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
            pull=[0.02] * len(direction_counts)
        )
    ])
    
    pie_fig.update_layout(
        title=dict(
            text='🧭 Migration Direction Frequency',
            font=dict(size=16, color='#f8fafc', family='Space Grotesk, sans-serif'),
            x=0.5
        ),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(26, 26, 46, 0.9)',
            bordercolor='rgba(255,255,255,0.1)',
            font=dict(color='#94a3b8', size=10)
        ),
        margin=dict(t=60, b=80, l=40, r=40)
    )
    
    # Figure 2: Polar Bar Chart for Bearing Distribution
    bearing_bins = pd.cut(directions_df['Bearing_Degrees'], bins=8, labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    bearing_counts = bearing_bins.value_counts().reindex(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']).fillna(0)
    
    polar_fig = go.Figure(data=[
        go.Barpolar(
            r=bearing_counts.values,
            theta=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
            marker_color=colors[:8],
            marker_line_color='rgba(255,255,255,0.3)',
            marker_line_width=1,
            opacity=0.9,
            hovertemplate='<b>Direction:</b> %{theta}<br><b>Count:</b> %{r}<extra></extra>'
        )
    ])
    
    polar_fig.update_layout(
        title=dict(
            text='🎯 Bearing Distribution',
            font=dict(size=16, color='#f8fafc', family='Space Grotesk, sans-serif'),
            x=0.5
        ),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(26, 26, 46, 0.9)',
            bordercolor='rgba(255,255,255,0.1)',
            font=dict(color='#94a3b8', size=10)
        ),
        polar=dict(
            bgcolor='rgba(26, 26, 46, 0.8)',
            radialaxis=dict(
                showticklabels=True, 
                ticks='',
                gridcolor='rgba(255,255,255,0.1)',
                color='#94a3b8',
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                direction="clockwise",
                gridcolor='rgba(255,255,255,0.1)',
                color='#f8fafc',
                tickfont=dict(size=12, color='#f8fafc')
            )
        ),
        margin=dict(t=60, b=80, l=60, r=60)
    )
    
    return pie_fig, polar_fig

def create_cluster_heatmap(cluster_df):
    """Create cluster activity heatmap"""
    
    # Create pivot table for heatmap
    cluster_pivot = cluster_df.groupby(['year', 'cluster_rank']).agg({
        'event_count': 'sum',
        'avg_mag': 'mean'
    }).reset_index()
    
    heatmap_data = cluster_pivot.pivot(index='cluster_rank', columns='year', values='event_count').fillna(0)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Hot',
        colorbar=dict(
            title=dict(text="Event Count", font=dict(color='#f8fafc')),
            tickfont=dict(color='#94a3b8'),
            bgcolor='rgba(26, 26, 46, 0.9)',
            bordercolor='rgba(255,255,255,0.1)',
            borderwidth=1
        ),
        hovertemplate='<b>Year:</b> %{x}<br><b>Cluster:</b> %{y}<br><b>Events:</b> %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='🔥 Cluster Activity Heatmap',
            font=dict(size=20, color='#f8fafc', family='Space Grotesk, sans-serif'),
            x=0.5
        ),
        xaxis_title='Year',
        yaxis_title='Cluster Rank',
        height=420,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26, 26, 46, 0.6)',
        font=dict(color='#f8fafc'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', color='#94a3b8'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', color='#94a3b8')
    )
    
    return fig

def create_3d_visualization(df, year_range):
    """Create 3D visualization of earthquakes"""
    
    filtered = df[
        (df['year'] >= year_range[0]) & 
        (df['year'] <= year_range[1])
    ]
    
    # Sample if data is too large
    if len(filtered) > 2000:
        filtered = filtered.sample(2000)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=filtered['lon'],
        y=filtered['lat'],
        z=-filtered['depth'],  # Negative for depth below surface
        mode='markers',
        marker=dict(
            size=filtered['mag'] * 2,
            color=filtered['mag'],
            colorscale='Turbo',
            colorbar=dict(
                title=dict(text="Magnitude", font=dict(color='#f8fafc')),
                tickfont=dict(color='#94a3b8'),
                bgcolor='rgba(26, 26, 46, 0.9)',
                bordercolor='rgba(255,255,255,0.1)'
            ),
            opacity=0.8,
            line=dict(color='rgba(255,255,255,0.2)', width=0.5)
        ),
        text=filtered.apply(lambda x: f"<b>Location:</b> ({x['lat']:.2f}°, {x['lon']:.2f}°)<br><b>Depth:</b> {x['depth']:.1f} km<br><b>Magnitude:</b> {x['mag']:.1f}", axis=1),
        hoverinfo='text'
    )])
    
    fig.update_layout(
        title=dict(
            text='🌐 3D Earthquake Visualization',
            font=dict(size=22, color='#f8fafc', family='Space Grotesk, sans-serif'),
            x=0.5
        ),
        scene=dict(
            xaxis=dict(
                title='Longitude',
                backgroundcolor='rgba(26, 26, 46, 0.8)',
                gridcolor='rgba(255,255,255,0.1)',
                color='#94a3b8'
            ),
            yaxis=dict(
                title='Latitude',
                backgroundcolor='rgba(26, 26, 46, 0.8)',
                gridcolor='rgba(255,255,255,0.1)',
                color='#94a3b8'
            ),
            zaxis=dict(
                title='Depth (km)',
                backgroundcolor='rgba(26, 26, 46, 0.8)',
                gridcolor='rgba(255,255,255,0.1)',
                color='#94a3b8'
            ),
            bgcolor='rgba(10, 10, 15, 0.9)'
        ),
        height=550,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc')
    )
    
    return fig


def load_globe_coordinates():
    """Load coordinates data for globe visualization (actual vs predicted)"""
    try:
        cords_df = pd.read_csv('globe/cords.csv')
        cords_df['time'] = pd.to_datetime(cords_df['time'])
        cords_df['year'] = cords_df['time'].dt.year
        return cords_df
    except Exception as e:
        st.error(f"Could not load globe coordinates: {e}")
        return None


def create_globe_visualization(cords_df, selected_year, show_actual=True, show_predicted=True, rotation_lon=0, rotation_lat=20):
    """Create interactive 3D globe visualization with actual vs predicted earthquake locations"""
    
    # Filter data by year
    year_data = cords_df[cords_df['year'] == selected_year]
    
    fig = go.Figure()
    
    # Add actual earthquake points (red/orange)
    if show_actual and len(year_data) > 0:
        fig.add_trace(go.Scattergeo(
            lon=year_data['Actual_Lon'],
            lat=year_data['Actual_Lat'],
            mode='markers',
            marker=dict(
                size=12,
                color='#ef4444',
                opacity=0.9,
                symbol='circle',
                line=dict(color='#ffffff', width=1)
            ),
            name='Actual Earthquakes',
            text=year_data.apply(lambda x: f"<b>Actual Location</b><br>Lat: {x['Actual_Lat']:.3f}°<br>Lon: {x['Actual_Lon']:.3f}°<br>Date: {x['time'].strftime('%Y-%m-%d')}", axis=1),
            hoverinfo='text'
        ))
    
    # Add predicted earthquake points (green/lime)
    if show_predicted and len(year_data) > 0:
        fig.add_trace(go.Scattergeo(
            lon=year_data['Pred_Lon'],
            lat=year_data['Pred_Lat'],
            mode='markers',
            marker=dict(
                size=10,
                color='#22c55e',
                opacity=0.85,
                symbol='diamond',
                line=dict(color='#ffffff', width=1)
            ),
            name='Predicted Locations',
            text=year_data.apply(lambda x: f"<b>Predicted Location</b><br>Lat: {x['Pred_Lat']:.3f}°<br>Lon: {x['Pred_Lon']:.3f}°<br>Date: {x['time'].strftime('%Y-%m-%d')}", axis=1),
            hoverinfo='text'
        ))
    
    # Add connection lines between actual and predicted
    if show_actual and show_predicted and len(year_data) > 0:
        for _, row in year_data.iterrows():
            fig.add_trace(go.Scattergeo(
                lon=[row['Actual_Lon'], row['Pred_Lon']],
                lat=[row['Actual_Lat'], row['Pred_Lat']],
                mode='lines',
                line=dict(color='rgba(249, 115, 22, 0.4)', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Configure globe projection
    fig.update_geos(
        projection_type="orthographic",
        projection_rotation=dict(lon=rotation_lon, lat=rotation_lat, roll=0),
        showland=True,
        landcolor='rgb(40, 40, 60)',
        showocean=True,
        oceancolor='rgb(15, 25, 45)',
        showcountries=True,
        countrycolor='rgba(255, 255, 255, 0.2)',
        showcoastlines=True,
        coastlinecolor='rgba(255, 255, 255, 0.3)',
        showlakes=True,
        lakecolor='rgb(15, 25, 45)',
        showrivers=False,
        bgcolor='rgba(0,0,0,0)',
        framecolor='rgba(255, 255, 255, 0.1)',
        framewidth=1,
        lonaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)', gridwidth=0.5),
        lataxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)', gridwidth=0.5)
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'🌍 Interactive Globe - Year {selected_year}',
            font=dict(size=24, color='#f8fafc', family='Space Grotesk, sans-serif'),
            x=0.5,
            y=0.95
        ),
        height=650,
        paper_bgcolor='rgba(10, 10, 15, 0.95)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.05,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(26, 26, 46, 0.9)',
            bordercolor='rgba(255, 255, 255, 0.2)',
            borderwidth=1,
            font=dict(color='#f8fafc', size=12)
        ),
        margin=dict(l=0, r=0, t=60, b=60),
        showlegend=True
    )
    
    return fig


def create_globe_statistics(cords_df, selected_year):
    """Create statistics charts for globe data"""
    year_data = cords_df[cords_df['year'] == selected_year]
    
    if len(year_data) == 0:
        return None, None
    
    # Calculate distances between actual and predicted
    distances = []
    for _, row in year_data.iterrows():
        # Haversine approximation for distance in km
        lat1, lon1 = np.radians(row['Actual_Lat']), np.radians(row['Actual_Lon'])
        lat2, lon2 = np.radians(row['Pred_Lat']), np.radians(row['Pred_Lon'])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Earth radius in km
        distances.append(r * c)
    
    year_data = year_data.copy()
    year_data['distance_km'] = distances
    
    # Distance distribution histogram
    dist_fig = go.Figure(data=[go.Histogram(
        x=year_data['distance_km'],
        nbinsx=15,
        marker=dict(
            color='#f97316',
            line=dict(color='#ef4444', width=1)
        ),
        opacity=0.85,
        hovertemplate='<b>Distance:</b> %{x:.1f} km<br><b>Count:</b> %{y}<extra></extra>'
    )])
    
    dist_fig.update_layout(
        title=dict(text='📏 Prediction Error Distribution', font=dict(color='#f8fafc', size=16)),
        xaxis_title='Distance Error (km)',
        yaxis_title='Count',
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26, 26, 46, 0.6)',
        font=dict(color='#f8fafc'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', color='#94a3b8'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', color='#94a3b8')
    )
    
    # Events per month
    year_data['month'] = year_data['time'].dt.month
    monthly_counts = year_data.groupby('month').size().reindex(range(1, 13), fill_value=0)
    
    monthly_fig = go.Figure(data=[go.Bar(
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y=monthly_counts.values,
        marker=dict(
            color=monthly_counts.values,
            colorscale='YlOrRd',
            line=dict(color='#ef4444', width=1)
        ),
        hovertemplate='<b>%{x}</b><br>Events: %{y}<extra></extra>'
    )])
    
    monthly_fig.update_layout(
        title=dict(text='📅 Monthly Event Distribution', font=dict(color='#f8fafc', size=16)),
        xaxis_title='Month',
        yaxis_title='Number of Events',
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26, 26, 46, 0.6)',
        font=dict(color='#f8fafc'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', color='#94a3b8'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', color='#94a3b8')
    )
    
    return dist_fig, monthly_fig, year_data['distance_km'].mean(), year_data['distance_km'].min(), year_data['distance_km'].max()


# Main Dashboard
def main():
    # Header with animation
    st.markdown("""
    <div class="dashboard-header">
        <h1>🌍 Earthquake Cluster Analysis</h1>
        <p>Interactive visualization of seismic activity patterns and cluster migrations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data with custom spinner
    with st.spinner('🔄 Loading earthquake data...'):
        earthquake_df = load_earthquake_data()
        cluster_df = load_cluster_summary()
        directions_df = load_cluster_directions()
        plate_data = load_plate_boundaries()
    
    if earthquake_df is None:
        st.error("❌ Failed to load data. Please check if the required CSV files exist.")
        return
    
    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <span style="font-size: 2.5rem;">🌍</span>
            <h2 style="color: #f8fafc; margin-top: 0.5rem; font-family: 'Space Grotesk', sans-serif;">Control Panel</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Year range slider
        min_year = int(earthquake_df['year'].min())
        max_year = int(earthquake_df['year'].max())
        st.markdown("#### 📅 Time Period")
        year_range = st.slider(
            "Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(1980, 2010),
            help="Filter earthquakes by year range",
            label_visibility="collapsed"
        )
        st.caption(f"Selected: {year_range[0]} - {year_range[1]}")
        
        st.markdown("#### 📊 Magnitude Filter")
        mag_range = st.slider(
            "Magnitude Range",
            min_value=0.0,
            max_value=10.0,
            value=(0.0, 10.0),
            step=0.5,
            help="Filter earthquakes by magnitude",
            label_visibility="collapsed"
        )
        st.caption(f"Range: {mag_range[0]} - {mag_range[1]}")
        
        st.markdown("---")
        
        # Quick stats with improved styling
        st.markdown("#### 📈 Live Statistics")
        filtered_df = earthquake_df[
            (earthquake_df['year'] >= year_range[0]) & 
            (earthquake_df['year'] <= year_range[1]) &
            (earthquake_df['mag'] >= mag_range[0]) &
            (earthquake_df['mag'] <= mag_range[1])
        ]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔢 Total", f"{len(filtered_df):,}")
            st.metric("📏 Max Mag", f"{filtered_df['mag'].max():.1f}")
        with col2:
            st.metric("📈 Avg Mag", f"{filtered_df['mag'].mean():.2f}")
            st.metric("🕳️ Avg Depth", f"{filtered_df['depth'].mean():.0f} km")
        
        st.markdown("---")
        
        # Info section
        st.markdown("""
        <div style="background: rgba(99, 102, 241, 0.1); padding: 1rem; border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.3);">
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">
                <strong style="color: #f97316;">💡 Tip:</strong> Use the sliders above to filter the data. All charts will update automatically.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #64748b; font-size: 0.8rem;">
            <p>Built with ❤️ using</p>
            <p><strong>Streamlit + Plotly</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content with tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🗺️ Map View", 
        "📊 Statistics", 
        "🧭 Migration",
        "🔥 Clusters",
        "🌐 3D View",
        "🌍 Globe",
        "🎯 Predictions"
    ])
    
    with tab1:
        st.markdown('<p class="section-header">🗺️ Interactive Earthquake Map</p>', unsafe_allow_html=True)
        map_fig = create_main_map(earthquake_df, cluster_df, plate_data, year_range, mag_range)
        st.plotly_chart(map_fig, use_container_width=True, config=PLOTLY_CONFIG)
        
        # Map legend cards
        st.markdown("#### 📋 Map Legend")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value" style="color: #ef4444;">━━</div>
                <div class="metric-label">Plate Boundaries</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value" style="color: #f97316;">★</div>
                <div class="metric-label">Cluster Centers</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value" style="color: #22c55e;">●</div>
                <div class="metric-label">Low Magnitude</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value" style="color: #ef4444;">●</div>
                <div class="metric-label">High Magnitude</div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<p class="section-header">📊 Temporal & Statistical Analysis</p>', unsafe_allow_html=True)
        
        # Temporal charts
        temporal_fig = create_temporal_analysis(earthquake_df, year_range)
        st.plotly_chart(temporal_fig, use_container_width=True, config=PLOTLY_CONFIG)
        
        st.markdown('<p class="section-header">📈 Magnitude Distribution</p>', unsafe_allow_html=True)
        mag_fig = create_magnitude_distribution(earthquake_df, year_range)
        st.plotly_chart(mag_fig, use_container_width=True, config=PLOTLY_CONFIG)
    
    with tab3:
        st.markdown('<p class="section-header">🧭 Cluster Migration Patterns</p>', unsafe_allow_html=True)
        
        if directions_df is not None:
            # Migration map on top
            st.markdown("#### 🗺️ Migration Flow Map")
            migration_fig = create_cluster_migration_map(directions_df, year_range)
            st.plotly_chart(migration_fig, use_container_width=True, config=PLOTLY_CONFIG)
            
            st.markdown("---")
            
            # Direction analysis charts side by side
            st.markdown("#### 📊 Direction Analysis")
            pie_fig, polar_fig = create_direction_analysis(directions_df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(pie_fig, use_container_width=True, config=PLOTLY_CONFIG)
            with col2:
                st.plotly_chart(polar_fig, use_container_width=True, config=PLOTLY_CONFIG)
            
            st.markdown("---")
            
            # Direction data table with styling
            st.markdown('<p class="section-header">📋 Migration Data Table</p>', unsafe_allow_html=True)
            filtered_directions = directions_df[
                (directions_df['Year'] >= year_range[0]) & 
                (directions_df['Year'] <= year_range[1])
            ]
            st.dataframe(
                filtered_directions[['Year', 'From_Cluster', 'To_Cluster', 'Direction', 'Bearing_Degrees']],
                use_container_width=True,
                height=300,
                hide_index=True
            )
        else:
            st.warning("⚠️ Migration data not available. Please check if cluster_directions_final.csv exists.")
    
    with tab4:
        st.markdown('<p class="section-header">🔥 Cluster Analysis</p>', unsafe_allow_html=True)
        
        if cluster_df is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                heatmap_fig = create_cluster_heatmap(cluster_df)
                st.plotly_chart(heatmap_fig, use_container_width=True, config=PLOTLY_CONFIG)
            
            with col2:
                # Top clusters by event count
                st.markdown("#### 🏆 Top Clusters by Event Count")
                top_clusters = cluster_df.nlargest(10, 'event_count')[
                    ['year', 'cluster_rank', 'event_count', 'avg_mag', 'centroid_lat', 'centroid_lon']
                ]
                st.dataframe(top_clusters, use_container_width=True, height=380, hide_index=True)
            
            # Cluster summary stats with enhanced cards
            st.markdown('<p class="section-header">📊 Cluster Summary Statistics</p>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_clusters = len(cluster_df)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_clusters}</div>
                    <div class="metric-label">Total Clusters</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                avg_events = cluster_df['event_count'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{avg_events:.1f}</div>
                    <div class="metric-label">Avg Events/Cluster</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                max_events = cluster_df['event_count'].max()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{max_events}</div>
                    <div class="metric-label">Max Events</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                years_with_clusters = cluster_df['year'].nunique()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{years_with_clusters}</div>
                    <div class="metric-label">Years Analyzed</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Cluster data not available. Please check if cluster_summary_1980_2005.csv exists.")
    
    with tab5:
        st.markdown('<p class="section-header">🌐 3D Earthquake Visualization</p>', unsafe_allow_html=True)
        
        # Info box with dark styling
        st.markdown("""
        <div style="background: rgba(99, 102, 241, 0.1); padding: 1rem; border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.3); margin-bottom: 1rem;">
            <p style="color: #94a3b8; font-size: 0.95rem; margin: 0;">
                <strong style="color: #f97316;">🎮 Controls:</strong> Drag to rotate • Scroll to zoom • Right-click drag to pan • Depth shown as negative Z-axis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        fig_3d = create_3d_visualization(earthquake_df, year_range)
        st.plotly_chart(fig_3d, use_container_width=True, config=PLOTLY_CONFIG)
        
        # Depth analysis with dark theme
        st.markdown('<p class="section-header">🔬 Depth Analysis</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            depth_hist = go.Figure(data=[go.Histogram(
                x=earthquake_df['depth'],
                nbinsx=50,
                marker=dict(
                    color='#ef4444',
                    line=dict(color='#f97316', width=1)
                ),
                opacity=0.85,
                hovertemplate='<b>Depth:</b> %{x} km<br><b>Count:</b> %{y}<extra></extra>'
            )])
            depth_hist.update_layout(
                title=dict(text='📏 Depth Distribution', font=dict(color='#f8fafc', size=16)),
                xaxis_title='Depth (km)',
                yaxis_title='Count',
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26, 26, 46, 0.6)',
                font=dict(color='#f8fafc'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)', color='#94a3b8'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', color='#94a3b8')
            )
            st.plotly_chart(depth_hist, use_container_width=True, config=PLOTLY_CONFIG)
        
        with col2:
            # Depth categories
            earthquake_df['depth_category'] = pd.cut(
                earthquake_df['depth'],
                bins=[0, 70, 300, 700],
                labels=['Shallow (0-70km)', 'Intermediate (70-300km)', 'Deep (300-700km)']
            )
            depth_counts = earthquake_df['depth_category'].value_counts()
            
            depth_pie = go.Figure(data=[go.Pie(
                labels=depth_counts.index,
                values=depth_counts.values,
                hole=0.5,
                marker=dict(
                    colors=['#22c55e', '#f59e0b', '#ef4444'],
                    line=dict(color='#1a1a2e', width=2)
                ),
                textfont=dict(color='#f8fafc'),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )])
            depth_pie.update_layout(
                title=dict(text='🥧 Depth Categories', font=dict(color='#f8fafc', size=16)),
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f8fafc'),
                legend=dict(
                    bgcolor='rgba(26, 26, 46, 0.9)',
                    bordercolor='rgba(255,255,255,0.1)',
                    font=dict(color='#94a3b8')
                )
            )
            st.plotly_chart(depth_pie, use_container_width=True, config=PLOTLY_CONFIG)
    
    with tab6:
        st.markdown('<p class="section-header">🌍 Interactive Globe Visualization</p>', unsafe_allow_html=True)
        
        # Load globe coordinates
        cords_df = load_globe_coordinates()
        
        if cords_df is not None:
            # Info box
            st.markdown("""
            <div style="background: rgba(239, 68, 68, 0.1); padding: 1rem; border-radius: 12px; border: 1px solid rgba(239, 68, 68, 0.3); margin-bottom: 1rem;">
                <p style="color: #94a3b8; font-size: 0.95rem; margin: 0;">
                    <strong style="color: #ef4444;">🎮 Controls:</strong> Drag globe to rotate • Scroll to zoom • 
                    <span style="color: #ef4444;">●</span> Red = Actual earthquakes • 
                    <span style="color: #22c55e;">◆</span> Green = Predicted locations
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Globe controls
            st.markdown("#### 🎛️ Globe Controls")
            globe_col1, globe_col2, globe_col3, globe_col4 = st.columns(4)
            
            with globe_col1:
                available_years = sorted(cords_df['year'].unique())
                globe_year = st.selectbox(
                    "📅 Select Year",
                    options=available_years,
                    index=0,
                    help="Choose the year to display earthquake data"
                )
            
            with globe_col2:
                rotation_lon = st.slider(
                    "🔄 Rotation (Longitude)",
                    min_value=-180,
                    max_value=180,
                    value=85,  # Center on India/Nepal region
                    help="Rotate the globe horizontally"
                )
            
            with globe_col3:
                rotation_lat = st.slider(
                    "📐 Tilt (Latitude)",
                    min_value=-90,
                    max_value=90,
                    value=25,  # Tilt to show Himalayan region better
                    help="Tilt the globe vertically"
                )
            
            with globe_col4:
                st.markdown("**Display Options**")
                show_actual = st.checkbox("Show Actual", value=True)
                show_predicted = st.checkbox("Show Predicted", value=True)
            
            # Create and display globe
            globe_fig = create_globe_visualization(
                cords_df, 
                globe_year, 
                show_actual=show_actual, 
                show_predicted=show_predicted,
                rotation_lon=rotation_lon,
                rotation_lat=rotation_lat
            )
            st.plotly_chart(globe_fig, use_container_width=True, config=PLOTLY_CONFIG)
            
            # Statistics section
            st.markdown("#### 📊 Prediction Accuracy Metrics")
            
            stats_result = create_globe_statistics(cords_df, globe_year)
            
            if stats_result[0] is not None:
                dist_fig, monthly_fig, avg_dist, min_dist, max_dist = stats_result
                
                # Metric cards
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                year_data = cords_df[cords_df['year'] == globe_year]
                
                with metric_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color: #ef4444;">{len(year_data)}</div>
                        <div class="metric-label">Total Events</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color: #f97316;">{avg_dist:.1f} km</div>
                        <div class="metric-label">Avg. Prediction Error</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color: #22c55e;">{min_dist:.1f} km</div>
                        <div class="metric-label">Best Prediction</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color: #dc2626;">{max_dist:.1f} km</div>
                        <div class="metric-label">Worst Prediction</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Charts
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    st.plotly_chart(dist_fig, use_container_width=True, config=PLOTLY_CONFIG)
                
                with chart_col2:
                    st.plotly_chart(monthly_fig, use_container_width=True, config=PLOTLY_CONFIG)
            else:
                st.info(f"📭 No earthquake events recorded for year {globe_year}")
        else:
            st.warning("⚠️ Globe coordinates data not available. Please check if 'globe/cords.csv' exists.")
    
    with tab7:
        st.markdown('<p class="section-header">🎯 Stress Zone Predictions (XGBoost Model)</p>', unsafe_allow_html=True)
        
        # Info box
        st.markdown("""
        <div style="background: rgba(99, 102, 241, 0.1); padding: 1rem; border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.3); margin-bottom: 1rem;">
            <p style="color: #94a3b8; font-size: 0.95rem; margin: 0;">
                <strong style="color: #f97316;">🧠 Model Info:</strong> XGBoost classifier trained on 1980-2005 data, tested on 2006-2011 data.
                Predicts which sector (8×8 grid) will experience the next stress zone migration.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Run predictions
        with st.spinner('🔄 Running stress zone prediction model...'):
            results_df, metrics, feature_importance, GRID_DIM, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = run_stress_prediction()
        
        if results_df is not None and metrics is not None:
            # Metrics cards
            st.markdown("#### 📊 Model Performance")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics['total']}</div>
                    <div class="metric-label">Total Predictions</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #22c55e;">{metrics['top1_pct']:.1f}%</div>
                    <div class="metric-label">Exact Match (Top 1)</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #f59e0b;">{metrics['top3_pct']:.1f}%</div>
                    <div class="metric-label">Top 3 Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #f97316;">{metrics['top5_pct']:.1f}%</div>
                    <div class="metric-label">Top 5 Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Visualization columns
            col1, col2 = st.columns(2)
            
            with col1:
                # Actual vs Predicted Map
                st.markdown("#### 🗺️ Actual vs Predicted Zones")
                
                pred_map = go.Figure()
                
                # Draw connection lines
                for _, row in results_df.iterrows():
                    color = '#22c55e' if row['Rank'] == 1 else '#f59e0b' if row['Rank'] <= 3 else '#ef4444'
                    pred_map.add_trace(go.Scattergeo(
                        lon=[row['Real_Lon'], row['Pred_Lon']],
                        lat=[row['Real_Lat'], row['Pred_Lat']],
                        mode='lines',
                        line=dict(width=1, color=color),
                        opacity=0.4,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                # Actual points
                pred_map.add_trace(go.Scattergeo(
                    lon=results_df['Real_Lon'],
                    lat=results_df['Real_Lat'],
                    mode='markers',
                    marker=dict(size=10, color='#3b82f6', symbol='circle', line=dict(width=1, color='white')),
                    name='Actual',
                    text=results_df.apply(lambda x: f"<b>Year:</b> {int(x['Year'])}<br><b>Sector:</b> {int(x['Real_Sector'])}<br><b>Location:</b> ({x['Real_Lat']:.1f}°, {x['Real_Lon']:.1f}°)", axis=1),
                    hoverinfo='text'
                ))
                
                # Predicted points
                pred_map.add_trace(go.Scattergeo(
                    lon=results_df['Pred_Lon'],
                    lat=results_df['Pred_Lat'],
                    mode='markers',
                    marker=dict(size=10, color='#ef4444', symbol='x', line=dict(width=2, color='white')),
                    name='Predicted',
                    text=results_df.apply(lambda x: f"<b>Year:</b> {int(x['Year'])}<br><b>Predicted Sector:</b> {int(x['Pred_Sector'])}<br><b>Rank:</b> {int(x['Rank'])}<br><b>Confidence:</b> {x['Confidence']:.1f}%", axis=1),
                    hoverinfo='text'
                ))
                
                pred_map.update_geos(
                    projection_type="mercator",
                    showland=True,
                    landcolor='#1e293b',
                    countrycolor='#334155',
                    coastlinecolor='#475569',
                    showocean=True,
                    oceancolor='#0f172a',
                    center=dict(lat=25, lon=82),
                    projection_scale=4,
                    bgcolor='rgba(0,0,0,0)'
                )
                
                pred_map.update_layout(
                    height=450,
                    margin=dict(l=0, r=0, t=30, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    legend=dict(
                        yanchor="top", y=0.99, xanchor="left", x=0.01,
                        bgcolor='rgba(26, 26, 46, 0.95)',
                        bordercolor='rgba(255,255,255,0.1)',
                        font=dict(color='#f8fafc')
                    )
                )
                st.plotly_chart(pred_map, use_container_width=True, config=PLOTLY_CONFIG)
            
            with col2:
                # Feature Importance
                st.markdown("#### 🔬 Feature Importance")
                
                if feature_importance:
                    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                    feature_names = [f[0] for f in sorted_features]
                    feature_values = [f[1] for f in sorted_features]
                    
                    feat_fig = go.Figure(data=[go.Bar(
                        x=feature_values,
                        y=feature_names,
                        orientation='h',
                        marker=dict(
                            color=feature_values,
                            colorscale='YlOrRd',
                            line=dict(color='rgba(255,255,255,0.3)', width=1)
                        ),
                        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
                    )])
                    
                    feat_fig.update_layout(
                        height=450,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(26, 26, 46, 0.6)',
                        font=dict(color='#f8fafc'),
                        xaxis=dict(title='Importance', gridcolor='rgba(255,255,255,0.05)', color='#94a3b8'),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', color='#94a3b8'),
                        margin=dict(l=120, r=20, t=30, b=40)
                    )
                    st.plotly_chart(feat_fig, use_container_width=True, config=PLOTLY_CONFIG)
            
            st.markdown("---")
            
            # Prediction results table
            st.markdown("#### 📋 Prediction Results (2006-2011)")
            
            # Add status emoji column
            results_display = results_df.copy()
            results_display['Status'] = results_display['Rank'].apply(
                lambda x: '🎯 Exact' if x == 1 else '✅ Top 3' if x <= 3 else '⚠️ Top 5' if x <= 5 else '❌ Miss'
            )
            results_display['Year'] = results_display['Year'].astype(int)
            results_display['Real_Sector'] = results_display['Real_Sector'].astype(int)
            results_display['Pred_Sector'] = results_display['Pred_Sector'].astype(int)
            results_display['Confidence'] = results_display['Confidence'].round(1).astype(str) + '%'
            
            st.dataframe(
                results_display[['Year', 'Real_Sector', 'Pred_Sector', 'Rank', 'Confidence', 'Status']].rename(columns={
                    'Real_Sector': 'Actual Sector',
                    'Pred_Sector': 'Predicted Sector'
                }),
                use_container_width=True,
                height=300,
                hide_index=True
            )
            
            # Accuracy by year
            st.markdown("#### 📈 Accuracy by Year")
            yearly_accuracy = results_df.groupby('Year').apply(
                lambda x: pd.Series({
                    'Total': len(x),
                    'Top1': (x['Rank'] == 1).sum(),
                    'Top3': (x['Rank'] <= 3).sum(),
                    'Top5': (x['Rank'] <= 5).sum()
                })
            ).reset_index()
            yearly_accuracy['Top1_Pct'] = (yearly_accuracy['Top1'] / yearly_accuracy['Total'] * 100).round(1)
            yearly_accuracy['Top3_Pct'] = (yearly_accuracy['Top3'] / yearly_accuracy['Total'] * 100).round(1)
            
            acc_fig = go.Figure()
            acc_fig.add_trace(go.Bar(
                x=yearly_accuracy['Year'],
                y=yearly_accuracy['Top1_Pct'],
                name='Exact Match',
                marker_color='#22c55e'
            ))
            acc_fig.add_trace(go.Bar(
                x=yearly_accuracy['Year'],
                y=yearly_accuracy['Top3_Pct'],
                name='Top 3',
                marker_color='#f59e0b'
            ))
            
            acc_fig.update_layout(
                height=350,
                barmode='group',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26, 26, 46, 0.6)',
                font=dict(color='#f8fafc'),
                xaxis=dict(title='Year', gridcolor='rgba(255,255,255,0.05)', color='#94a3b8'),
                yaxis=dict(title='Accuracy (%)', gridcolor='rgba(255,255,255,0.05)', color='#94a3b8'),
                legend=dict(
                    bgcolor='rgba(26, 26, 46, 0.9)',
                    bordercolor='rgba(255,255,255,0.1)',
                    font=dict(color='#94a3b8')
                )
            )
            st.plotly_chart(acc_fig, use_container_width=True, config=PLOTLY_CONFIG)
            
        else:
            st.warning("⚠️ Could not run prediction model. Please ensure `test_set.csv` and `cluster_directions_final.csv` exist.")
        
        # ============ VISUALIZATION GALLERIES ============
        st.markdown("---")
        st.markdown('<p class="section-header">🖼️ Stress Zone Visualization Gallery</p>', unsafe_allow_html=True)
        
        # Gallery CSS
        st.markdown("""
        <style>
        .gallery-container {
            background: rgba(26, 26, 46, 0.6);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 1rem;
        }
        .gallery-title {
            color: #f8fafc;
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .image-card {
            background: rgba(15, 23, 42, 0.8);
            border-radius: 12px;
            padding: 0.75rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: all 0.3s ease;
        }
        .image-card:hover {
            border-color: rgba(99, 102, 241, 0.5);
            transform: translateY(-2px);
        }
        .image-label {
            color: #94a3b8;
            font-size: 0.85rem;
            text-align: center;
            margin-top: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Section 1: Yearly Stress Predictions (2006-2010 comparisons)
        with st.expander("📊 Yearly Stress Prediction Comparisons (Test Period: 2006-2010)", expanded=True):
            st.markdown("""
            <div style="background: rgba(34, 197, 94, 0.1); padding: 0.75rem; border-radius: 8px; border-left: 3px solid #22c55e; margin-bottom: 1rem;">
                <p style="color: #94a3b8; margin: 0; font-size: 0.9rem;">
                    <strong style="color: #22c55e;">ℹ️ About:</strong> These visualizations compare actual vs predicted stress zones for each test year. 
                    Green indicates correct predictions, red indicates misses.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            pred_years = [2006, 2007, 2008, 2009, 2010]
            cols = st.columns(3)
            for idx, year in enumerate(pred_years):
                with cols[idx % 3]:
                    img_path = f"yearly_stress_predictions/stress_comparison_{year}.png"
                    if os.path.exists(img_path):
                        st.image(img_path, caption=f"🎯 {year} - Actual vs Predicted", use_container_width=True)
                    else:
                        st.info(f"Image for {year} not found")
        
        # Section 2: Historical Yearly Heatmaps (Training Period)
        with st.expander("🗺️ Historical Stress Heatmaps (Training Period: 1980-2005)", expanded=False):
            st.markdown("""
            <div style="background: rgba(249, 115, 22, 0.1); padding: 0.75rem; border-radius: 8px; border-left: 3px solid #f97316; margin-bottom: 1rem;">
                <p style="color: #94a3b8; margin: 0; font-size: 0.9rem;">
                    <strong style="color: #f97316;">ℹ️ About:</strong> Annual stress zone distributions used for model training. 
                    Darker regions indicate higher seismic activity concentration.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Get all available years from the folder
            heatmap_years = sorted([int(f.replace('stress_map_', '').replace('.png', '')) 
                                   for f in os.listdir('yearly_heatmaps') if f.startswith('stress_map_')])
            
            # Year selector
            selected_decade = st.radio(
                "Select Decade:",
                ["1980s", "1990s", "2000s"],
                horizontal=True,
                key="heatmap_decade"
            )
            
            if selected_decade == "1980s":
                display_years = [y for y in heatmap_years if 1980 <= y < 1990]
            elif selected_decade == "1990s":
                display_years = [y for y in heatmap_years if 1990 <= y < 2000]
            else:
                display_years = [y for y in heatmap_years if 2000 <= y <= 2005]
            
            cols = st.columns(4)
            for idx, year in enumerate(display_years):
                with cols[idx % 4]:
                    img_path = f"yearly_heatmaps/stress_map_{year}.png"
                    if os.path.exists(img_path):
                        st.image(img_path, caption=f"📅 {year}", use_container_width=True)
        
        # Section 3: Stress Migration Maps
        with st.expander("🔄 Stress Migration Maps (1980-2005)", expanded=False):
            st.markdown("""
            <div style="background: rgba(245, 158, 11, 0.1); padding: 0.75rem; border-radius: 8px; border-left: 3px solid #f59e0b; margin-bottom: 1rem;">
                <p style="color: #94a3b8; margin: 0; font-size: 0.9rem;">
                    <strong style="color: #f59e0b;">ℹ️ About:</strong> These maps show how stress zones migrate over time, 
                    illustrating the directional patterns of seismic activity movement.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            migration_years = sorted([int(f.replace('stress_map_', '').replace('.png', '')) 
                                      for f in os.listdir('stress_migration_maps') if f.startswith('stress_map_')])
            
            # Slider for year selection
            if migration_years:
                selected_mig_year = st.select_slider(
                    "Select Year:",
                    options=migration_years,
                    value=migration_years[len(migration_years)//2],
                    key="migration_year"
                )
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    img_path = f"stress_migration_maps/stress_map_{selected_mig_year}.png"
                    if os.path.exists(img_path):
                        st.image(img_path, caption=f"🔄 Stress Migration - {selected_mig_year}", use_container_width=True)
        
        # Section 4: Energy Transfer Plots
        with st.expander("⚡ Energy Transfer Visualizations", expanded=False):
            st.markdown("""
            <div style="background: rgba(239, 68, 68, 0.1); padding: 0.75rem; border-radius: 8px; border-left: 3px solid #ef4444; margin-bottom: 1rem;">
                <p style="color: #94a3b8; margin: 0; font-size: 0.9rem;">
                    <strong style="color: #ef4444;">ℹ️ About:</strong> Energy transfer plots showing seismic energy distribution 
                    and propagation patterns across the study region.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            energy_pages = sorted([int(f.replace('energy_map_page_', '').replace('.png', '')) 
                                   for f in os.listdir('energy_transfer_plots') if f.startswith('energy_map_page_')])
            
            if energy_pages:
                selected_page = st.select_slider(
                    "Select Page:",
                    options=energy_pages,
                    value=energy_pages[0],
                    key="energy_page"
                )
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    img_path = f"energy_transfer_plots/energy_map_page_{selected_page}.png"
                    if os.path.exists(img_path):
                        st.image(img_path, caption=f"⚡ Energy Transfer - Page {selected_page}", use_container_width=True)
        
        # Section 5: Model Output Visualizations
        with st.expander("📈 Model Output Visualizations", expanded=False):
            st.markdown("""
            <div style="background: rgba(168, 85, 247, 0.1); padding: 0.75rem; border-radius: 8px; border-left: 3px solid #a855f7; margin-bottom: 1rem;">
                <p style="color: #94a3b8; margin: 0; font-size: 0.9rem;">
                    <strong style="color: #a855f7;">ℹ️ About:</strong> Additional model outputs including feature importance, 
                    error analysis, and comparative heatmaps.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            model_images = {
                'feature_importance.png': ('🔬 Feature Importance', 'Shows which features contribute most to predictions'),
                'stress_zone_scatter.png': ('📊 Prediction Scatter', 'Actual vs predicted stress zone comparison'),
                'stress_zone_error_map.png': ('❌ Error Analysis', 'Geographic distribution of prediction errors'),
                'stress_zone_heatmap_comparison.png': ('🗺️ Heatmap Comparison', 'Side-by-side actual vs predicted heatmaps')
            }
            
            cols = st.columns(2)
            col_idx = 0
            for img_file, (title, desc) in model_images.items():
                if os.path.exists(img_file):
                    with cols[col_idx % 2]:
                        st.markdown(f"**{title}**")
                        st.caption(desc)
                        st.image(img_file, use_container_width=True)
                    col_idx += 1
    
    # Footer with dark theme
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">🌍 <strong>Earthquake Cluster Analysis Dashboard</strong></p>
        <p style="font-size: 0.9rem; color: #64748b;">Built with Streamlit & Plotly | Data source: USGS Earthquake Catalog</p>
        <p style="font-size: 0.8rem; color: #475569; margin-top: 0.5rem;">© 2024 Seismic Research Project</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
