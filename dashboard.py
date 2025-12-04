"""
Earthquake Stress Migration Analysis
Professional Dashboard for Seismic Pattern Visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Seismic Analysis Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Dark Theme - Seismic Color Palette
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --bg-dark: #0c0c0e;
        --bg-card: #141418;
        --bg-elevated: #1c1c22;
        --text-primary: #f4f4f5;
        --text-secondary: #9ca3af;
        --text-muted: #6b7280;
        --accent: #c2410c;
        --accent-light: #ea580c;
        --accent-glow: rgba(234, 88, 12, 0.15);
        --teal: #0d9488;
        --teal-light: #14b8a6;
        --amber: #d97706;
        --success: #059669;
        --warning: #ca8a04;
        --danger: #dc2626;
        --border: #27272a;
        --border-light: #3f3f46;
    }
    
    .stApp {
        background: linear-gradient(180deg, var(--bg-dark) 0%, #0a0a0c 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main .block-container {
        padding: 1.5rem 2rem;
        max-width: 100%;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Header */
    .header-container {
        background: linear-gradient(135deg, #1c1917 0%, #0c0a09 100%);
        padding: 1.75rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid var(--border);
        box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--accent-light), transparent);
        opacity: 0.5;
    }
    
    .header-title {
        font-size: 1.625rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #fff 0%, #d4d4d8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .header-subtitle {
        font-size: 0.875rem;
        color: var(--text-muted);
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border);
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }
    
    .metric-value {
        font-size: 1.875rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Section titles */
    .section-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .section-title::before {
        content: '';
        width: 3px;
        height: 14px;
        background: var(--accent);
        border-radius: 2px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-card) 0%, var(--bg-dark) 100%);
        border-right: 1px solid var(--border);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-secondary);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card);
        padding: 6px;
        border-radius: 10px;
        gap: 4px;
        border: 1px solid var(--border);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 10px 20px;
        color: var(--text-muted);
        font-size: 0.85rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-secondary);
        background: var(--bg-elevated);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent) 0%, #9a3412 100%) !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(194, 65, 12, 0.3);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.25rem;
        color: var(--teal-light);
        font-weight: 600;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-muted);
        font-size: 0.75rem;
    }
    
    /* Image gallery */
    .heatmap-gallery {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 1rem;
    }
    
    .heatmap-item {
        background: var(--bg-card);
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid var(--border);
        transition: all 0.25s ease;
    }
    
    .heatmap-item:hover {
        transform: translateY(-3px);
        border-color: var(--accent);
        box-shadow: 0 8px 24px rgba(194, 65, 12, 0.15);
    }
    
    .heatmap-label {
        padding: 0.875rem;
        font-size: 0.8rem;
        color: var(--text-secondary);
        text-align: center;
        border-top: 1px solid var(--border);
        background: var(--bg-elevated);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-dark); }
    ::-webkit-scrollbar-thumb { 
        background: var(--border-light); 
        border-radius: 4px;
        border: 2px solid var(--bg-dark);
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
    
    /* Buttons & Inputs */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent) 0%, #9a3412 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(194, 65, 12, 0.4);
        transform: translateY(-1px);
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: var(--bg-card);
        padding: 0.5rem;
        border-radius: 8px;
        border: 1px solid var(--border);
    }
    
    /* Sliders */
    .stSlider [data-baseweb="slider"] {
        margin-top: 0.5rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--bg-card);
        border-radius: 8px;
        border: 1px solid var(--border);
    }
    
    /* DataFrame */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid var(--border);
    }
    
    /* Select boxes and multiselect */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: var(--bg-card);
        border-color: var(--border);
        border-radius: 8px;
    }
    
    .stSelectbox label,
    .stMultiSelect label {
        color: var(--text-secondary) !important;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Slider styling */
    .stSlider label {
        color: var(--text-secondary) !important;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    /* Radio button styling */
    .stRadio label {
        color: var(--text-secondary) !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: var(--bg-elevated);
        color: var(--text-primary);
        border: 1px solid var(--border);
        border-radius: 8px;
    }
    
    .stDownloadButton > button:hover {
        background: var(--bg-card);
        border-color: var(--accent);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: var(--accent) !important;
    }
    
    /* Info/Warning/Error boxes */
    .stAlert {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
    }
    
    /* Image containers */
    .stImage {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Plotly charts container */
    .js-plotly-plot {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# Data Loading Functions
@st.cache_data
def load_earthquake_data():
    try:
        df = pd.read_csv('data/coordinates.csv')
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time', 'lat', 'lon'])
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        df['mag'] = df['mag'].fillna(0)
        return df
    except Exception as e:
        st.error(f"Error loading coordinates: {e}")
        return None


@st.cache_data
def load_cluster_directions():
    try:
        return pd.read_csv('data/cluster_directions_final.csv')
    except:
        return None


@st.cache_data
def load_plate_boundaries():
    try:
        with open('data/eu_in_plates.geojson', 'r') as f:
            return json.load(f)
    except:
        return None


@st.cache_data
def load_prediction_data():
    try:
        pred_df = pd.read_csv('data/predicted_heatmap_data.csv')
        grid_df = pd.read_csv('data/heatmap_grid_density.csv')
        return pred_df, grid_df
    except:
        return None, None


@st.cache_data
def load_globe_data():
    """Load globe prediction data"""
    try:
        return pd.read_csv('globe/cords.csv')
    except:
        return None


@st.cache_data
def run_prediction_model():
    """XGBoost stress zone prediction"""
    try:
        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder
        
        GRID_DIM, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 8, 15, 35, 65, 100
        
        def get_sector(lat, lon):
            if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
                return -1
            y = min(int((lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * GRID_DIM), GRID_DIM - 1)
            x = min(int((lon - LON_MIN) / (LON_MAX - LON_MIN) * GRID_DIM), GRID_DIM - 1)
            return y * GRID_DIM + x
        
        def get_center(sector):
            y, x = sector // GRID_DIM, sector % GRID_DIM
            return (LAT_MIN + (y + 0.5) * (LAT_MAX - LAT_MIN) / GRID_DIM,
                    LON_MIN + (x + 0.5) * (LON_MAX - LON_MIN) / GRID_DIM)
        
        def make_features(df, lookback=3):
            features = []
            df = df.sort_values('From_Time').reset_index(drop=True)
            for i in range(lookback, len(df)):
                curr = df.iloc[i]
                tgt = get_sector(curr['To_Lat'], curr['To_Lon'])
                src = get_sector(curr['From_Lat'], curr['From_Lon'])
                if tgt == -1 or src == -1:
                    continue
                f = {'Year': curr['Year'], 'Month': curr['From_Time'].month,
                     'Current_Sector': src, 'Current_Bearing': curr['Bearing_Degrees'],
                     'Current_Angle': curr['Angle_wrt_X_Axis']}
                valid = True
                for lag in range(1, lookback + 1):
                    p = df.iloc[i - lag]
                    ps = get_sector(p['To_Lat'], p['To_Lon'])
                    if ps == -1:
                        valid = False
                        break
                    f[f'Lag{lag}_Sector'] = ps
                    f[f'Lag{lag}_Bearing'] = p['Bearing_Degrees']
                    f[f'Lag{lag}_Angle'] = p['Angle_wrt_X_Axis']
                    f[f'Lag{lag}_DaysAgo'] = max((curr['From_Time'] - p['From_Time']).days, 1)
                if valid:
                    f['Target_Sector'] = tgt
                    features.append(f)
            return pd.DataFrame(features)
        
        train = pd.read_csv('data/cluster_directions_final.csv')
        train['From_Time'] = pd.to_datetime(train['From_Time'], format='mixed')
        train = train[train['Year'] <= 2005].sort_values('From_Time')
        
        test = pd.read_csv('data/test_set.csv')
        test['From_Time'] = pd.to_datetime(test['From_Time'], format='mixed')
        test = test.sort_values('From_Time')
        
        train_f, test_f = make_features(train), make_features(test)
        X_train = train_f.drop(columns=['Target_Sector', 'Year'])
        y_train = train_f['Target_Sector']
        X_test = test_f.drop(columns=['Target_Sector', 'Year'])
        y_test = test_f['Target_Sector']
        
        le = LabelEncoder()
        le.fit(pd.concat([y_train, y_test]))
        
        model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbosity=0)
        model.fit(X_train, le.transform(y_train))
        
        probs = model.predict_proba(X_test)
        y_test_enc = le.transform(y_test)
        
        results, hits = [], {'top1': 0, 'top3': 0, 'top5': 0}
        for i in range(len(X_test)):
            ranked = np.argsort(probs[i])[::-1]
            rank = np.where(ranked == y_test_enc[i])[0][0] + 1 if y_test_enc[i] in ranked else 99
            if rank == 1: hits['top1'] += 1
            if rank <= 3: hits['top3'] += 1
            if rank <= 5: hits['top5'] += 1
            
            pred = le.inverse_transform([ranked[0]])[0]
            real = y_test.iloc[i]
            results.append({
                'Year': test_f.iloc[i]['Year'], 'Rank': rank,
                'Real_Sector': real, 'Pred_Sector': pred,
                'Real_Lat': get_center(real)[0], 'Real_Lon': get_center(real)[1],
                'Pred_Lat': get_center(pred)[0], 'Pred_Lon': get_center(pred)[1],
                'Confidence': probs[i][ranked[0]] * 100
            })
        
        total = len(X_test)
        metrics = {'total': total, 'top1': hits['top1']/total*100, 
                   'top3': hits['top3']/total*100, 'top5': hits['top5']/total*100}
        return pd.DataFrame(results), metrics, GRID_DIM, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
    except Exception as e:
        return None, None, None, None, None, None, None


# Chart helpers
CHART_CONFIG = {'displayModeBar': True, 'displaylogo': False}

def dark_layout(height=400):
    return dict(
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,24,0.9)',
        font=dict(family='Inter', color='#f4f4f5', size=11),
        margin=dict(l=50, r=30, t=40, b=40),
        hoverlabel=dict(bgcolor='#1c1c22', font_size=11, bordercolor='#3f3f46', font_color='#f4f4f5')
    )


def get_heatmap_files(folder):
    """Get list of heatmap image files from a folder"""
    path = Path(folder)
    if path.exists():
        files = sorted([f for f in path.glob('*.png')])
        return files
    return []


# Main Application
def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">Seismic Stress Migration Analysis</h1>
        <p class="header-subtitle">Indian Subcontinent | 1980-2011 | XGBoost Prediction Model</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    eq_df = load_earthquake_data()
    dir_df = load_cluster_directions()
    plates = load_plate_boundaries()
    pred_df, grid_df = load_prediction_data()
    globe_df = load_globe_data()
    
    if eq_df is None:
        st.error("Failed to load earthquake data. Ensure data/coordinates.csv exists.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Filters")
        min_yr, max_yr = int(eq_df['year'].min()), int(eq_df['year'].max())
        year_range = st.slider("Year Range", min_yr, max_yr, (1980, 2010))
        mag_range = st.slider("Magnitude", 0.0, 10.0, (0.0, 10.0), 0.5)
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        
        filtered = eq_df[
            (eq_df['year'].between(*year_range)) &
            (eq_df['mag'].between(*mag_range))
        ]
        
        st.metric("Total Events", f"{len(filtered):,}")
        st.metric("Avg Magnitude", f"{filtered['mag'].mean():.2f}")
        st.metric("Max Magnitude", f"{filtered['mag'].max():.1f}")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "Globe", "Migration", "Heatmaps", "Predictions", "Data"
    ])
    
    # TAB 1: Overview
    with tab1:
        st.markdown('<p class="section-title">Earthquake Distribution</p>', unsafe_allow_html=True)
        
        # Map
        fig = go.Figure()
        
        if plates:
            for feat in plates['features']:
                coords = feat['geometry']['coordinates']
                fig.add_trace(go.Scattergeo(
                    lon=[c[0] for c in coords], lat=[c[1] for c in coords],
                    mode='lines', line=dict(width=1.5, color='#ef4444'),
                    name=feat['properties']['Name'], showlegend=False
                ))
        
        fig.add_trace(go.Scattergeo(
            lon=filtered['lon'], lat=filtered['lat'],
            mode='markers',
            marker=dict(
                size=filtered['mag'] * 1.5 + 3,
                color=filtered['mag'],
                colorscale='YlOrRd',
                colorbar=dict(title='Mag', thickness=15, len=0.5),
                opacity=0.7, line=dict(width=0.5, color='rgba(255,255,255,0.3)')
            ),
            text=filtered.apply(lambda x: f"M{x['mag']:.1f} | {x['time'].strftime('%Y-%m-%d')}", axis=1),
            hoverinfo='text', name='Events'
        ))
        
        fig.update_geos(
            projection_type="mercator",
            showland=True, landcolor='#18181b',
            showocean=True, oceancolor='#09090b',
            showcoastlines=True, coastlinecolor='#3f3f46',
            showcountries=True, countrycolor='#3f3f46',
            center=dict(lat=27, lon=82), projection_scale=6,
            bgcolor='rgba(0,0,0,0)'
        )
        fig.update_layout(**dark_layout(500))
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
        
        # Stats cards
        col1, col2, col3, col4 = st.columns(4)
        stats = [
            (f"{len(filtered):,}", "Total Events"),
            (f"{filtered['mag'].mean():.2f}", "Avg Magnitude"),
            (f"{filtered['mag'].max():.1f}", "Max Magnitude"),
            (f"{filtered['depth'].mean():.0f} km", "Avg Depth")
        ]
        for col, (val, label) in zip([col1, col2, col3, col4], stats):
            col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
        
        # Temporal trends
        st.markdown("---")
        st.markdown('<p class="section-title">Temporal Analysis</p>', unsafe_allow_html=True)
        
        yearly = filtered.groupby('year').agg({'mag': ['count', 'mean']}).reset_index()
        yearly.columns = ['year', 'count', 'avg_mag']
        
        fig2 = make_subplots(rows=1, cols=2, subplot_titles=['Annual Event Count', 'Average Magnitude Trend'])
        fig2.add_trace(go.Bar(x=yearly['year'], y=yearly['count'], marker_color='#0d9488', showlegend=False), row=1, col=1)
        fig2.add_trace(go.Scatter(x=yearly['year'], y=yearly['avg_mag'], mode='lines+markers', 
                                   line=dict(color='#ea580c', width=2), marker=dict(size=6, color='#ea580c'), showlegend=False), row=1, col=2)
        fig2.update_layout(**dark_layout(300))
        fig2.update_xaxes(gridcolor='#27272a', color='#9ca3af')
        fig2.update_yaxes(gridcolor='#27272a', color='#9ca3af')
        st.plotly_chart(fig2, use_container_width=True, config=CHART_CONFIG)
    
    # TAB 2: 3D Globe
    with tab2:
        # Globe header with gradient
        st.markdown("""
        <div style="background: linear-gradient(135deg, #0c0a09 0%, #1c1917 50%, #0c0a09 100%); 
                    padding: 2rem; border-radius: 16px; margin-bottom: 1.5rem; 
                    border: 1px solid #27272a; text-align: center;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);">
            <h2 style="color: #f4f4f5; margin: 0 0 0.5rem 0; font-size: 1.5rem; font-weight: 600; letter-spacing: -0.02em;">
                Interactive 3D Globe
            </h2>
            <p style="color: #6b7280; margin: 0; font-size: 0.875rem;">
                Seismic activity visualization with actual vs predicted stress zones
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Controls row
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1])
        
        with ctrl_col1:
            view_mode = st.selectbox(
                "Projection",
                ["Orthographic (3D Globe)", "Natural Earth", "Mercator"],
                index=0
            )
        
        with ctrl_col2:
            show_data = st.multiselect(
                "Show Layers",
                ["Earthquakes", "Predictions", "Plate Boundaries"],
                default=["Earthquakes", "Predictions", "Plate Boundaries"]
            )
        
        with ctrl_col3:
            color_scheme = st.selectbox(
                "Color Scheme",
                ["Inferno", "Plasma", "Viridis", "Hot", "Turbo"],
                index=0
            )
        
        # Map projection type
        proj_map = {
            "Orthographic (3D Globe)": "orthographic",
            "Natural Earth": "natural earth",
            "Mercator": "mercator"
        }
        
        # Create 3D globe with earthquake data
        globe_fig = go.Figure()
        
        # Add earthquake events to globe
        if "Earthquakes" in show_data:
            sample_size = min(3000, len(filtered))
            if sample_size > 0:
                globe_sample = filtered.sample(n=sample_size, random_state=42) if len(filtered) > sample_size else filtered
                
                globe_fig.add_trace(go.Scattergeo(
                    lon=globe_sample['lon'],
                    lat=globe_sample['lat'],
                    mode='markers',
                    marker=dict(
                        size=globe_sample['mag'] * 2.5 + 3,
                        color=globe_sample['mag'],
                        colorscale=color_scheme,
                        colorbar=dict(
                            title=dict(text='Magnitude', font=dict(size=12, color='#9ca3af')),
                            thickness=12, len=0.4, x=1.01,
                            bgcolor='rgba(20,20,24,0.8)',
                            bordercolor='#27272a', borderwidth=1,
                            tickfont=dict(color='#9ca3af', size=10)
                        ),
                        opacity=0.85,
                        line=dict(width=0.5, color='rgba(255,255,255,0.2)')
                    ),
                    text=globe_sample.apply(lambda x: f"<b>M {x['mag']:.1f}</b><br>{x['time'].strftime('%b %d, %Y')}<br>Depth: {x['depth']:.0f} km<br>Lat: {x['lat']:.2f}, Lon: {x['lon']:.2f}", axis=1),
                    hoverinfo='text',
                    name='Earthquakes'
                ))
        
        # Add prediction vs actual if available
        if "Predictions" in show_data and globe_df is not None:
            # Draw lines between actual and predicted first (so they appear behind markers)
            for _, row in globe_df.iterrows():
                globe_fig.add_trace(go.Scattergeo(
                    lon=[row['Actual_Lon'], row['Pred_Lon']],
                    lat=[row['Actual_Lat'], row['Pred_Lat']],
                    mode='lines',
                    line=dict(width=1.5, color='rgba(234, 88, 12, 0.4)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            globe_fig.add_trace(go.Scattergeo(
                lon=globe_df['Actual_Lon'],
                lat=globe_df['Actual_Lat'],
                mode='markers',
                marker=dict(size=12, color='#14b8a6', symbol='circle', 
                           line=dict(color='#fff', width=2),
                           opacity=0.95),
                name='Actual Location',
                text=globe_df.apply(lambda x: f"<b>ACTUAL</b><br>Lat: {x['Actual_Lat']:.3f}<br>Lon: {x['Actual_Lon']:.3f}", axis=1),
                hoverinfo='text'
            ))
            
            globe_fig.add_trace(go.Scattergeo(
                lon=globe_df['Pred_Lon'],
                lat=globe_df['Pred_Lat'],
                mode='markers',
                marker=dict(size=10, color='#ea580c', symbol='diamond', 
                           line=dict(color='#fff', width=2),
                           opacity=0.95),
                name='Predicted Location',
                text=globe_df.apply(lambda x: f"<b>PREDICTED</b><br>Lat: {x['Pred_Lat']:.3f}<br>Lon: {x['Pred_Lon']:.3f}", axis=1),
                hoverinfo='text'
            ))
        
        # Add plate boundaries
        if "Plate Boundaries" in show_data and plates:
            for feat in plates['features']:
                coords = feat['geometry']['coordinates']
                globe_fig.add_trace(go.Scattergeo(
                    lon=[c[0] for c in coords], lat=[c[1] for c in coords],
                    mode='lines', line=dict(width=2, color='#dc2626'),
                    name='Plate Boundary', showlegend=False, hoverinfo='skip'
                ))
        
        # Globe projection settings
        globe_fig.update_geos(
            projection_type=proj_map[view_mode],
            showland=True, landcolor='#1c1917',
            showocean=True, oceancolor='#09090b',
            showcoastlines=True, coastlinecolor='#52525b',
            coastlinewidth=1,
            showcountries=True, countrycolor='#3f3f46',
            countrywidth=0.5,
            showlakes=True, lakecolor='#09090b',
            showrivers=False,
            projection_rotation=dict(lon=82, lat=22, roll=0),
            bgcolor='rgba(0,0,0,0)',
            lataxis=dict(range=[5, 45]) if view_mode != "Orthographic (3D Globe)" else None,
            lonaxis=dict(range=[60, 105]) if view_mode != "Orthographic (3D Globe)" else None,
        )
        
        globe_fig.update_layout(
            height=650,
            paper_bgcolor='rgba(12,10,9,1)',
            plot_bgcolor='rgba(12,10,9,1)',
            font=dict(family='Inter', color='#f4f4f5', size=11),
            margin=dict(l=0, r=0, t=10, b=10),
            legend=dict(
                orientation='h', y=-0.02, x=0.5, xanchor='center',
                bgcolor='rgba(28,25,23,0.95)', bordercolor='#3f3f46', borderwidth=1,
                font=dict(color='#f4f4f5', size=11),
                itemsizing='constant'
            ),
            hoverlabel=dict(
                bgcolor='rgba(28,28,34,0.95)', 
                font_size=12, 
                bordercolor='#52525b', 
                font_color='#f4f4f5',
                font_family='Inter'
            )
        )
        
        st.plotly_chart(globe_fig, use_container_width=True, config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['orbitRotation', 'resetGeo'],
            'scrollZoom': True
        })
        
        # Stats row below globe
        st.markdown("---")
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #ea580c;">{len(filtered):,}</div>
                <div class="metric-label">Events Displayed</div>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_col2:
            if globe_df is not None:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #14b8a6;">{len(globe_df)}</div>
                    <div class="metric-label">Prediction Points</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">--</div>
                    <div class="metric-label">Prediction Points</div>
                </div>
                """, unsafe_allow_html=True)
        
        with stat_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #d97706;">{filtered['mag'].max():.1f}</div>
                <div class="metric-label">Max Magnitude</div>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{year_range[0]}-{year_range[1]}</div>
                <div class="metric-label">Time Period</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Instructions
        with st.expander("How to use the 3D Globe"):
            st.markdown("""
            <div style="color: #9ca3af; font-size: 0.875rem; line-height: 1.8;">
                <p><strong style="color: #f4f4f5;">Navigation:</strong></p>
                <ul style="margin-left: 1rem;">
                    <li><strong>Rotate:</strong> Click and drag on the globe</li>
                    <li><strong>Zoom:</strong> Use mouse scroll wheel or pinch gesture</li>
                    <li><strong>Reset:</strong> Double-click to reset view</li>
                </ul>
                <p style="margin-top: 1rem;"><strong style="color: #f4f4f5;">Legend:</strong></p>
                <ul style="margin-left: 1rem;">
                    <li><span style="color: #14b8a6;">&#9679;</span> <strong>Teal circles:</strong> Actual earthquake cluster centroids</li>
                    <li><span style="color: #ea580c;">&#9670;</span> <strong>Orange diamonds:</strong> Model predicted locations</li>
                    <li><span style="color: #dc2626;">&#8212;</span> <strong>Red lines:</strong> Tectonic plate boundaries</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # TAB 3: Migration
    with tab3:
        st.markdown('<p class="section-title">Stress Migration Patterns</p>', unsafe_allow_html=True)
        
        if dir_df is not None:
            dir_filtered = dir_df[(dir_df['Year'] >= year_range[0]) & (dir_df['Year'] <= year_range[1])]
            
            mig_fig = go.Figure()
            
            if plates:
                for feat in plates['features']:
                    coords = feat['geometry']['coordinates']
                    mig_fig.add_trace(go.Scattergeo(
                        lon=[c[0] for c in coords], lat=[c[1] for c in coords],
                        mode='lines', line=dict(width=1, color='rgba(239,68,68,0.4)'),
                        showlegend=False, hoverinfo='skip'
                    ))
            
            for _, row in dir_filtered.iterrows():
                mig_fig.add_trace(go.Scattergeo(
                    lon=[row['From_Lon'], row['To_Lon']],
                    lat=[row['From_Lat'], row['To_Lat']],
                    mode='lines+markers',
                    line=dict(width=2, color='#14b8a6'),
                    marker=dict(size=[6, 10], symbol=['circle', 'triangle-up'], color='#14b8a6'),
                    text=f"{int(row['Year'])}: {row['Direction']}",
                    hoverinfo='text', showlegend=False
                ))
            
            mig_fig.update_geos(
                projection_type="mercator",
                showland=True, landcolor='#18181b',
                showocean=True, oceancolor='#09090b',
                center=dict(lat=27, lon=82), projection_scale=6,
                bgcolor='rgba(0,0,0,0)'
            )
            mig_fig.update_layout(**dark_layout(450))
            st.plotly_chart(mig_fig, use_container_width=True, config=CHART_CONFIG)
            
            # Direction stats
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<p class="section-title">Direction Distribution</p>', unsafe_allow_html=True)
                dir_counts = dir_filtered['Direction'].value_counts()
                seismic_colors = ['#ea580c', '#0d9488', '#d97706', '#059669', '#dc2626', '#ca8a04', '#14b8a6', '#c2410c']
                pie = go.Figure(data=[go.Pie(
                    labels=dir_counts.index, values=dir_counts.values,
                    hole=0.45, marker=dict(colors=seismic_colors, line=dict(color='#18181b', width=2)),
                    textinfo='label+percent', textfont=dict(size=10, color='#f4f4f5')
                )])
                pie.update_layout(**dark_layout(300))
                st.plotly_chart(pie, use_container_width=True, config=CHART_CONFIG)
            
            with col2:
                st.markdown('<p class="section-title">Summary</p>', unsafe_allow_html=True)
                st.metric("Total Migrations", len(dir_filtered))
                if len(dir_counts) > 0:
                    st.metric("Primary Direction", dir_counts.index[0])
                st.metric("Period", f"{int(dir_filtered['Year'].min())} - {int(dir_filtered['Year'].max())}")
        else:
            st.info("Migration data not available")
    
    # TAB 4: Heatmaps Gallery
    with tab4:
        st.markdown('<p class="section-title">Stress Heatmap Visualizations</p>', unsafe_allow_html=True)
        
        heatmap_type = st.radio(
            "Select Heatmap Type",
            ["Training Period (1980-2005)", "Prediction Period (2006-2011)"],
            horizontal=True
        )
        
        if heatmap_type == "Training Period (1980-2005)":
            folder = "yearly_heatmaps"
            files = get_heatmap_files(folder)
            
            if files:
                st.markdown(f"**{len(files)} heatmaps available** - Showing earthquake density by year")
                
                # Year selector
                years = [int(f.stem.split('_')[-1]) for f in files]
                
                view_mode = st.radio("View Mode", ["Single Year", "Gallery"], horizontal=True)
                
                if view_mode == "Single Year":
                    selected_year = st.select_slider("Select Year", options=years)
                    selected_file = [f for f in files if str(selected_year) in f.stem][0]
                    
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col2:
                        st.image(str(selected_file), caption=f"Stress Distribution - {selected_year}", use_container_width=True)
                else:
                    # Gallery view
                    cols = st.columns(4)
                    for i, f in enumerate(files):
                        with cols[i % 4]:
                            year = f.stem.split('_')[-1]
                            st.image(str(f), caption=year, use_container_width=True)
            else:
                st.warning("No heatmap files found in yearly_heatmaps/")
        
        else:
            folder = "yearly_stress_predictions"
            files = get_heatmap_files(folder)
            
            if files:
                st.markdown(f"**{len(files)} prediction comparisons** - Actual vs Predicted stress zones")
                
                years = [int(f.stem.split('_')[-1]) for f in files]
                
                view_mode = st.radio("View Mode", ["Single Year", "Gallery"], horizontal=True, key="pred_view")
                
                if view_mode == "Single Year":
                    selected_year = st.select_slider("Select Year", options=years, key="pred_year")
                    selected_file = [f for f in files if str(selected_year) in f.stem][0]
                    
                    st.image(str(selected_file), caption=f"Actual vs Predicted - {selected_year}", use_container_width=True)
                else:
                    cols = st.columns(3)
                    for i, f in enumerate(files):
                        with cols[i % 3]:
                            year = f.stem.split('_')[-1]
                            st.image(str(f), caption=year, use_container_width=True)
            else:
                st.warning("No prediction heatmaps found in yearly_stress_predictions/")
        
        # Additional heatmap folders
        st.markdown("---")
        with st.expander("Additional Visualizations"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Stress Migration Maps**")
                mig_files = get_heatmap_files("stress_migration_maps")
                if mig_files:
                    selected = st.selectbox("Select", [f.stem for f in mig_files], key="mig_select")
                    sel_file = [f for f in mig_files if f.stem == selected][0]
                    st.image(str(sel_file), use_container_width=True)
            
            with col2:
                st.markdown("**Energy Transfer Plots**")
                energy_files = get_heatmap_files("energy_transfer_plots")
                if energy_files:
                    selected = st.selectbox("Select", [f.stem for f in energy_files], key="energy_select")
                    sel_file = [f for f in energy_files if f.stem == selected][0]
                    st.image(str(sel_file), use_container_width=True)
    
    # TAB 5: Predictions
    with tab5:
        st.markdown('<p class="section-title">XGBoost Prediction Model</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1c1917 0%, #141418 100%); padding: 1.25rem; border-radius: 10px; border-left: 3px solid #ea580c; margin-bottom: 1.5rem; border: 1px solid #27272a;">
            <p style="color: #9ca3af; margin: 0; font-size: 0.85rem; line-height: 1.6;">
                <strong style="color: #f4f4f5;">Model:</strong> XGBoost classifier trained on 1980-2005 migration data. 
                Predicts next stress zone location on an 8x8 grid covering the Indian subcontinent.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner('Running model...'):
            results, metrics, *grid_params = run_prediction_model()
        
        if results is not None:
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            metric_data = [
                (str(metrics['total']), "Test Events"),
                (f"{metrics['top1']:.1f}%", "Exact Match"),
                (f"{metrics['top3']:.1f}%", "Top 3 Accuracy"),
                (f"{metrics['top5']:.1f}%", "Top 5 Accuracy")
            ]
            colors = ['#f4f4f5', '#059669', '#d97706', '#0d9488']
            for col, (val, label), color in zip([col1, col2, col3, col4], metric_data, colors):
                col.markdown(f'<div class="metric-card"><div class="metric-value" style="color: {color};">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Map
            st.markdown('<p class="section-title">Prediction Results Map</p>', unsafe_allow_html=True)
            
            pred_fig = go.Figure()
            
            # Lines
            for _, row in results.iterrows():
                color = '#059669' if row['Rank'] == 1 else '#d97706' if row['Rank'] <= 3 else '#dc2626'
                pred_fig.add_trace(go.Scattergeo(
                    lon=[row['Real_Lon'], row['Pred_Lon']], lat=[row['Real_Lat'], row['Pred_Lat']],
                    mode='lines', line=dict(width=1.5, color=color, dash='dot'),
                    opacity=0.6, showlegend=False, hoverinfo='skip'
                ))
            
            # Actual
            pred_fig.add_trace(go.Scattergeo(
                lon=results['Real_Lon'], lat=results['Real_Lat'],
                mode='markers', marker=dict(size=10, color='#14b8a6', symbol='circle', line=dict(color='#fff', width=1)),
                name='Actual', text=results.apply(lambda x: f"Actual | {int(x['Year'])} | Sector {int(x['Real_Sector'])}", axis=1),
                hoverinfo='text'
            ))
            
            # Predicted
            pred_fig.add_trace(go.Scattergeo(
                lon=results['Pred_Lon'], lat=results['Pred_Lat'],
                mode='markers', marker=dict(size=8, color='#ea580c', symbol='x', line=dict(width=2)),
                name='Predicted', text=results.apply(lambda x: f"Predicted | Conf: {x['Confidence']:.1f}% | Rank: {int(x['Rank'])}", axis=1),
                hoverinfo='text'
            ))
            
            pred_fig.update_geos(
                projection_type="natural earth",
                showland=True, landcolor='#18181b',
                showocean=True, oceancolor='#09090b',
                center=dict(lat=25, lon=82), projection_scale=4,
                lonaxis=dict(range=[65, 100]), lataxis=dict(range=[15, 35]),
                bgcolor='rgba(0,0,0,0)'
            )
            pred_fig.update_layout(**dark_layout(450),
                                    legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center', bgcolor='rgba(0,0,0,0)', font=dict(color='#fff')))
            st.plotly_chart(pred_fig, use_container_width=True, config=CHART_CONFIG)
            
            # Yearly breakdown
            st.markdown('<p class="section-title">Performance by Year</p>', unsafe_allow_html=True)
            yearly_perf = results.groupby('Year').apply(
                lambda x: pd.Series({'Total': len(x), 'Correct': (x['Rank'] == 1).sum(), 'Top3': (x['Rank'] <= 3).sum()}),
                include_groups=False
            ).reset_index()
            
            perf_fig = go.Figure()
            perf_fig.add_trace(go.Bar(x=yearly_perf['Year'], y=yearly_perf['Total'], name='Total', marker_color='#27272a'))
            perf_fig.add_trace(go.Bar(x=yearly_perf['Year'], y=yearly_perf['Top3'], name='Top 3', marker_color='#d97706'))
            perf_fig.add_trace(go.Bar(x=yearly_perf['Year'], y=yearly_perf['Correct'], name='Exact', marker_color='#059669'))
            perf_fig.update_layout(**dark_layout(280), barmode='overlay', 
                                    legend=dict(orientation='h', y=1.1, x=0.5, xanchor='center', bgcolor='rgba(0,0,0,0)'))
            perf_fig.update_xaxes(gridcolor='#27272a', color='#9ca3af', dtick=1)
            perf_fig.update_yaxes(gridcolor='#27272a', color='#9ca3af')
            st.plotly_chart(perf_fig, use_container_width=True, config=CHART_CONFIG)
        else:
            st.warning("Could not run prediction model. Check data files.")
    
    # TAB 6: Data
    with tab6:
        st.markdown('<p class="section-title">Data Explorer</p>', unsafe_allow_html=True)
        
        # Sub-tabs for data section
        data_view = st.radio("View", ["3D Visualization", "Data Tables"], horizontal=True)
        
        if data_view == "3D Visualization":
            st.markdown("""
            <div style="background: linear-gradient(135deg, #0c0a09 0%, #1c1917 50%, #0c0a09 100%); 
                        padding: 1.5rem; border-radius: 12px; margin: 1rem 0; 
                        border: 1px solid #27272a; text-align: center;">
                <h3 style="color: #f4f4f5; margin: 0 0 0.5rem 0; font-size: 1.25rem; font-weight: 600;">
                    3D Earthquake Distribution
                </h3>
                <p style="color: #6b7280; margin: 0; font-size: 0.8rem;">
                    Latitude × Longitude × Depth visualization with magnitude coloring
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Controls
            ctrl1, ctrl2, ctrl3 = st.columns(3)
            with ctrl1:
                max_points = st.slider("Max Points", 500, 5000, 2000, 100)
            with ctrl2:
                color_by = st.selectbox("Color by", ["Magnitude", "Depth", "Year"])
            with ctrl3:
                point_size = st.slider("Point Size", 2, 10, 4)
            
            # Sample data
            sample_3d = filtered.sample(n=min(max_points, len(filtered)), random_state=42) if len(filtered) > max_points else filtered
            
            # Create 3D scatter plot
            color_col = {'Magnitude': 'mag', 'Depth': 'depth', 'Year': 'year'}[color_by]
            color_scale = {'Magnitude': 'Inferno', 'Depth': 'Viridis', 'Year': 'Plasma'}[color_by]
            
            fig_3d = go.Figure(data=[go.Scatter3d(
                x=sample_3d['lon'],
                y=sample_3d['lat'],
                z=-sample_3d['depth'],  # Negative so deeper = lower
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=sample_3d[color_col],
                    colorscale=color_scale,
                    colorbar=dict(
                        title=dict(text=color_by, font=dict(size=12, color='#9ca3af')),
                        thickness=15, len=0.6,
                        bgcolor='rgba(20,20,24,0.9)',
                        bordercolor='#3f3f46', borderwidth=1,
                        tickfont=dict(color='#9ca3af', size=10)
                    ),
                    opacity=0.8,
                    line=dict(width=0.5, color='rgba(255,255,255,0.1)')
                ),
                text=sample_3d.apply(
                    lambda x: f"<b>M {x['mag']:.1f}</b><br>"
                              f"Date: {x['time'].strftime('%Y-%m-%d')}<br>"
                              f"Lat: {x['lat']:.3f}<br>"
                              f"Lon: {x['lon']:.3f}<br>"
                              f"Depth: {x['depth']:.1f} km", axis=1
                ),
                hoverinfo='text',
                name='Earthquakes'
            )])
            
            fig_3d.update_layout(
                height=650,
                paper_bgcolor='rgba(12,10,9,1)',
                plot_bgcolor='rgba(12,10,9,1)',
                font=dict(family='Inter', color='#f4f4f5', size=11),
                margin=dict(l=0, r=0, t=30, b=0),
                scene=dict(
                    xaxis=dict(
                        title=dict(text='Longitude', font=dict(color='#f4f4f5')),
                        backgroundcolor='rgba(20,20,24,0.8)',
                        gridcolor='#27272a',
                        showbackground=True,
                        tickfont=dict(color='#9ca3af')
                    ),
                    yaxis=dict(
                        title=dict(text='Latitude', font=dict(color='#f4f4f5')),
                        backgroundcolor='rgba(20,20,24,0.8)',
                        gridcolor='#27272a',
                        showbackground=True,
                        tickfont=dict(color='#9ca3af')
                    ),
                    zaxis=dict(
                        title=dict(text='Depth (km)', font=dict(color='#f4f4f5')),
                        backgroundcolor='rgba(20,20,24,0.8)',
                        gridcolor='#27272a',
                        showbackground=True,
                        tickfont=dict(color='#9ca3af')
                    ),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.2)
                    ),
                    aspectmode='manual',
                    aspectratio=dict(x=1.5, y=1, z=0.7)
                ),
                hoverlabel=dict(
                    bgcolor='rgba(28,28,34,0.95)', 
                    font_size=12, 
                    bordercolor='#52525b', 
                    font_color='#f4f4f5',
                    font_family='Inter'
                )
            )
            
            st.plotly_chart(fig_3d, use_container_width=True, config={
                'displayModeBar': True,
                'displaylogo': False,
                'scrollZoom': True
            })
            
            # Stats below 3D plot
            st.markdown("---")
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #ea580c;">{len(sample_3d):,}</div>
                    <div class="metric-label">Points Shown</div>
                </div>
                """, unsafe_allow_html=True)
            with s2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #14b8a6;">{sample_3d['depth'].mean():.1f} km</div>
                    <div class="metric-label">Avg Depth</div>
                </div>
                """, unsafe_allow_html=True)
            with s3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #d97706;">{sample_3d['depth'].max():.1f} km</div>
                    <div class="metric-label">Max Depth</div>
                </div>
                """, unsafe_allow_html=True)
            with s4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{sample_3d['mag'].mean():.2f}</div>
                    <div class="metric-label">Avg Magnitude</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Instructions
            with st.expander("How to navigate the 3D plot"):
                st.markdown("""
                <div style="color: #9ca3af; font-size: 0.875rem; line-height: 1.8;">
                    <ul style="margin-left: 1rem;">
                        <li><strong>Rotate:</strong> Click and drag to rotate the view</li>
                        <li><strong>Zoom:</strong> Scroll to zoom in/out</li>
                        <li><strong>Pan:</strong> Right-click and drag to pan</li>
                        <li><strong>Reset:</strong> Double-click to reset camera</li>
                    </ul>
                    <p style="margin-top: 0.5rem; color: #6b7280;">
                        <em>Note: Depth axis is inverted (deeper earthquakes appear lower)</em>
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        else:  # Data Tables
            st.markdown("---")
            data_type = st.radio("Dataset", ["Earthquakes", "Migrations", "Predictions"], horizontal=True, key="data_tables")
            
            if data_type == "Earthquakes":
                st.dataframe(filtered[['time', 'lat', 'lon', 'depth', 'mag', 'year']].sort_values('time', ascending=False), 
                            use_container_width=True, height=400)
                csv = filtered.to_csv(index=False)
                st.download_button("Download CSV", csv, "earthquake_data.csv", "text/csv")
            
            elif data_type == "Migrations" and dir_df is not None:
                st.dataframe(dir_df, use_container_width=True, height=400)
                csv = dir_df.to_csv(index=False)
                st.download_button("Download CSV", csv, "migration_data.csv", "text/csv")
            
            elif data_type == "Predictions" and pred_df is not None:
                st.dataframe(pred_df, use_container_width=True, height=400)
                csv = pred_df.to_csv(index=False)
                st.download_button("Download CSV", csv, "prediction_data.csv", "text/csv")


if __name__ == "__main__":
    main()
