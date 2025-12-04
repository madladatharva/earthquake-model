"""
Earthquake Stress Migration Analysis Dashboard
Professional visualization of seismic activity patterns and XGBoost predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Earthquake Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Dark Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --bg-primary: #0d1117;
        --bg-secondary: #161b22;
        --bg-card: #21262d;
        --text-primary: #e6edf3;
        --text-secondary: #8b949e;
        --accent: #58a6ff;
        --border: #30363d;
        --success: #3fb950;
        --warning: #d29922;
        --danger: #f85149;
    }
    
    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 100%;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Header */
    .dashboard-header {
        background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    
    .dashboard-header h1 {
        font-size: 1.75rem;
        font-weight: 600;
        color: white;
        margin: 0 0 0.5rem 0;
    }
    
    .dashboard-header p {
        font-size: 0.95rem;
        color: rgba(255,255,255,0.85);
        margin: 0;
    }
    
    /* Cards */
    .stat-card {
        background: var(--bg-card);
        padding: 1.25rem;
        border-radius: 8px;
        border: 1px solid var(--border);
    }
    
    .stat-value {
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.25rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-secondary);
        padding: 4px;
        border-radius: 8px;
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 6px;
        padding: 10px 20px;
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--accent) !important;
        color: white !important;
    }
    
    /* Section headers */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: var(--accent);
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-secondary); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# Data Loading
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
    except:
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
def run_prediction_model():
    """Run XGBoost stress zone prediction"""
    try:
        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder
        
        GRID_DIM = 8
        LAT_MIN, LAT_MAX = 15, 35
        LON_MIN, LON_MAX = 65, 100
        
        def get_sector_id(lat, lon):
            if lat < LAT_MIN or lat > LAT_MAX or lon < LON_MIN or lon > LON_MAX:
                return -1
            y_idx = min(int((lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * GRID_DIM), GRID_DIM - 1)
            x_idx = min(int((lon - LON_MIN) / (LON_MAX - LON_MIN) * GRID_DIM), GRID_DIM - 1)
            return y_idx * GRID_DIM + x_idx
        
        def get_sector_center(sector_id):
            y_idx, x_idx = sector_id // GRID_DIM, sector_id % GRID_DIM
            lat = LAT_MIN + (y_idx + 0.5) * (LAT_MAX - LAT_MIN) / GRID_DIM
            lon = LON_MIN + (x_idx + 0.5) * (LON_MAX - LON_MIN) / GRID_DIM
            return lat, lon
        
        def create_features(df, lookback=3):
            features_list = []
            df_sorted = df.sort_values('From_Time').reset_index(drop=True)
            
            for i in range(lookback, len(df_sorted)):
                current = df_sorted.iloc[i]
                target = get_sector_id(current['To_Lat'], current['To_Lon'])
                source = get_sector_id(current['From_Lat'], current['From_Lon'])
                if target == -1 or source == -1:
                    continue
                
                features = {
                    'Year': current['Year'],
                    'Month': current['From_Time'].month,
                    'Current_Sector': source,
                    'Current_Bearing': current['Bearing_Degrees'],
                    'Current_Angle': current['Angle_wrt_X_Axis']
                }
                
                valid = True
                for lag in range(1, lookback + 1):
                    past = df_sorted.iloc[i - lag]
                    past_sector = get_sector_id(past['To_Lat'], past['To_Lon'])
                    if past_sector == -1:
                        valid = False
                        break
                    features[f'Lag{lag}_Sector'] = past_sector
                    features[f'Lag{lag}_Bearing'] = past['Bearing_Degrees']
                    features[f'Lag{lag}_Angle'] = past['Angle_wrt_X_Axis']
                    features[f'Lag{lag}_DaysAgo'] = max((current['From_Time'] - past['From_Time']).days, 1)
                
                if valid:
                    features['Target_Sector'] = target
                    features_list.append(features)
            
            return pd.DataFrame(features_list)
        
        train_df = pd.read_csv('data/cluster_directions_final.csv')
        train_df['From_Time'] = pd.to_datetime(train_df['From_Time'], format='mixed')
        train_df = train_df[train_df['Year'] <= 2005].sort_values('From_Time')
        
        test_df = pd.read_csv('data/test_set.csv')
        test_df['From_Time'] = pd.to_datetime(test_df['From_Time'], format='mixed')
        test_df = test_df.sort_values('From_Time')
        
        train_features = create_features(train_df)
        test_features = create_features(test_df)
        
        X_train = train_features.drop(columns=['Target_Sector', 'Year'])
        y_train = train_features['Target_Sector']
        X_test = test_features.drop(columns=['Target_Sector', 'Year'])
        y_test = test_features['Target_Sector']
        
        le = LabelEncoder()
        le.fit(pd.concat([y_train, y_test]))
        y_train_enc, y_test_enc = le.transform(y_train), le.transform(y_test)
        
        model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbosity=0)
        model.fit(X_train, y_train_enc)
        
        probs = model.predict_proba(X_test)
        results = []
        hits = {'top1': 0, 'top3': 0, 'top5': 0}
        
        for i in range(len(X_test)):
            real_idx = y_test_enc[i]
            ranked = np.argsort(probs[i])[::-1]
            rank = np.where(ranked == real_idx)[0][0] + 1 if real_idx in ranked else 99
            
            if rank == 1: hits['top1'] += 1
            if rank <= 3: hits['top3'] += 1
            if rank <= 5: hits['top5'] += 1
            
            pred_sector = le.inverse_transform([ranked[0]])[0]
            real_sector = y_test.iloc[i]
            
            results.append({
                'Year': test_features.iloc[i]['Year'],
                'Real_Sector': real_sector,
                'Pred_Sector': pred_sector,
                'Rank': rank,
                'Real_Lat': get_sector_center(real_sector)[0],
                'Real_Lon': get_sector_center(real_sector)[1],
                'Pred_Lat': get_sector_center(pred_sector)[0],
                'Pred_Lon': get_sector_center(pred_sector)[1],
                'Confidence': probs[i][ranked[0]] * 100
            })
        
        total = len(X_test)
        metrics = {
            'total': total,
            'top1_pct': hits['top1'] / total * 100,
            'top3_pct': hits['top3'] / total * 100,
            'top5_pct': hits['top5'] / total * 100
        }
        
        return pd.DataFrame(results), metrics, GRID_DIM, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
    except Exception as e:
        return None, None, None, None, None, None, None

# Chart configuration
CHART_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
    'toImageButtonOptions': {'format': 'png', 'scale': 2}
}

def dark_layout():
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(33, 38, 45, 0.8)',
        font=dict(family='Inter, sans-serif', color='#e6edf3', size=12),
        margin=dict(l=60, r=30, t=50, b=50),
        hoverlabel=dict(bgcolor='#21262d', font_size=12, bordercolor='#30363d')
    )

# Main App
def main():
    # Header
    st.markdown("""
    <div class="dashboard-header">
        <h1>Earthquake Stress Migration Analysis</h1>
        <p>Seismic activity patterns and XGBoost-based zone predictions for the Indian subcontinent</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    earthquake_df = load_earthquake_data()
    directions_df = load_cluster_directions()
    plate_data = load_plate_boundaries()
    pred_df, grid_df = load_prediction_data()
    
    if earthquake_df is None:
        st.error("Failed to load data. Please ensure CSV files are in the data/ folder.")
        return
    
    # Sidebar filters
    with st.sidebar:
        st.markdown("### Filters")
        
        min_yr, max_yr = int(earthquake_df['year'].min()), int(earthquake_df['year'].max())
        year_range = st.slider("Year Range", min_yr, max_yr, (1980, 2010))
        mag_range = st.slider("Magnitude", 0.0, 10.0, (0.0, 10.0), 0.5)
        
        st.markdown("---")
        st.markdown("### Summary")
        
        filtered = earthquake_df[
            (earthquake_df['year'].between(*year_range)) &
            (earthquake_df['mag'].between(*mag_range))
        ]
        
        st.metric("Events", f"{len(filtered):,}")
        st.metric("Avg Magnitude", f"{filtered['mag'].mean():.2f}")
        st.metric("Max Magnitude", f"{filtered['mag'].max():.1f}")
        st.metric("Avg Depth", f"{filtered['depth'].mean():.0f} km")
    
    # Main tabs - simplified to 4 essential views
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Migration Analysis", "Predictions", "Data"])
    
    # TAB 1: Overview
    with tab1:
        st.markdown('<p class="section-title">Earthquake Distribution Map</p>', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        # Plate boundaries
        if plate_data:
            for feature in plate_data['features']:
                coords = feature['geometry']['coordinates']
                fig.add_trace(go.Scattergeo(
                    lon=[c[0] for c in coords], lat=[c[1] for c in coords],
                    mode='lines', line=dict(width=2, color='#f85149'),
                    name=feature['properties']['Name'], showlegend=False, hoverinfo='name'
                ))
        
        # Earthquakes
        fig.add_trace(go.Scattergeo(
            lon=filtered['lon'], lat=filtered['lat'],
            mode='markers',
            marker=dict(
                size=filtered['mag'] * 2 + 3,
                color=filtered['mag'],
                colorscale='YlOrRd',
                colorbar=dict(title='Mag', tickfont=dict(color='#8b949e')),
                opacity=0.7,
                line=dict(width=0.5, color='rgba(255,255,255,0.3)')
            ),
            text=filtered.apply(lambda x: f"M{x['mag']:.1f} | {x['time'].strftime('%Y-%m-%d')} | Depth: {x['depth']:.0f}km", axis=1),
            hoverinfo='text', name='Earthquakes'
        ))
        
        fig.update_geos(
            projection_type="mercator",
            showland=True, landcolor='#21262d',
            showocean=True, oceancolor='#0d1117',
            showcoastlines=True, coastlinecolor='#30363d',
            showcountries=True, countrycolor='#30363d',
            center=dict(lat=27, lon=82), projection_scale=6,
            bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_layout(
            height=550, margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(33,38,45,0.9)', font=dict(color='#e6edf3'))
        )
        
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
        
        # Stats row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{len(filtered):,}</div><div class="stat-label">Total Events</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{filtered["mag"].mean():.2f}</div><div class="stat-label">Avg Magnitude</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{filtered["mag"].max():.1f}</div><div class="stat-label">Max Magnitude</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{filtered["depth"].mean():.0f} km</div><div class="stat-label">Avg Depth</div></div>', unsafe_allow_html=True)
        
        # Temporal analysis
        st.markdown("---")
        st.markdown('<p class="section-title">Temporal Trends</p>', unsafe_allow_html=True)
        
        yearly = filtered.groupby('year').agg({'mag': ['count', 'mean']}).reset_index()
        yearly.columns = ['year', 'count', 'avg_mag']
        
        fig2 = make_subplots(rows=1, cols=2, subplot_titles=['Events per Year', 'Average Magnitude Trend'])
        
        fig2.add_trace(go.Bar(x=yearly['year'], y=yearly['count'], marker=dict(color=yearly['count'], colorscale='Blues'), showlegend=False), row=1, col=1)
        fig2.add_trace(go.Scatter(x=yearly['year'], y=yearly['avg_mag'], mode='lines+markers', line=dict(color='#58a6ff', width=2), marker=dict(size=6), showlegend=False), row=1, col=2)
        
        fig2.update_layout(**dark_layout(), height=350, showlegend=False)
        fig2.update_xaxes(gridcolor='rgba(48,54,61,0.5)', color='#8b949e')
        fig2.update_yaxes(gridcolor='rgba(48,54,61,0.5)', color='#8b949e')
        
        st.plotly_chart(fig2, use_container_width=True, config=CHART_CONFIG)
    
    # TAB 2: Migration Analysis
    with tab2:
        st.markdown('<p class="section-title">Stress Migration Patterns</p>', unsafe_allow_html=True)
        
        if directions_df is not None:
            dir_filtered = directions_df[(directions_df['Year'] >= year_range[0]) & (directions_df['Year'] <= year_range[1])]
            
            # Migration map
            mig_fig = go.Figure()
            
            if plate_data:
                for feature in plate_data['features']:
                    coords = feature['geometry']['coordinates']
                    mig_fig.add_trace(go.Scattergeo(
                        lon=[c[0] for c in coords], lat=[c[1] for c in coords],
                        mode='lines', line=dict(width=1.5, color='rgba(248, 81, 73, 0.5)'),
                        showlegend=False, hoverinfo='skip'
                    ))
            
            # Migration arrows
            for _, row in dir_filtered.iterrows():
                mig_fig.add_trace(go.Scattergeo(
                    lon=[row['From_Lon'], row['To_Lon']],
                    lat=[row['From_Lat'], row['To_Lat']],
                    mode='lines+markers',
                    line=dict(width=2, color='#58a6ff'),
                    marker=dict(size=[8, 12], symbol=['circle', 'triangle-up'], color='#58a6ff'),
                    text=f"{row['Year']}: {row['Direction']}",
                    hoverinfo='text', showlegend=False
                ))
            
            mig_fig.update_geos(
                projection_type="mercator",
                showland=True, landcolor='#21262d',
                showocean=True, oceancolor='#0d1117',
                center=dict(lat=27, lon=82), projection_scale=6,
                bgcolor='rgba(0,0,0,0)'
            )
            
            mig_fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(mig_fig, use_container_width=True, config=CHART_CONFIG)
            
            # Direction distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<p class="section-title">Direction Distribution</p>', unsafe_allow_html=True)
                dir_counts = dir_filtered['Direction'].value_counts()
                
                pie_fig = go.Figure(data=[go.Pie(
                    labels=dir_counts.index, values=dir_counts.values,
                    hole=0.4, marker=dict(colors=px.colors.qualitative.Set2),
                    textinfo='label+percent', textposition='outside',
                    textfont=dict(color='#e6edf3', size=11)
                )])
                pie_fig.update_layout(**dark_layout(), height=350, showlegend=False)
                st.plotly_chart(pie_fig, use_container_width=True, config=CHART_CONFIG)
            
            with col2:
                st.markdown('<p class="section-title">Migration Statistics</p>', unsafe_allow_html=True)
                st.metric("Total Migrations", len(dir_filtered))
                st.metric("Most Common Direction", dir_counts.index[0] if len(dir_counts) > 0 else "N/A")
                st.metric("Years Covered", f"{dir_filtered['Year'].min():.0f} - {dir_filtered['Year'].max():.0f}")
        else:
            st.info("Migration data not available.")
    
    # TAB 3: Predictions
    with tab3:
        st.markdown('<p class="section-title">XGBoost Stress Zone Predictions</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: rgba(88, 166, 255, 0.1); padding: 1rem; border-radius: 8px; border: 1px solid rgba(88, 166, 255, 0.3); margin-bottom: 1.5rem;">
            <p style="color: #8b949e; margin: 0; font-size: 0.9rem;">
                <strong style="color: #58a6ff;">Model:</strong> XGBoost classifier trained on 1980-2005 data, tested on 2006-2011. 
                Predicts which sector in an 8×8 grid will experience the next stress zone migration.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner('Running prediction model...'):
            results_df, metrics, GRID_DIM, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = run_prediction_model()
        
        if results_df is not None:
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="stat-card"><div class="stat-value">{metrics["total"]}</div><div class="stat-label">Test Events</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="stat-card"><div class="stat-value" style="color: #3fb950;">{metrics["top1_pct"]:.1f}%</div><div class="stat-label">Exact Match</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="stat-card"><div class="stat-value" style="color: #d29922;">{metrics["top3_pct"]:.1f}%</div><div class="stat-label">Top 3 Accuracy</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="stat-card"><div class="stat-value" style="color: #58a6ff;">{metrics["top5_pct"]:.1f}%</div><div class="stat-label">Top 5 Accuracy</div></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Prediction map
            st.markdown('<p class="section-title">Actual vs Predicted Zones</p>', unsafe_allow_html=True)
            
            pred_fig = go.Figure()
            
            # Connection lines
            for _, row in results_df.iterrows():
                color = '#3fb950' if row['Rank'] == 1 else '#d29922' if row['Rank'] <= 3 else '#f85149'
                pred_fig.add_trace(go.Scattergeo(
                    lon=[row['Real_Lon'], row['Pred_Lon']], lat=[row['Real_Lat'], row['Pred_Lat']],
                    mode='lines', line=dict(width=1.5, color=color, dash='dot'),
                    opacity=0.6, showlegend=False, hoverinfo='skip'
                ))
            
            # Actual points
            pred_fig.add_trace(go.Scattergeo(
                lon=results_df['Real_Lon'], lat=results_df['Real_Lat'],
                mode='markers', marker=dict(size=12, color='#58a6ff', symbol='circle', line=dict(color='white', width=1)),
                name='Actual', text=results_df.apply(lambda x: f"Actual | Year: {x['Year']:.0f} | Sector: {x['Real_Sector']:.0f}", axis=1),
                hoverinfo='text'
            ))
            
            # Predicted points
            pred_fig.add_trace(go.Scattergeo(
                lon=results_df['Pred_Lon'], lat=results_df['Pred_Lat'],
                mode='markers', marker=dict(size=10, color='#f85149', symbol='x', line=dict(color='#f85149', width=2)),
                name='Predicted', text=results_df.apply(lambda x: f"Predicted | Conf: {x['Confidence']:.1f}% | Rank: {x['Rank']:.0f}", axis=1),
                hoverinfo='text'
            ))
            
            pred_fig.update_geos(
                projection_type="natural earth",
                showland=True, landcolor='#21262d',
                showocean=True, oceancolor='#0d1117',
                center=dict(lat=25, lon=82), projection_scale=4,
                lonaxis=dict(range=[65, 100]), lataxis=dict(range=[15, 35]),
                bgcolor='rgba(0,0,0,0)'
            )
            
            pred_fig.update_layout(
                height=500, margin=dict(l=0, r=0, t=30, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, bgcolor='rgba(33,38,45,0.9)', font=dict(color='#e6edf3'))
            )
            
            st.plotly_chart(pred_fig, use_container_width=True, config=CHART_CONFIG)
            
            # Year breakdown
            st.markdown('<p class="section-title">Yearly Performance</p>', unsafe_allow_html=True)
            
            yearly_perf = results_df.groupby('Year').apply(lambda x: pd.Series({
                'Total': len(x),
                'Correct': (x['Rank'] == 1).sum(),
                'Top3': (x['Rank'] <= 3).sum()
            }), include_groups=False).reset_index()
            
            perf_fig = go.Figure()
            perf_fig.add_trace(go.Bar(x=yearly_perf['Year'], y=yearly_perf['Total'], name='Total', marker_color='#30363d'))
            perf_fig.add_trace(go.Bar(x=yearly_perf['Year'], y=yearly_perf['Top3'], name='Top 3', marker_color='#d29922'))
            perf_fig.add_trace(go.Bar(x=yearly_perf['Year'], y=yearly_perf['Correct'], name='Exact', marker_color='#3fb950'))
            
            perf_fig.update_layout(**dark_layout(), height=300, barmode='overlay', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5))
            perf_fig.update_xaxes(gridcolor='rgba(48,54,61,0.5)', color='#8b949e', dtick=1)
            perf_fig.update_yaxes(gridcolor='rgba(48,54,61,0.5)', color='#8b949e')
            
            st.plotly_chart(perf_fig, use_container_width=True, config=CHART_CONFIG)
        else:
            st.warning("Could not run prediction model. Check if required data files exist.")
    
    # TAB 4: Data
    with tab4:
        st.markdown('<p class="section-title">Earthquake Data</p>', unsafe_allow_html=True)
        
        display_cols = ['time', 'lat', 'lon', 'depth', 'mag', 'year']
        st.dataframe(
            filtered[display_cols].sort_values('time', ascending=False).head(500),
            use_container_width=True,
            height=400
        )
        
        # Download
        csv = filtered.to_csv(index=False)
        st.download_button("Download Filtered Data", csv, "earthquake_data.csv", "text/csv")
        
        if directions_df is not None:
            st.markdown("---")
            st.markdown('<p class="section-title">Migration Data</p>', unsafe_allow_html=True)
            st.dataframe(directions_df.head(100), use_container_width=True, height=300)

if __name__ == "__main__":
    main()
