"""
Earthquake Stress & Cluster Modeling Dashboard
New, cleaner layout with professional dark theme
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# -------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Earthquake Stress & Cluster Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# GLOBAL THEME (PROFESSIONAL DARK)
# -------------------------------------------------------------------

PRIMARY_BG = "#050816"       # page background
PANEL_BG = "#0b1020"         # panels
CARD_BG = "#111827"          # cards
ACCENT = "#0ea5e9"           # cyan
ACCENT_SOFT = "#38bdf8"
ACCENT_WARN = "#f97316"
TEXT_PRIMARY = "#e5e7eb"
TEXT_SECONDARY = "#9ca3af"
GRID_COLOR = "rgba(148, 163, 184, 0.25)"

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "earthquake_dashboard",
        "height": 800,
        "width": 1200,
        "scale": 2,
    },
}


def base_layout():
    """Common layout for all figures."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CARD_BG,
        font=dict(family="Inter, system-ui, -apple-system, BlinkMacSystemFont",
                  color=TEXT_PRIMARY),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    )


# -------------------------------------------------------------------
# CUSTOM CSS (NO EMOJIS, SUBTLE STYLING)
# -------------------------------------------------------------------

st.markdown(
    f"""
    <style>
    .stApp {{
        background: radial-gradient(circle at top, #020617 0, #020617 40%, {PRIMARY_BG} 100%);
        color: {TEXT_PRIMARY};
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .main .block-container {{
        padding-top: 1.25rem;
        max-width: 1500px;
    }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #020617 0%, #020617 40%, #020617 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.35);
    }}
    .app-header {{
        padding: 1rem 1.25rem 1.5rem 1.25rem;
        border-radius: 18px;
        background: radial-gradient(circle at 0% 0%, rgba(14,165,233,0.25), transparent 60%),
                    radial-gradient(circle at 100% 0%, rgba(249,115,22,0.2), transparent 55%),
                    linear-gradient(135deg, #020617, #020617);
        border: 1px solid rgba(148, 163, 184, 0.25);
    }}
    .app-header h1 {{
        font-size: 1.75rem;
        margin: 0 0 0.5rem 0;
        letter-spacing: 0.03em;
    }}
    .app-header p {{
        margin: 0;
        color: {TEXT_SECONDARY};
        font-size: 0.9rem;
    }}
    .metric-card {{
        background: {CARD_BG};
        border-radius: 14px;
        padding: 0.75rem 0.9rem;
        border: 1px solid rgba(31,41,55,0.9);
    }}
    .metric-label {{
        font-size: 0.7rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {TEXT_SECONDARY};
        margin-bottom: 0.15rem;
    }}
    .metric-value {{
        font-size: 1.25rem;
        font-weight: 600;
        color: {TEXT_PRIMARY};
    }}
    .metric-sub {{
        font-size: 0.75rem;
        color: {TEXT_SECONDARY};
    }}
    .section-title {{
        font-size: 1.05rem;
        font-weight: 600;
        margin: 0.3rem 0 0.5rem 0;
        border-left: 3px solid {ACCENT};
        padding-left: 0.5rem;
    }}
    .section-subtitle {{
        font-size: 0.8rem;
        color: {TEXT_SECONDARY};
        margin-bottom: 0.4rem;
    }}
    .info-panel {{
        background: {CARD_BG};
        border-radius: 12px;
        border: 1px solid rgba(15,23,42,0.9);
        padding: 0.75rem 0.9rem;
        font-size: 0.78rem;
        color: {TEXT_SECONDARY};
    }}
    .info-panel strong {{
        color: {ACCENT_SOFT};
    }}
    .footer-text {{
        margin-top: 1.5rem;
        font-size: 0.7rem;
        color: {TEXT_SECONDARY};
        text-align: center;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.25rem;
        background: transparent;
        border-bottom: 1px solid rgba(31,41,55,0.9);
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 0.45rem 0.9rem;
        border-radius: 999px;
        font-size: 0.85rem;
        color: {TEXT_SECONDARY};
    }}
    .stTabs [aria-selected="true"] {{
        background: rgba(15,23,42,0.9) !important;
        border: 1px solid rgba(56,189,248,0.6) !important;
        color: {TEXT_PRIMARY} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# DATA LOADERS
# -------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_earthquakes():
    df = pd.read_csv("coordinates.csv")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "lat", "lon"])
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    if "mag" in df.columns:
        df["mag"] = df["mag"].astype(float).fillna(0)
    else:
        df["mag"] = 0.0
    if "depth" in df.columns:
        df["depth"] = df["depth"].astype(float).fillna(0)
    else:
        df["depth"] = 0.0
    return df


@st.cache_data(show_spinner=False)
def load_cluster_summary():
    try:
        df = pd.read_csv("train_set.csv")
        return df
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_cluster_directions():
    try:
        df = pd.read_csv("cluster_directions_final.csv")
        return df
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_test_set():
    try:
        df = pd.read_csv("test_set.csv")
        df["From_Time"] = pd.to_datetime(df["From_Time"], format="mixed")
        return df
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_globe_cords():
    path = os.path.join("globe", "cords.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["year"] = df["time"].dt.year
    return df


@st.cache_data(show_spinner=False)
def run_stress_model():
    """
    Use the in-dashboard version of the XGBoost sector prediction,
    similar to stress_predictor_model.py but returning DataFrames
    and metrics for visualization.
    """
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder

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
        df_sorted = df.sort_values("From_Time").reset_index(drop=True)
        for i in range(lookback, len(df_sorted)):
            current = df_sorted.iloc[i]

            target_sector = get_sector_id(current["To_Lat"], current["To_Lon"])
            if target_sector == -1:
                continue

            current_sector = get_sector_id(current["From_Lat"], current["From_Lon"])
            if current_sector == -1:
                continue

            features = dict(
                Year=current["Year"],
                Month=current["From_Time"].month,
                Current_Sector=current_sector,
                Current_Bearing=current["Bearing_Degrees"],
                Current_Angle=current["Angle_wrt_X_Axis"],
            )

            valid = True
            for lag in range(1, lookback + 1):
                past = df_sorted.iloc[i - lag]
                past_sector = get_sector_id(past["To_Lat"], past["To_Lon"])
                if past_sector == -1:
                    valid = False
                    break
                prefix = f"Lag{lag}_"
                features[prefix + "Sector"] = past_sector
                features[prefix + "Bearing"] = past["Bearing_Degrees"]
                features[prefix + "Angle"] = past["Angle_wrt_X_Axis"]
                delta_days = (current["From_Time"] - past["From_Time"]).days
                features[prefix + "DaysAgo"] = max(delta_days, 1)

            if valid:
                features["Target_Sector"] = target_sector
                features_list.append(features)

        return pd.DataFrame(features_list)

    # load training / test from files
    train_df = pd.read_csv("cluster_directions_final.csv")
    train_df["From_Time"] = pd.to_datetime(train_df["From_Time"], format="mixed")
    # keep historical portion
    train_df = train_df[train_df["Year"] <= 2005].sort_values("From_Time")

    test_df = pd.read_csv("test_set.csv")
    test_df["From_Time"] = pd.to_datetime(test_df["From_Time"], format="mixed")
    test_df = test_df.sort_values("From_Time")

    train_feat = create_features(train_df, lookback=3)
    test_feat = create_features(test_df, lookback=3)

    if train_feat.empty or test_feat.empty:
        return None, None, None, GRID_DIM, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX

    X_train = train_feat.drop(columns=["Target_Sector", "Year"])
    y_train = train_feat["Target_Sector"]
    X_test = test_feat.drop(columns=["Target_Sector", "Year"])
    y_test = test_feat["Target_Sector"]

    le = LabelEncoder()
    le.fit(pd.concat([y_train, y_test]))
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    model = XGBClassifier(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train_enc)

    probs = model.predict_proba(X_test)

    rows = []
    top1 = top3 = top5 = 0
    total = len(X_test)

    for i in range(total):
        real_idx = y_test_enc[i]
        real_sector = y_test.iloc[i]
        year = test_feat.iloc[i]["Year"]
        p = probs[i]
        ranked_idx = np.argsort(p)[::-1]
        rank_pos = np.where(ranked_idx == real_idx)[0]
        rank = int(rank_pos[0] + 1) if len(rank_pos) > 0 else 99

        if rank == 1:
            top1 += 1
        if rank <= 3:
            top3 += 1
        if rank <= 5:
            top5 += 1

        best_idx = ranked_idx[0]
        pred_sector = le.inverse_transform([best_idx])[0]
        pred_lat, pred_lon = get_sector_center(pred_sector)
        real_lat, real_lon = get_sector_center(real_sector)

        rows.append(
            dict(
                Year=year,
                Real_Sector=int(real_sector),
                Pred_Sector=int(pred_sector),
                Rank=int(rank),
                Real_Lat=real_lat,
                Real_Lon=real_lon,
                Pred_Lat=pred_lat,
                Pred_Lon=pred_lon,
                Confidence=float(p[best_idx] * 100.0),
            )
        )

    results = pd.DataFrame(rows)

    feature_importance = dict(
        zip(X_train.columns, model.feature_importances_.tolist())
    )
    metrics = dict(
        total=total,
        top1=top1,
        top3=top3,
        top5=top5,
        top1_pct=0 if total == 0 else top1 / total * 100,
        top3_pct=0 if total == 0 else top3 / total * 100,
        top5_pct=0 if total == 0 else top5 / total * 100,
    )

    return results, metrics, feature_importance, GRID_DIM, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX


# -------------------------------------------------------------------
# VISUAL HELPERS
# -------------------------------------------------------------------


def main_map(df, year_range, mag_range):
    filtered = df[
        (df["year"].between(year_range[0], year_range[1]))
        & (df["mag"].between(mag_range[0], mag_range[1]))
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scattergeo(
            lon=filtered["lon"],
            lat=filtered["lat"],
            mode="markers",
            marker=dict(
                size=4 + filtered["mag"] * 1.8,
                color=filtered["mag"],
                colorscale="Viridis",
                colorbar=dict(
                    title="Magnitude",
                    tickcolor=TEXT_SECONDARY,
                    titlefont=dict(color=TEXT_PRIMARY),
                ),
                opacity=0.75,
                line=dict(width=0.5, color="rgba(15,23,42,0.7)"),
            ),
            hovertemplate=(
                "Magnitude: %{marker.size:.1f}<br>"
                "Time: %{text}<br>"
                "Lat: %{lat:.2f}, Lon: %{lon:.2f}<extra></extra>"
            ),
            text=filtered["time"].dt.strftime("%Y-%m-%d %H:%M"),
            name="Earthquakes",
        )
    )

    fig.update_geos(
        projection_type="mercator",
        showland=True,
        landcolor="#020617",
        showocean=True,
        oceancolor="#020617",
        coastlinecolor="#4b5563",
        countrycolor="#4b5563",
        bgcolor="rgba(0,0,0,0)",
        center=dict(lat=28, lon=85),
        projection_scale=7,
    )

    fig.update_layout(
        **base_layout(),
        height=520,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(bgcolor="rgba(15,23,42,0.85)"),
    )
    return fig


def magnitude_panels(df, year_range):
    filtered = df[df["year"].between(year_range[0], year_range[1])]

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "histogram"}, {"type": "box"}]],
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.15,
    )

    fig.add_trace(
        go.Histogram(
            x=filtered["mag"],
            nbinsx=30,
            marker=dict(
                color=ACCENT_SOFT,
                line=dict(color="rgba(15,23,42,1.0)", width=0.6),
            ),
            opacity=0.85,
            name="Magnitude",
        ),
        row=1,
        col=1,
    )

    years = sorted(filtered["year"].unique())
    palette = px.colors.sequential.Blues

    for i, y in enumerate(years):
        sub = filtered[filtered["year"] == y]
        fig.add_trace(
            go.Box(
                y=sub["mag"],
                name=str(y),
                marker_color=palette[i % len(palette)],
                boxpoints=False,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        **base_layout(),
        height=360,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Magnitude", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=1, col=2)
    fig.update_yaxes(title_text="Magnitude", row=1, col=2)
    return fig


def temporal_panels(df, year_range):
    filtered = df[df["year"].between(year_range[0], year_range[1])]

    yearly = (
        filtered.groupby("year")["mag"]
        .agg(["count", "mean", "max"])
        .reset_index()
        .rename(columns={"count": "events", "mean": "avg_mag", "max": "max_mag"})
    )
    monthly = (
        filtered.groupby("month")["mag"].size().reindex(range(1, 13), fill_value=0)
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "scatter"}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    fig.add_trace(
        go.Bar(
            x=yearly["year"],
            y=yearly["events"],
            marker=dict(
                color=yearly["events"],
                colorscale="Teal",
                line=dict(color="rgba(15,23,42,1.0)", width=0.6),
            ),
            name="Events",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=yearly["year"],
            y=yearly["avg_mag"],
            mode="lines+markers",
            line=dict(color=ACCENT_WARN, width=2.0),
            marker=dict(size=7, color=ACCENT_WARN),
            name="Average magnitude",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Bar(
            x=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            y=monthly.values,
            marker=dict(
                color=monthly.values,
                colorscale="Viridis",
                line=dict(color="rgba(15,23,42,1.0)", width=0.5),
            ),
            name="By month",
        ),
        row=2,
        col=1,
    )

    # depth vs magnitude scatter
    sample = (
        filtered.sample(1200, random_state=42)
        if len(filtered) > 1200
        else filtered
    )
    fig.add_trace(
        go.Scatter(
            x=sample["depth"],
            y=sample["mag"],
            mode="markers",
            marker=dict(
                size=5,
                color=sample["mag"],
                colorscale="Magma",
                opacity=0.7,
                line=dict(color="rgba(15,23,42,1.0)", width=0.4),
            ),
            name="Depth vs magnitude",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        **base_layout(),
        height=620,
        showlegend=False,
    )

    fig.update_xaxes(title="Year", row=1, col=1)
    fig.update_yaxes(title="Events", row=1, col=1)
    fig.update_xaxes(title="Year", row=1, col=2)
    fig.update_yaxes(title="Average magnitude", row=1, col=2)
    fig.update_xaxes(title="Month", row=2, col=1)
    fig.update_yaxes(title="Events", row=2, col=1)
    fig.update_xaxes(title="Depth (km)", row=2, col=2)
    fig.update_yaxes(title="Magnitude", row=2, col=2)

    return fig


def prediction_feature_importance(feature_importance: dict):
    if not feature_importance:
        return None
    items = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:12]
    names = [i[0] for i in items]
    vals = [i[1] for i in items]

    fig = go.Figure(
        data=[
            go.Bar(
                x=vals,
                y=names,
                orientation="h",
                marker=dict(
                    color=vals,
                    colorscale="Blues",
                    line=dict(color="rgba(15,23,42,1.0)", width=0.6),
                ),
            )
        ]
    )
    fig.update_layout(
        **base_layout(),
        height=460,
        margin=dict(l=130, r=20, t=10, b=40),
        xaxis_title="Importance",
    )
    return fig


def prediction_map(results_df):
    fig = go.Figure()

    for _, row in results_df.iterrows():
        color = "#22c55e" if row["Rank"] == 1 else (
            "#fbbf24" if row["Rank"] <= 3 else (
                "#f97316" if row["Rank"] <= 5 else "#f87171"
            )
        )
        fig.add_trace(
            go.Scattergeo(
                lon=[row["Real_Lon"], row["Pred_Lon"]],
                lat=[row["Real_Lat"], row["Pred_Lat"]],
                mode="lines",
                line=dict(color=color, width=0.7),
                opacity=0.35,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scattergeo(
            lon=results_df["Real_Lon"],
            lat=results_df["Real_Lat"],
            mode="markers",
            name="Actual next zone",
            marker=dict(
                size=7,
                color="#3b82f6",
                line=dict(color="white", width=0.6),
            ),
            hovertemplate=(
                "Year: %{text}<br>"
                "Lat: %{lat:.1f}, Lon: %{lon:.1f}<extra></extra>"
            ),
            text=results_df["Year"].astype(int),
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lon=results_df["Pred_Lon"],
            lat=results_df["Pred_Lat"],
            mode="markers",
            name="Predicted next zone",
            marker=dict(
                size=7,
                color="#f97316",
                symbol="x",
                line=dict(color="white", width=0.9),
            ),
            hovertemplate=(
                "Year: %{customdata[0]}<br>"
                "Rank: %{customdata[1]}<br>"
                "Confidence: %{customdata[2]:.1f}%<extra></extra>"
            ),
            customdata=np.stack(
                [
                    results_df["Year"].astype(int).values,
                    results_df["Rank"].astype(int).values,
                    results_df["Confidence"].values,
                ],
                axis=1,
            ),
        )
    )

    fig.update_geos(
        projection_type="mercator",
        showland=True,
        landcolor="#020617",
        showocean=True,
        oceancolor="#020617",
        coastlinecolor="#4b5563",
        countrycolor="#4b5563",
        center=dict(lat=26, lon=82),
        projection_scale=4.3,
        bgcolor="rgba(0,0,0,0)",
    )

    fig.update_layout(
        **base_layout(),
        height=460,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(
            bgcolor="rgba(15,23,42,0.9)",
            bordercolor="rgba(31,41,55,1.0)",
            borderwidth=1,
        ),
    )
    return fig


def globe_figure(cords_df, selected_year):
    year_data = cords_df[cords_df["year"] == selected_year]
    fig = go.Figure()

    if year_data.empty:
        fig.update_layout(
            **base_layout(),
            height=420,
            annotations=[
                dict(
                    text=f"No globe data for {selected_year}",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(color=TEXT_SECONDARY),
                )
            ],
        )
        return fig

    fig.add_trace(
        go.Scattergeo(
            lon=year_data["Actual_Lon"],
            lat=year_data["Actual_Lat"],
            mode="markers",
            name="Actual",
            marker=dict(
                size=7,
                color="#f97316",
                line=dict(color="white", width=0.6),
            ),
            hovertemplate=(
                "Actual<br>"
                "Lat: %{lat:.2f}, Lon: %{lon:.2f}<br>"
                "%{text}<extra></extra>"
            ),
            text=year_data["time"].dt.strftime("%Y-%m-%d"),
        )
    )

    fig.add_trace(
        go.Scattergeo(
            lon=year_data["Pred_Lon"],
            lat=year_data["Pred_Lat"],
            mode="markers",
            name="Predicted",
            marker=dict(
                size=7,
                color="#22c55e",
                symbol="diamond",
                line=dict(color="white", width=0.6),
            ),
            hovertemplate=(
                "Predicted<br>"
                "Lat: %{lat:.2f}, Lon: %{lon:.2f}<br>"
                "%{text}<extra></extra>"
            ),
            text=year_data["time"].dt.strftime("%Y-%m-%d"),
        )
    )

    fig.update_geos(
        projection_type="orthographic",
        showland=True,
        landcolor="#020617",
        showocean=True,
        oceancolor="#020617",
        coastlinecolor="#4b5563",
        countrycolor="#4b5563",
        bgcolor="rgba(0,0,0,0)",
    )

    fig.update_layout(
        **base_layout(),
        height=420,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(
            bgcolor="rgba(15,23,42,0.9)",
            bordercolor="rgba(31,41,55,1.0)",
            borderwidth=1,
        ),
    )
    return fig


# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------


def main():
    # Header
    st.markdown(
        """
        <div class="app-header">
            <h1>Earthquake Stress & Cluster Modeling</h1>
            <p>
                Exploratory dashboard for the seismic clustering pipeline: raw earthquakes, 
                derived clusters, directional migration and sector-level stress predictions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Loading data"):
        eq = load_earthquakes()
        clusters = load_cluster_summary()
        directions = load_cluster_directions()
        test_set = load_test_set()
        globe_df = load_globe_cords()

    if eq is None or eq.empty:
        st.error("Unable to load coordinates.csv. The dashboard needs this file.")
        return

    # Sidebar filters
    with st.sidebar:
        st.markdown(
            "<div class='section-title'>Global filters</div>",
            unsafe_allow_html=True,
        )

        min_year, max_year = int(eq["year"].min()), int(eq["year"].max())
        year_sel = st.slider(
            "Year range",
            min_value=min_year,
            max_value=max_year,
            value=(max(1980, min_year), min(2015, max_year)),
        )

        st.markdown("---")
        mag_min = float(eq["mag"].min())
        mag_max = max(float(eq["mag"].max()), mag_min + 0.5)
        mag_sel = st.slider(
            "Magnitude range",
            min_value=float(np.floor(mag_min)),
            max_value=float(np.ceil(mag_max)),
            value=(float(np.floor(mag_min)), float(np.ceil(mag_max))),
            step=0.5,
        )

        st.markdown("---")
        filtered = eq[
            (eq["year"].between(year_sel[0], year_sel[1]))
            & (eq["mag"].between(mag_sel[0], mag_sel[1]))
        ]

        st.markdown(
            "<div class='section-title'>Snapshot</div>",
            unsafe_allow_html=True,
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Events in filter</div>
                    <div class="metric-value">{len(filtered):,}</div>
                    <div class="metric-sub">From {year_sel[0]} to {year_sel[1]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_b:
            if not filtered.empty:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Magnitude (avg / max)</div>
                        <div class="metric-value">
                            {filtered["mag"].mean():.2f} / {filtered["mag"].max():.1f}
                        </div>
                        <div class="metric-sub">Within active filters</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="metric-card">
                        <div class="metric-label">Magnitude (avg / max)</div>
                        <div class="metric-value">–</div>
                        <div class="metric-sub">No events in current filter</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown(
            """
            <div class="info-panel">
                <strong>Guidance</strong><br/>
                • Adjust the time and magnitude filters above to align all views.<br/>
                • The "Model" tab operates on precomputed cluster-direction data and 
                  uses the same grid logic as your training script.<br/>
                • Globe view uses globe/cords.csv if present.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Tabs
    tab_overview, tab_stats, tab_clusters, tab_model, tab_globe, tab_data = st.tabs(
        [
            "Overview map",
            "Temporal statistics",
            "Cluster structure",
            "Stress model",
            "Globe comparison",
            "Data inspection",
        ]
    )

    # ------------------------------------------------------------------
    # TAB: OVERVIEW
    # ------------------------------------------------------------------
    with tab_overview:
        st.markdown(
            "<div class='section-title'>Spatial distribution</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='section-subtitle'>Filtered by global controls in the sidebar.</div>",
            unsafe_allow_html=True,
        )

        fig_map = main_map(eq, year_sel, mag_sel)
        st.plotly_chart(fig_map, use_container_width=True, config=PLOTLY_CONFIG)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                "<div class='section-title'>Magnitude profile</div>",
                unsafe_allow_html=True,
            )
            fig_mag = magnitude_panels(eq, year_sel)
            st.plotly_chart(fig_mag, use_container_width=True, config=PLOTLY_CONFIG)
        with c2:
            st.markdown(
                "<div class='section-title'>Depth distribution</div>",
                unsafe_allow_html=True,
            )
            depth_fig = go.Figure(
                data=[
                    go.Histogram(
                        x=eq["depth"],
                        nbinsx=40,
                        marker=dict(
                            color=ACCENT_SOFT,
                            line=dict(color="rgba(15,23,42,1.0)", width=0.5),
                        ),
                        opacity=0.85,
                    )
                ]
            )
            depth_fig.update_layout(
                **base_layout(),
                height=360,
                xaxis_title="Depth (km)",
                yaxis_title="Count",
            )
            st.plotly_chart(depth_fig, use_container_width=True, config=PLOTLY_CONFIG)

    # ------------------------------------------------------------------
    # TAB: TEMPORAL STATS
    # ------------------------------------------------------------------
    with tab_stats:
        st.markdown(
            "<div class='section-title'>Temporal behaviour</div>",
            unsafe_allow_html=True,
        )
        fig_t = temporal_panels(eq, year_sel)
        st.plotly_chart(fig_t, use_container_width=True, config=PLOTLY_CONFIG)

    # ------------------------------------------------------------------
    # TAB: CLUSTERS
    # ------------------------------------------------------------------
    with tab_clusters:
        st.markdown(
            "<div class='section-title'>Cluster summary (train_set.csv)</div>",
            unsafe_allow_html=True,
        )
        if clusters is None or clusters.empty:
            st.info("train_set.csv not available or empty.")
        else:
            # heatmap: clusters by year vs event_count
            try:
                tmp = (
                    clusters.groupby(["year", "cluster_rank"])["event_count"]
                    .sum()
                    .reset_index()
                )
                pivot = tmp.pivot(
                    index="cluster_rank", columns="year", values="event_count"
                ).fillna(0)

                heat = go.Figure(
                    data=[
                        go.Heatmap(
                            z=pivot.values,
                            x=pivot.columns.astype(str),
                            y=pivot.index.astype(str),
                            colorscale="Inferno",
                            colorbar=dict(title="Events"),
                        )
                    ]
                )
                heat.update_layout(
                    **base_layout(),
                    height=400,
                    xaxis_title="Year",
                    yaxis_title="Cluster rank",
                )
                st.plotly_chart(heat, use_container_width=True, config=PLOTLY_CONFIG)
            except Exception:
                st.info("Unable to build cluster heatmap from train_set.csv.")

            st.markdown(
                "<div class='section-title'>Top clusters by activity</div>",
                unsafe_allow_html=True,
            )
            top = clusters.sort_values("event_count", ascending=False).head(12)
            st.dataframe(
                top[
                    [
                        "year",
                        "cluster_rank",
                        "event_count",
                        "avg_mag",
                        "centroid_lat",
                        "centroid_lon",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    # ------------------------------------------------------------------
    # TAB: MODEL (STRESS PREDICTION)
    # ------------------------------------------------------------------
    with tab_model:
        st.markdown(
            "<div class='section-title'>Sector-level stress prediction</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='section-subtitle'>Using the same XGBoost logic as stress_predictor_model.py, with cluster_directions_final.csv + test_set.csv.</div>",
            unsafe_allow_html=True,
        )

        if directions is None or test_set is None:
            st.info(
                "cluster_directions_final.csv and/or test_set.csv are missing; "
                "model view cannot be constructed."
            )
        else:
            with st.spinner("Running XGBoost sector model"):
                results_df, metrics, feat_imp, *_ = run_stress_model()

            if results_df is None or metrics is None:
                st.info("Model did not return valid predictions.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Prediction count</div>
                        <div class="metric-value">{metrics["total"]}</div>
                        <div class="metric-sub">Test samples</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                c2.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Exact sector</div>
                        <div class="metric-value">{metrics["top1_pct"]:.1f}%</div>
                        <div class="metric-sub">Rank = 1</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                c3.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Within top 3</div>
                        <div class="metric-value">{metrics["top3_pct"]:.1f}%</div>
                        <div class="metric-sub">True sector rank ≤ 3</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                c4.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Within top 5</div>
                        <div class="metric-value">{metrics["top5_pct"]:.1f}%</div>
                        <div class="metric-sub">True sector rank ≤ 5</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("---")
                col_l, col_r = st.columns([1.2, 1])

                with col_l:
                    st.markdown(
                        "<div class='section-title'>Actual vs predicted sectors</div>",
                        unsafe_allow_html=True,
                    )
                    pm = prediction_map(results_df)
                    st.plotly_chart(pm, use_container_width=True, config=PLOTLY_CONFIG)

                with col_r:
                    st.markdown(
                        "<div class='section-title'>Feature importance</div>",
                        unsafe_allow_html=True,
                    )
                    fi_fig = prediction_feature_importance(feat_imp)
                    if fi_fig is not None:
                        st.plotly_chart(
                            fi_fig, use_container_width=True, config=PLOTLY_CONFIG
                        )

                st.markdown("---")
                st.markdown(
                    "<div class='section-title'>Per-year accuracy</div>",
                    unsafe_allow_html=True,
                )
                by_year = (
                    results_df.groupby("Year")
                    .apply(
                        lambda x: pd.Series(
                            dict(
                                total=len(x),
                                top1=(x["Rank"] == 1).sum(),
                                top3=(x["Rank"] <= 3).sum(),
                            )
                        )
                    )
                    .reset_index()
                )
                by_year["top1_pct"] = (by_year["top1"] / by_year["total"] * 100).round(1)
                by_year["top3_pct"] = (by_year["top3"] / by_year["total"] * 100).round(1)

                acc_fig = go.Figure()
                acc_fig.add_trace(
                    go.Bar(
                        x=by_year["Year"].astype(int),
                        y=by_year["top1_pct"],
                        name="Exact",
                        marker_color="#22c55e",
                    )
                )
                acc_fig.add_trace(
                    go.Bar(
                        x=by_year["Year"].astype(int),
                        y=by_year["top3_pct"],
                        name="Top 3",
                        marker_color="#fbbf24",
                    )
                )
                acc_fig.update_layout(
                    **base_layout(),
                    barmode="group",
                    height=360,
                    xaxis_title="Year",
                    yaxis_title="Accuracy (%)",
                    legend=dict(
                        bgcolor="rgba(15,23,42,0.9)",
                        bordercolor="rgba(31,41,55,1.0)",
                        borderwidth=1,
                    ),
                )
                st.plotly_chart(acc_fig, use_container_width=True, config=PLOTLY_CONFIG)

                st.markdown(
                    "<div class='section-title'>Prediction table</div>",
                    unsafe_allow_html=True,
                )
                display = results_df.copy()
                display["Year"] = display["Year"].astype(int)
                display["Confidence (%)"] = display["Confidence"].round(1)
                st.dataframe(
                    display[
                        [
                            "Year",
                            "Real_Sector",
                            "Pred_Sector",
                            "Rank",
                            "Confidence (%)",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

    # ------------------------------------------------------------------
    # TAB: GLOBE
    # ------------------------------------------------------------------
    with tab_globe:
        st.markdown(
            "<div class='section-title'>Actual vs predicted positions (globe/cords.csv)</div>",
            unsafe_allow_html=True,
        )
        if globe_df is None or globe_df.empty:
            st.info("globe/cords.csv not present or empty.")
        else:
            years = sorted(globe_df["year"].dropna().unique())
            year_choice = st.selectbox("Year", options=years)
            gf = globe_figure(globe_df, year_choice)
            st.plotly_chart(gf, use_container_width=True, config=PLOTLY_CONFIG)

    # ------------------------------------------------------------------
    # TAB: DATA
    # ------------------------------------------------------------------
    with tab_data:
        st.markdown(
            "<div class='section-title'>Raw inputs and derived tables</div>",
            unsafe_allow_html=True,
        )
        subset_cols = ["time", "lat", "lon", "mag", "depth", "year"]
        subset_cols = [c for c in subset_cols if c in eq.columns]
        st.markdown("Coordinates (sample):")
        st.dataframe(
            eq[subset_cols].head(500),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("cluster_directions_final.csv (sample):")
            if directions is None:
                st.info("cluster_directions_final.csv not found.")
            else:
                st.dataframe(directions.head(300), use_container_width=True)

        with col2:
            st.markdown("train_set.csv and test_set.csv (samples):")
            if clusters is not None:
                st.caption("train_set.csv")
                st.dataframe(
                    clusters.head(200),
                    use_container_width=True,
                )
            if test_set is not None:
                st.caption("test_set.csv")
                st.dataframe(
                    test_set.head(200),
                    use_container_width=True,
                )

    st.markdown(
        "<div class='footer-text'>Dashboard layout generated for agbuddy7/earthquak-model using existing CSV inputs and model code.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()