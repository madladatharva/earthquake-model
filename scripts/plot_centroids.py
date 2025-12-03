#!/usr/bin/env python3
"""
Plot predicted earthquake centroids with plate boundaries and large circles.

Usage:
    python plot_centroids.py

Options (edit variables near top of file):
    - DEFAULT_RADIUS_KM: radius for circles around each centroid (in km)
    - OUTPUT_FILE: output PNG filename

If `predicted_events.csv` contains a `radius_km` column it will be used per-point.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Configuration
PRED_CSV = 'predicted_events.csv'
PLATE_GEOJSON = 'eu_in_plates.geojson'
OUTPUT_FILE = 'centroids_with_plates.png'

POINT_SIZE = 300  # plot size for centroids (user requested large circles/markers)


def load_predicted_centroids(csv_path):
    df = pd.read_csv(csv_path)
    # Ensure columns exist
    expected = ['latitude_pred', 'longitude_pred']
    for col in expected:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in {csv_path}")

    # If there are duplicate centroid coordinates across simulations, collapse to unique
    # Keep a representative time_pred (first occurrence) so we can extract a year label
    if 'sim_id' in df.columns:
        grouped = df.groupby(['latitude_pred', 'longitude_pred']).agg(
            n_events=('sim_id', 'count'),
            time_pred=('time_pred', 'first')
        ).reset_index()
    else:
        grouped = df.groupby(['latitude_pred', 'longitude_pred']).agg(
            n_events=('latitude_pred', 'size'),
            time_pred=('time_pred', 'first') if 'time_pred' in df.columns else ('latitude_pred', 'first')
        ).reset_index()

    # Extract year (if time_pred like 'YYYY' or 'YYYY-MM' or 'YYYY-MM-DD')
    def extract_year(x):
        try:
            s = str(x)
            return s.split('-')[0]
        except Exception:
            return ''

    grouped['year'] = grouped['time_pred'].apply(extract_year)

    # Keep simple DataFrame with lat/lon and optional radius_km if present in original file
    if 'radius_km' in df.columns:
        # Map radius from first occurrence
        radii = df.drop_duplicates(subset=['latitude_pred', 'longitude_pred'])[['latitude_pred', 'longitude_pred', 'radius_km']]
        grouped = grouped.merge(radii, on=['latitude_pred', 'longitude_pred'], how='left')
    else:
        grouped['radius_km'] = np.nan

    return grouped


def load_plate_boundaries(geojson_path):
    if not os.path.exists(geojson_path):
        print(f"Plate GeoJSON not found at '{geojson_path}' — continuing without plate boundaries.")
        return []

    with open(geojson_path, 'r', encoding='utf-8') as f:
        gj = json.load(f)

    segments = []
    for feat in gj.get('features', []):
        geom = feat.get('geometry')
        if not geom:
            continue
        gtype = geom.get('type')
        coords = geom.get('coordinates')
        if gtype == 'LineString':
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            segments.append((lons, lats))
        elif gtype == 'MultiLineString':
            for line in coords:
                lons = [c[0] for c in line]
                lats = [c[1] for c in line]
                segments.append((lons, lats))
    return segments


def plot_centroids(df_centroids, plate_segments, output_file=OUTPUT_FILE):
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot plate boundaries
    for i, (lons, lats) in enumerate(plate_segments):
        ax.plot(lons, lats, color='red', linewidth=1.5, alpha=0.8, label='EU-IN plate boundary' if i == 0 else None)

    # Plot centroids individually with per-point colors and labels (size ~300 as requested)
    # Create a color map
    num = len(df_centroids)
    cmap = plt.get_cmap('tab20')

    for i, row in df_centroids.iterrows():
        lat = row['latitude_pred']
        lon = row['longitude_pred']
        year = str(row.get('year', ''))
        color = cmap(i % 20)
        ax.scatter(lon, lat, s=POINT_SIZE, c=[color], alpha=0.8, edgecolors='k', linewidth=0.5, zorder=5)
        if year:
            ax.annotate(year, (lon, lat), textcoords='offset points', xytext=(5, 5), fontsize=9, zorder=6)

    # Draw large circles around each centroid
 
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_title('Predicted Earthquake Centroids with Plate Boundaries')
    ax.grid(True, alpha=0.3)

    # Create legend manually
    handles = []
    if plate_segments:
        handles.append(plt.Line2D([0], [0], color='red', lw=2, label='EU-IN plate boundary'))
    handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Centroid', markerfacecolor='blue', markersize=8, markeredgecolor='k'))
    handles.append(plt.Line2D([0], [0], color='orange', lw=2, label=f'Radius ({default_radius_km} km)'))
    ax.legend(handles=handles, loc='upper right')

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")


def load_all_predictions(csv_path):
    """Load the full predicted_events.csv without collapsing — used for month-wise plotting."""
    df = pd.read_csv(csv_path)
    if 'time_pred' not in df.columns:
        raise ValueError("predicted_events.csv must contain a 'time_pred' column for month-wise plotting")
    # Normalize time_pred to YYYY-MM if it's YYYY or YYYY-MM-DD
    df['month'] = df['time_pred'].astype(str).apply(lambda s: str(s).split('-')[0] + ('-' + str(s).split('-')[1] if len(str(s).split('-')) > 1 else ''))
    return df


def plot_monthly_facets(df_all, plate_segments, out_prefix='centroids_monthly'):
    """Create faceted maps, one subplot per month (YYYY-MM).

    If there are more months than fit comfortably, this saves multiple paginated PNGs.
    """
    months = sorted(df_all['month'].unique())
    if not months:
        print("No months found in data — skipping monthly plots")
        return []

    print(f"Creating month-wise plots for {len(months)} months")

    # Layout: choose up to 12 panels per page
    panels_per_page = 12
    pages = (len(months) + panels_per_page - 1) // panels_per_page
    saved_files = []

    for page in range(pages):
        start = page * panels_per_page
        end = min(start + panels_per_page, len(months))
        page_months = months[start:end]

        n = len(page_months)
        cols = 4 if n >= 4 else n
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols + 2, 3.5 * rows + 1))
        axes = np.array(axes).reshape(-1)

        for ax_idx, month in enumerate(page_months):
            ax = axes[ax_idx]
            # Plot plate boundaries
            for (lons, lats) in plate_segments:
                ax.plot(lons, lats, color='red', linewidth=1.2, alpha=0.8)

            month_df = df_all[df_all['month'] == month]
            # Plot all centroids for this month
            cmap = plt.get_cmap('tab20')
            for i, row in month_df.iterrows():
                lon = row['longitude_pred']
                lat = row['latitude_pred']
                color = cmap(i % 20)
                ax.scatter(lon, lat, s=POINT_SIZE, c=[color], alpha=0.8, edgecolors='k', linewidth=0.4)

            # If radius column exists in monthly df, draw circles around unique centroids
            if 'radius_km' in month_df.columns:
                uniq = month_df.drop_duplicates(subset=['latitude_pred', 'longitude_pred'])
                for _, crow in uniq.iterrows():
                    rkm = crow.get('radius_km', default_radius_km)
                    if pd.isna(rkm):
                        rkm = default_radius_km
                    rdeg = rkm / 111.0
                    circ = plt.Circle((crow['longitude_pred'], crow['latitude_pred']), rdeg, fill=False, edgecolor='orange', linewidth=1.5, alpha=0.6)
                    ax.add_patch(circ)

            ax.set_title(month)
            ax.set_xlabel('Lon')
            ax.set_ylabel('Lat')
            ax.grid(True, alpha=0.2)

        # Hide any unused axes
        for j in range(len(page_months), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        out_file = f"{out_prefix}_page{page+1}.png"
        fig.savefig(out_file, dpi=200, bbox_inches='tight')
        saved_files.append(out_file)
        print(f"Saved monthly plot page: {out_file}")
        plt.close(fig)

    return saved_files


if __name__ == '__main__':
    print('Loading predicted centroids...')
    centroids = load_predicted_centroids(PRED_CSV)
    print(f'Found {len(centroids)} unique centroids')

    print('Loading plate boundaries...')
    plates = load_plate_boundaries(PLATE_GEOJSON)
    print(f'Loaded {len(plates)} plate boundary segments')

    print('Plotting...')
    plot_centroids(centroids, plates, DEFAULT_RADIUS_KM, OUTPUT_FILE)

    # Also create month-wise faceted plots using the raw predictions
    try:
        print('\nCreating month-wise plots...')
        all_df = load_all_predictions(PRED_CSV)
        saved = plot_monthly_facets(all_df, plates, DEFAULT_RADIUS_KM, out_prefix='centroids_monthly')
        if saved:
            print('Monthly plots saved:', ', '.join(saved))
    except Exception as e:
        print('Could not create month-wise plots:', e)
