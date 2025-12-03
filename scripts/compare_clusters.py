#!/usr/bin/env python3
"""
Earthquake Cluster Comparison Tool
Compares real earthquake cluster data with predicted cluster data from CSV files.

Author: GitHub Copilot
Description: Analyzes differences between actual and predicted earthquake clusters,
            calculates accuracy metrics, and creates visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from geopy.distance import geodesic
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial.distance import cdist
import seaborn as sns
from datetime import datetime, timedelta

class EarthquakeClusterComparator:
    def __init__(self, real_csv_path, predicted_csv_path, plate_boundaries_path=None):
        """
        Initialize the comparator with paths to real and predicted data
        
        Parameters:
        - real_csv_path: Path to CSV with real earthquake cluster data
        - predicted_csv_path: Path to CSV with predicted earthquake cluster data  
        - plate_boundaries_path: Optional path to GeoJSON with plate boundaries
        """
        self.real_csv_path = real_csv_path
        self.predicted_csv_path = predicted_csv_path
        self.plate_boundaries_path = plate_boundaries_path
        
        self.real_data = None
        self.predicted_data = None
        self.plate_boundaries = None
        self.comparison_results = {}
        
    def load_data(self):
        """Load real and predicted earthquake cluster data"""
        print("📂 Loading real earthquake cluster data...")
        self.real_data = self._load_cluster_csv(self.real_csv_path, "real")
        
        print("📂 Loading predicted earthquake cluster data...")
        self.predicted_data = self._load_cluster_csv(self.predicted_csv_path, "predicted")
        
        if self.plate_boundaries_path:
            print("📂 Loading plate boundaries...")
            self.plate_boundaries = self._load_plate_boundaries()
            
        print(f"✅ Data loaded successfully!")
        print(f"   Real clusters: {len(self.real_data)}")
        print(f"   Predicted clusters: {len(self.predicted_data)}")
        
    def _load_cluster_csv(self, csv_path, data_type):
        """Load cluster data from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            
            # Standardize column names
            column_mapping = {
                'lat': 'latitude', 'latitude_pred': 'latitude',
                'lon': 'longitude', 'longitude_pred': 'longitude',
                'time_pred': 'time'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Ensure required columns exist
            required_cols = ['latitude', 'longitude']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in {data_type} data: {missing_cols}")
            
            # Parse time if available
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
                df = df.dropna(subset=['time'])
                df['year'] = df['time'].dt.year
                df['month'] = df['time'].dt.month
                df['year_month'] = df['time'].dt.strftime('%Y-%m')
            
            # Clean coordinates
            df = df.dropna(subset=['latitude', 'longitude'])
            df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]
            df = df[(df['longitude'] >= -180) & (df['longitude'] <= 180)]
            
            print(f"   ✅ Loaded {len(df)} {data_type} cluster points")
            return df
            
        except Exception as e:
            print(f"   ❌ Error loading {data_type} data: {e}")
            return None
            
    def _load_plate_boundaries(self):
        """Load plate boundaries from GeoJSON"""
        try:
            with open(self.plate_boundaries_path, 'r') as f:
                data = json.load(f)
            
            boundaries = []
            for feature in data['features']:
                coords = feature['geometry']['coordinates']
                boundaries.append([(coord[0], coord[1]) for coord in coords])
            
            return boundaries
        except Exception as e:
            print(f"   ❌ Error loading plate boundaries: {e}")
            return None
    
    def calculate_spatial_accuracy(self, max_distance_km=100):
        """
        Calculate spatial accuracy metrics between real and predicted clusters
        Focus on location accuracy rather than cluster count matching
        
        Parameters:
        - max_distance_km: Maximum distance to consider a prediction as a match
        """
        print(f"\n📊 Calculating spatial accuracy (max distance: {max_distance_km} km)...")
        print("   Note: Focusing on location accuracy, not cluster count matching")
        
        real_coords = self.real_data[['latitude', 'longitude']].values
        pred_coords = self.predicted_data[['latitude', 'longitude']].values
        
        # Calculate distance matrix between all real and predicted points
        distances_km = []
        for real_point in real_coords:
            for pred_point in pred_coords:
                dist = geodesic(real_point, pred_point).kilometers
                distances_km.append(dist)
        
        distances_matrix = np.array(distances_km).reshape(len(real_coords), len(pred_coords))
        
        # Find closest predictions for each real cluster
        closest_distances = []
        matched_predictions = 0
        
        for i, real_point in enumerate(real_coords):
            min_distance = np.min(distances_matrix[i, :])
            closest_distances.append(min_distance)
            
            if min_distance <= max_distance_km:
                matched_predictions += 1
        
        # Calculate spatial accuracy metrics (location-based, not count-based)
        mean_distance = np.mean(closest_distances)
        median_distance = np.median(closest_distances)
        std_distance = np.std(closest_distances)
        location_accuracy = (matched_predictions / len(real_coords)) * 100
        
        self.comparison_results['spatial_accuracy'] = {
            'mean_distance_km': mean_distance,
            'median_distance_km': median_distance,
            'std_distance_km': std_distance,
            'location_accuracy_percentage': location_accuracy,
            'matched_predictions': matched_predictions,
            'total_real_clusters': len(real_coords),
            'total_predicted_clusters': len(pred_coords),
            'closest_distances': closest_distances
        }
        
        print(f"   📈 Mean distance: {mean_distance:.2f} km")
        print(f"   📈 Median distance: {median_distance:.2f} km")
        print(f"   📈 Location accuracy: {location_accuracy:.1f}% (within {max_distance_km} km)")
        print(f"   📈 Real clusters with close predictions: {matched_predictions}/{len(real_coords)}")
        print(f"   📈 Total predicted clusters available: {len(pred_coords)}")
        
    def calculate_temporal_accuracy(self):
        """Calculate temporal accuracy focusing on timing patterns, not cluster counts"""
        if 'time' not in self.real_data.columns or 'time' not in self.predicted_data.columns:
            print("⚠️  Temporal accuracy calculation skipped - time data not available in both datasets")
            return
            
        print("\n📊 Calculating temporal accuracy...")
        print("   Note: Focusing on timing patterns, not exact cluster counts")
        
        # Create time-based comparison focusing on when clusters occur, not how many
        real_times = set(self.real_data['year_month'].unique())
        pred_times = set(self.predicted_data['year_month'].unique())
        
        # Calculate temporal overlap
        temporal_overlap = len(real_times.intersection(pred_times))
        total_real_periods = len(real_times)
        temporal_coverage = (temporal_overlap / total_real_periods) * 100 if total_real_periods > 0 else 0
        
        # For periods where both have data, look at spatial-temporal matching
        common_periods = real_times.intersection(pred_times)
        period_matches = []
        
        for period in common_periods:
            real_period_data = self.real_data[self.real_data['year_month'] == period]
            pred_period_data = self.predicted_data[self.predicted_data['year_month'] == period]
            
            # For each real cluster in this period, find closest predicted cluster in same period
            if len(real_period_data) > 0 and len(pred_period_data) > 0:
                real_coords = real_period_data[['latitude', 'longitude']].values
                pred_coords = pred_period_data[['latitude', 'longitude']].values
                
                min_distances = []
                for real_point in real_coords:
                    distances = [geodesic(real_point, pred_point).kilometers for pred_point in pred_coords]
                    min_distances.append(min(distances))
                
                avg_distance = np.mean(min_distances)
                period_matches.append({
                    'period': period,
                    'avg_distance_km': avg_distance,
                    'real_clusters': len(real_period_data),
                    'pred_clusters': len(pred_period_data)
                })
        
        # Calculate overall temporal-spatial accuracy
        if period_matches:
            avg_temporal_distance = np.mean([pm['avg_distance_km'] for pm in period_matches])
        else:
            avg_temporal_distance = float('inf')
        
        self.comparison_results['temporal_accuracy'] = {
            'temporal_coverage_percentage': temporal_coverage,
            'periods_with_overlap': temporal_overlap,
            'total_real_periods': total_real_periods,
            'avg_temporal_spatial_distance_km': avg_temporal_distance,
            'period_matches': period_matches,
            'real_periods': list(real_times),
            'predicted_periods': list(pred_times)
        }
        
        print(f"   📈 Temporal coverage: {temporal_coverage:.1f}% ({temporal_overlap}/{total_real_periods} periods)")
        print(f"   📈 Average spatial distance in matching periods: {avg_temporal_distance:.2f} km")
        print(f"   📈 Periods analyzed: {len(period_matches)}")
        
    def calculate_time_shifted_comparison(self, real_start_month="2009-12", pred_start_month="2010-07", max_distance_km=200):
        """
        Calculate spatial comparison with time shift assumption
        Maps real data starting from real_start_month to predicted data starting from pred_start_month
        """
        print(f"📊 Calculating time-shifted spatial comparison...")
        print(f"   Mapping {real_start_month} (real) to {pred_start_month} (predicted)")
        
        # Convert month strings to datetime for easier manipulation
        real_start = datetime.strptime(real_start_month, "%Y-%m")
        pred_start = datetime.strptime(pred_start_month, "%Y-%m")
        
        # Get unique real periods sorted
        real_periods = sorted(self.real_data['year_month'].unique())
        pred_periods = sorted(self.predicted_data['year_month'].unique())
        
        # Find the index of our starting periods
        try:
            real_start_idx = real_periods.index(real_start_month)
        except ValueError:
            print(f"   ⚠️ Real start month {real_start_month} not found in data")
            return
            
        try:
            pred_start_idx = pred_periods.index(pred_start_month)
        except ValueError:
            print(f"   ⚠️ Predicted start month {pred_start_month} not found in data")
            return
        
        # Create mapping between real and predicted periods
        time_shifted_matches = []
        mapping_details = []
        
        # Map sequential periods
        for i in range(min(len(real_periods) - real_start_idx, len(pred_periods) - pred_start_idx)):
            real_period = real_periods[real_start_idx + i]
            pred_period = pred_periods[pred_start_idx + i]
            
            real_period_data = self.real_data[self.real_data['year_month'] == real_period]
            pred_period_data = self.predicted_data[self.predicted_data['year_month'] == pred_period]
            
            if len(real_period_data) > 0 and len(pred_period_data) > 0:
                # Calculate spatial comparison for this mapped period
                real_coords = real_period_data[['latitude', 'longitude']].values
                pred_coords = pred_period_data[['latitude', 'longitude']].values
                
                min_distances = []
                closest_matches = []
                
                for real_point in real_coords:
                    distances = [geodesic(real_point, pred_point).kilometers for pred_point in pred_coords]
                    min_dist = min(distances)
                    closest_idx = np.argmin(distances)
                    min_distances.append(min_dist)
                    closest_matches.append({
                        'real_lat': real_point[0],
                        'real_lon': real_point[1],
                        'pred_lat': pred_coords[closest_idx][0],
                        'pred_lon': pred_coords[closest_idx][1],
                        'distance_km': min_dist
                    })
                
                avg_distance = np.mean(min_distances)
                time_shifted_matches.append({
                    'real_period': real_period,
                    'pred_period': pred_period,
                    'avg_distance_km': avg_distance,
                    'real_clusters': len(real_period_data),
                    'pred_clusters': len(pred_period_data),
                    'matches': closest_matches
                })
                
                mapping_details.extend([{
                    'real_period': real_period,
                    'pred_period': pred_period,
                    'real_lat': match['real_lat'],
                    'real_lon': match['real_lon'],
                    'pred_lat': match['pred_lat'],
                    'pred_lon': match['pred_lon'],
                    'distance_km': match['distance_km']
                } for match in closest_matches])
        
        # Calculate summary statistics
        if time_shifted_matches:
            all_distances = []
            for match in time_shifted_matches:
                all_distances.extend([m['distance_km'] for m in match['matches']])
            
            mean_distance = np.mean(all_distances)
            median_distance = np.median(all_distances)
            std_distance = np.std(all_distances)
            accuracy_threshold = (np.sum(np.array(all_distances) <= max_distance_km) / len(all_distances)) * 100
            
            self.comparison_results['time_shifted_comparison'] = {
                'mapping': f"{real_start_month} → {pred_start_month}",
                'total_mapped_periods': len(time_shifted_matches),
                'mean_distance_km': mean_distance,
                'median_distance_km': median_distance,
                'std_distance_km': std_distance,
                'accuracy_threshold_percentage': accuracy_threshold,
                'max_distance_km': max_distance_km,
                'period_matches': time_shifted_matches,
                'detailed_matches': mapping_details
            }
            
            print(f"   📈 Mapped periods: {len(time_shifted_matches)}")
            print(f"   📈 Mean distance: {mean_distance:.2f} km")
            print(f"   📈 Accuracy (within {max_distance_km}km): {accuracy_threshold:.1f}%")
        else:
            print("   ⚠️ No time-shifted matches found")
        
    def create_comparison_visualizations(self, output_folder='comparison_results'):
        """Create comprehensive comparison visualizations"""
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\n🎨 Creating comparison visualizations in '{output_folder}'...")
        
        # 1. Spatial comparison map
        self._plot_spatial_comparison(output_folder)
        
        # 2. Distance distribution
        self._plot_distance_distribution(output_folder)
        
        # 3. Temporal comparison (if available)
        if 'temporal_accuracy' in self.comparison_results:
            self._plot_temporal_comparison(output_folder)
            
        # 4. Time-shifted comparison (if available)
        if 'time_shifted_comparison' in self.comparison_results:
            self._plot_time_shifted_comparison(output_folder)
            
        # 5. Summary statistics
        self._create_summary_report(output_folder)
        
    def _plot_spatial_comparison(self, output_folder):
        """Plot spatial comparison between real and predicted clusters"""
        plt.figure(figsize=(15, 10))
        
        # Plot plate boundaries if available
        if self.plate_boundaries:
            for boundary in self.plate_boundaries:
                lons, lats = zip(*boundary)
                plt.plot(lons, lats, 'red', linewidth=1, alpha=0.7, label='Plate Boundaries')
        
        # Plot real clusters
        plt.scatter(self.real_data['longitude'], self.real_data['latitude'], 
                   c='blue', s=100, alpha=0.7, marker='o', 
                   label=f'Real Clusters ({len(self.real_data)})', edgecolors='black')
        
        # Plot predicted clusters
        plt.scatter(self.predicted_data['longitude'], self.predicted_data['latitude'], 
                   c='red', s=100, alpha=0.7, marker='^', 
                   label=f'Predicted Clusters ({len(self.predicted_data)})', edgecolors='black')
        
        # Draw lines between closest pairs
        real_coords = self.real_data[['latitude', 'longitude']].values
        pred_coords = self.predicted_data[['latitude', 'longitude']].values
        
        for real_point in real_coords:
            distances = [geodesic(real_point, pred_point).kilometers for pred_point in pred_coords]
            closest_idx = np.argmin(distances)
            closest_pred = pred_coords[closest_idx]
            
            plt.plot([real_point[1], closest_pred[1]], [real_point[0], closest_pred[0]], 
                    'gray', linewidth=0.5, alpha=0.5)
        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Spatial Comparison: Real vs Predicted Earthquake Clusters')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_folder}/spatial_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_distance_distribution(self, output_folder):
        """Plot distribution of distances between real and predicted clusters"""
        if 'spatial_accuracy' not in self.comparison_results:
            return
            
        distances = self.comparison_results['spatial_accuracy']['closest_distances']
        
        plt.figure(figsize=(12, 8))
        
        # Histogram
        plt.subplot(2, 2, 1)
        plt.hist(distances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(distances), color='red', linestyle='--', label=f'Mean: {np.mean(distances):.1f} km')
        plt.axvline(np.median(distances), color='orange', linestyle='--', label=f'Median: {np.median(distances):.1f} km')
        plt.xlabel('Distance (km)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Closest Distances')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(2, 2, 2)
        plt.boxplot(distances, vert=True)
        plt.ylabel('Distance (km)')
        plt.title('Distance Distribution (Box Plot)')
        plt.grid(True, alpha=0.3)
        
        # Cumulative distribution
        plt.subplot(2, 2, 3)
        sorted_distances = np.sort(distances)
        cumulative_pct = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances) * 100
        plt.plot(sorted_distances, cumulative_pct, linewidth=2)
        plt.xlabel('Distance (km)')
        plt.ylabel('Cumulative Percentage')
        plt.title('Cumulative Distance Distribution')
        plt.grid(True, alpha=0.3)
        
        # Accuracy vs threshold
        plt.subplot(2, 2, 4)
        thresholds = np.linspace(0, np.max(distances), 50)
        accuracies = [(np.sum(np.array(distances) <= threshold) / len(distances)) * 100 for threshold in thresholds]
        plt.plot(thresholds, accuracies, linewidth=2, color='green')
        plt.xlabel('Distance Threshold (km)')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy vs Distance Threshold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_folder}/distance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_temporal_comparison(self, output_folder):
        """Plot temporal comparison focusing on timing patterns and spatial accuracy over time"""
        temporal_data = self.comparison_results['temporal_accuracy']
        
        plt.figure(figsize=(15, 10))
        
        # Period coverage comparison
        plt.subplot(2, 2, 1)
        real_periods = set(temporal_data['real_periods'])
        pred_periods = set(temporal_data['predicted_periods'])
        all_periods = sorted(real_periods.union(pred_periods))
        
        real_coverage = [1 if period in real_periods else 0 for period in all_periods]
        pred_coverage = [1 if period in pred_periods else 0 for period in all_periods]
        
        x_pos = np.arange(len(all_periods))
        plt.bar(x_pos - 0.2, real_coverage, 0.4, label='Real Data', alpha=0.7, color='blue')
        plt.bar(x_pos + 0.2, pred_coverage, 0.4, label='Predicted Data', alpha=0.7, color='red')
        plt.xlabel('Time Periods')
        plt.ylabel('Data Availability')
        plt.title('Temporal Coverage: Real vs Predicted')
        plt.xticks(x_pos[::max(1, len(all_periods)//10)], 
                  [all_periods[i] for i in range(0, len(all_periods), max(1, len(all_periods)//10))], 
                  rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Distance over time for matching periods
        if temporal_data['period_matches']:
            plt.subplot(2, 2, 2)
            periods = [pm['period'] for pm in temporal_data['period_matches']]
            distances = [pm['avg_distance_km'] for pm in temporal_data['period_matches']]
            
            plt.plot(periods, distances, 'g-o', linewidth=2, markersize=6)
            plt.xlabel('Time Period')
            plt.ylabel('Average Distance (km)')
            plt.title('Spatial Accuracy Over Time')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        # Cluster availability per period
        if temporal_data['period_matches']:
            plt.subplot(2, 2, 3)
            periods = [pm['period'] for pm in temporal_data['period_matches']]
            real_counts = [pm['real_clusters'] for pm in temporal_data['period_matches']]
            pred_counts = [pm['pred_clusters'] for pm in temporal_data['period_matches']]
            
            x_pos = np.arange(len(periods))
            plt.bar(x_pos - 0.2, real_counts, 0.4, label='Real Clusters', alpha=0.7, color='blue')
            plt.bar(x_pos + 0.2, pred_counts, 0.4, label='Predicted Clusters', alpha=0.7, color='red')
            plt.xlabel('Time Period')
            plt.ylabel('Number of Clusters')
            plt.title('Cluster Availability in Matching Periods')
            plt.xticks(x_pos, periods, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Summary statistics
        plt.subplot(2, 2, 4)
        plt.axis('off')
        summary_text = f"""
TEMPORAL ANALYSIS SUMMARY

Total Real Periods: {temporal_data['total_real_periods']}
Periods with Predictions: {temporal_data['periods_with_overlap']}
Temporal Coverage: {temporal_data['temporal_coverage_percentage']:.1f}%

Average Spatial Distance: {temporal_data['avg_temporal_spatial_distance_km']:.2f} km
(in periods with both real and predicted data)

Periods Analyzed: {len(temporal_data['period_matches'])}
"""
        plt.text(0.1, 0.8, summary_text, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f'{output_folder}/temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_time_shifted_comparison(self, output_folder):
        """Plot time-shifted comparison analysis"""
        time_shifted_data = self.comparison_results['time_shifted_comparison']
        
        plt.figure(figsize=(18, 12))
        
        # 1. Spatial map with time-shifted matches
        plt.subplot(2, 3, 1)
        
        # Plot plate boundaries if available
        if self.plate_boundaries:
            for boundary in self.plate_boundaries:
                lons, lats = zip(*boundary)
                plt.plot(lons, lats, 'gray', linewidth=1, alpha=0.5)
        
        # Plot all predicted clusters (light)
        plt.scatter(self.predicted_data['longitude'], self.predicted_data['latitude'], 
                   c='lightcoral', s=20, alpha=0.3, marker='^', label='All Predicted')
        
        # Plot time-shifted matches
        for match_data in time_shifted_data['detailed_matches']:
            # Plot real cluster
            plt.scatter(match_data['real_lon'], match_data['real_lat'], 
                       c='blue', s=100, alpha=0.8, marker='o', edgecolors='black')
            # Plot corresponding predicted cluster
            plt.scatter(match_data['pred_lon'], match_data['pred_lat'], 
                       c='red', s=100, alpha=0.8, marker='^', edgecolors='black')
            # Draw connection line
            plt.plot([match_data['real_lon'], match_data['pred_lon']], 
                    [match_data['real_lat'], match_data['pred_lat']], 
                    'purple', alpha=0.6, linewidth=1)
        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Time-Shifted Spatial Matches\n{time_shifted_data["mapping"]}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 2. Distance distribution for time-shifted matches
        plt.subplot(2, 3, 2)
        distances = [match['distance_km'] for match in time_shifted_data['detailed_matches']]
        plt.hist(distances, bins=15, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(np.mean(distances), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(distances):.1f} km')
        plt.axvline(np.median(distances), color='orange', linestyle='--', 
                   label=f'Median: {np.median(distances):.1f} km')
        plt.xlabel('Distance (km)')
        plt.ylabel('Frequency')
        plt.title('Time-Shifted Distance Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Period-by-period accuracy
        plt.subplot(2, 3, 3)
        periods = [match['real_period'] for match in time_shifted_data['period_matches']]
        avg_distances = [match['avg_distance_km'] for match in time_shifted_data['period_matches']]
        
        plt.plot(periods, avg_distances, 'g-o', linewidth=2, markersize=6)
        threshold = time_shifted_data.get('max_distance_km', 200)
        plt.axhline(threshold, color='red', linestyle='--', alpha=0.7, label=f'{threshold}km threshold')
        plt.xlabel('Real Data Period')
        plt.ylabel('Average Distance (km)')
        plt.title('Accuracy Over Time-Shifted Periods')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Mapping timeline
        plt.subplot(2, 3, 4)
        real_periods = [match['real_period'] for match in time_shifted_data['period_matches']]
        pred_periods = [match['pred_period'] for match in time_shifted_data['period_matches']]
        
        y_pos = np.arange(len(real_periods))
        plt.barh(y_pos - 0.2, [1]*len(real_periods), 0.4, label='Real Data', alpha=0.7, color='blue')
        plt.barh(y_pos + 0.2, [1]*len(pred_periods), 0.4, label='Predicted Data', alpha=0.7, color='red')
        
        plt.yticks(y_pos, [f"{real} → {pred}" for real, pred in zip(real_periods, pred_periods)])
        plt.xlabel('Data Availability')
        plt.title('Time Period Mapping')
        plt.legend()
        
        # 5. Cumulative accuracy
        plt.subplot(2, 3, 5)
        sorted_distances = np.sort(distances)
        cumulative_pct = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances) * 100
        plt.plot(sorted_distances, cumulative_pct, linewidth=2, color='purple')
        threshold = time_shifted_data.get('max_distance_km', 200)
        plt.axvline(threshold, color='red', linestyle='--', alpha=0.7, label=f'{threshold}km threshold')
        plt.xlabel('Distance (km)')
        plt.ylabel('Cumulative Percentage')
        plt.title('Cumulative Distance Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Summary statistics
        plt.subplot(2, 3, 6)
        plt.axis('off')
        threshold = time_shifted_data.get('max_distance_km', 200)
        accuracy_key = 'accuracy_threshold_percentage' if 'accuracy_threshold_percentage' in time_shifted_data else 'accuracy_100km_percentage'
        summary_text = f"""
TIME-SHIFTED ANALYSIS SUMMARY

Mapping: {time_shifted_data['mapping']}
Mapped Periods: {time_shifted_data['total_mapped_periods']}

Mean Distance: {time_shifted_data['mean_distance_km']:.2f} km
Median Distance: {time_shifted_data['median_distance_km']:.2f} km
Std Deviation: {time_shifted_data['std_distance_km']:.2f} km

Accuracy (≤{threshold}km): {time_shifted_data[accuracy_key]:.1f}%

Total Matches: {len(time_shifted_data['detailed_matches'])}
"""
        plt.text(0.1, 0.8, summary_text, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f'{output_folder}/time_shifted_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_summary_report(self, output_folder):
        """Create a comprehensive summary report"""
        report_path = f'{output_folder}/comparison_summary.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("EARTHQUAKE CLUSTER COMPARISON SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Data summary
            f.write("DATA SUMMARY:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Real clusters: {len(self.real_data)}\n")
            f.write(f"Predicted clusters: {len(self.predicted_data)}\n\n")
            
            # Spatial accuracy
            if 'spatial_accuracy' in self.comparison_results:
                spatial = self.comparison_results['spatial_accuracy']
                f.write("SPATIAL ACCURACY:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Mean distance to closest prediction: {spatial['mean_distance_km']:.2f} km\n")
                f.write(f"Median distance: {spatial['median_distance_km']:.2f} km\n")
                f.write(f"Standard deviation: {spatial['std_distance_km']:.2f} km\n")
                f.write(f"Location accuracy (within 200km): {spatial['location_accuracy_percentage']:.1f}%\n")
                f.write(f"Real clusters with close predictions: {spatial['matched_predictions']}/{spatial['total_real_clusters']}\n")
                f.write(f"Total predicted clusters available: {spatial['total_predicted_clusters']}\n\n")
            
            # Temporal accuracy
            if 'temporal_accuracy' in self.comparison_results:
                temporal = self.comparison_results['temporal_accuracy']
                f.write("TEMPORAL ACCURACY:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Temporal coverage: {temporal['temporal_coverage_percentage']:.1f}%\n")
                f.write(f"Periods with overlap: {temporal['periods_with_overlap']}/{temporal['total_real_periods']}\n")
                f.write(f"Average spatial distance in matching periods: {temporal['avg_temporal_spatial_distance_km']:.2f} km\n\n")
            
            # Time-shifted accuracy
            if 'time_shifted_comparison' in self.comparison_results:
                time_shifted = self.comparison_results['time_shifted_comparison']
                f.write("TIME-SHIFTED SPATIAL ACCURACY:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Time mapping: {time_shifted['mapping']}\n")
                f.write(f"Mapped periods: {time_shifted['total_mapped_periods']}\n")
                f.write(f"Mean distance: {time_shifted['mean_distance_km']:.2f} km\n")
                f.write(f"Median distance: {time_shifted['median_distance_km']:.2f} km\n")
                f.write(f"Standard deviation: {time_shifted['std_distance_km']:.2f} km\n")
                threshold = time_shifted.get('max_distance_km', 200)
                accuracy_key = 'accuracy_threshold_percentage' if 'accuracy_threshold_percentage' in time_shifted else 'accuracy_100km_percentage'
                f.write(f"Accuracy (within {threshold}km): {time_shifted[accuracy_key]:.1f}%\n\n")
            
            f.write("FILES GENERATED:\n")
            f.write("-" * 20 + "\n")
            f.write("- spatial_comparison.png: Map showing real vs predicted clusters\n")
            f.write("- distance_analysis.png: Distance distribution analysis\n")
            if 'temporal_accuracy' in self.comparison_results:
                f.write("- temporal_analysis.png: Temporal comparison analysis\n")
            if 'time_shifted_comparison' in self.comparison_results:
                f.write("- time_shifted_analysis.png: Time-shifted comparison analysis\n")
            f.write("- comparison_summary.txt: This summary report\n")
        
        print(f"   ✅ Summary report saved to: {report_path}")
        
    def save_detailed_results(self, output_folder='comparison_results'):
        """Save detailed comparison results to CSV"""
        os.makedirs(output_folder, exist_ok=True)
        
        # Save spatial comparison details
        if 'spatial_accuracy' in self.comparison_results:
            real_coords = self.real_data[['latitude', 'longitude']].values
            pred_coords = self.predicted_data[['latitude', 'longitude']].values
            
            detailed_results = []
            for i, real_point in enumerate(real_coords):
                distances = [geodesic(real_point, pred_point).kilometers for pred_point in pred_coords]
                closest_idx = np.argmin(distances)
                closest_distance = distances[closest_idx]
                closest_pred = pred_coords[closest_idx]
                
                detailed_results.append({
                    'real_lat': real_point[0],
                    'real_lon': real_point[1],
                    'closest_pred_lat': closest_pred[0],
                    'closest_pred_lon': closest_pred[1],
                    'distance_km': closest_distance,
                    'within_200km': closest_distance <= 200
                })
            
            pd.DataFrame(detailed_results).to_csv(f'{output_folder}/detailed_spatial_comparison.csv', index=False)
            print(f"   ✅ Detailed spatial results saved to: {output_folder}/detailed_spatial_comparison.csv")
        
        # Save temporal comparison details
        if 'temporal_accuracy' in self.comparison_results:
            temporal_data = self.comparison_results['temporal_accuracy']
            
            # Create detailed temporal comparison DataFrame
            temporal_details = []
            if 'period_matches' in temporal_data and temporal_data['period_matches']:
                for match in temporal_data['period_matches']:
                    temporal_details.append({
                        'period': match['period'],
                        'real_clusters': match['real_clusters'],
                        'pred_clusters': match['pred_clusters'],
                        'avg_distance_km': match['avg_distance_km']
                    })
            
            if temporal_details:
                pd.DataFrame(temporal_details).to_csv(f'{output_folder}/temporal_comparison.csv', index=False)
                print(f"   ✅ Temporal comparison saved to: {output_folder}/temporal_comparison.csv")
            else:
                print("   ⚠️ No temporal matches found to save")
        
        # Save time-shifted comparison details
        if 'time_shifted_comparison' in self.comparison_results:
            time_shifted_data = self.comparison_results['time_shifted_comparison']
            
            if 'detailed_matches' in time_shifted_data and time_shifted_data['detailed_matches']:
                pd.DataFrame(time_shifted_data['detailed_matches']).to_csv(f'{output_folder}/time_shifted_detailed.csv', index=False)
                print(f"   ✅ Time-shifted detailed results saved to: {output_folder}/time_shifted_detailed.csv")
                
                # Also save period mapping summary
                period_summary = []
                for match in time_shifted_data['period_matches']:
                    period_summary.append({
                        'real_period': match['real_period'],
                        'pred_period': match['pred_period'],
                        'avg_distance_km': match['avg_distance_km'],
                        'real_clusters': match['real_clusters'],
                        'pred_clusters': match['pred_clusters']
                    })
                pd.DataFrame(period_summary).to_csv(f'{output_folder}/time_shifted_periods.csv', index=False)
                print(f"   ✅ Time-shifted period mapping saved to: {output_folder}/time_shifted_periods.csv")
            else:
                print("   ⚠️ No time-shifted matches found to save")

def main():
    """Main function to run the earthquake cluster comparison"""
    print("🌍 EARTHQUAKE CLUSTER COMPARISON TOOL")
    print("=" * 50)
    
    # File paths - UPDATE THESE PATHS TO YOUR FILES
    real_csv = 'real.csv'  # Replace with your real cluster data CSV
    predicted_csv = 'predicted_events.csv'   # Replace with your predicted cluster data CSV
    plate_boundaries = 'eu_in_plates.geojson'  # Optional: plate boundaries file
    
    # Check if files exist
    if not os.path.exists(real_csv):
        print(f"❌ Real cluster data file not found: {real_csv}")
        print("Please update the 'real_csv' variable with the correct path to your real cluster data.")
        return
        
    if not os.path.exists(predicted_csv):
        print(f"❌ Predicted cluster data file not found: {predicted_csv}")
        print("Please update the 'predicted_csv' variable with the correct path to your predicted cluster data.")
        return
    
    # Initialize comparator
    comparator = EarthquakeClusterComparator(
        real_csv_path=real_csv,
        predicted_csv_path=predicted_csv,
        plate_boundaries_path=plate_boundaries if os.path.exists(plate_boundaries) else None
    )
    
    # Load data
    comparator.load_data()
    
    if comparator.real_data is None or comparator.predicted_data is None:
        print("❌ Failed to load data. Please check your CSV files.")
        return
    
    # Calculate accuracy metrics
    comparator.calculate_spatial_accuracy(max_distance_km=200)
    comparator.calculate_temporal_accuracy()
    
    # Calculate time-shifted comparison (Dec-2009 real → July-2010 predicted)
    comparator.calculate_time_shifted_comparison(real_start_month="2009-12", pred_start_month="2010-07", max_distance_km=200)
    
    # Create visualizations
    comparator.create_comparison_visualizations()
    
    # Save detailed results
    comparator.save_detailed_results()
    
    print("\n🎉 Comparison analysis complete!")
    print("Check the 'comparison_results' folder for all outputs.")

if __name__ == "__main__":
    main()