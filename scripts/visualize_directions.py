import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os

def main():
    # Load data
    csv_file = "cluster_directions_1980_2005.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)
    df['From_Time'] = pd.to_datetime(df['From_Time'], format='mixed')

    # Create output directory for arrow images
    output_dir = "arrow_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # -------------------------------------------------------
    # Part 1: Arrow Grids (9 per page)
    # -------------------------------------------------------
    rows_per_page = 9
    num_pages = math.ceil(len(df) / rows_per_page)
    
    print(f"Generating {num_pages} pages of arrow plots...")

    for page in range(num_pages):
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle(f'Cluster Directions (Page {page+1}/{num_pages})', fontsize=16)
        axes = axes.flatten()
        
        start_idx = page * rows_per_page
        end_idx = min((page + 1) * rows_per_page, len(df))
        subset = df.iloc[start_idx:end_idx]
        
        # Loop through the 9 (or fewer) items for this page
        for i in range(rows_per_page):
            ax = axes[i]
            
            # If we have data for this slot
            if i < len(subset):
                row = subset.iloc[i]
                angle = row['Angle_wrt_X_Axis']
                year = row['Year']
                c_from = row['From_Cluster']
                c_to = row['To_Cluster']
                
                # Convert angle to radians for plotting
                # Angle is w.r.t X-axis (0 is East, 90 is North)
                rad = np.radians(angle)
                dx = np.cos(rad)
                dy = np.sin(rad)
                
                # Draw arrow
                ax.arrow(0, 0, dx, dy, head_width=0.1, head_length=0.1, fc='blue', ec='blue', length_includes_head=True)
                
                # Setup plot limits and style
                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-1.2, 1.2)
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Label
                ax.set_title(f"{year}: {c_from}->{c_to}\nAngle: {angle}°", fontsize=10)
                
                # Add reference lines (crosshair)
                ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
                ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
                
                # Add compass labels for reference
                ax.text(1.1, 0, 'E', ha='center', va='center', fontsize=8, color='gray')
                ax.text(0, 1.1, 'N', ha='center', va='center', fontsize=8, color='gray')
            else:
                # Hide unused subplots
                ax.axis('off')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(output_dir, f'arrows_page_{page+1}.png')
        plt.savefig(save_path)
        plt.close()

    print(f"Saved arrow plots to folder '{output_dir}'")

    # -------------------------------------------------------
    # Part 2: Advanced Visualizations
    # -------------------------------------------------------
    print("Generating advanced visualizations...")
    
    # Convert angles to radians for calculations
    # Angle is w.r.t X-axis (0=East, 90=North)
    df['rad'] = np.radians(df['Angle_wrt_X_Axis'])
    
    # Calculate components
    # U = East-West component (Cos)
    # V = North-South component (Sin)
    U = np.cos(df['rad'])
    V = np.sin(df['rad'])

    # -------------------------------------------------------
    # Plot 1: Quiver Plot (Arrow Plot over Time)
    # -------------------------------------------------------
    plt.figure(figsize=(15, 4))
    # Plot arrows along a timeline
    # We use a dummy y-axis (0) for all arrows
    plt.quiver(df['From_Time'], [0]*len(df), U, V, 
               color='purple', scale=20, width=0.003, headwidth=4)
    
    plt.title("1. Quiver Plot: Direction of Movement over Time")
    plt.yticks([]) # Hide Y axis
    plt.xlabel("Time")
    plt.ylim(-0.5, 0.5) # Give some space around the arrows
    plt.tight_layout()
    plt.savefig('plot1_quiver.png')
    plt.close()
    print("Saved 'plot1_quiver.png'")

    # -------------------------------------------------------
    # Plot 2: Polar Scatter Plot (Radius = Time)
    # -------------------------------------------------------
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    
    # Map time to radius. We use the index or year to expand outwards
    # Using Year for clearer concentric circles
    min_year = df['Year'].min()
    radius = df['Year'] - min_year + 1 
    
    sc = ax.scatter(df['rad'], radius, c=df['Year'], cmap='viridis', alpha=0.8, s=50, edgecolors='k')
    
    # Configure Polar Axis
    ax.set_theta_zero_location("E") # 0 degrees is East
    ax.set_theta_direction(1)       # Counter-clockwise
    # Simplified axis labels to avoid potential layout hangs
    ax.set_rlabel_position(45)
    
    plt.colorbar(sc, label='Year')
    plt.title("2. Polar Scatter Plot (Radius = Time)\nCenter = Oldest, Outer = Newest")
    plt.savefig('plot2_polar_scatter.png')
    plt.close()
    print("Saved 'plot2_polar_scatter.png'")

    # -------------------------------------------------------
    # Plot 3: Rose Diagram (Wind Rose)
    # -------------------------------------------------------
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    
    # Create bins (e.g., 16 directions)
    num_bins = 16
    bins = np.linspace(0.0, 2 * np.pi, num_bins + 1)
    hist, _ = np.histogram(df['rad'], bins=bins)
    width = (2 * np.pi) / num_bins
    
    # Plot bars
    bars = ax.bar(bins[:-1], hist, width=width, bottom=0.0, color='skyblue', edgecolor='black', alpha=0.7)
    
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    plt.title(f"3. Rose Diagram (Frequency of Directions)\n{num_bins} Bins")
    plt.savefig('plot3_rose_diagram.png')
    plt.close()
    print("Saved 'plot3_rose_diagram.png'")

    # -------------------------------------------------------
    # Plot 4: Sine/Cosine Decomposition
    # -------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # North-South (Sin)
    ax1.plot(df['From_Time'], V, 'b-o', label='North-South (Sin)', markersize=4)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel("North-South Component\n(+1=N, -1=S)")
    ax1.set_ylim(-1.2, 1.2)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # East-West (Cos)
    ax2.plot(df['From_Time'], U, 'r-o', label='East-West (Cos)', markersize=4)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel("East-West Component\n(+1=E, -1=W)")
    ax2.set_xlabel("Time")
    ax2.set_ylim(-1.2, 1.2)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    plt.suptitle("4. Sine/Cosine Decomposition (Smooth Trends)")
    plt.tight_layout()
    plt.savefig('plot4_decomposition.png')
    plt.close()
    print("Saved 'plot4_decomposition.png'")

if __name__ == "__main__":
    main()
