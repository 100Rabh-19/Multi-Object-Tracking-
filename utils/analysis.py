# utils/analysis.py
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import cv2
from collections import defaultdict


def plot_metrics_comparison(results_files: List[str], output_path: str = 'output/metrics_comparison.png'):
    """
    Compare tracking metrics across multiple runs
    
    Args:
        results_files: List of paths to metrics CSV files
        output_path: Path to save the comparison plot
    """
    # Load results from all files
    all_results = []
    names = []
    
    for file_path in results_files:
        try:
            # Extract name from file path
            name = os.path.basename(file_path).split('_metrics')[0]
            names.append(name)
            
            # Load metrics
            df = pd.read_csv(file_path)
            all_results.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not all_results:
        print("No valid results files found.")
        return
    
    # Combine results
    combined_df = pd.concat(all_results, keys=names).reset_index(level=0)
    combined_df.rename(columns={'level_0': 'Method'}, inplace=True)
    
    # Select key metrics
    key_metrics = ['mota', 'motp', 'idf1', 'precision', 'recall', 'fps']
    plot_df = combined_df[['Method'] + key_metrics].melt(
        id_vars=['Method'], 
        value_vars=key_metrics,
        var_name='Metric', 
        value_name='Value'
    )
    
    # Create plot
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    # Create grouped bar plot
    ax = sns.barplot(x='Metric', y='Value', hue='Method', data=plot_df)
    
    # Customize plot
    plt.title('Tracking Performance Metrics Comparison', fontsize=16)
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Method')
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to {output_path}")


def plot_trajectory_analysis(tracking_results_file: str, output_path: str = 'output/trajectory_analysis.png'):
    """
    Analyze and visualize object trajectories
    
    Args:
        tracking_results_file: Path to tracking results CSV
        output_path: Path to save the trajectory visualization
    """
    # Load tracking data
    # Expected format: frame_id, track_id, x1, y1, x2, y2, class_id, etc.
    try:
        df = pd.read_csv(tracking_results_file)
    except Exception as e:
        print(f"Error loading tracking results: {e}")
        return
    
    # Extract trajectory data
    trajectories = {}
    
    # Group by track_id
    for track_id, group in df.groupby('track_id'):
        # Sort by frame_id
        group = group.sort_values('frame_id')
        
        # Calculate center points
        x_centers = (group['x1'] + group['x2']) / 2
        y_centers = (group['y1'] + group['y2']) / 2
        
        trajectories[track_id] = {
            'x': x_centers.values,
            'y': y_centers.values,
            'frames': group['frame_id'].values,
            'class_id': group['class_id'].iloc[0]  # Assume class doesn't change
        }
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Generate colors for each track
    unique_classes = df['class_id'].unique()
    class_colors = {cls: plt.cm.tab10(i % 10) for i, cls in enumerate(unique_classes)}
    
    # Plot each trajectory
    for track_id, data in trajectories.items():
        color = class_colors[data['class_id']]
        plt.plot(data['x'], data['y'], '-', color=color, alpha=0.7, linewidth=1)
        plt.plot(data['x'][0], data['y'][0], 'o', color=color)  # Start point
        plt.plot(data['x'][-1], data['y'][-1], 's', color=color)  # End point
    
    # Create legend for classes
    legend_elements = [plt.Line2D([0], [0], color=color, lw=2, label=f'Class {cls}') 
                       for cls, color in class_colors.items()]
    
    plt.legend(handles=legend_elements)
    plt.title('Object Trajectories', fontsize=16)
    plt.xlabel('X coordinate', fontsize=14)
    plt.ylabel('Y coordinate', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Invert y-axis (image coordinates)
    plt.gca().invert_yaxis()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Trajectory analysis saved to {output_path}")


def generate_track_statistics(tracking_results_file: str, output_path: str = 'output/track_statistics.csv'):
    """
    Generate statistics about tracks (duration, displacement, speed, etc.)
    
    Args:
        tracking_results_file: Path to tracking results CSV
        output_path: Path to save the statistics
    """
    # Load tracking data
    try:
        df = pd.read_csv(tracking_results_file)
    except Exception as e:
        print(f"Error loading tracking results: {e}")
        return
    
    # Group by track_id
    track_stats = []
    
    for track_id, group in df.groupby('track_id'):
        # Sort by frame_id
        group = group.sort_values('frame_id')
        
        # Track duration (in frames)
        start_frame = group['frame_id'].iloc[0]
        end_frame = group['frame_id'].iloc[-1]
        duration = end_frame - start_frame + 1
        
        # Calculate centers
        x_centers = (group['x1'] + group['x2']) / 2
        y_centers = (group['y1'] + group['y2']) / 2
        
        # Calculate bounding box sizes
        widths = group['x2'] - group['x1']
        heights = group['y2'] - group['y1']
        areas = widths * heights
        
        # Start and end positions
        start_pos = (x_centers.iloc[0], y_centers.iloc[0])
        end_pos = (x_centers.iloc[-1], y_centers.iloc[-1])
        
        # Total displacement (Euclidean distance from start to end)
        displacement = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
        # Total path length
        path_length = 0
        for i in range(1, len(x_centers)):
            segment_length = np.sqrt((x_centers.iloc[i] - x_centers.iloc[i-1])**2 + 
                                    (y_centers.iloc[i] - y_centers.iloc[i-1])**2)
            path_length += segment_length
        
        # Average speed (pixels per frame)
        avg_speed = path_length / duration if duration > 0 else 0
        
        # Average size
        avg_width = widths.mean()
        avg_height = heights.mean()
        avg_area = areas.mean()
        
        # Class information
        class_id = group['class_id'].iloc[0]  # Assume class doesn't change
        
        # Add to statistics
        track_stats.append({
            'track_id': track_id,
            'class_id': class_id,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'duration': duration,
            'start_x': start_pos[0],
            'start_y': start_pos[1],
            'end_x': end_pos[0],
            'end_y': end_pos[1],
            'displacement': displacement,
            'path_length': path_length,
            'straightness': displacement / path_length if path_length > 0 else 0,
            'avg_speed': avg_speed,
            'avg_width': avg_width,
            'avg_height': avg_height,
            'avg_area': avg_area
        })
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(track_stats)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    stats_df.to_csv(output_path, index=False)
    
    print(f"Track statistics saved to {output_path}")
    
    return stats_df


def visualize_track_density(tracking_results_file: str, video_dimensions: Tuple[int, int], 
                          output_path: str = 'output/track_density.png'):
    """
    Create a heatmap showing the density of tracked objects
    
    Args:
        tracking_results_file: Path to tracking results CSV
        video_dimensions: (width, height) of the video
        output_path: Path to save the heatmap
    """
    try:
        df = pd.read_csv(tracking_results_file)
    except Exception as e:
        print(f"Error loading tracking results: {e}")
        return
    
    width, height = video_dimensions
    
    # Calculate center points of all bounding boxes
    df['center_x'] = (df['x1'] + df['x2']) / 2
    df['center_y'] = (df['y1'] + df['y2']) / 2
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    
    # Use a 2D histogram to create density map
    heatmap, xedges, yedges = np.histogram2d(
        df['center_x'], df['center_y'], 
        bins=[50, 50], 
        range=[[0, width], [0, height]]
    )
    
    # Invert the y-axis for image coordinates
    plt.imshow(
        heatmap.T, 
        origin='lower', 
        extent=[0, width, 0, height],
        cmap='viridis',
        interpolation='gaussian'
    )
    
    plt.title('Track Density Heatmap', fontsize=16)
    plt.xlabel('X coordinate', fontsize=14)
    plt.ylabel('Y coordinate', fontsize=14)
    plt.colorbar(label='Count')
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Track density heatmap saved to {output_path}")


def export_tracking_results(tracks_by_frame: Dict[int, List[Dict]], output_path: str):
    """
    Export tracking results to CSV file
    
    Args:
        tracks_by_frame: Dictionary mapping frame_id to list of track dictionaries
        output_path: Path to save the CSV file
    """
    # Flatten the data
    rows = []
    
    for frame_id, tracks in tracks_by_frame.items():
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            
            row = {
                'frame_id': frame_id,
                'track_id': track['track_id'],
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'confidence': track['confidence'],
                'class_id': track['class_id'],
                'class_name': track['class_name']
            }
            
            rows.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Sort by frame_id and track_id
    df = df.sort_values(['frame_id', 'track_id'])
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Tracking results exported to {output_path}")


def create_track_timeline_visualization(tracking_results_file: str, output_path: str = 'output/track_timeline.png'):
    """
    Create a timeline visualization showing when each track is active
    
    Args:
        tracking_results_file: Path to tracking results CSV
        output_path: Path to save the visualization
    """
    try:
        df = pd.read_csv(tracking_results_file)
    except Exception as e:
        print(f"Error loading tracking results: {e}")
        return
    
    # Group by track_id
    track_timelines = []
    
    for track_id, group in df.groupby('track_id'):
        # Get start and end frames
        start_frame = group['frame_id'].min()
        end_frame = group['frame_id'].max()
        class_id = group['class_id'].iloc[0]  # Assume class doesn't change
        
        track_timelines.append({
            'track_id': track_id,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'duration': end_frame - start_frame + 1,
            'class_id': class_id
        })
    
    # Convert to DataFrame and sort
    timeline_df = pd.DataFrame(track_timelines)
    timeline_df = timeline_df.sort_values('start_frame')
    
    # Get unique classes for coloring
    unique_classes = df['class_id'].unique()
    class_colors = {cls: plt.cm.tab10(i % 10) for i, cls in enumerate(unique_classes)}
    
    # Adjust figure size based on number of tracks
    num_tracks = len(timeline_df)
    fig_height = max(10, num_tracks * 0.4)  # 0.4 inches per track, minimum 10 inches
    plt.figure(figsize=(14, fig_height))
    
    # Plot each track as a horizontal line
    for i, (_, track) in enumerate(timeline_df.iterrows()):
        color = class_colors[track['class_id']]
        plt.hlines(
            y=i, 
            xmin=track['start_frame'], 
            xmax=track['end_frame'], 
            colors=color, 
            linewidth=6, 
            alpha=0.7
        )
    
    # Create legend for classes
    legend_elements = [plt.Line2D([0], [0], color=color, lw=4, label=f'Class {cls}') 
                       for cls, color in class_colors.items()]
    
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Track Timeline Visualization', fontsize=16)
    plt.xlabel('Frame', fontsize=14)
    plt.ylabel('Track ID (sorted by appearance)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Set y-ticks to track IDs with proper spacing
    plt.yticks(
        range(len(timeline_df)), 
        [f"ID: {tid}" for tid in timeline_df['track_id']],
        fontsize=8
    )
    
    # Add more padding for y-axis labels
    plt.subplots_adjust(right=0.85, left=0.15)
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Track timeline visualization saved to {output_path}")


def extract_track_clips(video_path: str, tracking_results_file: str, output_dir: str = 'output/track_clips'):
    """
    Extract video clips for each tracked object
    
    Args:
        video_path: Path to source video
        tracking_results_file: Path to tracking results CSV
        output_dir: Directory to save clips
    """
    try:
        df = pd.read_csv(tracking_results_file)
    except Exception as e:
        print(f"Error loading tracking results: {e}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if width == 0 or height == 0 or fps == 0:
        print(f"Error: Invalid video properties. Width: {width}, Height: {height}, FPS: {fps}")
        cap.release()
        return
    
    # Group by track_id
    for track_id, group in df.groupby('track_id'):
        # Sort by frame_id
        group = group.sort_values('frame_id')
        
        # Get track info
        start_frame = group['frame_id'].min()
        end_frame = group['frame_id'].max()
        class_id = group['class_id'].iloc[0]
        class_name = group['class_name'].iloc[0]
        
        # Skip short tracks
        if end_frame - start_frame < 10:
            continue
        
        print(f"Processing track {track_id} (class: {class_name}, frames: {start_frame}-{end_frame})")
        
        # Create output path
        output_path = os.path.join(output_dir, f"track_{track_id}_class_{class_name}_{start_frame}_{end_frame}.mp4")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Using H.264 codec which is more widely supported
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            print(f"Error: Could not create video writer for {output_path}")
            continue
        
        # Create a frame-to-bbox mapping
        frame_to_bbox = {row['frame_id']: (row['x1'], row['y1'], row['x2'], row['y2']) 
                        for _, row in group.iterrows()}
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Process frames
        for frame_id in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get bbox for this frame
            if frame_id in frame_to_bbox:
                x1, y1, x2, y2 = map(int, frame_to_bbox[frame_id])
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f"ID: {track_id} {class_name}"
                cv2.putText(
                    frame, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
            
            # Write frame
            writer.write(frame)
        
        # Release writer
        writer.release()
        
    # Release video
    cap.release()
    
    print(f"Track clips extracted to {output_dir}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Track analysis utilities')
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['compare', 'trajectory', 'stats', 'density', 'timeline', 'clips'],
                        help='Analysis mode')
    parser.add_argument('--input', type=str, required=True, 
                        help='Input file(s), comma-separated for compare mode')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output path')
    parser.add_argument('--video', type=str, default=None,
                        help='Video path for clip extraction')
    parser.add_argument('--width', type=int, default=1920,
                        help='Video width for density map')
    parser.add_argument('--height', type=int, default=1080,
                        help='Video height for density map')
    
    args = parser.parse_args()
    
    if args.mode == 'compare':
        files = args.input.split(',')
        output = args.output or 'output/metrics_comparison.png'
        plot_metrics_comparison(files, output)
    
    elif args.mode == 'trajectory':
        output = args.output or 'output/trajectory_analysis.png'
        plot_trajectory_analysis(args.input, output)
    
    elif args.mode == 'stats':
        output = args.output or 'output/track_statistics.csv'
        generate_track_statistics(args.input, output)
    
    elif args.mode == 'density':
        output = args.output or 'output/track_density.png'
        visualize_track_density(args.input, (args.width, args.height), output)
    
    elif args.mode == 'timeline':
        output = args.output or 'output/track_timeline.png'
        create_track_timeline_visualization(args.input, output)
    
    elif args.mode == 'clips':
        if not args.video:
            print("Error: --video argument is required for clips mode")
        else:
            output = args.output or 'output/track_clips'
            extract_track_clips(args.video, args.input, output)