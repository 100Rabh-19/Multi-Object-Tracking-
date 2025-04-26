# utils/dataset_converter.py
import os
import csv
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path


def mot_to_kitti(mot_file: str, output_file: str):
    """
    Convert MOTChallenge format to KITTI format
    
    Args:
        mot_file: Path to MOTChallenge GT file
        output_file: Path to output KITTI format file
    """
    # Read MOT file
    try:
        df = pd.read_csv(
            mot_file, 
            header=None, 
            names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class', 'visibility'],
            sep=','
        )
    except Exception as e:
        print(f"Error reading MOT file: {e}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert and write to KITTI format
    with open(output_file, 'w') as f:
        for _, row in df.iterrows():
            # Extract data
            frame_id = int(row['frame']) - 1  # KITTI is 0-indexed
            track_id = int(row['id'])
            x1 = float(row['bb_left'])
            y1 = float(row['bb_top'])
            w = float(row['bb_width'])
            h = float(row['bb_height'])
            x2 = x1 + w
            y2 = y1 + h
            
            # Map MOT class to KITTI class
            mot_class = int(row['class'])
            kitti_class = "Pedestrian" if mot_class == 1 else "Person_sitting" if mot_class == 7 else "Car" if mot_class == 3 else "DontCare"
            
            # KITTI format
            # Format: frame track_id class truncation occlusion alpha x1 y1 x2 y2 h w l x y z rotation
            # We'll set some default values for 3D info
            truncation = 0
            occlusion = 0
            alpha = -10
            h_3d, w_3d, l_3d = 1.5, 0.6, 0.8  # Default 3D dimensions
            x_3d, y_3d, z_3d = -1000, -1000, -1000  # Unknown 3D position
            rotation_y = -10
            
            # Write line
            f.write(f"{frame_id} {track_id} {kitti_class} {truncation} {occlusion} {alpha} "
                    f"{x1} {y1} {x2} {y2} {h_3d} {w_3d} {l_3d} {x_3d} {y_3d} {z_3d} {rotation_y}\n")
    
    print(f"Converted MOT file to KITTI format: {output_file}")


def kitti_to_mot(kitti_file: str, output_file: str):
    """
    Convert KITTI format to MOTChallenge format
    
    Args:
        kitti_file: Path to KITTI format file
        output_file: Path to output MOTChallenge GT file
    """
    # Read KITTI file
    kitti_data = []
    try:
        with open(kitti_file, 'r') as f:
            for line in f:
                fields = line.strip().split()
                if len(fields) < 16:
                    continue
                kitti_data.append(fields)
    except Exception as e:
        print(f"Error reading KITTI file: {e}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert and write to MOT format
    with open(output_file, 'w') as f:
        for fields in kitti_data:
            # Extract data
            frame_id = int(fields[0]) + 1  # MOT is 1-indexed
            track_id = int(fields[1])
            obj_type = fields[2]
            
            # Get bounding box
            x1 = float(fields[6])
            y1 = float(fields[7])
            x2 = float(fields[8])
            y2 = float(fields[9])
            w = x2 - x1
            h = y2 - y1
            
            # Map KITTI class to MOT class
            mot_class = 1 if obj_type == "Pedestrian" else 7 if obj_type == "Person_sitting" else 3 if obj_type == "Car" else 10
            
            # MOT format
            # Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
            conf = 1.0  # Assume GT has 100% confidence
            visibility = 1.0  # Assume fully visible
            
            # Write line
            f.write(