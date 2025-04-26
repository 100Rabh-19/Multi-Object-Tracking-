# evaluation/metrics.py
import numpy as np
import motmetrics as mm
from typing import List, Dict, Tuple, Optional, Union
import pandas as pd
import time

class MOTEvaluator:
    """
    Evaluator for Multi-Object Tracking
    Calculates standard MOT metrics including MOTA, MOTP, IDF1, etc.
    """
    def __init__(self):
        """
        Initialize MOT evaluator
        """
        # Create metrics accumulator
        self.acc = mm.MOTAccumulator(auto_id=True)
        
        # Track FPS measurements
        self.frame_times = []
        self.start_time = None
        
    def reset(self):
        """
        Reset the evaluator for new evaluation
        """
        self.acc = mm.MOTAccumulator(auto_id=True)
        self.frame_times = []
        self.start_time = None
    
    def start_frame(self):
        """
        Start timing a new frame
        """
        self.start_time = time.time()
    
    def end_frame(self):
        """
        End timing for current frame
        """
        if self.start_time is not None:
            frame_duration = time.time() - self.start_time
            self.frame_times.append(frame_duration)
            self.start_time = None
    
    def get_fps(self) -> float:
        """
        Calculate frames per second
        
        Returns:
            Average FPS over all recorded frames
        """
        if not self.frame_times:
            return 0.0
        
        mean_frame_time = np.mean(self.frame_times)
        return 1.0 / mean_frame_time if mean_frame_time > 0 else 0.0
    
    def update(
        self,
        gt_objects: List[Dict],
        predicted_objects: List[Dict],
        frame_id: int = None
    ):
        """
        Update metrics with ground truth and predictions for one frame
        
        Args:
            gt_objects: List of ground truth objects with 'track_id' and 'bbox' keys
            predicted_objects: List of predicted objects with 'track_id' and 'bbox' keys
            frame_id: Optional frame ID (auto-incremented if None)
        """
        # Extract object IDs and boxes
        gt_ids = [obj['track_id'] for obj in gt_objects]
        pred_ids = [obj['track_id'] for obj in predicted_objects]
        
        # Convert bounding boxes to (x, y, width, height) format for distance calculation
        gt_boxes = []
        for obj in gt_objects:
            x1, y1, x2, y2 = obj['bbox']
            gt_boxes.append([x1, y1, x2 - x1, y2 - y1]) 
            
        pred_boxes = []
        for obj in predicted_objects:
            x1, y1, x2, y2 = obj['bbox']
            pred_boxes.append([x1, y1, x2 - x1, y2 - y1])
        
        # Calculate IoU distances between GT and predictions
        distances = mm.distances.iou_matrix(
            gt_boxes, pred_boxes, 
            max_iou=0.5  # Max IoU threshold for considering a match
        )
        
        # Update accumulator
        self.acc.update(gt_ids, pred_ids, distances, frame_id=frame_id)
    
    def compute_metrics(self) -> Dict:
        """
        Compute MOT metrics from accumulated data
        
        Returns:
            Dictionary with computed metrics
        """
        mh = mm.metrics.create()
        summary = mh.compute(
            self.acc, 
            metrics=[
                'num_frames', 'mota', 'motp', 'idf1', 
                'mostly_tracked', 'partially_tracked', 'mostly_lost',
                'num_false_positives', 'num_misses', 'num_switches',
                'precision', 'recall'
            ],
            name='Summary'
        )
        
        # Convert results to dictionary
        metrics = summary.to_dict('records')[0]
        
        # Add FPS information
        metrics['fps'] = self.get_fps()
        
        return metrics
    
    def print_metrics(self, metrics: Dict = None):
        """
        Print formatted metrics
        
        Args:
            metrics: Metrics dictionary (if None, compute metrics)
        """
        if metrics is None:
            metrics = self.compute_metrics()
        
        print("\n=== MOT Performance Metrics ===")
        print(f"MOTA: {metrics['mota']:.2f}%")
        print(f"MOTP: {metrics['motp']:.2f}%")
        print(f"IDF1: {metrics['idf1']:.2f}%")
        print(f"Precision: {metrics['precision']:.2f}%")
        print(f"Recall: {metrics['recall']:.2f}%")
        print(f"Mostly Tracked: {metrics['mostly_tracked']}")
        print(f"Partially Tracked: {metrics['partially_tracked']}")
        print(f"Mostly Lost: {metrics['mostly_lost']}")
        print(f"False Positives: {metrics['num_false_positives']}")
        print(f"Misses: {metrics['num_misses']}")
        print(f"ID Switches: {metrics['num_switches']}")
        print(f"FPS: {metrics['fps']:.2f}")
        print("==============================\n")
        
        return metrics


class GroundTruthLoader:
    """
    Load ground truth annotations from standard datasets
    """
    def __init__(self, dataset_type: str = "MOTChallenge"):
        """
        Initialize GT loader
        
        Args:
            dataset_type: Type of dataset ('MOTChallenge', 'KITTI', etc.)
        """
        self.dataset_type = dataset_type
    
    def load_sequence(self, gt_file_path: str) -> Dict[int, List[Dict]]:
        """
        Load ground truth for a sequence
        
        Args:
            gt_file_path: Path to ground truth file
            
        Returns:
            Dictionary mapping frame_id to list of ground truth objects
        """
        if self.dataset_type == "MOTChallenge":
            return self._load_mot_challenge(gt_file_path)
        elif self.dataset_type == "KITTI":
            return self._load_kitti(gt_file_path)
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
    
    def _load_mot_challenge(self, gt_file_path: str) -> Dict[int, List[Dict]]:
        """
        Load MOTChallenge format ground truth
        Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
        
        Args:
            gt_file_path: Path to ground truth file
            
        Returns:
            Dictionary mapping frame_id to list of ground truth objects
        """
        gt_data = {}
        
        try:
            # Read the ground truth file
            df = pd.read_csv(
                gt_file_path, 
                header=None, 
                names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class', 'visibility'],
                sep=','
            )
            
            # Group by frame
            for frame, group in df.groupby('frame'):
                gt_objects = []
                
                for _, row in group.iterrows():
                    # Convert from <x,y,w,h> to <x1,y1,x2,y2>
                    x1 = int(row['bb_left'])
                    y1 = int(row['bb_top'])
                    x2 = int(row['bb_left'] + row['bb_width'])
                    y2 = int(row['bb_top'] + row['bb_height'])
                    
                    # Only include valid objects (visibility > 0)
                    if row['visibility'] > 0:
                        gt_objects.append({
                            'track_id': int(row['id']),
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(row['conf']),
                            'class_id': int(row['class']),
                            'class_name': self._get_class_name(int(row['class']))
                        })
                
                gt_data[int(frame)] = gt_objects
                
        except Exception as e:
            print(f"Error loading MOTChallenge ground truth: {e}")
            return {}
        
        return gt_data
    
    def _load_kitti(self, gt_file_path: str) -> Dict[int, List[Dict]]:
        """
        Load KITTI format ground truth
        
        Args:
            gt_file_path: Path to ground truth file
            
        Returns:
            Dictionary mapping frame_id to list of ground truth objects
        """
        gt_data = {}
        
        try:
            # Assuming each line contains annotations for one frame
            # Format varies, but typically includes frame, track ID, class, and bbox
            # Adapt this based on the specific KITTI format you're using
            
            with open(gt_file_path, 'r') as f:
                for line in f:
                    fields = line.strip().split(' ')
                    
                    if len(fields) < 6:
                        continue
                    
                    frame = int(fields[0])
                    track_id = int(fields[1])
                    obj_type = fields[2]
                    
                    # Extract bounding box (format depends on KITTI variant)
                    # This is a simplified version - adjust as needed
                    x1 = float(fields[6])
                    y1 = float(fields[7])
                    x2 = float(fields[8])
                    y2 = float(fields[9])
                    
                    # Create or append to frame's object list
                    if frame not in gt_data:
                        gt_data[frame] = []
                        
                    gt_data[frame].append({
                        'track_id': track_id,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'class_id': self._get_kitti_class_id(obj_type),
                        'class_name': obj_type,
                        'confidence': 1.0  # Ground truth is always 100% confidence
                    })
                    
        except Exception as e:
            print(f"Error loading KITTI ground truth: {e}")
            return {}
        
        return gt_data
    
    def _get_class_name(self, class_id: int) -> str:
        """
        Get class name from class ID for MOTChallenge
        
        Args:
            class_id: Class ID
            
        Returns:
            Class name
        """
        # MOTChallenge default class mapping
        class_map = {
            1: 'pedestrian',
            2: 'person on vehicle',
            3: 'car',
            4: 'bicycle',
            5: 'motorbike',
            6: 'non motorized vehicle',
            7: 'static person',
            8: 'distractor',
            9: 'occluder',
            10: 'occluder on the ground',
            11: 'occluder full',
            12: 'reflection'
        }
        
        return class_map.get(class_id, f'unknown_{class_id}')
    
    def _get_kitti_class_id(self, class_name: str) -> int:
        """
        Get class ID from class name for KITTI
        
        Args:
            class_name: Class name
            
        Returns:
            Class ID
        """
        # KITTI default class mapping
        class_map = {
            'Car': 1,
            'Van': 2,
            'Truck': 3,
            'Pedestrian': 4,
            'Person_sitting': 5,
            'Cyclist': 6,
            'Tram': 7,
            'Misc': 8,
            'DontCare': 9
        }
        
        return class_map.get(class_name, 0)