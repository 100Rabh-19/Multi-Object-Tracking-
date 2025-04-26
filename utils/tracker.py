# utils/tracker.py
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional, Union, Any
import cv2

class Detection:
    """
    Single detection object with bounding box and other metadata
    """
    def __init__(
        self, 
        bbox: List[int], 
        confidence: float, 
        class_id: int,
        class_name: str,
        feature: Optional[np.ndarray] = None
    ):
        """
        Initialize detection
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            confidence: Detection confidence
            class_id: Class ID
            class_name: Class name
            feature: Feature vector for appearance matching (used in DeepSORT)
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.feature = feature
        
        # Convert bbox [x1, y1, x2, y2] to [x, y, w, h] format
        self.to_xyah()
        
    def to_xyah(self) -> np.ndarray:
        """
        Convert bbox to [x, y, aspect_ratio, height] format for Kalman filter
        """
        x1, y1, x2, y2 = self.bbox
        w, h = x2 - x1, y2 - y1
        x_center, y_center = x1 + w/2, y1 + h/2
        aspect_ratio = w / float(h)
        
        self.xyah = np.array([x_center, y_center, aspect_ratio, h])
        return self.xyah
    
    def to_tlbr(self) -> List[int]:
        """
        Convert centered format back to [x1, y1, x2, y2]
        """
        x, y, aspect_ratio, h = self.xyah
        w = aspect_ratio * h
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        
        return [x1, y1, x2, y2]


class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bounding boxes.
    """
    count = 0
    
    def __init__(self, detection: Detection, max_age: int = 1):
        """
        Initialize a tracker using initial bounding box
        
        Args:
            detection: Initial detection
            max_age: Maximum number of frames to keep alive without matching
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix (motion model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix (maps state to measurement space)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise covariance
        self.kf.R[2:, 2:] *= 10.0
        
        # Process noise covariance
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initial state covariance
        self.kf.P[4:, 4:] *= 1000.0  # Give high uncertainty to velocities
        self.kf.P *= 10.0
        
        # Initialize state with detection
        self.kf.x[:4] = detection.xyah.reshape(4, 1)
        
        # Tracking metadata
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1  # Number of total hits including the first detection
        self.hit_streak = 1  # Number of consecutive hits
        self.age = 0  # Number of frames since first detection
        
        # Store detection info
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.last_confidence = detection.confidence
        self.last_detection = detection
        self.max_age = max_age
        
        # For DeepSORT
        self.features = []
        if detection.feature is not None:
            self.features.append(detection.feature)
        
    def update(self, detection: Detection) -> None:
        """
        Update tracker with matched detection
        
        Args:
            detection: Matched detection
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        # Update Kalman filter state
        self.kf.update(detection.xyah.reshape(4, 1))
        
        # Update metadata
        self.last_detection = detection
        self.last_confidence = detection.confidence
        
        # Update feature history for appearance matching (DeepSORT)
        if detection.feature is not None:
            self.features.append(detection.feature)
            if len(self.features) > 100:  # Limit feature history
                self.features.pop(0)
    
    def predict(self) -> np.ndarray:
        """
        Advances the state vector and returns the predicted bounding box
        
        Returns:
            Predicted bounding box in [x1, y1, x2, y2] format
        """
        # Increment trackers' age
        self.age += 1
        self.time_since_update += 1
        self.hit_streak = 0 if self.time_since_update > 0 else self.hit_streak
        
        # Get predicted location from Kalman filter
        self.kf.predict()
        self.history.append(self.state_as_tlbr())
        
        return self.history[-1]
    
    def state_as_tlbr(self) -> np.ndarray:
        """
        Convert Kalman filter state to bounding box in [x1, y1, x2, y2] format
        
        Returns:
            Bounding box in [x1, y1, x2, y2] format
        """
        x, y, aspect_ratio, height = self.kf.x[:4].flatten()
        width = aspect_ratio * height
        
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        x1 = x - width / 2
        y1 = y - height / 2
        x2 = x + width / 2
        y2 = y + height / 2
        
        return np.array([x1, y1, x2, y2])
    
    def get_state(self) -> Dict:
        """
        Returns the current state as a dictionary
        
        Returns:
            Dictionary with tracking info
        """
        bbox = self.state_as_tlbr().astype(int).tolist()
        return {
            'track_id': self.id,
            'bbox': bbox,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.last_confidence,
            'age': self.age,
            'time_since_update': self.time_since_update
        }


def iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Compute IOU between two bounding boxes
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IOU score
    """
    # Get the coordinates of bounding boxes
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Get the coordinates of the intersection rectangle
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    # Return 0 if there's no intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Compute intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Compute union area
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - intersection_area
    
    # Compute IOU
    iou = intersection_area / union_area
    
    return iou


class SORT:
    """
    Simple Online and Realtime Tracking (SORT) algorithm
    """
    def __init__(
        self,
        max_age: int = 30,  # Maximum frames to keep track alive without matching
        min_hits: int = 3,  # Minimum hits needed before track is established
        iou_threshold: float = 0.3,  # IOU threshold for match consideration
        use_class_info: bool = True  # Use class information for tracking
    ):
        """
        Initialize SORT tracker
        
        Args:
            max_age: Maximum frames to keep track alive without matching
            min_hits: Minimum hits needed before track is established
            iou_threshold: IOU threshold for match consideration
            use_class_info: Use class information for tracking
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.use_class_info = use_class_info
        
        self.trackers = []  # List of active trackers
        self.frame_count = 0
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update trackers with new detections
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            List of track dictionaries
        """
        self.frame_count += 1
        
        # Convert detections to internal format
        detection_objs = []
        for det in detections:
            detection_objs.append(
                Detection(
                    bbox=det['bbox'],
                    confidence=det['confidence'],
                    class_id=det['class_id'],
                    class_name=det['class_name'],
                    feature=det.get('feature', None)
                )
            )
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        # Remove invalid trackers
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # Calculate IoU distance matrix between detections and predictions
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            detection_objs, trks
        )
        
        # Update matched trackers with assigned detections
        for t, d in matched:
            self.trackers[t].update(detection_objs[d])
        
        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detection_objs[i], max_age=self.max_age)
            self.trackers.append(trk)
        
        # Get outputs from trackers
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            
            # Only return confirmed tracks
            if ((trk.time_since_update < 1) and 
                (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(d)
            
            # Remove dead tracklets
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        return ret
    
    def _associate_detections_to_trackers(
        self, 
        detections: List[Detection], 
        trackers: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Assigns detections to tracked objects using IoU
        
        Args:
            detections: List of detections
            trackers: Numpy array of predicted tracker locations
            
        Returns:
            Tuple of matches, unmatched_detections, unmatched_trackers
        """
        if len(trackers) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(trackers)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(detections), len(trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                # Check class compatibility if using class info
                if self.use_class_info and det.class_id != self.trackers[t].class_id:
                    iou_matrix[d, t] = 0.0
                else:
                    iou_matrix[d, t] = iou(det.to_tlbr(), trk)
        
        # Apply Hungarian algorithm for optimal assignment
        matched_indices = []
        if min(iou_matrix.shape) > 0:
            # Linear assignment using Hungarian algorithm
            row_indices, col_indices = linear_sum_assignment(-iou_matrix)
            for row, col in zip(row_indices, col_indices):
                # Filter matches with low IoU
                if iou_matrix[row, col] >= self.iou_threshold:
                    matched_indices.append((col, row))
        
        # Find unmatched detections and trackers
        unmatched_detections = [d for d in range(len(detections)) if not any(d == det for _, det in matched_indices)]
        unmatched_trackers = [t for t in range(len(trackers)) if not any(t == trk for trk, _ in matched_indices)]
        
        # Reorder matches to be (tracker_idx, detection_idx)
        matches = [(t, d) for t, d in matched_indices]
        
        return matches, unmatched_detections, unmatched_trackers


class DeepSORT(SORT):
    """
    DeepSORT tracking algorithm with appearance features
    """
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        use_class_info: bool = True,
        max_cosine_distance: float = 0.2,
        nn_budget: int = 100  # Maximum size of appearance descriptor gallery
    ):
        """
        Initialize DeepSORT tracker
        
        Args:
            max_age: Maximum frames to keep track alive without matching
            min_hits: Minimum hits needed before track is established
            iou_threshold: IOU threshold for match consideration
            use_class_info: Use class information for tracking
            max_cosine_distance: Maximum cosine distance for feature matching
            nn_budget: Maximum size of appearance descriptor gallery
        """
        super().__init__(max_age, min_hits, iou_threshold, use_class_info)
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
    
    def _associate_detections_to_trackers(
        self, 
        detections: List[Detection], 
        trackers: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Assigns detections to tracked objects using both IoU and appearance features
        
        Args:
            detections: List of detections
            trackers: Numpy array of predicted tracker locations
            
        Returns:
            Tuple of matches, unmatched_detections, unmatched_trackers
        """
        if len(trackers) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(trackers)))
        
        # First, perform matching based on IoU
        iou_matrix = np.zeros((len(detections), len(trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                # Check class compatibility
                if self.use_class_info and det.class_id != self.trackers[t].class_id:
                    iou_matrix[d, t] = 0.0
                else:
                    iou_matrix[d, t] = iou(det.to_tlbr(), trk)
        
        # Get high IoU matches
        iou_matched_indices = []
        if min(iou_matrix.shape) > 0:
            row_indices, col_indices = linear_sum_assignment(-iou_matrix)
            for row, col in zip(row_indices, col_indices):
                if iou_matrix[row, col] >= self.iou_threshold:
                    iou_matched_indices.append((col, row))
        
        # Find unmatched detections and trackers after IoU matching
        unmatched_detections = [d for d in range(len(detections)) if not any(d == det for _, det in iou_matched_indices)]
        unmatched_trackers = [t for t in range(len(trackers)) if not any(t == trk for trk, _ in iou_matched_indices)]
        
        # If no appearance features, fall back to IoU matching
        if len(unmatched_detections) == 0 or len(unmatched_trackers) == 0:
            return iou_matched_indices, unmatched_detections, unmatched_trackers
        
        # Filter detections and trackers with features for appearance matching
        feature_detections = [d for d in unmatched_detections if detections[d].feature is not None]
        feature_trackers = [t for t in unmatched_trackers if len(self.trackers[t].features) > 0]
        
        if len(feature_detections) == 0 or len(feature_trackers) == 0:
            return iou_matched_indices, unmatched_detections, unmatched_trackers
        
        # Calculate appearance similarity using cosine distance
        cost_matrix = np.zeros((len(feature_detections), len(feature_trackers)))
        
        for i, d in enumerate(feature_detections):
            det_feature = detections[d].feature
            
            for j, t in enumerate(feature_trackers):
                # Use mean of tracker's feature history
                track_features = np.array(self.trackers[t].features)
                track_feature = np.mean(track_features, axis=0)
                
                # Calculate cosine similarity
                similarity = np.dot(det_feature, track_feature) / (
                    np.linalg.norm(det_feature) * np.linalg.norm(track_feature)
                )
                
                # Convert to distance (1 - similarity)
                cost_matrix[i, j] = 1.0 - similarity
        
        # Match features using Hungarian algorithm
        feature_matches = []
        if min(cost_matrix.shape) > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            for row, col in zip(row_indices, col_indices):
                # Only keep matches with small distance
                if cost_matrix[row, col] < self.max_cosine_distance:
                    # Map back to original indices
                    feature_matches.append((feature_trackers[col], feature_detections[row]))
        
        # Update unmatched lists
        for t, d in feature_matches:
            unmatched_detections.remove(d)
            unmatched_trackers.remove(t)
        
        # Combine IoU and feature matches
        matches = iou_matched_indices + feature_matches
        
        return matches, unmatched_detections, unmatched_trackers


def visualize_tracks(
    frame: np.ndarray, 
    tracks: List[Dict], 
    color_map: Dict = None,
    thickness: int = 2, 
    text_size: float = 0.5, 
    text_thickness: int = 2
) -> np.ndarray:
    """
    Visualize tracks on an image
    
    Args:
        frame: Image to visualize on
        tracks: List of track dictionaries
        color_map: Dictionary mapping track_id to BGR color tuples
        thickness: Bounding box thickness
        text_size: Size of text
        text_thickness: Thickness of text
        
    Returns:
        Frame with visualized tracks
    """
    vis_frame = frame.copy()
    
    # Create color map if not provided
    if color_map is None:
        np.random.seed(42)  # For consistent colors
        color_map = {}
    
    # Draw bounding boxes and labels
    for track in tracks:
        track_id = track['track_id']
        x1, y1, x2, y2 = track['bbox']
        class_name = track['class_name']
        confidence = track['confidence']
        
        # Get color (persistent color per track ID)
        if track_id not in color_map:
            color_map[track_id] = tuple(map(int, np.random.randint(0, 255, 3)))
        color = color_map[track_id]
        
        # Draw bounding box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label background
        label = f"ID: {track_id} {class_name}: {confidence:.2f}"
        text_size_px, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)
        cv2.rectangle(vis_frame, (x1, y1 - text_size_px[1] - 5), (x1 + text_size_px[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(
            vis_frame, 
            label, 
            (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            text_size, 
            (255, 255, 255), 
            text_thickness
        )
        
    return vis_frame, color_map