# detectors/yolo_detector.py
import torch
import numpy as np
from ultralytics import YOLO
import cv2
from typing import List, Dict, Tuple, Optional, Union

class YOLODetector:
    def __init__(
        self, 
        model_path: str = 'yolov8n.pt',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = '',  # Auto-select device
        classes: Optional[List[int]] = None  # Filter by class IDs
    ):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            device: Device to run model on ('cpu', '0', '0,1', etc.)
            classes: List of class IDs to detect (None for all classes)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        
        # Load YOLO model
        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Running YOLOv8 detector on device: {self.device}")
        
        # Get class names if available
        self.class_names = self.model.names
        print(f"Available classes: {self.class_names}")
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in a frame
        
        Args:
            frame: Image frame (BGR format from OpenCV)
            
        Returns:
            List of dictionaries containing:
                - bbox: [x1, y1, x2, y2] (absolute pixel coordinates)
                - confidence: detection confidence
                - class_id: class ID of the detected object
                - class_name: class name of the detected object
        """
        # Run YOLOv8 inference
        results = self.model(
            frame, 
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False
        )
        
        detections = []
        
        # Process detections
        for result in results:
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                # Get detection info
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names[class_id]
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                }
                detections.append(detection)
                
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """
        Detect objects in a batch of frames
        
        Args:
            frames: List of image frames
            
        Returns:
            List of detection results for each frame
        """
        results = []
        for frame in frames:
            results.append(self.detect(frame))
        return results
    
    def visualize_detections(
        self, 
        frame: np.ndarray, 
        detections: List[Dict],
        color_map: Dict = None,
        thickness: int = 2,
        text_size: float = 0.5,
        text_thickness: int = 2
    ) -> np.ndarray:
        """
        Visualize detections on an image
        
        Args:
            frame: Image to visualize on
            detections: List of detection dictionaries
            color_map: Dictionary mapping class_id to BGR color tuples
            thickness: Bounding box thickness
            text_size: Size of text
            text_thickness: Thickness of text
            
        Returns:
            Frame with visualized detections
        """
        vis_frame = frame.copy()
        
        # Create color map if not provided
        if color_map is None:
            np.random.seed(42)  # For consistent colors
            color_map = {}
            for detection in detections:
                class_id = detection['class_id']
                if class_id not in color_map:
                    color_map[class_id] = tuple(map(int, np.random.randint(0, 255, 3)))
        
        # Draw bounding boxes and labels
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_id = detection['class_id']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Get color
            color = color_map.get(class_id, (0, 255, 0))  # Default to green
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
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
            
        return vis_frame