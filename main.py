# main.py
import os
import cv2
import argparse
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import yaml

# Custom modules
from detectors.yolo_detector import YOLODetector
from utils.tracker import SORT, DeepSORT, visualize_tracks, Detection
from evaluation.metrics import MOTEvaluator, GroundTruthLoader

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Multi-Object Tracking with YOLO and SORT/DeepSORT')
    
    # Input/Output
    parser.add_argument('--input', type=str, required=True, help='Path to input video file or camera index')
    parser.add_argument('--output', type=str, default='output/output.mp4', help='Path to output video file')
    parser.add_argument('--save_results', action='store_true', help='Save tracking results to file')
    parser.add_argument('--show', action='store_true', help='Display output frames')
    
    # Detection
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to YOLO model')
    parser.add_argument('--conf_thresh', type=float, default=0.25, help='Detection confidence threshold')
    parser.add_argument('--iou_thresh', type=float, default=0.45, help='Detection NMS IoU threshold')
    parser.add_argument('--classes', type=int, nargs='+', default=None, help='Filter by class IDs')
    
    # Tracking
    parser.add_argument('--tracker', type=str, default='sort', choices=['sort', 'deepsort'], help='Tracking algorithm')
    parser.add_argument('--max_age', type=int, default=30, help='Maximum frames to keep track alive without matching')
    parser.add_argument('--min_hits', type=int, default=3, help='Minimum hits needed before track is established')
    parser.add_argument('--iou_tracking', type=float, default=0.3, help='IoU threshold for tracking')
    parser.add_argument('--use_class_info', action='store_true', help='Use class information for tracking')
    
    # DeepSORT specific
    parser.add_argument('--max_cosine_dist', type=float, default=0.2, help='Max cosine distance for feature matching')
    parser.add_argument('--nn_budget', type=int, default=100, help='Maximum size of appearance descriptor gallery')
    
    # Evaluation
    parser.add_argument('--eval', action='store_true', help='Evaluate tracking performance')
    parser.add_argument('--gt_file', type=str, default=None, help='Path to ground truth file')
    parser.add_argument('--dataset_type', type=str, default='MOTChallenge', choices=['MOTChallenge', 'KITTI'], 
                        help='Dataset type for evaluation')
    
    # Other
    parser.add_argument('--device', type=str, default='', help='Device to run on (e.g., cpu, 0, 0,1)')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    
    args = parser.parse_args()
    
    # Load from config file if provided (overrides command line args)
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            
        # Update args with config values (non-None values in config take precedence)
        for key, value in config.items():
            if value is not None:
                setattr(args, key, value)
    
    return args

def load_video(input_path):
    """
    Load video capture from file or camera
    
    Args:
        input_path: Path to video file or camera index
        
    Returns:
        Video capture object, width, height, fps
    """
    # Check if input is a camera index
    if input_path.isdigit():
        cap = cv2.VideoCapture(int(input_path))
    else:
        cap = cv2.VideoCapture(input_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {input_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    return cap, width, height, fps

def setup_output_writer(output_path, width, height, fps):
    """
    Setup video writer for output
    
    Args:
        output_path: Path to output video file
        width: Frame width
        height: Frame height
        fps: Frame rate
        
    Returns:
        Video writer object
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    return writer

def main():
    """Main execution function"""
    # Parse arguments
    args = parse_args()
    
    # Load video input
    print(f"Loading video from: {args.input}")
    cap, width, height, fps = load_video(args.input)
    
    # Setup output video writer if needed
    writer = None
    if args.output:
        writer = setup_output_writer(args.output, width, height, fps)
    
    # Initialize detector
    print(f"Initializing YOLOv8 detector with model: {args.model}")
    detector = YOLODetector(
        model_path=args.model,
        conf_threshold=args.conf_thresh,
        iou_threshold=args.iou_thresh,
        device=args.device,
        classes=args.classes
    )
    
    # Initialize tracker
    if args.tracker.lower() == 'sort':
        print("Initializing SORT tracker")
        tracker = SORT(
            max_age=args.max_age,
            min_hits=args.min_hits,
            iou_threshold=args.iou_tracking,
            use_class_info=args.use_class_info
        )
    else:  # DeepSORT
        print("Initializing DeepSORT tracker")
        tracker = DeepSORT(
            max_age=args.max_age,
            min_hits=args.min_hits,
            iou_threshold=args.iou_tracking,
            use_class_info=args.use_class_info,
            max_cosine_distance=args.max_cosine_dist,
            nn_budget=args.nn_budget
        )
    
    # Initialize evaluator if evaluation is enabled
    evaluator = None
    gt_loader = None
    gt_data = None
    
    if args.eval and args.gt_file:
        print(f"Initializing evaluator with ground truth from: {args.gt_file}")
        evaluator = MOTEvaluator()
        gt_loader = GroundTruthLoader(dataset_type=args.dataset_type)
        gt_data = gt_loader.load_sequence(args.gt_file)
    
    # Process the video
    frame_idx = 0
    color_map = {}  # For consistent colors across frames
    
    print("Starting video processing...")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Start timing for FPS calculation
        if evaluator:
            evaluator.start_frame()
        
        # 1. Object Detection
        detections = detector.detect(frame)
        
        # 2. Object Tracking
        tracks = tracker.update(detections)
        
        # End timing for FPS calculation
        if evaluator:
            evaluator.end_frame()
            
            # Update evaluator with current frame's tracking results
            if gt_data and frame_idx in gt_data:
                evaluator.update(gt_data[frame_idx], tracks, frame_idx)
        
        # 3. Visualization
        if args.show or writer:
            # Visualize tracks on frame
            vis_frame, color_map = visualize_tracks(frame, tracks, color_map)
            
            # Display frame
            if args.show:
                cv2.imshow('Multi-Object Tracking', vis_frame)
                
                # Exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write frame to output video
            if writer:
                writer.write(vis_frame)
        
        # Print progress every 100 frames
        if frame_idx % 100 == 0:
            fps = evaluator.get_fps() if evaluator else 0
            print(f"Processed {frame_idx} frames (FPS: {fps:.2f})")
    
    # Print final evaluation metrics
    if evaluator:
        metrics = evaluator.compute_metrics()
        evaluator.print_metrics(metrics)
        
        # Save metrics to file
        if args.save_results:
            metrics_file = os.path.splitext(args.output)[0] + '_metrics.csv'
            pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
            print(f"Saved metrics to: {metrics_file}")
    
    # Release resources
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print("Processing complete!")

if __name__ == '__main__':
    main()