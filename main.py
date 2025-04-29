# main.py
import os
import cv2
import argparse
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import yaml
import pandas as pd

# Custom modules
from detectors.yolo_detector import YOLODetector
from utils.tracker import SORT, DeepSORT, visualize_tracks, Detection
from evaluation.metrics import MOTEvaluator, GroundTruthLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Object Tracking with YOLO and SORT/DeepSORT')
    
    parser.add_argument('--input', type=str, required=True, help='Path to input video file or camera index')
    parser.add_argument('--output', type=str, default='output/output.mp4', help='Path to output video file')
    parser.add_argument('--save_results', action='store_true', help='Save tracking results to file')
    parser.add_argument('--show', action='store_true', help='Display output frames')
    
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to YOLO model')
    parser.add_argument('--conf_thresh', type=float, default=0.25, help='Detection confidence threshold')
    parser.add_argument('--iou_thresh', type=float, default=0.45, help='Detection NMS IoU threshold')
    parser.add_argument('--classes', type=int, nargs='+', default=None, help='Filter by class IDs')
    
    parser.add_argument('--tracker', type=str, default='sort', choices=['sort', 'deepsort'], help='Tracking algorithm')
    parser.add_argument('--max_age', type=int, default=30, help='Maximum frames to keep track alive without matching')
    parser.add_argument('--min_hits', type=int, default=3, help='Minimum hits needed before track is established')
    parser.add_argument('--iou_tracking', type=float, default=0.3, help='IoU threshold for tracking')
    parser.add_argument('--use_class_info', action='store_true', help='Use class information for tracking')
    
    parser.add_argument('--max_cosine_dist', type=float, default=0.2, help='Max cosine distance for feature matching')
    parser.add_argument('--nn_budget', type=int, default=100, help='Maximum size of appearance descriptor gallery')
    
    parser.add_argument('--eval', action='store_true', help='Evaluate tracking performance')
    parser.add_argument('--gt_file', type=str, default=None, help='Path to ground truth file')
    parser.add_argument('--dataset_type', type=str, default='MOTChallenge', choices=['MOTChallenge', 'KITTI'], 
                        help='Dataset type for evaluation')
    
    parser.add_argument('--device', type=str, default='', help='Device to run on (e.g., cpu, 0, 0,1)')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    
    args = parser.parse_args()
    
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if value is not None:
                setattr(args, key, value)
    
    return args

def load_video(input_path):
    if isinstance(input_path, str) and input_path.isdigit():
        cap = cv2.VideoCapture(int(input_path))
    else:
        cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {input_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps <= 0:
        fps = 30.0
    
    print(f"Video properties: {width}x{height} @ {fps}fps")
    return cap, width, height, fps

def setup_output_writer(output_path, width, height, fps):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return writer

def main():
    args = parse_args()
    
    detector = YOLODetector(
        model_path=args.model,
        conf_threshold=args.conf_thresh,
        iou_threshold=args.iou_thresh
    )
    
    if args.tracker.lower() == 'sort':
        tracker = SORT(
            max_age=args.max_age,
            min_hits=args.min_hits,
            iou_threshold=args.iou_thresh
        )
    else:
        tracker = DeepSORT(
            max_age=args.max_age,
            min_hits=args.min_hits,
            max_iou_distance=args.iou_thresh,
            max_cosine_distance=args.max_cosine_dist,
            nn_budget=args.nn_budget
        )
    
    cap, width, height, fps = load_video(args.input)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    writer = None
    if args.output:
        writer = setup_output_writer(args.output, width, height, fps)
    
    evaluator = None
    gt_loader = None
    gt_data = None
    
    if args.eval and args.gt_file:
        print(f"Initializing evaluator with ground truth from: {args.gt_file}")
        evaluator = MOTEvaluator()
        gt_loader = GroundTruthLoader(dataset_type=args.dataset_type)
        gt_data = gt_loader.load_sequence(args.gt_file)
    
    print("Starting video processing...")
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        detections = detector.detect(frame)
        print(f"Detected {len(detections)} objects: {[d['class_name'] for d in detections]}")
        
        tracks = tracker.update(detections) if detections else []
        print(f"Active tracks: {len(tracks)}")
        
        for track in tracks:
            print("Track content:", track)
            if isinstance(track, dict):
                x1, y1, x2, y2 = map(int, track['bbox'])
                track_id = track['track_id']
                class_name = track['class_name']
                label = f"{class_name} ID:{track_id}"

                # Draw track bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                x1, y1, x2, y2, track_id = map(int, track)
                label = f"ID: {track_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw detections (optional, green)
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if args.show:
            cv2.imshow('Multi-Object Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if writer:
            writer.write(frame)
        
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"FPS: {fps:.2f}, Tracks: {len(tracks)}")
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    if evaluator:
        metrics = evaluator.compute_metrics()
        evaluator.print_metrics(metrics)
        if args.save_results:
            metrics_file = os.path.splitext(args.output)[0] + '_metrics.csv'
            pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
            print(f"Saved metrics to: {metrics_file}")
    
    print("Processing complete!")

if __name__ == '__main__':
    main()
