# Multi-Object Tracking System

This project implements a real-time multi-object tracking system using YOLOv8 for object detection combined with SORT/DeepSORT tracking algorithms. The system is designed to accurately detect and track multiple objects across video frames with high performance.

## Features

- **Real-time object detection** using YOLOv8
- **Multiple tracking algorithms**:
  - SORT (Simple Online and Realtime Tracking)
  - DeepSORT (with appearance features)
- **Performance evaluation** using standard MOT metrics:
  - MOTA (Multiple Object Tracking Accuracy)
  - MOTP (Multiple Object Tracking Precision)
  - IDF1 (ID F1 Score)
  - FPS (Frames Per Second)
- **Support for benchmark datasets**:
  - MOTChallenge
  - KITTI

## Project Structure

```
multi-object-tracking/
│
├── data/                           # Contains datasets like MOTChallenge or KITTI
│
├── models/                         # Pretrained YOLO models and any custom-trained ones
│
├── utils/                          # Utility functions (e.g., visualization, IOU, Kalman filter)
│   └── tracker.py                  # Multi-object tracking logic (e.g., SORT/DeepSORT)
│
├── detectors/                      # Object detection logic
│   └── yolo_detector.py            # Wrapper for YOLO model
│
├── evaluation/                     # Evaluation metrics like MOTA, MOTP, IDF1, etc.
│   └── metrics.py
│
├── output/                         # Output videos and logs
│
├── main.py                         # Main execution script
│
├── requirements.txt                # Python dependencies
│
└── README.md                       # Project overview and instructions
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multi-object-tracking.git
   cd multi-object-tracking
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download YOLOv8 weights (if not using the default):
   ```bash
   # The code will automatically download YOLOv8n if not specified
   # For custom models, place them in the models/ directory
   ```

## Usage

### Basic Usage

Run the tracker on a video file:

```bash
python main.py --input path/to/video.mp4 --output output/result.mp4 --show
```

### Advanced Options

```bash
python main.py --input path/to/video.mp4 \
               --output output/result.mp4 \
               --model models/yolov8m.pt \
               --conf_thresh 0.25 \
               --tracker deepsort \
               --max_age 30 \
               --min_hits 3 \
               --show
```

### Using Config File

You can also specify options via a YAML config file:

```bash
python main.py --config configs/my_config.yaml
```

Example config file (`configs/my_config.yaml`):

```yaml
input: path/to/video.mp4
output: output/result.mp4
model: models/yolov8m.pt
conf_thresh: 0.25
tracker: deepsort
max_age: 30
min_hits: 3
show: true
```

### Evaluation

To evaluate tracking performance against ground truth:

```bash
python main.py --input path/to/video.mp4 \
               --eval \
               --gt_file path/to/gt.txt \
               --dataset_type MOTChallenge
```

## Key Parameters

### Detection Parameters

- `--model`: Path to YOLO model (default: yolov8n.pt)
- `--conf_thresh`: Detection confidence threshold (default: 0.25)
- `--iou_thresh`: NMS IoU threshold (default: 0.45)
- `--classes`: Filter specific classes (e.g., --classes 0 2 3)

### Tracking Parameters

- `--tracker`: Tracking algorithm ('sort' or 'deepsort')
- `--max_age`: Maximum frames to keep track alive without matching (default: 30)
- `--min_hits`: Minimum hits needed before track is established (default: 3)
- `--iou_tracking`: IoU threshold for tracking (default: 0.3)
- `--use_class_info`: Use class information for tracking

### DeepSORT Specific Parameters

- `--max_cosine_dist`: Max cosine distance for feature matching (default: 0.2)
- `--nn_budget`: Maximum size of appearance descriptor gallery (default: 100)

## Performance Considerations

- **Higher detection confidence** improves precision but reduces recall
- **Smaller object detection models** run faster but may be less accurate
- **SORT** is faster than DeepSORT but may have more ID switches
- **Higher max_age** values maintain tracks longer during occlusions but may cause false positives
- **Use GPU acceleration** when available for real-time performance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- SORT algorithm by Alex Bewley
- DeepSORT algorithm by Nicolai Wojke
- MOTChallenge and KITTI benchmark datasets