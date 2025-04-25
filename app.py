from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import uuid
import time
from werkzeug.utils import secure_filename

# Add core to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detection.detector import ObjectDetector
from core.tracking.tracker import MultiObjectTracker

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload

# Initialize models
detector = ObjectDetector()
tracker = MultiObjectTracker()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'MOT API is running'})

@app.route('/api/detect', methods=['POST'])
def detect_objects():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{str(uuid.uuid4())}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Create uploads directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        file.save(filepath)
        
        # Process video with detection and tracking
        try:
            results = process_video(filepath)
            return jsonify(results)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

def process_video(filepath):
    # Placeholder for video processing logic
    # In a real implementation, this would use the detector and tracker
    
    # Example results structure
    results = {
        'job_id': str(uuid.uuid4()),
        'status': 'completed',
        'processing_time': 2.5,  # seconds
        'frames_processed': 120,
        'objects_detected': {
            'car': 5,
            'person': 12,
            'bicycle': 2
        },
        'tracking_results': [
            {
                'frame_id': 1,
                'objects': [
                    {'id': 1, 'class': 'person', 'bbox': [10, 20, 50, 100], 'confidence': 0.92},
                    {'id': 2, 'class': 'car', 'bbox': [100, 150, 300, 200], 'confidence': 0.89}
                ]
            }
            # Additional frames would follow
        ]
    }
    
    return results

@app.route('/api/models', methods=['GET'])
def list_models():
    """List available detection and tracking models"""
    available_models = {
        'detectors': ['yolov8', 'fasterrcnn'],
        'trackers': ['deepsort', 'bytetrack', 'strongsort']
    }
    
    return jsonify(available_models)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)), debug=False)