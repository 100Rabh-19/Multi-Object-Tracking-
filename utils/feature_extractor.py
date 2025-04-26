# utils/feature_extractor.py
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union

class FeatureExtractor:
    """
    Extract appearance features from object detections for DeepSORT
    Uses a pre-trained CNN to extract feature vectors
    """
    def __init__(
        self,
        model_name: str = 'resnet18',
        input_size: Tuple[int, int] = (128, 64),  # Width, Height
        device: str = '',
        use_cuda: bool = True
    ):
        """
        Initialize feature extractor
        
        Args:
            model_name: Name of backbone model ('resnet18', 'mobilenet_v2', etc.)
            input_size: Input size for the model (width, height)
            device: Device to run on ('cpu', '0', '0,1', etc.)
            use_cuda: Whether to use CUDA if available
        """
        self.input_size = input_size
        
        # Set device
        if device:
            self.device = torch.device(f"cuda:{device}" if device.isdigit() else device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")
        
        print(f"FeatureExtractor using device: {self.device}")
        
        # Initialize model
        self.model = self._build_model(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _build_model(self, model_name: str) -> nn.Module:
        """
        Build the feature extraction model
        
        Args:
            model_name: Name of backbone model
            
        Returns:
            PyTorch model for feature extraction
        """
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            # Remove the final classification layer
            model = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 512
        
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 2048
        
        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 1280
        
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        return model
    
    @torch.no_grad()
    def extract_features(
        self, 
        frame: np.ndarray, 
        detections: List[Dict]
    ) -> List[Dict]:
        """
        Extract features for detected objects in a frame
        
        Args:
            frame: Image frame (BGR format from OpenCV)
            detections: List of detection dictionaries with 'bbox' key
            
        Returns:
            List of detection dictionaries with added 'feature' key
        """
        # Convert to RGB (PyTorch models expect RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract image patches
        patches = []
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Clip to frame boundaries
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(frame.shape[1], int(x2))
            y2 = min(frame.shape[0], int(y2))
            
            # Skip invalid boxes
            if x1 >= x2 or y1 >= y2:
                patches.append(None)
                continue
            
            # Extract patch
            patch = frame_rgb[y1:y2, x1:x2]
            patches.append(patch)
        
        # Process valid patches
        valid_patches = [p for p in patches if p is not None]
        
        if not valid_patches:
            # No valid patches, return original detections
            return detections
        
        # Preprocess patches
        processed_patches = [self.transform(patch) for patch in valid_patches]
        
        # Stack patches into a batch
        batch = torch.stack(processed_patches).to(self.device)
        
        # Extract features
        features = self.model(batch)
        
        # Reshape and normalize features
        features = features.view(batch.size(0), -1).cpu().numpy()
        
        # L2 normalization
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        
        # Assign features to detections
        feature_idx = 0
        for i, detection in enumerate(detections):
            if patches[i] is not None:
                detection['feature'] = features[feature_idx]
                feature_idx += 1
            else:
                # For invalid patches, use zeros as placeholder
                detection['feature'] = np.zeros(self.feature_dim)
        
        return detections