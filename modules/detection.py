"""
AvianGuard IRAPS - Bird Detection Module
YOLOv8-based bird detection system for real-time video analysis.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from ultralytics import YOLO
import utils.config as config
from utils.helpers import setup_logging

logger = setup_logging()


class BirdDetector:
    """
    YOLOv8-based bird detection wrapper.
    Detects birds in video frames and returns bounding boxes with confidence scores.
    """
    
    def __init__(self, model_path: str = config.MODEL_PATH, 
                 confidence_threshold: float = config.CONFIDENCE_THRESHOLD):
        """
        Initialize the bird detector.
        
        Args:
            model_path: Path to YOLOv8 model weights
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = 'cuda' if config.USE_GPU else 'cpu'
        
        logger.info(f"Initializing BirdDetector with model: {model_path}")
        self._load_model()
    
    def _load_model(self) -> None:
        """Load YOLOv8 model."""
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            logger.warning(f"Failed to load model from {self.model_path}: {e}")
            logger.info("Falling back to default YOLOv8n model")
            try:
                self.model = YOLO('yolov8n.pt')
                logger.info("Default model loaded successfully")
            except Exception as e2:
                logger.error(f"Failed to load default model: {e2}")
                raise
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect birds in a single frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of detections, each containing:
                - bbox: (x1, y1, x2, y2)
                - confidence: detection confidence
                - class_id: class ID
                - class_name: class name
        """
        if self.model is None:
            logger.error("Model not loaded")
            return []
        
        detections = []
        
        try:
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            # Process results
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence and class
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    # Filter for bird-like classes (bird, or using general objects as proxy)
                    # COCO dataset: 14=bird, 15=cat, 16=dog (we use these as proxies for testing)
                    if class_id in config.BIRD_CLASS_IDS or class_name.lower() == 'bird':
                        detection = {
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name
                        }
                        detections.append(detection)
            
            logger.debug(f"Detected {len(detections)} birds in frame")
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
        
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Detect birds in a batch of frames for improved performance.
        
        Args:
            frames: List of input frames
            
        Returns:
            List of detection lists for each frame
        """
        if self.model is None:
            logger.error("Model not loaded")
            return [[] for _ in frames]
        
        all_detections = []
        
        try:
            # Run batch inference
            results = self.model(frames, conf=self.confidence_threshold, verbose=False)
            
            # Process each frame's results
            for result in results:
                frame_detections = []
                boxes = result.boxes
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    if class_id in config.BIRD_CLASS_IDS or class_name.lower() == 'bird':
                        detection = {
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name
                        }
                        frame_detections.append(detection)
                
                all_detections.append(frame_detections)
            
            logger.debug(f"Processed batch of {len(frames)} frames")
            
        except Exception as e:
            logger.error(f"Batch detection error: {e}")
            all_detections = [[] for _ in frames]
        
        return all_detections
    
    def visualize_detections(self, frame: np.ndarray, 
                            detections: List[Dict[str, Any]],
                            color: Tuple[int, int, int] = (0, 255, 0),
                            thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes on frame.
        
        Args:
            frame: Input frame
            detections: List of detections
            color: Bounding box color (BGR)
            thickness: Box line thickness
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Background for label
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_size[1] - 5),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Label text
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return annotated_frame


if __name__ == "__main__":
    # Simple test
    detector = BirdDetector()
    print("BirdDetector initialized successfully")
    
    # Create a test frame
    test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = detector.detect(test_frame)
    print(f"Test detection returned {len(detections)} detections")
