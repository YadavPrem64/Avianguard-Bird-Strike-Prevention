"""
AvianGuard IRAPS - Helper Utilities
Common utility functions used across the system.
"""

import cv2
import numpy as np
import logging
import os
from datetime import datetime
from typing import Tuple, List, Optional, Dict, Any
import utils.config as config


def setup_logging() -> logging.Logger:
    """
    Setup logging configuration for the application.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(config.LOG_FILE_PATH)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logger = logging.getLogger('AvianGuard')
    logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if config.LOG_TO_FILE:
        file_handler = logging.FileHandler(config.LOG_FILE_PATH)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def ensure_directory(path: str) -> None:
    """
    Ensure a directory exists, create it if it doesn't.
    
    Args:
        path: Directory path to ensure exists
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def calculate_bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """
    Calculate the center point of a bounding box.
    
    Args:
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        
    Returns:
        Tuple of (center_x, center_y)
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y


def calculate_bbox_area(bbox: Tuple[int, int, int, int]) -> float:
    """
    Calculate the area of a bounding box.
    
    Args:
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        
    Returns:
        Area in square pixels
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return width * height


def calculate_bbox_diagonal(bbox: Tuple[int, int, int, int]) -> float:
    """
    Calculate the diagonal length of a bounding box.
    
    Args:
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        
    Returns:
        Diagonal length in pixels
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return np.sqrt(width**2 + height**2)


def normalize_coordinates(
    x: float, 
    y: float, 
    frame_width: int, 
    frame_height: int
) -> Tuple[float, float]:
    """
    Normalize coordinates to [0, 1] range based on frame dimensions.
    
    Args:
        x: X coordinate
        y: Y coordinate
        frame_width: Frame width
        frame_height: Frame height
        
    Returns:
        Tuple of normalized (x, y) coordinates
    """
    norm_x = x / frame_width if frame_width > 0 else 0
    norm_y = y / frame_height if frame_height > 0 else 0
    return np.clip(norm_x, 0, 1), np.clip(norm_y, 0, 1)


def denormalize_coordinates(
    norm_x: float, 
    norm_y: float, 
    frame_width: int, 
    frame_height: int
) -> Tuple[int, int]:
    """
    Convert normalized coordinates back to pixel coordinates.
    
    Args:
        norm_x: Normalized X coordinate [0, 1]
        norm_y: Normalized Y coordinate [0, 1]
        frame_width: Frame width
        frame_height: Frame height
        
    Returns:
        Tuple of pixel (x, y) coordinates
    """
    x = int(norm_x * frame_width)
    y = int(norm_y * frame_height)
    return x, y


def calculate_euclidean_distance(
    point1: Tuple[float, float], 
    point2: Tuple[float, float]
) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
        
    Returns:
        Euclidean distance
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def get_risk_color(risk_score: float) -> Tuple[int, int, int]:
    """
    Get color corresponding to risk level (BGR format for OpenCV).
    
    Args:
        risk_score: Risk score from 0 to 100
        
    Returns:
        BGR color tuple
    """
    if risk_score < config.RISK_LOW:
        return config.COLOR_SAFE  # Green
    elif risk_score < config.RISK_MEDIUM:
        return config.COLOR_CAUTION  # Yellow
    elif risk_score < config.RISK_HIGH:
        return config.COLOR_DANGER  # Orange
    else:
        return config.COLOR_CRITICAL  # Red


def get_risk_level_name(risk_score: float) -> str:
    """
    Get human-readable risk level name.
    
    Args:
        risk_score: Risk score from 0 to 100
        
    Returns:
        Risk level name
    """
    if risk_score < config.RISK_LOW:
        return 'low'
    elif risk_score < config.RISK_MEDIUM:
        return 'medium'
    elif risk_score < config.RISK_HIGH:
        return 'high'
    else:
        return 'extreme'


def draw_text_with_background(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.6,
    thickness: int = 2,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    padding: int = 5
) -> None:
    """
    Draw text with a background rectangle for better visibility.
    
    Args:
        frame: Image frame to draw on
        text: Text to draw
        position: (x, y) position for text
        font_scale: Font scale
        thickness: Text thickness
        text_color: Text color (BGR)
        bg_color: Background color (BGR)
        padding: Padding around text
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    # Calculate background rectangle coordinates
    x, y = position
    rect_x1 = x - padding
    rect_y1 = y - text_height - padding
    rect_x2 = x + text_width + padding
    rect_y2 = y + baseline + padding
    
    # Draw background rectangle
    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
    
    # Draw text
    cv2.putText(frame, text, position, font, font_scale, text_color, thickness)


def resize_frame(
    frame: np.ndarray, 
    target_width: int = config.FRAME_WIDTH, 
    target_height: int = config.FRAME_HEIGHT
) -> np.ndarray:
    """
    Resize frame to target dimensions while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        target_width: Target width
        target_height: Target height
        
    Returns:
        Resized frame
    """
    return cv2.resize(frame, (target_width, target_height))


def calculate_iou(bbox1: Tuple[int, int, int, int], 
                  bbox2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bounding box (x1, y1, x2, y2)
        bbox2: Second bounding box (x1, y1, x2, y2)
        
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = bbox1_area + bbox2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 0:
        return "N/A"
    elif seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"


def create_timestamp() -> str:
    """
    Create a timestamp string for file naming.
    
    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def moving_average(data: List[float], window_size: int = 5) -> List[float]:
    """
    Calculate moving average of a list of values.
    
    Args:
        data: List of values
        window_size: Size of moving average window
        
    Returns:
        List of smoothed values
    """
    if len(data) < window_size:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size + 1)
        window = data[start_idx:i + 1]
        smoothed.append(np.mean(window))
    
    return smoothed


def interpolate_color(
    color1: Tuple[int, int, int],
    color2: Tuple[int, int, int],
    ratio: float
) -> Tuple[int, int, int]:
    """
    Interpolate between two colors based on ratio.
    
    Args:
        color1: First color (BGR)
        color2: Second color (BGR)
        ratio: Interpolation ratio [0, 1]
        
    Returns:
        Interpolated color (BGR)
    """
    ratio = np.clip(ratio, 0, 1)
    b = int(color1[0] * (1 - ratio) + color2[0] * ratio)
    g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
    r = int(color1[2] * (1 - ratio) + color2[2] * ratio)
    return (b, g, r)


class FPSCounter:
    """Simple FPS counter for performance monitoring."""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter.
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = []
        self.last_time = None
    
    def update(self) -> float:
        """
        Update FPS counter with current time.
        
        Returns:
            Current FPS
        """
        import time
        current_time = time.time()
        
        if self.last_time is not None:
            frame_time = current_time - self.last_time
            self.frame_times.append(frame_time)
            
            # Keep only recent frames
            if len(self.frame_times) > self.window_size:
                self.frame_times.pop(0)
        
        self.last_time = current_time
        
        # Calculate FPS
        if len(self.frame_times) > 0:
            avg_frame_time = np.mean(self.frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            return fps
        
        return 0.0
    
    def get_fps(self) -> float:
        """
        Get current FPS without updating.
        
        Returns:
            Current FPS
        """
        if len(self.frame_times) > 0:
            avg_frame_time = np.mean(self.frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            return fps
        return 0.0
