"""
AvianGuard IRAPS - Configuration Parameters
Central configuration file for all system parameters.
"""

# ============================================================================
# DETECTION PARAMETERS
# ============================================================================

# YOLOv8 Detection Settings
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for bird detection
NMS_THRESHOLD = 0.4  # Non-maximum suppression threshold
INPUT_SIZE = (640, 640)  # Model input size
MODEL_PATH = "data/models/yolov8n.pt"  # Default YOLOv8 model path
BIRD_CLASS_IDS = [14, 15, 16]  # COCO class IDs for birds (bird, cat, dog as proxy)

# ============================================================================
# FUZZY LOGIC RANGES
# ============================================================================

# Distance Estimation (in pixels, inverse of bounding box size)
DISTANCE_FAR = (100, 500)  # Far from aircraft
DISTANCE_MEDIUM = (50, 150)  # Medium distance
DISTANCE_CLOSE = (20, 70)  # Close to aircraft
DISTANCE_CRITICAL = (0, 30)  # Critical proximity

# Bird Count Categories
BIRD_COUNT_SINGLE = (1, 2)  # Single bird
BIRD_COUNT_FEW = (2, 5)  # Few birds
BIRD_COUNT_FLOCK = (5, 15)  # Small flock
BIRD_COUNT_SWARM = (15, 100)  # Large swarm

# Velocity Categories (pixels per frame)
VELOCITY_SLOW = (0, 5)  # Slow movement
VELOCITY_MEDIUM = (5, 15)  # Medium speed
VELOCITY_FAST = (15, 100)  # Fast approach

# Screen Position Categories (normalized 0-1)
POSITION_SAFE = (0.0, 0.3)  # Peripheral areas
POSITION_CAUTION = (0.3, 0.6)  # Intermediate zones
POSITION_DANGER = (0.6, 1.0)  # Critical center zones

# ============================================================================
# RISK ASSESSMENT THRESHOLDS
# ============================================================================

RISK_LOW = 30  # Below this is low risk (green)
RISK_MEDIUM = 70  # Below this is medium risk (yellow)
RISK_HIGH = 90  # Above medium is high risk (red)
RISK_EXTREME = 100  # Maximum risk level

# ============================================================================
# DANGER ZONES (Screen Percentages)
# ============================================================================

# Frame division for danger zones
SAFE_ZONE_MARGIN = 0.2  # Outer 20% of frame is safe zone
CAUTION_ZONE_MIDDLE = (0.2, 0.5)  # Middle zones are caution
CRITICAL_ZONE_CENTER = (0.4, 0.6)  # Center 40-60% is critical

# Engine intake zones (critical areas)
ENGINE_ZONES = [
    {'x': (0.15, 0.35), 'y': (0.3, 0.7), 'name': 'Left Engine'},
    {'x': (0.65, 0.85), 'y': (0.3, 0.7), 'name': 'Right Engine'}
]

# Cockpit window zone (highest priority)
COCKPIT_ZONE = {'x': (0.4, 0.6), 'y': (0.3, 0.7), 'name': 'Cockpit'}

# ============================================================================
# TRACKING PARAMETERS
# ============================================================================

# DeepSORT Tracking Configuration
MAX_TRACKING_AGE = 30  # Max frames to keep track without detection
MIN_TRACKING_HITS = 3  # Min detections before track is confirmed
IOU_THRESHOLD = 0.3  # Intersection over Union threshold
MAX_COSINE_DISTANCE = 0.3  # Max appearance descriptor distance

# Trajectory History
TRAJECTORY_HISTORY_LENGTH = 30  # Frames to keep in trajectory history
MIN_TRAJECTORY_POINTS = 5  # Minimum points needed for prediction

# ============================================================================
# KALMAN FILTER PARAMETERS
# ============================================================================

# State vector: [x, y, vx, vy]
DT = 1.0  # Time step (frames)
PROCESS_NOISE = 1.0  # Process noise covariance
MEASUREMENT_NOISE = 10.0  # Measurement noise covariance

# Prediction parameters
PREDICTION_HORIZON = 90  # Frames to predict into future (3 seconds at 30fps)
TRAJECTORY_POINTS_TO_DRAW = 30  # Number of predicted points to visualize

# ============================================================================
# TIME-TO-COLLISION PARAMETERS
# ============================================================================

# TTC Calculation
MIN_BBOX_GROWTH_RATE = 1.5  # Minimum growth rate to consider (pixels/frame)
TTC_CRITICAL_THRESHOLD = 3.0  # Seconds - critical alert threshold
TTC_WARNING_THRESHOLD = 5.0  # Seconds - warning threshold
ASSUMED_BIRD_SIZE_CM = 30.0  # Assumed average bird size in cm
CAMERA_FOCAL_LENGTH_PX = 500.0  # Approximate focal length in pixels

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================

# Video Processing
FRAME_WIDTH = 1280  # Output frame width
FRAME_HEIGHT = 720  # Output frame height
FPS = 30  # Frames per second
DISPLAY_FPS = True  # Show FPS on display

# Visualization Colors (BGR format for OpenCV)
COLOR_SAFE = (0, 255, 0)  # Green
COLOR_CAUTION = (0, 255, 255)  # Yellow
COLOR_DANGER = (0, 165, 255)  # Orange
COLOR_CRITICAL = (0, 0, 255)  # Red
COLOR_TRAJECTORY = (255, 0, 255)  # Magenta

# Bounding Box Settings
BBOX_THICKNESS = 2  # Thickness of bounding boxes
BBOX_FONT_SCALE = 0.6  # Font scale for labels
BBOX_FONT_THICKNESS = 2  # Font thickness

# ============================================================================
# ALERT SYSTEM SETTINGS
# ============================================================================

# Audio Alerts
ENABLE_AUDIO_ALERTS = True  # Enable/disable audio warnings
AUDIO_ALERT_COOLDOWN = 5.0  # Seconds between repeated alerts
AUDIO_RATE = 150  # Speech rate (words per minute)

# Alert Messages
ALERT_MESSAGES = {
    'low': 'Bird detected. Monitoring.',
    'medium': 'Warning! Bird approaching. Exercise caution.',
    'high': 'Alert! High collision risk. Take evasive action.',
    'extreme': 'DANGER! Extreme collision risk. Immediate action required!'
}

# ============================================================================
# DASHBOARD SETTINGS
# ============================================================================

# Streamlit Layout
SIDEBAR_WIDTH = 300  # Pixels
VIDEO_DISPLAY_WIDTH = 800  # Pixels
REFRESH_RATE = 0.033  # Seconds (30 FPS)

# Radar Map Settings
RADAR_SIZE = 200  # Pixels (square)
RADAR_RANGE = 500  # Pixels in real space represented

# Heatmap Settings
HEATMAP_DECAY = 0.95  # Decay factor for historical heatmap
HEATMAP_RESOLUTION = (50, 50)  # Grid resolution for heatmap

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Processing Optimization
SKIP_FRAMES = 1  # Process every Nth frame (1 = process all)
USE_GPU = True  # Use GPU acceleration if available
MAX_BIRDS_TO_TRACK = 50  # Maximum number of simultaneous tracks

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_TO_FILE = True  # Save logs to file
LOG_FILE_PATH = "logs/avianguard.log"

# ============================================================================
# TESTING & DEBUG SETTINGS
# ============================================================================

DEBUG_MODE = False  # Enable debug visualizations
SAVE_DEBUG_FRAMES = False  # Save annotated frames
DEBUG_OUTPUT_PATH = "output/debug/"
SHOW_DANGER_ZONES = True  # Visualize danger zones on display
SHOW_TRAJECTORY = True  # Show predicted trajectories
SHOW_TRACKING_IDS = True  # Show tracking IDs

# ============================================================================
# DATA PATHS
# ============================================================================

# Input/Output Paths
DEFAULT_VIDEO_PATH = "data/sample_videos/sample.mp4"
OUTPUT_VIDEO_PATH = "output/processed_video.mp4"
MODEL_WEIGHTS_PATH = "data/models/"
TEMP_PATH = "tmp/"
