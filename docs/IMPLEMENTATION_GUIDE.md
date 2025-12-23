# AvianGuard IRAPS - Implementation Guide

Complete step-by-step guide for setting up and implementing the AvianGuard system.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Installation](#installation)
4. [Module-by-Module Implementation](#module-by-module-implementation)
5. [Testing & Validation](#testing--validation)
6. [Configuration & Tuning](#configuration--tuning)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization](#performance-optimization)

---

## Prerequisites

### System Requirements

**Minimum:**
- Python 3.8 or higher
- 8GB RAM
- Dual-core CPU
- 2GB disk space

**Recommended:**
- Python 3.9+
- 16GB RAM
- Quad-core CPU or better
- NVIDIA GPU with CUDA support
- 5GB disk space

### Operating System

- ✅ Windows 10/11
- ✅ macOS 10.14+
- ✅ Ubuntu 18.04+ / Debian 10+
- ✅ Other Linux distributions

### Software Dependencies

- Git
- pip (Python package manager)
- (Optional) CUDA Toolkit 11.x for GPU acceleration

---

## Environment Setup

### Step 1: Install Python

**Windows:**
```bash
# Download from python.org
# Or use winget:
winget install Python.Python.3.11
```

**macOS:**
```bash
# Using Homebrew
brew install python@3.11
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3-pip python3-venv

# Fedora
sudo dnf install python3.11
```

### Step 2: Verify Installation

```bash
python --version  # Should show 3.8+
pip --version     # Should show pip version
```

---

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/YadavPrem64/Avianguard-Bird-Strike-Prevention.git
cd Avianguard-Bird-Strike-Prevention
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

# Verify activation (should show venv in prompt)
```

### Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# Verify installation
python -c "import cv2, numpy, ultralytics; print('Success!')"
```

### Step 4: (Optional) GPU Setup

If you have an NVIDIA GPU:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Module-by-Module Implementation

### Module 1: Configuration (`utils/config.py`)

Already implemented. Key parameters to understand:

```python
# Detection settings
CONFIDENCE_THRESHOLD = 0.5  # Lower = more detections, higher = fewer false positives
MODEL_PATH = "data/models/yolov8n.pt"

# Risk thresholds
RISK_LOW = 30    # Below this = low risk (green)
RISK_MEDIUM = 70 # Below this = medium risk (yellow)
RISK_HIGH = 90   # Above medium = high risk (red)

# Performance
SKIP_FRAMES = 1  # Process every Nth frame (1 = all frames)
USE_GPU = True   # Enable GPU acceleration
```

### Module 2: Detection Engine (`modules/detection.py`)

**Test the detector:**

```python
from modules.detection import BirdDetector
import cv2

# Initialize detector
detector = BirdDetector()

# Load test image
frame = cv2.imread('test_image.jpg')

# Detect birds
detections = detector.detect(frame)

# Print results
for det in detections:
    print(f"Detected {det['class_name']} with {det['confidence']:.2f} confidence")
    print(f"  Bounding box: {det['bbox']}")
```

### Module 3: Video Processor (`modules/video_processor.py`)

**Test video processing:**

```python
from modules.video_processor import VideoProcessor

# Initialize processor
processor = VideoProcessor(video_source='test.mp4')

# Open video
if processor.open_video():
    # Read first frame
    ret, frame = processor.read_frame()
    if ret:
        print(f"Frame shape: {frame.shape}")
    
    processor.release()
```

### Module 4: Fuzzy Logic Engine (`modules/fuzzy_logic.py`) ⭐

**Test risk assessment:**

```python
from modules.fuzzy_logic import FuzzyRiskAssessment

# Initialize fuzzy engine
fuzzy = FuzzyRiskAssessment()

# Test scenarios
test_cases = [
    (10, 20, 30, 0.8, "Extreme risk"),  # close, many, fast, danger
    (200, 1, 3, 0.1, "Low risk"),        # far, single, slow, safe
]

for distance, count, velocity, position, description in test_cases:
    risk, explanation = fuzzy.assess_risk(distance, count, velocity, position)
    print(f"{description}: Risk = {risk:.1f}%")
```

### Module 5: Trajectory Predictor (`modules/trajectory.py`) ⭐

**Test trajectory prediction:**

```python
from modules.trajectory import TrajectoryPredictor

# Initialize predictor
predictor = TrajectoryPredictor()

# Simulate bird movement
bird_id = 1
for t in range(20):
    x = 100 + t * 5
    y = 100 + t * 3
    predictor.update(bird_id, (x, y), timestamp=t)

# Predict future positions
future = predictor.predict_trajectory(bird_id, steps=30)
print(f"Predicted {len(future)} future positions")

# Get velocity
vx, vy = predictor.get_velocity(bird_id)
print(f"Estimated velocity: ({vx:.2f}, {vy:.2f}) px/frame")
```

### Module 6: TTC Calculator (`modules/collision_calc.py`) ⭐

**Test collision time calculation:**

```python
from modules.collision_calc import TimeToCollisionCalculator

# Initialize calculator
ttc_calc = TimeToCollisionCalculator()

# Simulate approaching bird (growing bbox)
bird_id = 1
for t in range(30):
    size = 50 + t * 5  # Growing bbox
    bbox = (300, 200, 300+size, 200+size)
    ttc_calc.update_bbox(bird_id, bbox, timestamp=t)

# Calculate TTC
ttc = ttc_calc.calculate_ttc(bird_id, method='optical')
if ttc:
    print(f"Time to collision: {ttc:.2f} seconds")
```

### Module 7: Multi-Bird Tracker (`modules/tracker.py`) ⭐

**Test tracking:**

```python
from modules.tracker import BirdTracker
import numpy as np

# Initialize tracker
tracker = BirdTracker()

# Create test frame
frame = np.zeros((720, 1280, 3), dtype=np.uint8)

# Simulate detections
detections = [
    {'bbox': (100, 100, 150, 150), 'confidence': 0.9, 'class_name': 'bird'},
    {'bbox': (300, 200, 350, 250), 'confidence': 0.85, 'class_name': 'bird'},
]

# Update tracker
tracks = tracker.update(detections, frame)

# Print tracks
for track in tracks:
    print(f"Track ID {track['track_id']}: {track['bbox']}")
```

### Module 8: Danger Zones (`modules/danger_zones.py`) ⭐

**Test zone classification:**

```python
from modules.danger_zones import DangerZoneClassifier

# Initialize classifier
classifier = DangerZoneClassifier()

# Test positions
test_points = [
    (640, 360, "Center"),
    (100, 100, "Top-left"),
    (1200, 650, "Bottom-right"),
]

for x, y, description in test_points:
    classification = classifier.classify_position(x, y)
    print(f"{description}: {classification['zone_type']} zone")
    print(f"  Danger score: {classification['danger_score']:.2f}")
```

---

## Testing & Validation

### Unit Tests

Run individual module tests:

```bash
# Test fuzzy logic
python -m unittest tests.test_fuzzy_logic

# Test trajectory predictor
python -m unittest tests.test_trajectory

# Run all tests
python -m unittest discover tests
```

### Integration Testing

Test the complete pipeline:

```bash
# With a test video
python main.py --input data/sample_videos/test.mp4 --output output/result.mp4

# With webcam
python main.py --camera 0
```

### Expected Behavior

✅ **System should:**
- Detect birds in video frames
- Assign consistent IDs across frames
- Draw color-coded bounding boxes
- Display risk scores
- Show trajectory predictions
- Generate alerts for high-risk scenarios

❌ **If you see issues:**
- No detections: Check video quality, adjust CONFIDENCE_THRESHOLD
- Tracking errors: Tune IOU_THRESHOLD, MAX_TRACKING_AGE
- Slow performance: Enable GPU, reduce resolution, skip frames

---

## Configuration & Tuning

### Detection Tuning

Adjust in `utils/config.py`:

```python
# More detections (may increase false positives)
CONFIDENCE_THRESHOLD = 0.3

# Fewer detections (more conservative)
CONFIDENCE_THRESHOLD = 0.7

# Use larger model for better accuracy
MODEL_PATH = "data/models/yolov8s.pt"
```

### Risk Assessment Tuning

```python
# More conservative (fewer alerts)
RISK_LOW = 40
RISK_MEDIUM = 80
RISK_HIGH = 95

# More sensitive (more alerts)
RISK_LOW = 20
RISK_MEDIUM = 60
RISK_HIGH = 85
```

### Performance Tuning

```python
# Process every other frame (2x faster)
SKIP_FRAMES = 2

# Reduce output resolution
FRAME_WIDTH = 960
FRAME_HEIGHT = 540

# Disable expensive features
SHOW_TRAJECTORY = False
SHOW_DANGER_ZONES = False
```

---

## Troubleshooting

### Common Issues

**Issue: "ModuleNotFoundError"**
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt
```

**Issue: "CUDA out of memory"**
```python
# Solution: Use CPU or smaller model
USE_GPU = False
MODEL_PATH = "data/models/yolov8n.pt"
```

**Issue: "Video not opening"**
```bash
# Solution: Check codec support
ffmpeg -i video.mp4  # Check video properties
# Convert if needed:
ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4
```

**Issue: "DeepSORT fails to initialize"**
```python
# Solution: Falls back to SimpleTracker automatically
# Or install DeepSORT dependencies manually
pip install deep-sort-realtime
```

**Issue: "Poor detection accuracy"**
```python
# Solutions:
# 1. Use larger model
MODEL_PATH = "data/models/yolov8m.pt"

# 2. Fine-tune on bird dataset
# 3. Adjust confidence threshold
CONFIDENCE_THRESHOLD = 0.4
```

---

## Performance Optimization

### GPU Acceleration

```python
# Verify GPU usage
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Enable in config
USE_GPU = True
```

### Batch Processing

For offline video analysis:

```python
# Process in batches for efficiency
from modules.detection import BirdDetector

detector = BirdDetector()
frames = [frame1, frame2, frame3, ...]  # Collect frames
detections_batch = detector.detect_batch(frames)
```

### Multi-threading

Process frames in parallel (advanced):

```python
from concurrent.futures import ThreadPoolExecutor

def process_frame(frame):
    # Your processing logic
    pass

with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(process_frame, frames)
```

---

## Next Steps

1. **Customize for Your Use Case**
   - Adjust risk thresholds
   - Define custom danger zones
   - Modify fuzzy rules

2. **Add Dashboard**
   - Implement Streamlit interface
   - Add real-time visualizations
   - Create interactive controls

3. **Deploy**
   - Package as executable
   - Deploy on edge device
   - Integrate with existing systems

4. **Enhance**
   - Add species classification
   - Integrate weather data
   - Implement historical analytics

---

## Resources

- **YOLOv8 Docs**: https://docs.ultralytics.com/
- **OpenCV Tutorials**: https://docs.opencv.org/4.x/d9/df8/tutorial_root.html
- **Fuzzy Logic Guide**: https://pythonhosted.org/scikit-fuzzy/
- **Kalman Filtering**: https://filterpy.readthedocs.io/

---

**Need Help?** Open an issue on GitHub or check the documentation in `docs/`
