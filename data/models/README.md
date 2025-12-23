# Model Weights for AvianGuard IRAPS

This directory stores YOLOv8 model weights for bird detection.

## 🤖 YOLOv8 Models

AvianGuard uses YOLOv8 for bird detection. The model will be automatically downloaded on first run.

### Default Model

The system uses **YOLOv8n** (nano) by default for speed:

```python
# In utils/config.py
MODEL_PATH = "data/models/yolov8n.pt"
```

### Available Models

| Model | Size | Speed | mAP | Use Case |
|-------|------|-------|-----|----------|
| YOLOv8n | 6.3MB | Fastest | 37.3 | Real-time, embedded |
| YOLOv8s | 22MB | Fast | 44.9 | Good balance |
| YOLOv8m | 52MB | Medium | 50.2 | Higher accuracy |
| YOLOv8l | 87MB | Slow | 52.9 | Best accuracy |
| YOLOv8x | 136MB | Slowest | 53.9 | Maximum performance |

## 📥 Manual Download

If automatic download fails, manually download models:

### Option 1: Using Python

```python
from ultralytics import YOLO

# Download and save model
model = YOLO('yolov8n.pt')  # Downloads automatically
# Model saved to current directory
```

### Option 2: Direct Download

Download from Ultralytics GitHub releases:

```bash
# YOLOv8n (recommended)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# YOLOv8s (better accuracy)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt

# Move to models directory
mv yolov8*.pt data/models/
```

### Option 3: Using pip

```bash
# Install ultralytics (already in requirements.txt)
pip install ultralytics

# Models will auto-download on first use
```

## 🎯 Bird Detection Classes

YOLOv8 is trained on COCO dataset, which includes:

- **Class 14**: Bird
- **Class 15**: Cat (can be used as proxy for testing)
- **Class 16**: Dog (can be used as proxy for testing)

The system is configured to detect these classes in `utils/config.py`:

```python
BIRD_CLASS_IDS = [14, 15, 16]  # Bird, cat, dog
```

## 🔧 Fine-tuning (Advanced)

For better bird detection, you can fine-tune YOLOv8 on bird-specific datasets:

### Datasets for Fine-tuning
- **AirBirds** - Aviation-specific bird dataset
- **CUB-200-2011** - 200 bird species
- **iNaturalist Birds** - Large-scale bird images

### Fine-tuning Process

```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov8n.pt')

# Train on custom bird dataset
model.train(
    data='birds.yaml',  # Your dataset config
    epochs=100,
    imgsz=640,
    batch=16
)

# Save fine-tuned model
model.save('data/models/yolov8n_birds_finetuned.pt')
```

## 📊 Model Performance

Expected performance on bird detection:

| Metric | YOLOv8n | YOLOv8s | YOLOv8m |
|--------|---------|---------|---------|
| FPS (GPU) | 80-100 | 60-80 | 40-60 |
| FPS (CPU) | 10-15 | 5-10 | 2-5 |
| Bird Detection Accuracy | ~85% | ~90% | ~92% |

## 🔄 Switching Models

To use a different model, update `utils/config.py`:

```python
# Use larger model for better accuracy
MODEL_PATH = "data/models/yolov8s.pt"

# Or use full path
MODEL_PATH = "/absolute/path/to/model.pt"
```

## 💾 Storage

Place your model files in this directory:

```
data/models/
├── README.md (this file)
├── yolov8n.pt (auto-downloaded)
├── yolov8s.pt (optional)
├── yolov8m.pt (optional)
└── custom_bird_model.pt (if fine-tuned)
```

## 🐛 Troubleshooting

**Issue**: Model not downloading
```bash
# Manually download and place in data/models/
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

**Issue**: CUDA out of memory
```python
# Use smaller model or reduce batch size
MODEL_PATH = "data/models/yolov8n.pt"
```

**Issue**: Slow inference
- Use GPU if available (set `USE_GPU = True` in config)
- Use YOLOv8n instead of larger models
- Reduce input resolution

## 📚 Resources

- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **Model Zoo**: https://github.com/ultralytics/ultralytics
- **Training Guide**: https://docs.ultralytics.com/modes/train/

## ⚙️ Configuration

Key settings in `utils/config.py`:

```python
MODEL_PATH = "data/models/yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5  # Detection confidence
INPUT_SIZE = (640, 640)     # Model input size
USE_GPU = True              # Use GPU acceleration
```

---

**Note**: The `.gitignore` file excludes `*.pt` files to avoid committing large model weights to the repository. Models will be downloaded automatically on first run.
