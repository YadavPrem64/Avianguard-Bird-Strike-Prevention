# AvianGuard IRAPS 🦅✈️

**Intelligent Risk Assessment & Prediction System for Bird Strike Prevention**

> A comprehensive AI-powered system combining computer vision, fuzzy logic, and predictive analytics to prevent bird strikes in aviation.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Overview

AvianGuard IRAPS is a **100% software-based bird strike prevention system** that uses hybrid AI to provide real-time risk assessment and predictive warnings for aircraft. The system processes video footage to detect birds, track their movements, predict trajectories, and calculate collision risk using advanced algorithms.

### Key Statistics
- **80% Original Custom Algorithms** (Fuzzy Logic, Trajectory Prediction, TTC Calculation)
- **Real-time Processing** (30 FPS capability)
- **Multi-Bird Tracking** (Up to 50 simultaneous tracks)
- **6 Danger Zones** (Safe, Caution, Critical classifications)

## 🚨 The Problem

Bird strikes pose severe safety and economic threats to aviation:
- **13,000+ strikes annually** in the US alone
- **$1.2 billion** in annual damages worldwide
- **219 human fatalities** since 1988

## 💡 Our Solution

A **3-phase hybrid AI system**:

1. **Detection & Tracking** (YOLOv8 + DeepSORT)
2. **Intelligence & Risk Assessment** ⭐ (Custom Fuzzy Logic Engine)
3. **Prediction & Warning** ⭐ (Kalman Filter + TTC Calculator)

## ✨ Key Features

- ✅ Real-time bird detection using YOLOv8
- ✅ Multi-bird tracking with DeepSORT
- ✅ Trajectory prediction (3 seconds ahead)
- ✅ Fuzzy logic risk assessment (0-100% score)
- ✅ Time-to-collision calculation
- ✅ Danger zone classification
- ✅ Visual and audio alerts
- ✅ Explainable AI output

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/YadavPrem64/Avianguard-Bird-Strike-Prevention.git
cd Avianguard-Bird-Strike-Prevention

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Process video file
python main.py --input video.mp4 --output result.mp4

# Use webcam
python main.py --camera 0

# Headless mode (no display)
python main.py --input video.mp4 --no-display
```

## 🛠️ Technology Stack

- **Python 3.8+** - Core language
- **YOLOv8** (Ultralytics) - Object detection
- **DeepSORT** - Multi-object tracking
- **scikit-fuzzy** - Fuzzy logic engine ⭐
- **FilterPy** - Kalman filtering ⭐
- **OpenCV** - Video processing
- **NumPy** - Numerical computing
- **Streamlit** - Dashboard interface

## 📁 Project Structure

```
avianguard-bird-strike-prevention/
├── modules/              # Core processing modules
│   ├── detection.py     # YOLOv8 detection
│   ├── fuzzy_logic.py  # Risk assessment ⭐
│   ├── trajectory.py   # Prediction ⭐
│   ├── collision_calc.py # TTC calculation ⭐
│   ├── tracker.py      # Multi-bird tracking ⭐
│   └── danger_zones.py # Zone classification ⭐
├── utils/               # Utilities
├── dashboard/           # Streamlit UI
├── docs/               # Documentation
├── tests/              # Unit tests
└── main.py             # Entry point
```

## 🎓 Academic Context

### Originality (80% Custom Work) ⭐

1. **Fuzzy Logic Risk Engine** - 20+ custom rules
2. **Kalman Filter Predictor** - Trajectory forecasting
3. **TTC Calculator** - Optical flow-based
4. **Danger Zone System** - 6-zone classification
5. **Complete Integration** - End-to-end pipeline

### Suitable For
- Final year B.Tech/B.E. projects
- M.Tech thesis work
- Research publications
- Industry projects

## 📚 Documentation

See `docs/` directory for:
- Implementation Guide
- Fuzzy Logic Design
- Algorithm Details
- Weekly Development Plan
- Presentation Guide

## 🔮 Future Work

- Species classification
- Weather integration
- Multi-camera fusion
- 3D trajectory visualization
- Cloud deployment

## 👥 Contributors

- **Prem Yadav** - [@YadavPrem64](https://github.com/YadavPrem64)

## 📄 License

MIT License - See LICENSE file for details

---

**Made with ❤️ for Aviation Safety**

⭐ Star this repo if you find it useful!
