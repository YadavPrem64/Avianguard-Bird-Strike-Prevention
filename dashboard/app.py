"""
AvianGuard IRAPS - Streamlit Dashboard
Interactive web-based interface for bird strike prevention system.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="AvianGuard IRAPS",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
    }
    .risk-low { color: #28a745; font-weight: bold; }
    .risk-medium { color: #ffc107; font-weight: bold; }
    .risk-high { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🦅 AvianGuard IRAPS - Bird Strike Prevention System</div>', 
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x150/667eea/ffffff?text=AvianGuard", 
             use_column_width=True)
    
    st.header("⚙️ System Configuration")
    
    # Input source selection
    input_source = st.radio(
        "Input Source",
        ["Upload Video", "Webcam", "Sample Video"],
        index=0
    )
    
    # Model selection
    model_size = st.selectbox(
        "Detection Model",
        ["YOLOv8n (Fast)", "YOLOv8s (Balanced)", "YOLOv8m (Accurate)"],
        index=0
    )
    
    # Risk thresholds
    st.subheader("Risk Thresholds")
    risk_low = st.slider("Low Risk Threshold", 0, 100, 30)
    risk_medium = st.slider("Medium Risk Threshold", 0, 100, 70)
    
    # Processing options
    st.subheader("Processing Options")
    show_trajectory = st.checkbox("Show Trajectory", value=True)
    show_zones = st.checkbox("Show Danger Zones", value=True)
    audio_alerts = st.checkbox("Enable Audio Alerts", value=True)
    
    # Start/Stop buttons
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶️ Start", use_container_width=True)
    with col2:
        stop_btn = st.button("⏹️ Stop", use_container_width=True)

# Main content area
main_col1, main_col2 = st.columns([2, 1])

with main_col1:
    st.header("📹 Live Video Feed")
    
    # Video display area
    video_placeholder = st.empty()
    
    # Display placeholder image
    placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder_img, "AvianGuard IRAPS", (150, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.putText(placeholder_img, "Click 'Start' to begin processing", (130, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    
    video_placeholder.image(placeholder_img, channels="BGR", use_column_width=True)
    
    # Status bar
    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        st.metric("System Status", "Idle", "Waiting")
    with status_col2:
        st.metric("FPS", "0", "")
    with status_col3:
        st.metric("Birds Detected", "0", "")

with main_col2:
    st.header("📊 Risk Assessment")
    
    # Risk meter
    risk_value = 0
    risk_level = "LOW"
    risk_color = "green"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 10px; color: white; margin-bottom: 1rem;">
        <h1 style="font-size: 4rem; margin: 0;">{risk_value}%</h1>
        <h3 style="margin: 0; color: {risk_color};">{risk_level} RISK</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Bird tracking info
    st.subheader("🐦 Active Tracks")
    track_placeholder = st.empty()
    track_placeholder.info("No birds currently tracked")
    
    # Alert panel
    st.subheader("⚠️ Alert Messages")
    alert_placeholder = st.empty()
    alert_placeholder.success("System ready. No alerts.")
    
    # Statistics
    st.subheader("📈 Session Statistics")
    stat_col1, stat_col2 = st.columns(2)
    with stat_col1:
        st.metric("Total Detections", "0")
        st.metric("Max Birds", "0")
    with stat_col2:
        st.metric("High Risk Events", "0")
        st.metric("Avg Risk Score", "0%")

# Radar mini-map
st.header("🗺️ Radar Map")
radar_col1, radar_col2 = st.columns([1, 1])

with radar_col1:
    # Create simple radar visualization
    radar_img = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Draw concentric circles
    center = (150, 150)
    for radius in [50, 100, 150]:
        cv2.circle(radar_img, center, radius, (100, 100, 100), 1)
    
    # Draw crosshairs
    cv2.line(radar_img, (0, 150), (300, 150), (100, 100, 100), 1)
    cv2.line(radar_img, (150, 0), (150, 300), (100, 100, 100), 1)
    
    # Draw aircraft (center)
    cv2.circle(radar_img, center, 10, (0, 255, 0), -1)
    cv2.putText(radar_img, "AIRCRAFT", (105, 170), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    st.image(radar_img, channels="BGR", caption="Bird Positions Relative to Aircraft")

with radar_col2:
    st.subheader("Danger Zone Legend")
    st.markdown("""
    - 🟢 **Safe Zone**: Peripheral areas (low risk)
    - 🟡 **Caution Zone**: Intermediate areas (medium risk)
    - 🔴 **Critical Zone**: Engine/cockpit areas (high risk)
    
    **Radar Range**: 500m
    """)

# Information tabs
st.header("ℹ️ System Information")
tab1, tab2, tab3, tab4 = st.tabs(["About", "Algorithm Details", "Configuration", "Help"])

with tab1:
    st.markdown("""
    ### About AvianGuard IRAPS
    
    **Intelligent Risk Assessment & Prediction System** for bird strike prevention in aviation.
    
    #### Key Features:
    - 🎯 Real-time bird detection using YOLOv8
    - 🔍 Multi-bird tracking with DeepSORT
    - 🧠 Fuzzy logic risk assessment (custom algorithm)
    - 📈 Trajectory prediction using Kalman filtering
    - ⏱️ Time-to-collision calculation
    - 🗺️ 6-zone danger classification
    - 🔊 Audio/visual alerts
    
    #### System Components:
    1. **Detection Engine**: YOLOv8-based bird detection
    2. **Intelligence Layer**: Custom fuzzy logic + Kalman filtering
    3. **Alert System**: Real-time risk-based warnings
    
    **Version**: 1.0.0  
    **License**: MIT
    """)

with tab2:
    st.markdown("""
    ### Algorithm Details
    
    #### 1. Fuzzy Logic Risk Assessment ⭐
    
    **Input Variables:**
    - Distance (0-500 pixels)
    - Bird Count (0-100)
    - Velocity (0-100 px/frame)
    - Position (0-1, danger score)
    
    **Output**: Risk Score (0-100%)
    
    **Rules**: 20+ fuzzy inference rules combining multiple factors
    
    #### 2. Trajectory Prediction ⭐
    
    - **Method**: Kalman Filter
    - **State Vector**: [x, y, vx, vy]
    - **Prediction Horizon**: 90 frames (3 seconds)
    
    #### 3. Time-to-Collision ⭐
    
    - **Optical Flow Method**: TTC = size / (dsize/dt)
    - **Monocular Distance**: Estimates from bbox size
    - **Ensemble**: Combines multiple methods
    
    #### 4. Danger Zones ⭐
    
    - Safe: Peripheral (20%)
    - Caution: Intermediate (40%)  
    - Critical: Center/engines (40%)
    """)

with tab3:
    st.markdown(f"""
    ### Current Configuration
    
    **Detection:**
    - Model: {model_size}
    - Confidence Threshold: 0.5
    - Input Size: 640x640
    
    **Risk Thresholds:**
    - Low: < {risk_low}%
    - Medium: {risk_low}% - {risk_medium}%
    - High: > {risk_medium}%
    
    **Processing:**
    - Show Trajectory: {show_trajectory}
    - Show Danger Zones: {show_zones}
    - Audio Alerts: {audio_alerts}
    - Skip Frames: 1 (process all)
    
    **Performance:**
    - Target FPS: 30
    - GPU Acceleration: Enabled
    - Max Tracks: 50
    """)

with tab4:
    st.markdown("""
    ### Help & Troubleshooting
    
    #### Getting Started:
    1. Select input source (video file or webcam)
    2. Adjust risk thresholds if needed
    3. Click "Start" to begin processing
    4. Monitor risk levels and alerts
    
    #### Common Issues:
    
    **No birds detected:**
    - Lower confidence threshold in config
    - Ensure good video quality
    - Check lighting conditions
    
    **Low FPS:**
    - Enable GPU acceleration
    - Use smaller model (YOLOv8n)
    - Increase skip frames
    
    **False alerts:**
    - Increase risk thresholds
    - Tune fuzzy logic rules
    - Filter by position
    
    #### Keyboard Shortcuts:
    - `Space`: Pause/Resume
    - `Q`: Quit
    - `S`: Save screenshot
    
    #### Support:
    - GitHub: [Report Issues](https://github.com/YadavPrem64/Avianguard-Bird-Strike-Prevention)
    - Documentation: Check `docs/` folder
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>AvianGuard IRAPS v1.0 | Built with ❤️ for Aviation Safety | 
    <a href="https://github.com/YadavPrem64/Avianguard-Bird-Strike-Prevention">GitHub</a></p>
</div>
""", unsafe_allow_html=True)

# Note for users
st.info("""
**Note**: This is the dashboard interface. To process video files, use the main.py script:
```bash
python main.py --input video.mp4 --output result.mp4
```

The dashboard provides visualization and configuration options. Full integration with live processing is coming soon!
""")
