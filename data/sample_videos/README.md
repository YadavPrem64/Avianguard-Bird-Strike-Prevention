# Sample Videos for AvianGuard IRAPS

This directory is for storing test videos for the bird strike prevention system.

## 📥 How to Obtain Test Videos

### Option 1: Download from YouTube

Use `yt-dlp` to download bird/aircraft videos:

```bash
# Install yt-dlp
pip install yt-dlp

# Download a video
yt-dlp -f "best[height<=720]" -o "sample.mp4" "YOUTUBE_URL_HERE"
```

**Recommended Search Terms:**
- "birds flying close camera"
- "aircraft runway birds"
- "bird flock flying"
- "seagulls approaching camera"
- "cockpit view landing"

### Option 2: Free Stock Footage Websites

Download bird videos from these sources:

- **Pexels** - https://www.pexels.com/search/videos/birds%20flying/
- **Pixabay** - https://pixabay.com/videos/search/birds/
- **Videvo** - https://www.videvo.net/free-stock-video-footage/birds/
- **Coverr** - https://coverr.co/s/bird

### Option 3: Use Webcam

Test with your webcam (simulating bird detection on moving objects):

```bash
python main.py --camera 0
```

### Option 4: Research Datasets

For academic work, consider these datasets:

- **AirBirds Dataset** - Bird detection in aviation context
- **COCO Dataset** - Contains bird class (class ID: 14)
- **CUB-200-2011** - Bird species dataset

## 📝 Video Recommendations

For best results, use videos with:

- ✅ Clear visibility of birds
- ✅ Birds at various distances
- ✅ Multiple birds in frame
- ✅ Moving camera or moving birds
- ✅ 720p or higher resolution
- ✅ 30 FPS or higher

Avoid:
- ❌ Very low light conditions
- ❌ Heavy motion blur
- ❌ Extremely far birds (tiny in frame)
- ❌ Very compressed videos

## 🎬 Sample Video Placement

Place your test videos in this directory:

```
data/sample_videos/
├── README.md (this file)
├── test_single_bird.mp4
├── test_flock.mp4
├── test_approach.mp4
└── ...
```

## ⚙️ Configuration

Update the default video path in `utils/config.py`:

```python
DEFAULT_VIDEO_PATH = "data/sample_videos/your_video.mp4"
```

## 📊 Expected Results

The system will:
1. Detect birds in each frame
2. Track them with consistent IDs
3. Predict their trajectories
4. Calculate collision risk
5. Generate alerts when risk is high
6. Save annotated output video

## 🔧 Troubleshooting

**Issue**: Video not loading
- Check file path is correct
- Ensure video codec is supported by OpenCV
- Try converting to MP4 with H.264 codec

**Issue**: No birds detected
- Verify video contains clear bird images
- Adjust `CONFIDENCE_THRESHOLD` in config.py
- Try with different test video

**Issue**: Poor tracking performance
- Increase video quality/resolution
- Adjust tracking parameters in config.py
- Ensure good lighting in video

## 💡 Tips

- Start with short videos (10-30 seconds) for testing
- Use videos with prominent birds (not tiny specks)
- Good examples: seagulls near beach, pigeons in city
- Aircraft cockpit footage works great if available
- For demos, use videos with birds approaching camera

---

**Need help?** Check the main documentation or open an issue on GitHub.
