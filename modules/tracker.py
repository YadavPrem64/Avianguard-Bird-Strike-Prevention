"""
AvianGuard IRAPS - Multi-Bird Tracker Module ⭐
CORE ORIGINAL WORK - Tracks multiple birds across video frames.

Implements DeepSORT-based tracking to maintain consistent IDs for birds
across frames, handling occlusions and temporary disappearances.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from deep_sort_realtime.deepsort_tracker import DeepSort
import utils.config as config
from utils.helpers import setup_logging

logger = setup_logging()


class BirdTracker:
    """
    ⭐ CUSTOM ALGORITHM - Multi-Bird Tracking System
    
    Implements DeepSORT-based tracking to maintain consistent bird IDs
    across frames, even with occlusions and temporary disappearances.
    """
    
    def __init__(self,
                 max_age: int = config.MAX_TRACKING_AGE,
                 min_hits: int = config.MIN_TRACKING_HITS,
                 iou_threshold: float = config.IOU_THRESHOLD):
        """
        Initialize bird tracker.
        
        Args:
            max_age: Maximum frames to keep track without detections
            min_hits: Minimum detections before track is confirmed
            iou_threshold: IoU threshold for matching
        """
        logger.info("Initializing Multi-Bird Tracker (DeepSORT)")
        
        # Initialize DeepSORT tracker
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=min_hits,
            nms_max_overlap=1.0,
            max_cosine_distance=config.MAX_COSINE_DISTANCE,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=config.USE_GPU
        )
        
        # Track metadata
        self.track_metadata = {}
        
        # Statistics
        self.total_tracks = 0
        self.active_tracks = 0
        
        logger.info("DeepSORT tracker initialized")
    
    def update(self, detections: List[Dict[str, Any]], 
              frame: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections from detector
                       Each detection should have 'bbox' and 'confidence'
            frame: Current frame (required for DeepSORT appearance features)
            
        Returns:
            List of tracked objects with consistent IDs
        """
        if frame is None:
            logger.warning("Frame not provided for tracking, features will be limited")
            # Create a dummy frame
            frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
        
        # Convert detections to format expected by DeepSORT
        # Format: [[x1, y1, x2, y2, confidence], ...]
        deepsort_detections = []
        
        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            
            # DeepSORT expects [left, top, width, height, confidence]
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            deepsort_detections.append(([x1, y1, width, height], conf, 'bird'))
        
        # Update tracker
        try:
            tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)
        except Exception as e:
            logger.error(f"Error updating tracker: {e}")
            return []
        
        # Convert tracks back to our format
        tracked_birds = []
        active_ids = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            active_ids.append(track_id)
            
            # Get bounding box
            ltwh = track.to_ltwh()
            x1 = int(ltwh[0])
            y1 = int(ltwh[1])
            x2 = int(ltwh[0] + ltwh[2])
            y2 = int(ltwh[1] + ltwh[3])
            
            bbox = (x1, y1, x2, y2)
            
            # Initialize metadata if new track
            if track_id not in self.track_metadata:
                self.track_metadata[track_id] = {
                    'first_seen': 0,
                    'last_seen': 0,
                    'detection_count': 0,
                    'max_confidence': 0.0
                }
                self.total_tracks += 1
                logger.info(f"New track created: ID {track_id}")
            
            # Update metadata
            self.track_metadata[track_id]['last_seen'] = track.time_since_update
            self.track_metadata[track_id]['detection_count'] += 1
            
            # Create tracked bird object
            tracked_bird = {
                'track_id': track_id,
                'bbox': bbox,
                'class_name': 'bird',
                'confidence': 0.9,  # Tracked objects get high confidence
                'time_since_update': track.time_since_update,
                'age': track.age,
                'state': track.state
            }
            
            tracked_birds.append(tracked_bird)
        
        self.active_tracks = len(tracked_birds)
        
        # Cleanup old metadata
        self._cleanup_metadata(active_ids)
        
        logger.debug(f"Tracking {self.active_tracks} birds")
        
        return tracked_birds
    
    def _cleanup_metadata(self, active_ids: List[int]) -> None:
        """Remove metadata for tracks that are no longer active."""
        inactive_ids = set(self.track_metadata.keys()) - set(active_ids)
        
        for track_id in inactive_ids:
            if track_id in self.track_metadata:
                # Keep metadata for recently lost tracks
                if self.track_metadata[track_id]['last_seen'] > config.MAX_TRACKING_AGE * 2:
                    del self.track_metadata[track_id]
                    logger.debug(f"Removed metadata for track {track_id}")
    
    def get_track_info(self, track_id: int) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific track.
        
        Args:
            track_id: Track ID
            
        Returns:
            Track metadata or None
        """
        return self.track_metadata.get(track_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get tracking statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'total_tracks': self.total_tracks,
            'active_tracks': self.active_tracks,
            'tracked_objects': len(self.track_metadata)
        }
    
    def reset(self) -> None:
        """Reset tracker to initial state."""
        self.tracker = DeepSort(
            max_age=config.MAX_TRACKING_AGE,
            n_init=config.MIN_TRACKING_HITS,
            nms_max_overlap=1.0,
            max_cosine_distance=config.MAX_COSINE_DISTANCE,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=config.USE_GPU
        )
        self.track_metadata = {}
        self.total_tracks = 0
        self.active_tracks = 0
        logger.info("Tracker reset")


class SimpleTracker:
    """
    Simplified tracker using IoU-based matching.
    Fallback option when DeepSORT is not available or needed.
    """
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        """
        Initialize simple tracker.
        
        Args:
            iou_threshold: IoU threshold for matching
            max_age: Maximum frames without detection
        """
        logger.info("Initializing Simple IoU Tracker")
        
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        
        # Active tracks: {track_id: {'bbox': ..., 'age': ..., 'missed': ...}}
        self.tracks = {}
        self.next_id = 1
        
        self.total_tracks = 0
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int],
                      bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, detections: List[Dict[str, Any]], 
              frame: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections
            frame: Current frame (unused in simple tracker)
            
        Returns:
            List of tracked objects
        """
        matched_tracks = set()
        matched_detections = set()
        
        # Match detections to existing tracks
        for track_id, track in list(self.tracks.items()):
            best_iou = 0.0
            best_det_idx = -1
            
            for det_idx, detection in enumerate(detections):
                if det_idx in matched_detections:
                    continue
                
                iou = self._calculate_iou(track['bbox'], detection['bbox'])
                
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx
            
            # Update track if match found
            if best_iou >= self.iou_threshold:
                self.tracks[track_id]['bbox'] = detections[best_det_idx]['bbox']
                self.tracks[track_id]['confidence'] = detections[best_det_idx]['confidence']
                self.tracks[track_id]['missed'] = 0
                self.tracks[track_id]['age'] += 1
                
                matched_tracks.add(track_id)
                matched_detections.add(best_det_idx)
            else:
                # No match, increment missed count
                self.tracks[track_id]['missed'] += 1
        
        # Create new tracks for unmatched detections
        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_detections:
                self.tracks[self.next_id] = {
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'age': 1,
                    'missed': 0
                }
                matched_tracks.add(self.next_id)
                self.next_id += 1
                self.total_tracks += 1
        
        # Remove old tracks
        to_remove = []
        for track_id, track in self.tracks.items():
            if track['missed'] > self.max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
        
        # Format output
        tracked_birds = []
        for track_id in matched_tracks:
            if track_id in self.tracks:
                track = self.tracks[track_id]
                tracked_birds.append({
                    'track_id': track_id,
                    'bbox': track['bbox'],
                    'confidence': track['confidence'],
                    'age': track['age'],
                    'class_name': 'bird'
                })
        
        logger.debug(f"Simple tracker: {len(tracked_birds)} active tracks")
        
        return tracked_birds
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        return {
            'total_tracks': self.total_tracks,
            'active_tracks': len(self.tracks)
        }


if __name__ == "__main__":
    # Test the tracker
    print("Testing Bird Tracker...")
    
    # Create test frame
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    try:
        tracker = BirdTracker()
        print("DeepSORT tracker initialized")
    except Exception as e:
        print(f"DeepSORT failed: {e}")
        print("Using Simple tracker instead")
        tracker = SimpleTracker()
    
    # Simulate detections
    for frame_num in range(10):
        # Simulate moving bird
        x = 100 + frame_num * 10
        y = 100 + frame_num * 5
        
        detections = [
            {
                'bbox': (x, y, x+50, y+50),
                'confidence': 0.9,
                'class_name': 'bird'
            }
        ]
        
        tracks = tracker.update(detections, test_frame)
        
        print(f"Frame {frame_num}: {len(tracks)} tracks")
        for track in tracks:
            print(f"  Track ID {track['track_id']}: {track['bbox']}")
    
    print(f"\nStatistics: {tracker.get_statistics()}")
