"""
AvianGuard IRAPS - Time-to-Collision Calculator ⭐
CORE ORIGINAL WORK - Calculates estimated time until bird collision.

Uses bounding box growth rate analysis to estimate when a bird will
collide with the aircraft.
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from collections import deque
import utils.config as config
from utils.helpers import setup_logging, calculate_bbox_area, calculate_bbox_diagonal

logger = setup_logging()


class TimeToCollisionCalculator:
    """
    ⭐ CUSTOM ALGORITHM - Time-to-Collision (TTC) Calculator
    
    Analyzes bounding box growth rate to estimate time until collision.
    Uses optical flow principles and assumed bird size to calculate TTC.
    """
    
    def __init__(self, history_size: int = 10):
        """
        Initialize TTC calculator.
        
        Args:
            history_size: Number of frames to keep in history for calculation
        """
        logger.info("Initializing Time-to-Collision Calculator")
        
        self.history_size = history_size
        
        # Store bbox history for each bird
        # bird_id -> deque of (timestamp, bbox, area)
        self.bbox_history = {}
        
        # Parameters from config
        self.min_growth_rate = config.MIN_BBOX_GROWTH_RATE
        self.assumed_bird_size = config.ASSUMED_BIRD_SIZE_CM
        self.focal_length = config.CAMERA_FOCAL_LENGTH_PX
    
    def update_bbox(self, bird_id: int, bbox: Tuple[int, int, int, int], 
                   timestamp: Optional[int] = None) -> None:
        """
        Update bounding box history for a bird.
        
        Args:
            bird_id: Unique bird tracking ID
            bbox: Bounding box (x1, y1, x2, y2)
            timestamp: Frame number or timestamp
        """
        if bird_id not in self.bbox_history:
            self.bbox_history[bird_id] = deque(maxlen=self.history_size)
        
        # Calculate bbox metrics
        area = calculate_bbox_area(bbox)
        diagonal = calculate_bbox_diagonal(bbox)
        
        # Store data
        self.bbox_history[bird_id].append({
            'timestamp': timestamp,
            'bbox': bbox,
            'area': area,
            'diagonal': diagonal
        })
    
    def calculate_growth_rate(self, bird_id: int) -> Optional[float]:
        """
        Calculate bounding box growth rate (pixels per frame).
        
        Args:
            bird_id: Unique bird tracking ID
            
        Returns:
            Growth rate in pixels/frame or None if insufficient data
        """
        if bird_id not in self.bbox_history:
            return None
        
        history = list(self.bbox_history[bird_id])
        
        if len(history) < 2:
            return None
        
        # Calculate growth rates between consecutive frames
        growth_rates = []
        
        for i in range(1, len(history)):
            prev_data = history[i-1]
            curr_data = history[i]
            
            # Use diagonal as size metric
            prev_size = prev_data['diagonal']
            curr_size = curr_data['diagonal']
            
            # Calculate time difference
            if prev_data['timestamp'] is not None and curr_data['timestamp'] is not None:
                dt = curr_data['timestamp'] - prev_data['timestamp']
                if dt <= 0:
                    dt = 1
            else:
                dt = 1
            
            # Growth rate
            growth = (curr_size - prev_size) / dt
            growth_rates.append(growth)
        
        # Return average growth rate
        avg_growth_rate = np.mean(growth_rates)
        
        logger.debug(f"Bird {bird_id}: Growth rate = {avg_growth_rate:.2f} px/frame")
        
        return avg_growth_rate
    
    def calculate_ttc_simple(self, bird_id: int) -> Optional[float]:
        """
        ⭐ CUSTOM ALGORITHM - Calculate TTC using simple growth rate method.
        
        This method assumes the bird is approaching directly and uses
        the rate of bbox expansion to estimate collision time.
        
        Formula: TTC = current_size / growth_rate
        (Time until size becomes infinite, i.e., collision)
        
        Args:
            bird_id: Unique bird tracking ID
            
        Returns:
            Time to collision in seconds, or None if not calculable
        """
        growth_rate = self.calculate_growth_rate(bird_id)
        
        if growth_rate is None or growth_rate < self.min_growth_rate:
            # Not approaching or approaching too slowly
            return None
        
        if bird_id not in self.bbox_history or len(self.bbox_history[bird_id]) == 0:
            return None
        
        # Get current bbox size
        current_data = list(self.bbox_history[bird_id])[-1]
        current_size = current_data['diagonal']
        
        # Simple TTC calculation
        # Time until the bird "fills the screen" (proxy for collision)
        # Assume screen diagonal as "collision size"
        collision_size = np.sqrt(config.FRAME_WIDTH**2 + config.FRAME_HEIGHT**2)
        
        ttc_frames = (collision_size - current_size) / growth_rate
        
        # Convert to seconds
        ttc_seconds = ttc_frames / config.FPS
        
        # Clamp to reasonable range
        ttc_seconds = np.clip(ttc_seconds, 0, 60)
        
        logger.debug(f"Bird {bird_id}: TTC (simple) = {ttc_seconds:.2f}s")
        
        return ttc_seconds
    
    def calculate_ttc_optical(self, bird_id: int) -> Optional[float]:
        """
        ⭐ CUSTOM ALGORITHM - Calculate TTC using optical expansion formula.
        
        Based on optical flow theory:
        TTC = Z / (dZ/dt) = size / (dsize/dt)
        
        Where Z is distance and dZ/dt is approach velocity.
        
        Args:
            bird_id: Unique bird tracking ID
            
        Returns:
            Time to collision in seconds, or None if not calculable
        """
        if bird_id not in self.bbox_history or len(self.bbox_history[bird_id]) < 2:
            return None
        
        history = list(self.bbox_history[bird_id])
        
        # Get recent measurements
        recent_history = history[-5:] if len(history) >= 5 else history
        
        if len(recent_history) < 2:
            return None
        
        # Calculate instantaneous TTC for each pair
        ttc_values = []
        
        for i in range(1, len(recent_history)):
            prev_data = recent_history[i-1]
            curr_data = recent_history[i]
            
            prev_size = prev_data['diagonal']
            curr_size = curr_data['diagonal']
            
            # Time difference
            if prev_data['timestamp'] is not None and curr_data['timestamp'] is not None:
                dt = (curr_data['timestamp'] - prev_data['timestamp']) / config.FPS
                if dt <= 0:
                    dt = 1.0 / config.FPS
            else:
                dt = 1.0 / config.FPS
            
            # Rate of expansion
            expansion_rate = (curr_size - prev_size) / dt
            
            # TTC calculation (optical flow formula)
            if expansion_rate > self.min_growth_rate / config.FPS:
                ttc = curr_size / expansion_rate
                ttc_values.append(ttc)
        
        if len(ttc_values) == 0:
            return None
        
        # Average TTC
        avg_ttc = np.median(ttc_values)  # Use median for robustness
        
        # Clamp to reasonable range
        avg_ttc = np.clip(avg_ttc, 0, 60)
        
        logger.debug(f"Bird {bird_id}: TTC (optical) = {avg_ttc:.2f}s")
        
        return avg_ttc
    
    def calculate_ttc_monocular(self, bird_id: int) -> Optional[float]:
        """
        ⭐ CUSTOM ALGORITHM - Calculate TTC using monocular distance estimation.
        
        Uses assumed bird size and focal length to estimate actual distance,
        then combines with velocity to calculate TTC.
        
        Formula: 
        - Distance (Z) = (real_size * focal_length) / pixel_size
        - Velocity (dZ/dt) from distance change over time
        - TTC = Z / (dZ/dt)
        
        Args:
            bird_id: Unique bird tracking ID
            
        Returns:
            Time to collision in seconds, or None if not calculable
        """
        if bird_id not in self.bbox_history or len(self.bbox_history[bird_id]) < 3:
            return None
        
        history = list(self.bbox_history[bird_id])
        
        # Calculate distances for recent frames
        distances = []
        timestamps = []
        
        for data in history[-5:]:
            pixel_size = data['diagonal']
            
            # Estimate distance using similar triangles
            # Z = (Real_Size * Focal_Length) / Pixel_Size
            estimated_distance = (self.assumed_bird_size * self.focal_length) / pixel_size
            
            distances.append(estimated_distance)
            timestamps.append(data['timestamp'] if data['timestamp'] is not None else 0)
        
        if len(distances) < 2:
            return None
        
        # Calculate approach velocity (negative distance change)
        velocities = []
        for i in range(1, len(distances)):
            dt = (timestamps[i] - timestamps[i-1]) / config.FPS if timestamps[i] != 0 else 1.0/config.FPS
            if dt <= 0:
                dt = 1.0 / config.FPS
            
            # Velocity (negative because distance is decreasing)
            velocity = (distances[i-1] - distances[i]) / dt
            velocities.append(velocity)
        
        # Average velocity
        avg_velocity = np.mean(velocities)
        
        # Current distance
        current_distance = distances[-1]
        
        # TTC calculation
        if avg_velocity > 0.1:  # Bird is approaching
            ttc = current_distance / avg_velocity
            ttc = np.clip(ttc, 0, 60)
            
            logger.debug(
                f"Bird {bird_id}: Distance={current_distance:.1f}cm, "
                f"Velocity={avg_velocity:.2f}cm/s, TTC={ttc:.2f}s"
            )
            
            return ttc
        else:
            # Not approaching or moving away
            return None
    
    def calculate_ttc(self, bird_id: int, method: str = 'optical') -> Optional[float]:
        """
        Calculate time to collision using specified method.
        
        Args:
            bird_id: Unique bird tracking ID
            method: Calculation method ('simple', 'optical', 'monocular', 'ensemble')
            
        Returns:
            Time to collision in seconds, or None if not calculable
        """
        if method == 'simple':
            return self.calculate_ttc_simple(bird_id)
        elif method == 'optical':
            return self.calculate_ttc_optical(bird_id)
        elif method == 'monocular':
            return self.calculate_ttc_monocular(bird_id)
        elif method == 'ensemble':
            # Combine multiple methods
            ttcs = []
            
            ttc_simple = self.calculate_ttc_simple(bird_id)
            if ttc_simple is not None:
                ttcs.append(ttc_simple)
            
            ttc_optical = self.calculate_ttc_optical(bird_id)
            if ttc_optical is not None:
                ttcs.append(ttc_optical)
            
            ttc_monocular = self.calculate_ttc_monocular(bird_id)
            if ttc_monocular is not None:
                ttcs.append(ttc_monocular)
            
            if len(ttcs) > 0:
                # Use median for robustness
                return np.median(ttcs)
            else:
                return None
        else:
            logger.warning(f"Unknown TTC method: {method}")
            return None
    
    def get_risk_level(self, ttc: Optional[float]) -> str:
        """
        Convert TTC to risk level.
        
        Args:
            ttc: Time to collision in seconds
            
        Returns:
            Risk level string
        """
        if ttc is None or ttc > config.TTC_WARNING_THRESHOLD:
            return 'low'
        elif ttc > config.TTC_CRITICAL_THRESHOLD:
            return 'medium'
        else:
            return 'high'
    
    def remove_bird(self, bird_id: int) -> None:
        """Remove history for a bird that's no longer tracked."""
        if bird_id in self.bbox_history:
            del self.bbox_history[bird_id]
            logger.debug(f"Removed TTC history for bird {bird_id}")
    
    def cleanup_old_birds(self, active_bird_ids: List[int]) -> None:
        """Remove history for birds that are no longer active."""
        inactive_ids = set(self.bbox_history.keys()) - set(active_bird_ids)
        
        for bird_id in inactive_ids:
            self.remove_bird(bird_id)


if __name__ == "__main__":
    # Test the TTC calculator
    print("Testing Time-to-Collision Calculator...")
    
    calculator = TimeToCollisionCalculator()
    
    # Simulate a bird approaching (growing bbox)
    bird_id = 1
    
    for t in range(30):
        # Simulate growing bounding box (bird approaching)
        # Start small, grow larger
        base_size = 50
        growth = t * 5  # Growing by 5 pixels per frame
        
        x1 = 300 - growth // 2
        y1 = 200 - growth // 2
        x2 = 300 + base_size + growth // 2
        y2 = 200 + base_size + growth // 2
        
        bbox = (x1, y1, x2, y2)
        
        calculator.update_bbox(bird_id, bbox, timestamp=t)
        
        # Calculate TTC after we have enough data
        if t >= 5:
            ttc_optical = calculator.calculate_ttc(bird_id, method='optical')
            ttc_simple = calculator.calculate_ttc(bird_id, method='simple')
            
            print(f"Frame {t}: BBox diagonal={calculate_bbox_diagonal(bbox):.1f}, "
                  f"TTC(optical)={ttc_optical:.2f}s, TTC(simple)={ttc_simple:.2f}s")
