"""
AvianGuard IRAPS - Danger Zone Classification Module ⭐
CORE ORIGINAL WORK - Classifies screen zones and assesses position-based risk.

Defines danger zones on the frame (safe, caution, critical areas) and
calculates risk multipliers based on bird positions.
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Any, Optional
import utils.config as config
from utils.helpers import setup_logging, normalize_coordinates

logger = setup_logging()


class DangerZoneClassifier:
    """
    ⭐ CUSTOM ALGORITHM - Danger Zone Classification System
    
    Divides the frame into zones with different risk levels:
    - Safe zones: Peripheral areas
    - Caution zones: Intermediate areas
    - Critical zones: Center areas, engine intakes, cockpit window
    """
    
    def __init__(self, frame_width: int = config.FRAME_WIDTH,
                 frame_height: int = config.FRAME_HEIGHT):
        """
        Initialize danger zone classifier.
        
        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        """
        logger.info("Initializing Danger Zone Classifier")
        
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Define zones
        self.zones = self._define_zones()
        
        logger.info(f"Defined {len(self.zones)} danger zones")
    
    def _define_zones(self) -> List[Dict[str, Any]]:
        """
        ⭐ CUSTOM ALGORITHM - Define danger zones on the frame.
        
        Returns:
            List of zone definitions
        """
        zones = []
        
        # ========== SAFE ZONES (Peripheral) ==========
        # Top edge
        zones.append({
            'name': 'Safe - Top Edge',
            'type': 'safe',
            'bounds': (0, 0, self.frame_width, int(self.frame_height * 0.15)),
            'risk_multiplier': 0.3,
            'color': (0, 255, 0),  # Green
            'priority': 1
        })
        
        # Bottom edge
        zones.append({
            'name': 'Safe - Bottom Edge',
            'type': 'safe',
            'bounds': (0, int(self.frame_height * 0.85), 
                      self.frame_width, self.frame_height),
            'risk_multiplier': 0.3,
            'color': (0, 255, 0),
            'priority': 1
        })
        
        # Left edge
        zones.append({
            'name': 'Safe - Left Edge',
            'type': 'safe',
            'bounds': (0, 0, int(self.frame_width * 0.15), self.frame_height),
            'risk_multiplier': 0.3,
            'color': (0, 255, 0),
            'priority': 1
        })
        
        # Right edge
        zones.append({
            'name': 'Safe - Right Edge',
            'type': 'safe',
            'bounds': (int(self.frame_width * 0.85), 0, 
                      self.frame_width, self.frame_height),
            'risk_multiplier': 0.3,
            'color': (0, 255, 0),
            'priority': 1
        })
        
        # ========== CAUTION ZONES (Intermediate) ==========
        # Upper caution
        zones.append({
            'name': 'Caution - Upper',
            'type': 'caution',
            'bounds': (int(self.frame_width * 0.2), int(self.frame_height * 0.15),
                      int(self.frame_width * 0.8), int(self.frame_height * 0.35)),
            'risk_multiplier': 0.6,
            'color': (0, 255, 255),  # Yellow
            'priority': 2
        })
        
        # Lower caution
        zones.append({
            'name': 'Caution - Lower',
            'type': 'caution',
            'bounds': (int(self.frame_width * 0.2), int(self.frame_height * 0.65),
                      int(self.frame_width * 0.8), int(self.frame_height * 0.85)),
            'risk_multiplier': 0.6,
            'color': (0, 255, 255),
            'priority': 2
        })
        
        # Left caution
        zones.append({
            'name': 'Caution - Left',
            'type': 'caution',
            'bounds': (int(self.frame_width * 0.15), int(self.frame_height * 0.2),
                      int(self.frame_width * 0.35), int(self.frame_height * 0.8)),
            'risk_multiplier': 0.6,
            'color': (0, 255, 255),
            'priority': 2
        })
        
        # Right caution
        zones.append({
            'name': 'Caution - Right',
            'type': 'caution',
            'bounds': (int(self.frame_width * 0.65), int(self.frame_height * 0.2),
                      int(self.frame_width * 0.85), int(self.frame_height * 0.8)),
            'risk_multiplier': 0.6,
            'color': (0, 255, 255),
            'priority': 2
        })
        
        # ========== CRITICAL ZONES (High Risk) ==========
        # Cockpit/Windshield zone (center)
        zones.append({
            'name': 'CRITICAL - Cockpit Window',
            'type': 'critical',
            'bounds': (int(self.frame_width * 0.35), int(self.frame_height * 0.35),
                      int(self.frame_width * 0.65), int(self.frame_height * 0.65)),
            'risk_multiplier': 1.5,
            'color': (0, 0, 255),  # Red
            'priority': 4
        })
        
        # Left engine intake zone
        zones.append({
            'name': 'CRITICAL - Left Engine',
            'type': 'critical',
            'bounds': (int(self.frame_width * 0.15), int(self.frame_height * 0.4),
                      int(self.frame_width * 0.35), int(self.frame_height * 0.7)),
            'risk_multiplier': 1.3,
            'color': (0, 0, 255),
            'priority': 3
        })
        
        # Right engine intake zone
        zones.append({
            'name': 'CRITICAL - Right Engine',
            'type': 'critical',
            'bounds': (int(self.frame_width * 0.65), int(self.frame_height * 0.4),
                      int(self.frame_width * 0.85), int(self.frame_height * 0.7)),
            'risk_multiplier': 1.3,
            'color': (0, 0, 255),
            'priority': 3
        })
        
        return zones
    
    def classify_position(self, x: float, y: float) -> Dict[str, Any]:
        """
        ⭐ CUSTOM ALGORITHM - Classify a point's danger level.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Zone classification with risk information
        """
        # Find which zone(s) the point is in
        matching_zones = []
        
        for zone in self.zones:
            x1, y1, x2, y2 = zone['bounds']
            if x1 <= x <= x2 and y1 <= y <= y2:
                matching_zones.append(zone)
        
        # If in multiple zones, use highest priority (highest number)
        if matching_zones:
            zone = max(matching_zones, key=lambda z: z['priority'])
        else:
            # Default to safe if not in any defined zone
            zone = {
                'name': 'Undefined - Assumed Safe',
                'type': 'safe',
                'risk_multiplier': 0.5,
                'color': (128, 128, 128),
                'priority': 0
            }
        
        # Calculate normalized danger score (0-1)
        # Based on distance from center
        center_x = self.frame_width / 2
        center_y = self.frame_height / 2
        
        # Normalized distance from center
        max_dist = np.sqrt((self.frame_width/2)**2 + (self.frame_height/2)**2)
        current_dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        center_proximity = 1.0 - (current_dist / max_dist)
        
        # Combine zone type with center proximity
        danger_score = zone['risk_multiplier'] * center_proximity
        danger_score = np.clip(danger_score, 0, 1)
        
        return {
            'zone_name': zone['name'],
            'zone_type': zone['type'],
            'risk_multiplier': zone['risk_multiplier'],
            'danger_score': danger_score,
            'color': zone['color'],
            'center_proximity': center_proximity
        }
    
    def classify_bbox(self, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Classify a bounding box's danger level.
        
        Uses the center point of the bounding box.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Zone classification
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        return self.classify_position(center_x, center_y)
    
    def get_zone_mask(self, zone_type: str) -> np.ndarray:
        """
        Create a binary mask for zones of a specific type.
        
        Args:
            zone_type: Type of zone ('safe', 'caution', 'critical')
            
        Returns:
            Binary mask (0 or 255)
        """
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        
        for zone in self.zones:
            if zone['type'] == zone_type:
                x1, y1, x2, y2 = zone['bounds']
                mask[y1:y2, x1:x2] = 255
        
        return mask
    
    def visualize_zones(self, frame: np.ndarray, 
                       alpha: float = 0.3,
                       show_labels: bool = True) -> np.ndarray:
        """
        Visualize danger zones on a frame.
        
        Args:
            frame: Input frame
            alpha: Transparency for zone overlay
            show_labels: Whether to show zone labels
            
        Returns:
            Frame with zone visualization
        """
        overlay = frame.copy()
        
        # Draw zones in order of priority (low to high)
        sorted_zones = sorted(self.zones, key=lambda z: z['priority'])
        
        for zone in sorted_zones:
            x1, y1, x2, y2 = zone['bounds']
            color = zone['color']
            
            # Draw semi-transparent rectangle
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            
            # Draw border
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Add label if requested
            if show_labels and zone['priority'] >= 3:  # Only label critical zones
                label = zone['name']
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                
                # Get text size
                (text_width, text_height), _ = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                
                # Position at top of zone
                text_x = x1 + 5
                text_y = y1 + text_height + 5
                
                # Background for text
                cv2.rectangle(
                    overlay,
                    (text_x - 2, text_y - text_height - 2),
                    (text_x + text_width + 2, text_y + 2),
                    (0, 0, 0),
                    -1
                )
                
                # Draw text
                cv2.putText(
                    overlay,
                    label,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness
                )
        
        # Blend with original frame
        result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return result
    
    def get_heatmap(self, resolution: Tuple[int, int] = (50, 50)) -> np.ndarray:
        """
        Generate a danger heatmap of the frame.
        
        Args:
            resolution: Grid resolution for heatmap
            
        Returns:
            Heatmap array (values 0-1)
        """
        grid_h, grid_w = resolution
        heatmap = np.zeros((grid_h, grid_w))
        
        # Calculate danger score for each grid cell
        for i in range(grid_h):
            for j in range(grid_w):
                # Convert grid coordinates to frame coordinates
                x = (j + 0.5) * (self.frame_width / grid_w)
                y = (i + 0.5) * (self.frame_height / grid_h)
                
                # Get danger score
                classification = self.classify_position(x, y)
                heatmap[i, j] = classification['danger_score']
        
        return heatmap
    
    def get_all_zones(self) -> List[Dict[str, Any]]:
        """Get all defined zones."""
        return self.zones
    
    def update_frame_size(self, width: int, height: int) -> None:
        """
        Update frame dimensions and recalculate zones.
        
        Args:
            width: New frame width
            height: New frame height
        """
        self.frame_width = width
        self.frame_height = height
        self.zones = self._define_zones()
        logger.info(f"Updated zones for new frame size: {width}x{height}")


if __name__ == "__main__":
    # Test the danger zone classifier
    print("Testing Danger Zone Classifier...")
    
    classifier = DangerZoneClassifier()
    
    # Test various positions
    test_points = [
        (50, 50, "Top-left corner"),
        (640, 360, "Center"),
        (300, 400, "Left engine area"),
        (980, 400, "Right engine area"),
        (1200, 680, "Bottom-right corner"),
    ]
    
    for x, y, description in test_points:
        classification = classifier.classify_position(x, y)
        print(f"\n{description} ({x}, {y}):")
        print(f"  Zone: {classification['zone_name']}")
        print(f"  Type: {classification['zone_type']}")
        print(f"  Danger Score: {classification['danger_score']:.2f}")
        print(f"  Risk Multiplier: {classification['risk_multiplier']}")
    
    # Generate heatmap
    heatmap = classifier.get_heatmap()
    print(f"\nGenerated danger heatmap: {heatmap.shape}")
    print(f"Min danger: {heatmap.min():.2f}, Max danger: {heatmap.max():.2f}")
    
    # Test visualization on a dummy frame
    test_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
    viz_frame = classifier.visualize_zones(test_frame, alpha=0.4)
    print(f"\nVisualized zones on frame: {viz_frame.shape}")
