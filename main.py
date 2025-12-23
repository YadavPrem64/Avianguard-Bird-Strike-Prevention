"""
AvianGuard IRAPS - Main Application
Integrated Bird Strike Prevention System

This is the main entry point that integrates all modules:
- Detection (YOLOv8)
- Video Processing
- Fuzzy Logic Risk Assessment ⭐
- Trajectory Prediction ⭐
- Time-to-Collision Calculation ⭐
- Multi-Bird Tracking ⭐
- Danger Zone Classification ⭐
"""

import cv2
import numpy as np
import argparse
import sys
from typing import Dict, Any, List
import time

# Import custom modules
from modules.detection import BirdDetector
from modules.video_processor import VideoProcessor
from modules.fuzzy_logic import FuzzyRiskAssessment
from modules.trajectory import TrajectoryPredictor
from modules.collision_calc import TimeToCollisionCalculator
from modules.tracker import BirdTracker, SimpleTracker
from modules.danger_zones import DangerZoneClassifier

import utils.config as config
from utils.helpers import (
    setup_logging, ensure_directory, get_risk_color,
    get_risk_level_name, draw_text_with_background, FPSCounter
)

logger = setup_logging()


class AvianGuardIRAPS:
    """
    Main AvianGuard Intelligent Risk Assessment & Prediction System.
    
    Integrates all detection, tracking, and intelligence modules to provide
    real-time bird strike prevention capabilities.
    """
    
    def __init__(self, video_source: str = None, output_path: str = None):
        """
        Initialize AvianGuard IRAPS.
        
        Args:
            video_source: Path to video file or camera index
            output_path: Path for output video
        """
        logger.info("=" * 80)
        logger.info("Initializing AvianGuard IRAPS - Bird Strike Prevention System")
        logger.info("=" * 80)
        
        self.video_source = video_source or config.DEFAULT_VIDEO_PATH
        self.output_path = output_path or config.OUTPUT_VIDEO_PATH
        
        # Initialize all modules
        logger.info("Loading detection module...")
        self.detector = BirdDetector()
        
        logger.info("Loading video processor...")
        self.video_processor = VideoProcessor(
            video_source=self.video_source,
            output_path=self.output_path
        )
        
        logger.info("Loading fuzzy logic risk assessment...")
        self.fuzzy_engine = FuzzyRiskAssessment()
        
        logger.info("Loading trajectory predictor...")
        self.trajectory_predictor = TrajectoryPredictor()
        
        logger.info("Loading collision calculator...")
        self.ttc_calculator = TimeToCollisionCalculator()
        
        logger.info("Loading multi-bird tracker...")
        try:
            self.tracker = BirdTracker()
        except Exception as e:
            logger.warning(f"DeepSORT initialization failed: {e}")
            logger.info("Falling back to Simple tracker")
            self.tracker = SimpleTracker()
        
        logger.info("Loading danger zone classifier...")
        self.danger_zones = DangerZoneClassifier()
        
        # FPS counter
        self.fps_counter = FPSCounter()
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'max_simultaneous_birds': 0,
            'high_risk_events': 0
        }
        
        logger.info("AvianGuard IRAPS initialized successfully")
        logger.info("=" * 80)
    
    def process_frame(self, frame: np.ndarray, frame_num: int) -> np.ndarray:
        """
        Process a single frame through the complete pipeline.
        
        Args:
            frame: Input frame
            frame_num: Frame number
            
        Returns:
            Annotated output frame
        """
        # Create output frame
        output_frame = frame.copy()
        
        # Step 1: Detect birds
        detections = self.detector.detect(frame)
        self.stats['total_detections'] += len(detections)
        
        # Step 2: Track birds across frames
        tracked_birds = self.tracker.update(detections, frame)
        
        # Update max simultaneous birds
        if len(tracked_birds) > self.stats['max_simultaneous_birds']:
            self.stats['max_simultaneous_birds'] = len(tracked_birds)
        
        # Step 3: Process each tracked bird
        risk_scores = []
        
        for bird in tracked_birds:
            track_id = bird.get('track_id', 0)
            bbox = bird['bbox']
            
            # Calculate center point
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Update trajectory predictor
            self.trajectory_predictor.update(track_id, (center_x, center_y), frame_num)
            
            # Update TTC calculator
            self.ttc_calculator.update_bbox(track_id, bbox, frame_num)
            
            # Get velocity and speed
            speed = self.trajectory_predictor.get_speed(track_id)
            
            # Classify danger zone
            zone_info = self.danger_zones.classify_bbox(bbox)
            
            # Calculate distance metric (inverse of bbox size)
            bbox_diagonal = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distance_metric = max(500 - bbox_diagonal, 0)  # Larger bbox = smaller distance
            
            # Assess risk using fuzzy logic ⭐
            risk_score, risk_explanation = self.fuzzy_engine.assess_risk(
                distance=distance_metric,
                bird_count=len(tracked_birds),
                velocity=speed,
                position=zone_info['danger_score']
            )
            
            risk_scores.append(risk_score)
            
            # Calculate time to collision ⭐
            ttc = self.ttc_calculator.calculate_ttc(track_id, method='optical')
            
            # Predict trajectory ⭐
            predictions = self.trajectory_predictor.predict_trajectory(
                track_id, 
                steps=config.TRAJECTORY_POINTS_TO_DRAW
            )
            
            # Visualize trajectory
            if config.SHOW_TRAJECTORY and len(predictions) > 0:
                # Draw trajectory line
                traj_points = [(int(px), int(py)) for px, py in predictions]
                for i in range(len(traj_points) - 1):
                    cv2.line(
                        output_frame,
                        traj_points[i],
                        traj_points[i + 1],
                        config.COLOR_TRAJECTORY,
                        2
                    )
            
            # Draw bounding box with risk-based color
            risk_color = get_risk_color(risk_score)
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), risk_color, 2)
            
            # Prepare label text
            label_lines = [
                f"ID:{track_id} Risk:{risk_score:.0f}%",
                f"Zone:{zone_info['zone_type'].upper()}",
            ]
            
            if ttc is not None and ttc < config.TTC_WARNING_THRESHOLD:
                label_lines.append(f"TTC:{ttc:.1f}s")
            
            # Draw label with background
            y_offset = y1 - 10
            for line in reversed(label_lines):
                draw_text_with_background(
                    output_frame,
                    line,
                    (x1, y_offset),
                    font_scale=0.5,
                    thickness=1,
                    text_color=(255, 255, 255),
                    bg_color=risk_color
                )
                y_offset -= 20
        
        # Step 4: Visualize danger zones (if enabled)
        if config.SHOW_DANGER_ZONES:
            output_frame = self.danger_zones.visualize_zones(
                output_frame, 
                alpha=0.15,
                show_labels=False
            )
        
        # Step 5: Draw overall system status
        self._draw_status_panel(output_frame, tracked_birds, risk_scores)
        
        # Update statistics
        self.stats['frames_processed'] += 1
        if len(risk_scores) > 0 and max(risk_scores) > config.RISK_HIGH:
            self.stats['high_risk_events'] += 1
        
        return output_frame
    
    def _draw_status_panel(self, frame: np.ndarray, 
                          birds: List[Dict], 
                          risk_scores: List[float]) -> None:
        """Draw status information panel on frame."""
        # Calculate overall risk
        if len(risk_scores) > 0:
            max_risk = max(risk_scores)
            avg_risk = np.mean(risk_scores)
        else:
            max_risk = 0
            avg_risk = 0
        
        # Get FPS
        fps = self.fps_counter.update()
        
        # Panel background
        panel_height = 120
        cv2.rectangle(
            frame,
            (10, 10),
            (400, 10 + panel_height),
            (0, 0, 0),
            -1
        )
        cv2.rectangle(
            frame,
            (10, 10),
            (400, 10 + panel_height),
            (255, 255, 255),
            2
        )
        
        # Status text
        status_lines = [
            f"AvianGuard IRAPS - Active",
            f"Birds Detected: {len(birds)}",
            f"Max Risk: {max_risk:.0f}% ({get_risk_level_name(max_risk).upper()})",
            f"Avg Risk: {avg_risk:.0f}%",
            f"FPS: {fps:.1f}"
        ]
        
        y_pos = 35
        for line in status_lines:
            color = (0, 255, 0) if max_risk < config.RISK_MEDIUM else \
                    (0, 255, 255) if max_risk < config.RISK_HIGH else (0, 0, 255)
            
            cv2.putText(
                frame,
                line,
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
            y_pos += 22
        
        # Risk meter
        meter_x = 420
        meter_y = 30
        meter_width = 200
        meter_height = 30
        
        # Background
        cv2.rectangle(
            frame,
            (meter_x, meter_y),
            (meter_x + meter_width, meter_y + meter_height),
            (50, 50, 50),
            -1
        )
        
        # Risk fill
        fill_width = int((max_risk / 100) * meter_width)
        risk_color = get_risk_color(max_risk)
        cv2.rectangle(
            frame,
            (meter_x, meter_y),
            (meter_x + fill_width, meter_y + meter_height),
            risk_color,
            -1
        )
        
        # Border
        cv2.rectangle(
            frame,
            (meter_x, meter_y),
            (meter_x + meter_width, meter_y + meter_height),
            (255, 255, 255),
            2
        )
        
        # Label
        cv2.putText(
            frame,
            "RISK METER",
            (meter_x, meter_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )
    
    def run(self, display: bool = True, save_output: bool = True) -> None:
        """
        Run the complete AvianGuard IRAPS pipeline.
        
        Args:
            display: Show output in window
            save_output: Save output video
        """
        logger.info("Starting AvianGuard IRAPS pipeline...")
        
        # Process video
        frames_processed = self.video_processor.process_video(
            frame_callback=self.process_frame,
            skip_frames=config.SKIP_FRAMES,
            save_output=save_output,
            display=display
        )
        
        # Log statistics
        logger.info("=" * 80)
        logger.info("Processing Complete - Statistics:")
        logger.info(f"  Frames Processed: {self.stats['frames_processed']}")
        logger.info(f"  Total Detections: {self.stats['total_detections']}")
        logger.info(f"  Max Simultaneous Birds: {self.stats['max_simultaneous_birds']}")
        logger.info(f"  High Risk Events: {self.stats['high_risk_events']}")
        logger.info("=" * 80)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AvianGuard IRAPS - Intelligent Bird Strike Prevention System"
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='Input video file or camera index (default: config.DEFAULT_VIDEO_PATH)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output video file path (default: config.OUTPUT_VIDEO_PATH)'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable video display window'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Disable output video saving'
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        default=None,
        help='Use camera with specified index'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Determine video source
    if args.camera is not None:
        video_source = str(args.camera)
    elif args.input:
        video_source = args.input
    else:
        video_source = None
    
    # Create output directory
    if args.output and not args.no_save:
        import os
        ensure_directory(os.path.dirname(args.output))
    
    try:
        # Initialize system
        avianguard = AvianGuardIRAPS(
            video_source=video_source,
            output_path=args.output
        )
        
        # Run pipeline
        avianguard.run(
            display=not args.no_display,
            save_output=not args.no_save
        )
        
        # Print final statistics
        stats = avianguard.get_statistics()
        print("\n" + "=" * 80)
        print("AvianGuard IRAPS - Session Complete")
        print("=" * 80)
        print(f"Total Frames: {stats['frames_processed']}")
        print(f"Total Birds Detected: {stats['total_detections']}")
        print(f"Maximum Birds Simultaneously: {stats['max_simultaneous_birds']}")
        print(f"High Risk Events: {stats['high_risk_events']}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
        print("\nSystem stopped by user")
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
