"""
AvianGuard IRAPS - Video Processor Module
Handles video input/output operations and frame processing pipeline.
"""

import cv2
import numpy as np
import os
from typing import Optional, Callable, Generator, Tuple
import utils.config as config
from utils.helpers import setup_logging, ensure_directory

logger = setup_logging()


class VideoProcessor:
    """
    Handles video input/output operations.
    Supports multiple video sources (file, webcam, stream).
    """
    
    def __init__(self, video_source: Optional[str] = None,
                 output_path: Optional[str] = None,
                 resize_output: bool = True):
        """
        Initialize video processor.
        
        Args:
            video_source: Path to video file, camera index, or stream URL
            output_path: Path for output video (optional)
            resize_output: Whether to resize frames to standard dimensions
        """
        self.video_source = video_source or config.DEFAULT_VIDEO_PATH
        self.output_path = output_path
        self.resize_output = resize_output
        
        self.capture = None
        self.writer = None
        self.frame_count = 0
        self.fps = config.FPS
        self.frame_width = config.FRAME_WIDTH
        self.frame_height = config.FRAME_HEIGHT
        
        logger.info(f"VideoProcessor initialized with source: {self.video_source}")
    
    def open_video(self) -> bool:
        """
        Open video source for reading.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to convert to integer (for camera index)
            source = int(self.video_source)
        except ValueError:
            # It's a file path or URL
            source = self.video_source
        
        self.capture = cv2.VideoCapture(source)
        
        if not self.capture.isOpened():
            logger.error(f"Failed to open video source: {self.video_source}")
            return False
        
        # Get video properties
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS)) or config.FPS
        original_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if not self.resize_output:
            self.frame_width = original_width
            self.frame_height = original_height
        
        logger.info(f"Video opened: {original_width}x{original_height} @ {self.fps} FPS")
        logger.info(f"Total frames: {total_frames}")
        logger.info(f"Output will be: {self.frame_width}x{self.frame_height}")
        
        return True
    
    def create_writer(self, output_path: Optional[str] = None) -> bool:
        """
        Create video writer for saving output.
        
        Args:
            output_path: Output video path (overrides constructor parameter)
            
        Returns:
            True if successful, False otherwise
        """
        if output_path:
            self.output_path = output_path
        
        if not self.output_path:
            logger.warning("No output path specified")
            return False
        
        # Ensure output directory exists
        ensure_directory(os.path.dirname(self.output_path))
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )
        
        if not self.writer.isOpened():
            logger.error(f"Failed to create video writer: {self.output_path}")
            return False
        
        logger.info(f"Video writer created: {self.output_path}")
        return True
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame from video source.
        
        Returns:
            Tuple of (success, frame)
        """
        if self.capture is None or not self.capture.isOpened():
            return False, None
        
        ret, frame = self.capture.read()
        
        if not ret:
            return False, None
        
        # Resize if needed
        if self.resize_output and (frame.shape[1] != self.frame_width or 
                                    frame.shape[0] != self.frame_height):
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        self.frame_count += 1
        return True, frame
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Write frame to output video.
        
        Args:
            frame: Frame to write
            
        Returns:
            True if successful, False otherwise
        """
        if self.writer is None or not self.writer.isOpened():
            logger.warning("Video writer not initialized")
            return False
        
        try:
            # Ensure frame has correct dimensions
            if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            
            self.writer.write(frame)
            return True
        except Exception as e:
            logger.error(f"Error writing frame: {e}")
            return False
    
    def frame_generator(self, skip_frames: int = 1) -> Generator[np.ndarray, None, None]:
        """
        Generate frames from video source.
        
        Args:
            skip_frames: Process every Nth frame (1 = process all)
            
        Yields:
            Video frames
        """
        frame_idx = 0
        
        while True:
            ret, frame = self.read_frame()
            
            if not ret:
                break
            
            # Skip frames if configured
            if frame_idx % skip_frames == 0:
                yield frame
            
            frame_idx += 1
    
    def process_video(self, 
                     frame_callback: Callable[[np.ndarray, int], np.ndarray],
                     skip_frames: int = config.SKIP_FRAMES,
                     save_output: bool = True,
                     display: bool = False) -> int:
        """
        Process video with a callback function.
        
        Args:
            frame_callback: Function that processes each frame
                           Takes (frame, frame_number) and returns processed frame
            skip_frames: Process every Nth frame
            save_output: Save processed video to output path
            display: Display frames in a window
            
        Returns:
            Number of frames processed
        """
        if not self.open_video():
            return 0
        
        if save_output and self.output_path:
            self.create_writer()
        
        frames_processed = 0
        frame_idx = 0
        
        try:
            logger.info("Starting video processing...")
            
            for frame in self.frame_generator(skip_frames):
                # Process frame with callback
                processed_frame = frame_callback(frame, frame_idx)
                
                # Write to output
                if save_output and self.writer:
                    self.write_frame(processed_frame)
                
                # Display if requested
                if display:
                    cv2.imshow('AvianGuard - Processing', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Processing interrupted by user")
                        break
                
                frames_processed += 1
                frame_idx += skip_frames
                
                # Log progress
                if frames_processed % 100 == 0:
                    logger.info(f"Processed {frames_processed} frames")
            
            logger.info(f"Video processing complete. Processed {frames_processed} frames")
            
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        except Exception as e:
            logger.error(f"Error during video processing: {e}")
        finally:
            self.release()
            if display:
                cv2.destroyAllWindows()
        
        return frames_processed
    
    def get_video_properties(self) -> dict:
        """
        Get properties of current video source.
        
        Returns:
            Dictionary of video properties
        """
        if self.capture is None or not self.capture.isOpened():
            return {}
        
        return {
            'fps': self.fps,
            'width': self.frame_width,
            'height': self.frame_height,
            'frame_count': int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            'current_frame': self.frame_count
        }
    
    def release(self) -> None:
        """Release video capture and writer resources."""
        if self.capture is not None:
            self.capture.release()
            logger.info("Video capture released")
        
        if self.writer is not None:
            self.writer.release()
            logger.info("Video writer released")


class WebcamProcessor(VideoProcessor):
    """Specialized processor for webcam input."""
    
    def __init__(self, camera_index: int = 0, output_path: Optional[str] = None):
        """
        Initialize webcam processor.
        
        Args:
            camera_index: Camera device index (0 for default)
            output_path: Path for output video (optional)
        """
        super().__init__(video_source=str(camera_index), output_path=output_path)
        self.camera_index = camera_index


class StreamProcessor(VideoProcessor):
    """Specialized processor for video streams (RTSP, HTTP, etc.)."""
    
    def __init__(self, stream_url: str, output_path: Optional[str] = None):
        """
        Initialize stream processor.
        
        Args:
            stream_url: Stream URL (RTSP, HTTP, etc.)
            output_path: Path for output video (optional)
        """
        super().__init__(video_source=stream_url, output_path=output_path)
        self.stream_url = stream_url


if __name__ == "__main__":
    # Simple test
    import sys
    
    def test_callback(frame, frame_num):
        """Test callback that adds frame number to frame."""
        cv2.putText(
            frame,
            f"Frame: {frame_num}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        return frame
    
    # Test with a dummy video file or webcam
    processor = VideoProcessor(video_source="0")  # Use webcam
    print("VideoProcessor initialized")
    
    if processor.open_video():
        print("Video opened successfully")
        ret, frame = processor.read_frame()
        if ret:
            print(f"Read test frame: {frame.shape}")
        processor.release()
    else:
        print("Could not open video source")
