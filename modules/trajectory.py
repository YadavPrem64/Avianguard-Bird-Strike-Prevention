"""
AvianGuard IRAPS - Trajectory Prediction Module ⭐
CORE ORIGINAL WORK - Kalman Filter-based bird trajectory prediction.

Predicts future bird positions to assess collision paths and provide
early warnings for evasive action.
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from typing import List, Tuple, Optional, Dict
import utils.config as config
from utils.helpers import setup_logging

logger = setup_logging()


class TrajectoryPredictor:
    """
    ⭐ CUSTOM ALGORITHM - Kalman Filter-based Trajectory Prediction
    
    Implements trajectory prediction for birds using a Kalman filter with
    constant velocity model. Predicts future positions to assess collision risk.
    """
    
    def __init__(self):
        """Initialize trajectory predictor."""
        logger.info("Initializing Trajectory Predictor")
        
        # Dictionary to store filters for each tracked bird
        self.filters = {}
        
        # History storage for visualization
        self.trajectories = {}
        
        # Prediction parameters
        self.dt = config.DT
        self.prediction_horizon = config.PREDICTION_HORIZON
        self.history_length = config.TRAJECTORY_HISTORY_LENGTH
    
    def create_filter(self, bird_id: int, initial_position: Tuple[float, float]) -> None:
        """
        Create a new Kalman filter for a bird.
        
        Args:
            bird_id: Unique bird tracking ID
            initial_position: Initial (x, y) position
        """
        # Create Kalman filter with 4 state variables: [x, y, vx, vy]
        kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (constant velocity model)
        # x_new = x + vx*dt
        # y_new = y + vy*dt
        # vx_new = vx
        # vy_new = vy
        kf.F = np.array([
            [1., 0., self.dt, 0.],
            [0., 1., 0., self.dt],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ])
        
        # Measurement matrix (we only measure position)
        kf.H = np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.]
        ])
        
        # Measurement noise covariance
        kf.R = np.eye(2) * config.MEASUREMENT_NOISE
        
        # Process noise covariance
        kf.Q = np.eye(4) * config.PROCESS_NOISE
        
        # Initial state covariance
        kf.P = np.eye(4) * 100
        
        # Initial state: position from detection, zero velocity
        x, y = initial_position
        kf.x = np.array([x, y, 0., 0.])
        
        # Store filter
        self.filters[bird_id] = kf
        
        # Initialize trajectory history
        self.trajectories[bird_id] = {
            'history': [(x, y)],
            'predictions': [],
            'timestamps': [0]
        }
        
        logger.debug(f"Created Kalman filter for bird {bird_id} at position ({x:.1f}, {y:.1f})")
    
    def update(self, bird_id: int, position: Tuple[float, float], 
               timestamp: Optional[int] = None) -> Tuple[float, float]:
        """
        Update filter with new measurement and predict current position.
        
        Args:
            bird_id: Unique bird tracking ID
            position: Measured (x, y) position
            timestamp: Frame timestamp (optional)
            
        Returns:
            Predicted (x, y) position
        """
        # Create filter if it doesn't exist
        if bird_id not in self.filters:
            self.create_filter(bird_id, position)
            return position
        
        kf = self.filters[bird_id]
        
        # Predict step
        kf.predict()
        
        # Update step with measurement
        measurement = np.array([position[0], position[1]])
        kf.update(measurement)
        
        # Get predicted position
        predicted_x = kf.x[0]
        predicted_y = kf.x[1]
        
        # Update trajectory history
        trajectory = self.trajectories[bird_id]
        trajectory['history'].append((predicted_x, predicted_y))
        if timestamp is not None:
            trajectory['timestamps'].append(timestamp)
        
        # Keep history limited
        if len(trajectory['history']) > self.history_length:
            trajectory['history'].pop(0)
            if len(trajectory['timestamps']) > 0:
                trajectory['timestamps'].pop(0)
        
        logger.debug(
            f"Bird {bird_id}: Measured ({position[0]:.1f}, {position[1]:.1f}), "
            f"Predicted ({predicted_x:.1f}, {predicted_y:.1f})"
        )
        
        return predicted_x, predicted_y
    
    def predict_trajectory(self, bird_id: int, 
                          steps: Optional[int] = None) -> List[Tuple[float, float]]:
        """
        ⭐ CUSTOM ALGORITHM - Predict future trajectory.
        
        Args:
            bird_id: Unique bird tracking ID
            steps: Number of time steps to predict (default: prediction_horizon)
            
        Returns:
            List of predicted (x, y) positions
        """
        if bird_id not in self.filters:
            logger.warning(f"No filter exists for bird {bird_id}")
            return []
        
        if steps is None:
            steps = self.prediction_horizon
        
        kf = self.filters[bird_id]
        
        # Make a copy of the current state to avoid modifying the filter
        x = kf.x.copy()
        F = kf.F.copy()
        
        # Predict future positions
        predictions = []
        for i in range(steps):
            # Propagate state forward
            x = F @ x
            predictions.append((x[0], x[1]))
        
        # Store predictions for visualization
        self.trajectories[bird_id]['predictions'] = predictions
        
        logger.debug(f"Predicted {len(predictions)} future positions for bird {bird_id}")
        
        return predictions
    
    def get_velocity(self, bird_id: int) -> Tuple[float, float]:
        """
        Get current velocity estimate for a bird.
        
        Args:
            bird_id: Unique bird tracking ID
            
        Returns:
            Velocity (vx, vy) in pixels per frame
        """
        if bird_id not in self.filters:
            return (0.0, 0.0)
        
        kf = self.filters[bird_id]
        return (kf.x[2], kf.x[3])
    
    def get_speed(self, bird_id: int) -> float:
        """
        Get current speed (magnitude of velocity) for a bird.
        
        Args:
            bird_id: Unique bird tracking ID
            
        Returns:
            Speed in pixels per frame
        """
        vx, vy = self.get_velocity(bird_id)
        return np.sqrt(vx**2 + vy**2)
    
    def get_trajectory_history(self, bird_id: int) -> List[Tuple[float, float]]:
        """
        Get historical trajectory for a bird.
        
        Args:
            bird_id: Unique bird tracking ID
            
        Returns:
            List of past (x, y) positions
        """
        if bird_id not in self.trajectories:
            return []
        
        return self.trajectories[bird_id]['history']
    
    def get_predictions(self, bird_id: int) -> List[Tuple[float, float]]:
        """
        Get predicted future trajectory for a bird.
        
        Args:
            bird_id: Unique bird tracking ID
            
        Returns:
            List of predicted (x, y) positions
        """
        if bird_id not in self.trajectories:
            return []
        
        return self.trajectories[bird_id]['predictions']
    
    def estimate_collision_point(self, bird_id: int,
                                 target_zone: Tuple[int, int, int, int],
                                 max_steps: int = 100) -> Optional[Tuple[int, Tuple[float, float]]]:
        """
        ⭐ CUSTOM ALGORITHM - Estimate if and when bird will collide with a zone.
        
        Args:
            bird_id: Unique bird tracking ID
            target_zone: Zone boundaries (x1, y1, x2, y2)
            max_steps: Maximum steps to check ahead
            
        Returns:
            Tuple of (step_number, collision_point) or None if no collision
        """
        predictions = self.predict_trajectory(bird_id, steps=max_steps)
        
        x1, y1, x2, y2 = target_zone
        
        for step, (px, py) in enumerate(predictions):
            if x1 <= px <= x2 and y1 <= py <= y2:
                logger.info(
                    f"Bird {bird_id} predicted to collide with zone at step {step}"
                )
                return (step, (px, py))
        
        return None
    
    def remove_tracker(self, bird_id: int) -> None:
        """
        Remove filter and trajectory for a bird that's no longer tracked.
        
        Args:
            bird_id: Unique bird tracking ID
        """
        if bird_id in self.filters:
            del self.filters[bird_id]
        
        if bird_id in self.trajectories:
            del self.trajectories[bird_id]
        
        logger.debug(f"Removed tracker for bird {bird_id}")
    
    def get_all_trajectories(self) -> Dict[int, Dict]:
        """
        Get all bird trajectories for visualization.
        
        Returns:
            Dictionary mapping bird_id to trajectory data
        """
        return self.trajectories
    
    def cleanup_old_trackers(self, active_bird_ids: List[int]) -> None:
        """
        Remove trackers for birds that are no longer active.
        
        Args:
            active_bird_ids: List of currently active bird IDs
        """
        inactive_ids = set(self.filters.keys()) - set(active_bird_ids)
        
        for bird_id in inactive_ids:
            self.remove_tracker(bird_id)
        
        if len(inactive_ids) > 0:
            logger.debug(f"Cleaned up {len(inactive_ids)} inactive trackers")


class SimpleTrajectoryPredictor:
    """
    Simplified trajectory predictor using linear extrapolation.
    Useful as a fallback when Kalman filter is not needed.
    """
    
    def __init__(self, history_length: int = 10):
        """
        Initialize simple predictor.
        
        Args:
            history_length: Number of past positions to keep
        """
        self.history_length = history_length
        self.trajectories = {}
    
    def update(self, bird_id: int, position: Tuple[float, float]) -> None:
        """Update position history for a bird."""
        if bird_id not in self.trajectories:
            self.trajectories[bird_id] = []
        
        self.trajectories[bird_id].append(position)
        
        # Keep history limited
        if len(self.trajectories[bird_id]) > self.history_length:
            self.trajectories[bird_id].pop(0)
    
    def predict_position(self, bird_id: int, 
                        steps_ahead: int = 30) -> Optional[Tuple[float, float]]:
        """
        Predict future position using linear extrapolation.
        
        Args:
            bird_id: Unique bird tracking ID
            steps_ahead: Number of frames to predict ahead
            
        Returns:
            Predicted (x, y) position or None
        """
        if bird_id not in self.trajectories or len(self.trajectories[bird_id]) < 2:
            return None
        
        history = self.trajectories[bird_id]
        
        # Calculate average velocity from recent history
        velocities_x = []
        velocities_y = []
        
        for i in range(1, len(history)):
            vx = history[i][0] - history[i-1][0]
            vy = history[i][1] - history[i-1][1]
            velocities_x.append(vx)
            velocities_y.append(vy)
        
        avg_vx = np.mean(velocities_x)
        avg_vy = np.mean(velocities_y)
        
        # Extrapolate
        current_x, current_y = history[-1]
        predicted_x = current_x + avg_vx * steps_ahead
        predicted_y = current_y + avg_vy * steps_ahead
        
        return (predicted_x, predicted_y)


if __name__ == "__main__":
    # Test the trajectory predictor
    print("Testing Trajectory Predictor...")
    
    predictor = TrajectoryPredictor()
    
    # Simulate a bird moving in a straight line
    bird_id = 1
    for t in range(20):
        # Bird moving diagonally
        x = 100 + t * 5
        y = 100 + t * 3
        
        current_pos = predictor.update(bird_id, (x, y), timestamp=t)
        print(f"Frame {t}: Measured ({x}, {y}), Filtered {current_pos}")
    
    # Predict future trajectory
    future = predictor.predict_trajectory(bird_id, steps=30)
    print(f"\nPredicted next 30 positions:")
    for i, (px, py) in enumerate(future[:10]):  # Show first 10
        print(f"  Step {i+1}: ({px:.1f}, {py:.1f})")
    
    # Get velocity
    vx, vy = predictor.get_velocity(bird_id)
    speed = predictor.get_speed(bird_id)
    print(f"\nEstimated velocity: ({vx:.2f}, {vy:.2f}), Speed: {speed:.2f} px/frame")
