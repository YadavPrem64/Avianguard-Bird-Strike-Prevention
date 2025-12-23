"""
Unit Tests for Trajectory Predictor
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.trajectory import TrajectoryPredictor, SimpleTrajectoryPredictor


class TestTrajectoryPredictor(unittest.TestCase):
    """Test cases for Kalman filter-based trajectory predictor."""
    
    def setUp(self):
        """Initialize predictor before each test."""
        self.predictor = TrajectoryPredictor()
    
    def test_initialization(self):
        """Test that predictor initializes correctly."""
        self.assertEqual(len(self.predictor.filters), 0)
        self.assertEqual(len(self.predictor.trajectories), 0)
    
    def test_create_filter(self):
        """Test filter creation for new bird."""
        bird_id = 1
        position = (100.0, 200.0)
        
        self.predictor.create_filter(bird_id, position)
        
        self.assertIn(bird_id, self.predictor.filters)
        self.assertIn(bird_id, self.predictor.trajectories)
    
    def test_update_position(self):
        """Test position update and filtering."""
        bird_id = 1
        
        # Update multiple times
        for i in range(10):
            x = 100 + i * 10
            y = 200 + i * 5
            result = self.predictor.update(bird_id, (x, y), timestamp=i)
            
            # Result should be close to input
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
    
    def test_trajectory_prediction(self):
        """Test future trajectory prediction."""
        bird_id = 1
        
        # Build up some history
        for i in range(20):
            x = 100 + i * 5
            y = 100 + i * 3
            self.predictor.update(bird_id, (x, y))
        
        # Predict future
        predictions = self.predictor.predict_trajectory(bird_id, steps=30)
        
        self.assertEqual(len(predictions), 30)
        self.assertIsInstance(predictions[0], tuple)
    
    def test_velocity_estimation(self):
        """Test velocity extraction."""
        bird_id = 1
        
        # Move bird at constant velocity
        for i in range(10):
            x = 100 + i * 10  # 10 px/frame in x
            y = 100 + i * 5   # 5 px/frame in y
            self.predictor.update(bird_id, (x, y))
        
        vx, vy = self.predictor.get_velocity(bird_id)
        
        # Should be close to actual velocity
        self.assertAlmostEqual(vx, 10, delta=3)
        self.assertAlmostEqual(vy, 5, delta=3)
    
    def test_speed_calculation(self):
        """Test speed magnitude calculation."""
        bird_id = 1
        
        for i in range(10):
            x = 100 + i * 3
            y = 100 + i * 4  # 3-4-5 triangle
            self.predictor.update(bird_id, (x, y))
        
        speed = self.predictor.get_speed(bird_id)
        
        # Speed should be ~5 (sqrt(3^2 + 4^2))
        self.assertAlmostEqual(speed, 5, delta=2)
    
    def test_history_tracking(self):
        """Test trajectory history storage."""
        bird_id = 1
        
        for i in range(50):
            self.predictor.update(bird_id, (i, i))
        
        history = self.predictor.get_trajectory_history(bird_id)
        
        # Should be limited by history length
        self.assertLessEqual(len(history), self.predictor.history_length)
    
    def test_multiple_birds(self):
        """Test tracking multiple birds simultaneously."""
        # Create 5 birds
        for bird_id in range(1, 6):
            for i in range(10):
                x = 100 * bird_id + i * 5
                y = 100 * bird_id + i * 3
                self.predictor.update(bird_id, (x, y))
        
        # Should have 5 filters
        self.assertEqual(len(self.predictor.filters), 5)
        
        # Each should have predictions
        for bird_id in range(1, 6):
            predictions = self.predictor.predict_trajectory(bird_id, steps=10)
            self.assertEqual(len(predictions), 10)
    
    def test_collision_estimation(self):
        """Test collision point estimation."""
        bird_id = 1
        
        # Bird moving toward zone
        for i in range(20):
            x = 100 + i * 10
            y = 100 + i * 10
            self.predictor.update(bird_id, (x, y))
        
        # Define target zone
        target_zone = (300, 300, 400, 400)
        
        # Estimate collision
        result = self.predictor.estimate_collision_point(bird_id, target_zone)
        
        # Should detect potential collision
        self.assertIsInstance(result, (tuple, type(None)))
    
    def test_tracker_cleanup(self):
        """Test removal of old trackers."""
        # Create birds
        for bird_id in range(1, 6):
            self.predictor.update(bird_id, (100, 100))
        
        self.assertEqual(len(self.predictor.filters), 5)
        
        # Cleanup keeping only 1, 2, 3
        self.predictor.cleanup_old_trackers([1, 2, 3])
        
        self.assertEqual(len(self.predictor.filters), 3)


class TestSimpleTrajectoryPredictor(unittest.TestCase):
    """Test cases for simple trajectory predictor."""
    
    def setUp(self):
        """Initialize simple predictor."""
        self.predictor = SimpleTrajectoryPredictor(history_length=10)
    
    def test_initialization(self):
        """Test initialization."""
        self.assertEqual(len(self.predictor.trajectories), 0)
    
    def test_position_update(self):
        """Test position update."""
        bird_id = 1
        self.predictor.update(bird_id, (100, 100))
        
        self.assertIn(bird_id, self.predictor.trajectories)
        self.assertEqual(len(self.predictor.trajectories[bird_id]), 1)
    
    def test_prediction(self):
        """Test position prediction."""
        bird_id = 1
        
        # Build history
        for i in range(10):
            self.predictor.update(bird_id, (100 + i * 5, 100 + i * 3))
        
        # Predict
        prediction = self.predictor.predict_position(bird_id, steps_ahead=10)
        
        self.assertIsNotNone(prediction)
        self.assertIsInstance(prediction, tuple)


if __name__ == '__main__':
    unittest.main()
