"""
Unit Tests for Fuzzy Logic Risk Assessment Engine
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.fuzzy_logic import FuzzyRiskAssessment


class TestFuzzyLogic(unittest.TestCase):
    """Test cases for fuzzy logic risk assessment."""
    
    def setUp(self):
        """Initialize fuzzy engine before each test."""
        self.fuzzy = FuzzyRiskAssessment()
    
    def test_initialization(self):
        """Test that fuzzy system initializes correctly."""
        self.assertIsNotNone(self.fuzzy.distance)
        self.assertIsNotNone(self.fuzzy.bird_count)
        self.assertIsNotNone(self.fuzzy.velocity)
        self.assertIsNotNone(self.fuzzy.position)
        self.assertIsNotNone(self.fuzzy.risk)
        self.assertIsNotNone(self.fuzzy.risk_simulation)
    
    def test_extreme_risk_scenario(self):
        """Test extreme risk: close distance, many birds, fast, dangerous position."""
        risk, explanation = self.fuzzy.assess_risk(
            distance=10,      # Critical distance
            bird_count=20,    # Swarm
            velocity=30,      # Fast
            position=0.9      # Danger zone
        )
        
        self.assertGreater(risk, 70, "Extreme scenario should produce high risk")
        self.assertEqual(explanation['risk_level'], 'extreme')
    
    def test_low_risk_scenario(self):
        """Test low risk: far distance, single bird, slow, safe position."""
        risk, explanation = self.fuzzy.assess_risk(
            distance=400,     # Far
            bird_count=1,     # Single
            velocity=2,       # Slow
            position=0.1      # Safe zone
        )
        
        self.assertLess(risk, 30, "Safe scenario should produce low risk")
        self.assertIn(explanation['risk_level'], ['negligible', 'low'])
    
    def test_medium_risk_scenario(self):
        """Test medium risk: medium distance, few birds, medium velocity."""
        risk, explanation = self.fuzzy.assess_risk(
            distance=100,     # Medium
            bird_count=5,     # Few/Flock
            velocity=10,      # Medium
            position=0.5      # Caution zone
        )
        
        self.assertGreater(risk, 20, "Medium scenario should be above negligible")
        self.assertLess(risk, 80, "Medium scenario should be below extreme")
    
    def test_input_validation(self):
        """Test that inputs are properly constrained."""
        # Test with out-of-range inputs
        risk, _ = self.fuzzy.assess_risk(
            distance=1000,    # Above max
            bird_count=200,   # Above max
            velocity=200,     # Above max
            position=2.0      # Above max
        )
        
        # Should still return valid risk score
        self.assertGreaterEqual(risk, 0)
        self.assertLessEqual(risk, 100)
    
    def test_bird_count_effect(self):
        """Test that higher bird count increases risk."""
        risk_single, _ = self.fuzzy.assess_risk(
            distance=100, bird_count=1, velocity=10, position=0.5
        )
        
        risk_swarm, _ = self.fuzzy.assess_risk(
            distance=100, bird_count=25, velocity=10, position=0.5
        )
        
        self.assertGreater(risk_swarm, risk_single, 
                          "More birds should increase risk")
    
    def test_distance_effect(self):
        """Test that closer distance increases risk."""
        risk_far, _ = self.fuzzy.assess_risk(
            distance=300, bird_count=5, velocity=10, position=0.5
        )
        
        risk_close, _ = self.fuzzy.assess_risk(
            distance=30, bird_count=5, velocity=10, position=0.5
        )
        
        self.assertGreater(risk_close, risk_far,
                          "Closer birds should increase risk")
    
    def test_velocity_effect(self):
        """Test that higher velocity increases risk."""
        risk_slow, _ = self.fuzzy.assess_risk(
            distance=100, bird_count=5, velocity=2, position=0.5
        )
        
        risk_fast, _ = self.fuzzy.assess_risk(
            distance=100, bird_count=5, velocity=40, position=0.5
        )
        
        self.assertGreater(risk_fast, risk_slow,
                          "Faster approach should increase risk")
    
    def test_position_effect(self):
        """Test that dangerous position increases risk."""
        risk_safe, _ = self.fuzzy.assess_risk(
            distance=100, bird_count=5, velocity=10, position=0.1
        )
        
        risk_danger, _ = self.fuzzy.assess_risk(
            distance=100, bird_count=5, velocity=10, position=0.9
        )
        
        self.assertGreater(risk_danger, risk_safe,
                          "Dangerous position should increase risk")
    
    def test_explanation_structure(self):
        """Test that explanation dictionary has required fields."""
        _, explanation = self.fuzzy.assess_risk(
            distance=100, bird_count=5, velocity=10, position=0.5
        )
        
        self.assertIn('inputs', explanation)
        self.assertIn('risk_score', explanation)
        self.assertIn('risk_level', explanation)
        self.assertIn('contributing_factors', explanation)
        self.assertIsInstance(explanation['contributing_factors'], list)
    
    def test_consistency(self):
        """Test that same inputs produce same outputs."""
        risk1, _ = self.fuzzy.assess_risk(
            distance=150, bird_count=7, velocity=15, position=0.6
        )
        
        risk2, _ = self.fuzzy.assess_risk(
            distance=150, bird_count=7, velocity=15, position=0.6
        )
        
        self.assertAlmostEqual(risk1, risk2, places=1,
                              msg="Same inputs should produce consistent outputs")


if __name__ == '__main__':
    unittest.main()
