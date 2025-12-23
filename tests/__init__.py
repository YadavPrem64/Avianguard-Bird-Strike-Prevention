"""
AvianGuard IRAPS - Test Suite
Unit tests for core modules.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

__all__ = ['test_fuzzy_logic', 'test_trajectory', 'test_collision_calc', 'test_danger_zones']
