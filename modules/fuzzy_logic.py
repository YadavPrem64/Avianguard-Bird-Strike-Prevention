"""
AvianGuard IRAPS - Fuzzy Logic Risk Assessment Engine ⭐
CORE ORIGINAL WORK - Custom fuzzy logic system for bird strike risk assessment.

This module implements a fuzzy control system that assesses collision risk based on:
- Distance (derived from bounding box size)
- Bird count
- Screen position (proximity to danger zones)
- Velocity (approach speed)
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from typing import Dict, Any, Tuple, Optional
import utils.config as config
from utils.helpers import setup_logging

logger = setup_logging()


class FuzzyRiskAssessment:
    """
    ⭐ CUSTOM ALGORITHM - Fuzzy Logic Risk Assessment Engine
    
    Implements a comprehensive fuzzy control system for assessing bird strike risk
    using multiple input parameters and a rule-based inference system.
    """
    
    def __init__(self):
        """Initialize the fuzzy risk assessment system."""
        logger.info("Initializing Fuzzy Logic Risk Assessment Engine")
        
        # Create fuzzy variables
        self.distance = None
        self.bird_count = None
        self.velocity = None
        self.position = None
        self.risk = None
        
        # Create control system
        self.risk_ctrl = None
        self.risk_simulation = None
        
        # Build the fuzzy system
        self._create_fuzzy_variables()
        self._create_fuzzy_rules()
        self._create_control_system()
        
        logger.info("Fuzzy system initialized successfully")
    
    def _create_fuzzy_variables(self) -> None:
        """Create fuzzy input and output variables with membership functions."""
        
        # ========== INPUT VARIABLE 1: DISTANCE ==========
        # Distance is inversely related to bounding box size
        # Larger bbox = closer bird = smaller distance value
        self.distance = ctrl.Antecedent(np.arange(0, 501, 1), 'distance')
        
        # Membership functions for distance
        self.distance['critical'] = fuzz.trapmf(
            self.distance.universe,
            [0, 0, 20, 40]
        )
        self.distance['close'] = fuzz.trimf(
            self.distance.universe,
            [20, 50, 100]
        )
        self.distance['medium'] = fuzz.trimf(
            self.distance.universe,
            [70, 125, 180]
        )
        self.distance['far'] = fuzz.trapmf(
            self.distance.universe,
            [150, 250, 500, 500]
        )
        
        # ========== INPUT VARIABLE 2: BIRD COUNT ==========
        self.bird_count = ctrl.Antecedent(np.arange(0, 101, 1), 'bird_count')
        
        # Membership functions for bird count
        self.bird_count['single'] = fuzz.trapmf(
            self.bird_count.universe,
            [0, 0, 1, 3]
        )
        self.bird_count['few'] = fuzz.trimf(
            self.bird_count.universe,
            [2, 4, 8]
        )
        self.bird_count['flock'] = fuzz.trimf(
            self.bird_count.universe,
            [6, 10, 18]
        )
        self.bird_count['swarm'] = fuzz.trapmf(
            self.bird_count.universe,
            [15, 25, 100, 100]
        )
        
        # ========== INPUT VARIABLE 3: VELOCITY ==========
        # Velocity in pixels per frame (bounding box growth rate)
        self.velocity = ctrl.Antecedent(np.arange(0, 101, 1), 'velocity')
        
        # Membership functions for velocity
        self.velocity['slow'] = fuzz.trapmf(
            self.velocity.universe,
            [0, 0, 3, 8]
        )
        self.velocity['medium'] = fuzz.trimf(
            self.velocity.universe,
            [5, 12, 20]
        )
        self.velocity['fast'] = fuzz.trapmf(
            self.velocity.universe,
            [15, 30, 100, 100]
        )
        
        # ========== INPUT VARIABLE 4: POSITION ==========
        # Normalized position value (0-1), where higher = more dangerous
        self.position = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'position')
        
        # Membership functions for screen position
        self.position['safe'] = fuzz.trapmf(
            self.position.universe,
            [0, 0, 0.2, 0.4]
        )
        self.position['caution'] = fuzz.trimf(
            self.position.universe,
            [0.3, 0.5, 0.7]
        )
        self.position['danger'] = fuzz.trapmf(
            self.position.universe,
            [0.6, 0.8, 1.0, 1.0]
        )
        
        # ========== OUTPUT VARIABLE: RISK SCORE ==========
        self.risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')
        
        # Membership functions for risk output
        self.risk['negligible'] = fuzz.trapmf(
            self.risk.universe,
            [0, 0, 10, 25]
        )
        self.risk['low'] = fuzz.trimf(
            self.risk.universe,
            [15, 30, 45]
        )
        self.risk['moderate'] = fuzz.trimf(
            self.risk.universe,
            [35, 50, 65]
        )
        self.risk['high'] = fuzz.trimf(
            self.risk.universe,
            [55, 70, 85]
        )
        self.risk['extreme'] = fuzz.trapmf(
            self.risk.universe,
            [75, 90, 100, 100]
        )
    
    def _create_fuzzy_rules(self) -> None:
        """
        ⭐ CUSTOM ALGORITHM - Define fuzzy inference rules.
        
        These rules encode expert knowledge about bird strike risk assessment.
        """
        
        self.rules = [
            # ===== EXTREME RISK RULES =====
            # Critical distance + swarm + fast = extreme danger
            ctrl.Rule(
                self.distance['critical'] & self.bird_count['swarm'] & 
                self.velocity['fast'],
                self.risk['extreme']
            ),
            
            # Critical distance + flock + danger position = extreme
            ctrl.Rule(
                self.distance['critical'] & self.bird_count['flock'] & 
                self.position['danger'],
                self.risk['extreme']
            ),
            
            # Close + swarm + danger position = extreme
            ctrl.Rule(
                self.distance['close'] & self.bird_count['swarm'] & 
                self.position['danger'],
                self.risk['extreme']
            ),
            
            # ===== HIGH RISK RULES =====
            # Close + fast + danger = high risk
            ctrl.Rule(
                self.distance['close'] & self.velocity['fast'] & 
                self.position['danger'],
                self.risk['high']
            ),
            
            # Critical distance + few birds = high risk
            ctrl.Rule(
                self.distance['critical'] & self.bird_count['few'],
                self.risk['high']
            ),
            
            # Close + flock = high risk
            ctrl.Rule(
                self.distance['close'] & self.bird_count['flock'],
                self.risk['high']
            ),
            
            # Medium distance + swarm + fast = high risk
            ctrl.Rule(
                self.distance['medium'] & self.bird_count['swarm'] & 
                self.velocity['fast'],
                self.risk['high']
            ),
            
            # Close + medium velocity + caution position = high
            ctrl.Rule(
                self.distance['close'] & self.velocity['medium'] & 
                self.position['caution'],
                self.risk['high']
            ),
            
            # ===== MODERATE RISK RULES =====
            # Medium distance + flock = moderate
            ctrl.Rule(
                self.distance['medium'] & self.bird_count['flock'],
                self.risk['moderate']
            ),
            
            # Medium distance + fast velocity = moderate
            ctrl.Rule(
                self.distance['medium'] & self.velocity['fast'],
                self.risk['moderate']
            ),
            
            # Close + single bird + safe position = moderate
            ctrl.Rule(
                self.distance['close'] & self.bird_count['single'] & 
                self.position['safe'],
                self.risk['moderate']
            ),
            
            # Medium + medium velocity + danger position = moderate
            ctrl.Rule(
                self.distance['medium'] & self.velocity['medium'] & 
                self.position['danger'],
                self.risk['moderate']
            ),
            
            # Close + slow velocity + safe position = moderate
            ctrl.Rule(
                self.distance['close'] & self.velocity['slow'] & 
                self.position['safe'],
                self.risk['moderate']
            ),
            
            # ===== LOW RISK RULES =====
            # Far + single = low
            ctrl.Rule(
                self.distance['far'] & self.bird_count['single'],
                self.risk['low']
            ),
            
            # Medium + single + slow = low
            ctrl.Rule(
                self.distance['medium'] & self.bird_count['single'] & 
                self.velocity['slow'],
                self.risk['low']
            ),
            
            # Far + few + safe = low
            ctrl.Rule(
                self.distance['far'] & self.bird_count['few'] & 
                self.position['safe'],
                self.risk['low']
            ),
            
            # Medium + slow + safe = low
            ctrl.Rule(
                self.distance['medium'] & self.velocity['slow'] & 
                self.position['safe'],
                self.risk['low']
            ),
            
            # ===== NEGLIGIBLE RISK RULES =====
            # Far + slow = negligible
            ctrl.Rule(
                self.distance['far'] & self.velocity['slow'],
                self.risk['negligible']
            ),
            
            # Far + safe position = negligible
            ctrl.Rule(
                self.distance['far'] & self.position['safe'],
                self.risk['negligible']
            ),
            
            # Far + single + safe = negligible (redundant but reinforces)
            ctrl.Rule(
                self.distance['far'] & self.bird_count['single'] & 
                self.position['safe'],
                self.risk['negligible']
            ),
        ]
        
        logger.info(f"Created {len(self.rules)} fuzzy inference rules")
    
    def _create_control_system(self) -> None:
        """Create the fuzzy control system and simulation."""
        self.risk_ctrl = ctrl.ControlSystem(self.rules)
        self.risk_simulation = ctrl.ControlSystemSimulation(self.risk_ctrl)
    
    def assess_risk(self, 
                   distance: float,
                   bird_count: int,
                   velocity: float,
                   position: float) -> Tuple[float, Dict[str, Any]]:
        """
        ⭐ CUSTOM ALGORITHM - Assess bird strike risk using fuzzy logic.
        
        Args:
            distance: Estimated distance metric (0-500, lower = closer)
            bird_count: Number of birds detected (0-100)
            velocity: Approach velocity in pixels/frame (0-100)
            position: Danger zone score (0-1, higher = more dangerous)
            
        Returns:
            Tuple of (risk_score, explanation_dict)
            - risk_score: Risk level from 0-100
            - explanation_dict: Details about the assessment
        """
        try:
            # Constrain inputs to valid ranges
            distance = np.clip(distance, 0, 500)
            bird_count = np.clip(bird_count, 0, 100)
            velocity = np.clip(velocity, 0, 100)
            position = np.clip(position, 0, 1)
            
            # Set inputs
            self.risk_simulation.input['distance'] = distance
            self.risk_simulation.input['bird_count'] = bird_count
            self.risk_simulation.input['velocity'] = velocity
            self.risk_simulation.input['position'] = position
            
            # Compute risk
            self.risk_simulation.compute()
            
            # Get output
            risk_score = self.risk_simulation.output['risk']
            
            # Create explanation
            explanation = {
                'inputs': {
                    'distance': distance,
                    'bird_count': bird_count,
                    'velocity': velocity,
                    'position': position
                },
                'risk_score': risk_score,
                'risk_level': self._get_risk_level(risk_score),
                'contributing_factors': self._identify_factors(
                    distance, bird_count, velocity, position
                )
            }
            
            logger.debug(
                f"Risk assessment: {risk_score:.1f}% "
                f"(D:{distance:.1f}, C:{bird_count}, V:{velocity:.1f}, P:{position:.2f})"
            )
            
            return risk_score, explanation
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            # Return moderate risk as safe fallback
            return 50.0, {
                'inputs': {'distance': distance, 'bird_count': bird_count,
                          'velocity': velocity, 'position': position},
                'risk_score': 50.0,
                'risk_level': 'moderate',
                'contributing_factors': ['Error in computation'],
                'error': str(e)
            }
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert numeric risk score to level name."""
        if risk_score < 25:
            return 'negligible'
        elif risk_score < 45:
            return 'low'
        elif risk_score < 65:
            return 'moderate'
        elif risk_score < 85:
            return 'high'
        else:
            return 'extreme'
    
    def _identify_factors(self, distance: float, bird_count: int,
                         velocity: float, position: float) -> list:
        """Identify which factors are contributing to risk."""
        factors = []
        
        if distance < 50:
            factors.append("Critical proximity")
        elif distance < 100:
            factors.append("Close distance")
        
        if bird_count > 15:
            factors.append("Large swarm detected")
        elif bird_count > 5:
            factors.append("Flock formation")
        elif bird_count > 2:
            factors.append("Multiple birds")
        
        if velocity > 20:
            factors.append("Fast approach speed")
        elif velocity > 10:
            factors.append("Moderate velocity")
        
        if position > 0.7:
            factors.append("High-danger zone")
        elif position > 0.4:
            factors.append("Caution zone")
        
        return factors if factors else ["No significant risk factors"]


if __name__ == "__main__":
    # Test the fuzzy logic system
    print("Testing Fuzzy Risk Assessment Engine...")
    
    fuzzy = FuzzyRiskAssessment()
    
    # Test cases
    test_cases = [
        (10, 20, 30, 0.8, "Critical + Swarm + Fast + Danger"),
        (50, 5, 15, 0.5, "Close + Flock + Medium + Caution"),
        (200, 1, 3, 0.2, "Far + Single + Slow + Safe"),
        (100, 10, 20, 0.6, "Medium + Flock + Fast + Danger"),
    ]
    
    for distance, count, velocity, position, description in test_cases:
        risk, explanation = fuzzy.assess_risk(distance, count, velocity, position)
        print(f"\n{description}:")
        print(f"  Risk Score: {risk:.1f}%")
        print(f"  Risk Level: {explanation['risk_level']}")
        print(f"  Factors: {', '.join(explanation['contributing_factors'])}")
