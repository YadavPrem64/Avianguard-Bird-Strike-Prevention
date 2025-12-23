"""
AvianGuard IRAPS - Core Modules
Contains detection, tracking, and intelligence modules for bird strike prevention.
"""

from . import detection
from . import video_processor
from . import fuzzy_logic
from . import trajectory
from . import collision_calc
from . import tracker
from . import danger_zones

__all__ = [
    'detection',
    'video_processor',
    'fuzzy_logic',
    'trajectory',
    'collision_calc',
    'tracker',
    'danger_zones'
]
