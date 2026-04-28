"""
merlin: Simulation-based inference for Euclid 3×2pt cosmology.
"""

from .simulator import Simulator, PriorSampler
from .network import Network

__all__ = ["Simulator", "PriorSampler", "Network"]
__version__ = "0.1.0"
