"""
HLoc-style relocalization module.

Provides hierarchical visual localization pipeline:
- Global place recognition (MixVPR)
- Local geometric matching (SuperPoint + LightGlue)
- 6-DoF pose recovery (PnP-RANSAC)
"""

from .map_manager import RelocalizationMap, Keyframe
from .global_descriptor import GlobalDescriptorExtractor
from .local_matcher import LocalMatcher
from .pose_solver import PoseSolver
from .hloc_pipeline import HLocPipeline

__all__ = [
    "RelocalizationMap",
    "Keyframe",
    "GlobalDescriptorExtractor",
    "LocalMatcher",
    "PoseSolver",
    "HLocPipeline",
]
