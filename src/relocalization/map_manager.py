"""
Map Manager: Storage and retrieval for relocalization database.

Manages:
- Keyframe metadata and features
- 3D point cloud
- Camera intrinsics
- Efficient retrieval via kNN on global descriptors
"""

import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Keyframe:
    """A single keyframe in the relocalization map."""
    
    id: int
    image_path: str
    timestamp: float
    pose_w2c: np.ndarray  # (4, 4) world-to-camera transformation
    gray_image: Optional[np.ndarray] = None  # (H, W) for visualization
    descriptor_global: Optional[np.ndarray] = None  # (D,) global descriptor (e.g., MixVPR)
    keypoints: Optional[np.ndarray] = None  # (K, 2) local keypoints
    descriptors_local: Optional[np.ndarray] = None  # (K, 256) local descriptors (e.g., SuperPoint)
    point_ids: Optional[np.ndarray] = None  # (K,) indices into 3D points, -1 if no 3D


class RelocalizationMap:
    """Persistent relocalization map database.
    
    Stores and manages:
    - Keyframes with descriptors
    - 3D sparse point cloud
    - Camera intrinsics
    - Metadata (build date, coverage, etc.)
    
    Supports:
    - Add/retrieve keyframes
    - Fast kNN retrieval by global descriptors
    - Save/load to/from disk
    """
    
    def __init__(self, map_dir: str):
        """Initialize map manager.
        
        Args:
            map_dir: Directory where map will be stored/loaded.
                     Will create if doesn't exist.
        """
        self.map_dir = Path(map_dir)
        self.map_dir.mkdir(parents=True, exist_ok=True)
        
        self.keyframes: List[Keyframe] = []
        self.points_3d: Optional[np.ndarray] = None  # (N, 3)
        self.K: Optional[np.ndarray] = None  # (3, 3) camera intrinsics
        self.metadata: dict = {
            "created": None,
            "num_keyframes": 0,
            "num_points_3d": 0,
            "coverage_area": None,
            "scale_estimate": None,
        }
    
    def add_keyframe(self, kf: Keyframe) -> int:
        """Add keyframe to map.
        
        Args:
            kf: Keyframe object
        
        Returns:
            Keyframe ID
        """
        kf.id = len(self.keyframes)
        self.keyframes.append(kf)
        return kf.id
    
    def set_points_3d(self, points: np.ndarray) -> None:
        """Set 3D point cloud.
        
        Args:
            points: (N, 3) array of 3D points in world frame
        """
        assert points.ndim == 2 and points.shape[1] == 3
        self.points_3d = points.astype(np.float32)
        self.metadata["num_points_3d"] = len(points)
    
    def set_intrinsics(self, K: np.ndarray) -> None:
        """Set camera intrinsic matrix.
        
        Args:
            K: (3, 3) camera intrinsics
        """
        assert K.shape == (3, 3)
        self.K = K.astype(np.float32)
    
    def get_top_k_candidates(self, query_desc: np.ndarray,
                            k: int = 20) -> List[Keyframe]:
        """Retrieve top-K most similar keyframes by global descriptor.
        
        Uses cosine similarity on L2-normalized descriptors.
        
        Args:
            query_desc: (D,) query global descriptor, L2-normalized
            k: Number of top candidates to return
        
        Returns:
            List of K keyframes, sorted by similarity (best first)
        """
        if not self.keyframes:
            return []
        
        if len(self.keyframes) < k:
            k = len(self.keyframes)
        
        # Extract global descriptors from keyframes
        corpus = np.array([
            kf.descriptor_global for kf in self.keyframes
            if kf.descriptor_global is not None
        ], dtype=np.float32)
        
        if len(corpus) == 0:
            return []
        
        # Cosine similarity: dot product on L2-normalized vectors
        # Assume both are already normalized
        similarities = corpus @ query_desc.astype(np.float32)
        
        # Get top-k indices
        top_indices = np.argsort(-similarities)[:k]
        
        # Return corresponding keyframes
        return [self.keyframes[idx] for idx in top_indices]
    
    def save(self) -> None:
        """Serialize map to disk.
        
        Saves:
        - keyframes.pkl: List of keyframe objects (with features)
        - points_3d.npy: (N, 3) 3D points
        - intrinsics.npy: (3, 3) camera matrix
        - metadata.json: Map metadata
        """
        # Save keyframes (pickled)
        keyframes_path = self.map_dir / "keyframes.pkl"
        with open(keyframes_path, 'wb') as f:
            pickle.dump(self.keyframes, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save 3D points
        if self.points_3d is not None:
            points_path = self.map_dir / "points_3d.npy"
            np.save(points_path, self.points_3d)
        
        # Save intrinsics
        if self.K is not None:
            intrinsics_path = self.map_dir / "intrinsics.npy"
            np.save(intrinsics_path, self.K)
        
        # Save metadata
        metadata_path = self.map_dir / "metadata.json"
        self.metadata["num_keyframes"] = len(self.keyframes)
        self.metadata["num_points_3d"] = len(self.points_3d) if self.points_3d is not None else 0
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def load(self) -> None:
        """Deserialize map from disk.
        
        Loads:
        - keyframes.pkl
        - points_3d.npy
        - intrinsics.npy
        - metadata.json
        """
        keyframes_path = self.map_dir / "keyframes.pkl"
        if keyframes_path.exists():
            with open(keyframes_path, 'rb') as f:
                self.keyframes = pickle.load(f)
        
        points_path = self.map_dir / "points_3d.npy"
        if points_path.exists():
            self.points_3d = np.load(points_path).astype(np.float32)
        
        intrinsics_path = self.map_dir / "intrinsics.npy"
        if intrinsics_path.exists():
            self.K = np.load(intrinsics_path).astype(np.float32)
        
        metadata_path = self.map_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
    
    def __len__(self) -> int:
        """Return number of keyframes."""
        return len(self.keyframes)
    
    def __getitem__(self, idx: int) -> Keyframe:
        """Get keyframe by index."""
        return self.keyframes[idx]
    
    def summary(self) -> str:
        """Return summary statistics."""
        num_kf = len(self.keyframes)
        num_pts = len(self.points_3d) if self.points_3d is not None else 0
        desc_dims = [
            kf.descriptor_global.shape[0] if kf.descriptor_global is not None else 0
            for kf in self.keyframes
        ]
        
        summary = (
            f"RelocalizationMap Summary:\n"
            f"  Keyframes: {num_kf}\n"
            f"  3D Points: {num_pts}\n"
            f"  Global descriptor dim: {desc_dims[0] if desc_dims else 'N/A'}\n"
            f"  Camera K: {self.K.shape if self.K is not None else 'Not set'}\n"
            f"  Map dir: {self.map_dir}"
        )
        return summary
