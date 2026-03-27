"""
Utilities for offline map building.

Provides:
- Keyframe selection strategies
- Map quality assessment
- Pose refinement helpers
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class KeyframeSelectionConfig:
    """Configuration for keyframe selection."""
    # Spatial: keyframes farther apart than this are kept (meters)
    min_translation_distance: float = 0.3
    # Rotation: keyframes rotated more than this are kept (degrees)  
    min_rotation_angle: float = 10.0
    # Temporal: keep every Nth frame if other criteria not met
    temporal_interval: int = 5
    # Descriptor: keep frames with low similarity to previous keyframe
    min_descriptor_distance: float = 0.3


def compute_pose_distance(pose1: np.ndarray, pose2: np.ndarray) -> Tuple[float, float]:
    """Compute translation and rotation distance between poses.
    
    Args:
        pose1: (4, 4) transformation matrix
        pose2: (4, 4) transformation matrix
    
    Returns:
        (translation_distance, rotation_angle_degrees)
    """
    # Translation distance
    t1 = pose1[:3, 3]
    t2 = pose2[:3, 3]
    translation_dist = np.linalg.norm(t2 - t1)
    
    # Rotation angle via trace
    R1 = pose1[:3, :3]
    R2 = pose2[:3, :3]
    R_rel = R2 @ R1.T
    
    # Angle of rotation
    trace = np.clip(np.trace(R_rel), -1, 3)
    rotation_angle = np.arccos((trace - 1) / 2)
    rotation_angle_deg = np.degrees(rotation_angle)
    
    return translation_dist, rotation_angle_deg


def compute_descriptor_distance(desc1: np.ndarray, desc2: np.ndarray) -> float:
    """Compute descriptor distance (L2 for normalized descriptors).
    
    Args:
        desc1: (D,) descriptor vector (assumed L2-normalized)
        desc2: (D,) descriptor vector (assumed L2-normalized)
    
    Returns:
        Distance in [0, 2] (0 = identical, 2 = opposite)
    """
    if desc1 is None or desc2 is None:
        return 2.0
    
    # For L2-normalized vectors: distance = sqrt(2(1 - dot_product))
    dot_product = np.dot(desc1, desc2)
    dot_product = np.clip(dot_product, -1, 1)
    distance = np.sqrt(2 * (1 - dot_product))
    
    return distance


class KeyframeSelector:
    """Select representative keyframes from image sequence."""
    
    def __init__(self, config: Optional[KeyframeSelectionConfig] = None):
        """Initialize selector.
        
        Args:
            config: Keyframe selection configuration
        """
        self.config = config or KeyframeSelectionConfig()
        self.selected_indices: List[int] = []
    
    def select(self, poses: List[np.ndarray], 
              descriptors: Optional[List[np.ndarray]] = None) -> List[int]:
        """Select representative keyframes.
        
        Strategy:
        1. Always include first frame
        2. Include frames where translation or rotation exceeded threshold
        3. Include frames with low descriptor similarity to previous keyframe
        4. Fall back to temporal sampling if sparse
        
        Args:
            poses: List of (4, 4) camera poses
            descriptors: Optional list of global descriptors
        
        Returns:
            List of selected frame indices
        """
        if not poses:
            return []
        
        selected = [0]  # Always include first frame
        last_pose = poses[0]
        last_desc = descriptors[0] if descriptors else None
        
        for idx in range(1, len(poses)):
            pose = poses[idx]
            desc = descriptors[idx] if descriptors else None
            
            # Spatial criterion
            trans_dist, rot_angle = compute_pose_distance(last_pose, pose)
            spatial_criterion = (
                trans_dist >= self.config.min_translation_distance or
                rot_angle >= self.config.min_rotation_angle
            )
            
            # Descriptor criterion
            descriptor_criterion = False
            if last_desc is not None and desc is not None:
                desc_dist = compute_descriptor_distance(last_desc, desc)
                descriptor_criterion = desc_dist >= self.config.min_descriptor_distance
            
            # Temporal criterion
            temporal_criterion = (idx % self.config.temporal_interval) == 0
            
            # Include if any criterion met
            if spatial_criterion or descriptor_criterion or temporal_criterion:
                selected.append(idx)
                last_pose = pose
                last_desc = desc
        
        self.selected_indices = selected
        return selected


class MapQualityAssessment:
    """Assess quality of relocalization map."""
    
    @staticmethod
    def assess_coverage(keyframes_poses: np.ndarray,
                       grid_size: float = 1.0) -> float:
        """Assess spatial coverage of keyframes.
        
        Divides workspace into grid and computes coverage ratio.
        
        Args:
            keyframes_poses: (N, 4, 4) array of camera poses
            grid_size: Size of grid cells (meters)
        
        Returns:
            Coverage ratio in [0, 1]
        """
        if len(keyframes_poses) == 0:
            return 0.0
        
        # Extract camera positions (world-to-camera: invert for camera position in world)
        positions = []
        for T_w2c in keyframes_poses:
            # Camera position in world frame: -R^T @ t
            R = T_w2c[:3, :3]
            t = T_w2c[:3, 3]
            cam_pos = -R.T @ t
            positions.append(cam_pos)
        
        positions = np.array(positions)
        
        # Compute grid bounds
        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)
        
        # Grid occupancy
        grid_dims = np.ceil((maxs - mins) / grid_size + 1).astype(int)
        occupancy = set()
        
        for pos in positions:
            grid_idx = tuple(np.floor((pos - mins) / grid_size).astype(int))
            occupancy.add(grid_idx)
        
        total_cells = np.prod(grid_dims)
        coverage = len(occupancy) / max(total_cells, 1)
        
        return float(coverage)
    
    @staticmethod
    def assess_point_density(points_3d: np.ndarray,
                            bounds: Optional[Tuple[float, float, float]] = None) -> float:
        """Assess density of 3D points.
        
        Args:
            points_3d: (M, 3) array of 3D points
            bounds: (width, depth, height) of scene (meters)
        
        Returns:
            Point density (points per cubic meter)
        """
        if points_3d is None or len(points_3d) == 0:
            return 0.0
        
        if bounds is None:
            bounds = (10.0, 10.0, 2.0)  # Default: 10m x 10m x 2m
        
        volume = np.prod(bounds)
        density = len(points_3d) / volume
        
        return float(density)
    
    @staticmethod
    def assess_descriptor_distribution(descriptors: List[np.ndarray]) -> float:
        """Assess diversity of global descriptors.
        
        Computes average pairwise descriptor distance.
        High diversity = good coverage of appearance variations.
        
        Args:
            descriptors: List of (D,) descriptor vectors
        
        Returns:
            Mean pairwise descriptor distance
        """
        descriptors = [d for d in descriptors if d is not None]
        
        if len(descriptors) < 2:
            return 0.0
        
        # Sample for efficiency
        sample_size = min(100, len(descriptors))
        sample_indices = np.random.choice(len(descriptors), sample_size, replace=False)
        
        distances = []
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                desc_i = descriptors[sample_indices[i]]
                desc_j = descriptors[sample_indices[j]]
                dist = compute_descriptor_distance(desc_i, desc_j)
                distances.append(dist)
        
        if distances:
            return float(np.mean(distances))
        return 0.0
    
    @staticmethod
    def full_assessment(keyframes_poses: np.ndarray,
                       points_3d: Optional[np.ndarray],
                       descriptors: Optional[List[np.ndarray]] = None) -> dict:
        """Full map quality assessment.
        
        Args:
            keyframes_poses: (N, 4, 4) camera pose matrices
            points_3d: (M, 3) 3D point coordinates
            descriptors: List of global descriptors
        
        Returns:
            Dict with quality metrics
        """
        metrics = {
            'num_keyframes': len(keyframes_poses),
            'num_3d_points': len(points_3d) if points_3d is not None else 0,
            'spatial_coverage': MapQualityAssessment.assess_coverage(keyframes_poses),
            'point_density': MapQualityAssessment.assess_point_density(points_3d),
            'descriptor_diversity': (
                MapQualityAssessment.assess_descriptor_distribution(descriptors)
                if descriptors else 0.0
            ),
        }
        
        # Overall score: weighted combination
        score = (
            0.3 * metrics['spatial_coverage'] +
            0.3 * min(metrics['point_density'] / 100, 1.0) +  # Normalize to [0, 1]
            0.2 * min(metrics['descriptor_diversity'] / 2, 1.0) +  # Normalize to [0, 1]
            0.2 * min(metrics['num_keyframes'] / 50, 1.0)  # Normalize to [0, 1]
        )
        metrics['quality_score'] = float(score)
        
        return metrics


def print_map_statistics(keyframes_poses: np.ndarray,
                        points_3d: Optional[np.ndarray] = None,
                        descriptors: Optional[List[np.ndarray]] = None) -> None:
    """Print map statistics to console.
    
    Args:
        keyframes_poses: (N, 4, 4) camera pose matrices
        points_3d: (M, 3) 3D point coordinates
        descriptors: List of global descriptors
    """
    metrics = MapQualityAssessment.full_assessment(
        keyframes_poses, points_3d, descriptors
    )
    
    print("\n" + "="*50)
    print("Map Quality Assessment")
    print("="*50)
    print(f"Number of keyframes:      {metrics['num_keyframes']}")
    print(f"Number of 3D points:      {metrics['num_3d_points']}")
    print(f"Spatial coverage:         {metrics['spatial_coverage']:.1%}")
    print(f"Point density:            {metrics['point_density']:.2f} pts/m³")
    print(f"Descriptor diversity:     {metrics['descriptor_diversity']:.2f}")
    print(f"Overall quality score:    {metrics['quality_score']:.2f}/1.00")
    print("="*50 + "\n")
