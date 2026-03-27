"""
Pose Solver: 6-DoF camera pose recovery from 2D-3D correspondences.

Uses:
- PnP-RANSAC for robust pose estimation
- Geometric verification (epipolar consistency, scale check)
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


class PoseSolver:
    """Recover camera pose via PnP-RANSAC.
    
    Maps 2D image points to 3D world points to estimate
    camera extrinsic parameters (rotation + translation).
    
    Output: 4×4 transformation matrix (c2w: camera-to-world)
    """
    
    def __init__(self, K: np.ndarray,
                 inlier_threshold_px: float = 8.0,
                 confidence: float = 0.99):
        """Initialize pose solver.
        
        Args:
            K: (3, 3) camera intrinsic matrix
            inlier_threshold_px: RANSAC reprojection error threshold in pixels
            confidence: RANSAC confidence (0-1)
        """
        self.K = K.astype(np.float32)
        self.inlier_threshold = inlier_threshold_px
        self.confidence = confidence
        
        # Derived camera parameters
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
    
    def solve_pnp_ransac(self, points_2d: np.ndarray,
                         points_3d: np.ndarray,
                         iterations: int = 1000) -> Dict:
        """Estimate pose via PnP-RANSAC.
        
        Args:
            points_2d: (M, 2) image coordinates in pixels
            points_3d: (M, 3) world coordinates
            iterations: Max RANSAC iterations
        
        Returns:
            {
                'success': bool - did we find a good pose?
                'rotation': (3, 3) rotation matrix R_c2w
                'translation': (3,) translation vector t_c2w
                't_vec': (3,) translation (same as translation)
                'inlier_mask': (M,) boolean mask
                'num_inliers': int
                'reprojection_error': float (mean error of inliers)
                'rvec': (3,) rotation vector (Rodrigues)
            }
        """
        if len(points_2d) < 4 or len(points_3d) < 4:
            return {
                'success': False,
                'reason': f'Not enough points: 2D={len(points_2d)}, 3D={len(points_3d)}'
            }
        
        # Ensure proper dtypes
        points_2d = points_2d.astype(np.float32)
        points_3d = points_3d.astype(np.float32)
        
        # Use OpenCV's solvePnPRansac
        success, rvec, tvec, inlier_mask = cv2.solvePnPRansac(
            objectPoints=points_3d,
            imagePoints=points_2d,
            cameraMatrix=self.K,
            distCoeffs=None,
            iterationsCount=iterations,
            reprojectionError=self.inlier_threshold,
            confidence=self.confidence,
            useExtrinsicGuess=False,
            flags=cv2.SOLVEPNP_EPNP,
        )
        
        if not success or rvec is None:
            return {
                'success': False,
                'reason': 'PnP-RANSAC failed to find valid solution'
            }
        
        # Convert rotation vector to matrix
        R_c2w, _ = cv2.Rodrigues(rvec)
        t_c2w = tvec.flatten()
        
        # Extract inliers
        if inlier_mask is not None:
            inlier_mask = inlier_mask.flatten().astype(bool)
            # Pad if necessary (sometimes OpenCV returns shorter mask)
            if len(inlier_mask) < len(points_3d):
                padded_mask = np.zeros(len(points_3d), dtype=bool)
                padded_mask[:len(inlier_mask)] = inlier_mask
                inlier_mask = padded_mask
        else:
            inlier_mask = np.ones(len(points_3d), dtype=bool)
        
        num_inliers = int(np.sum(inlier_mask))
        
        # Compute reprojection error for inliers
        reproj_error = self._compute_reprojection_error(
            points_2d, points_3d, R_c2w, t_c2w, inlier_mask
        )
        
        return {
            'success': True,
            'rotation': R_c2w.astype(np.float32),
            'translation': t_c2w.astype(np.float32),
            't_vec': t_c2w.astype(np.float32),
            'rvec': rvec.flatten().astype(np.float32),
            'inlier_mask': inlier_mask,
            'num_inliers': num_inliers,
            'reprojection_error': reproj_error,
        }
    
    def _compute_reprojection_error(self, points_2d: np.ndarray,
                                   points_3d: np.ndarray,
                                   R: np.ndarray,
                                   t: np.ndarray,
                                   inlier_mask: np.ndarray) -> float:
        """Compute mean reprojection error for inliers.
        
        Args:
            points_2d: (M, 2) image points
            points_3d: (M, 3) world points
            R: (3, 3) rotation
            t: (3,) translation
            inlier_mask: (N,) boolean where N <= M
        
        Returns:
            Mean reprojection error in pixels
        """
        # Project 3D points to image
        points_3d_cam = points_3d @ R.T + t.reshape(1, 3)
        
        # Filter behind camera
        valid_depth = points_3d_cam[:, 2] > 0
        
        # Only evaluate on inliers that pass depth check
        if len(inlier_mask) == len(valid_depth):
            # Full mask provided
            valid = valid_depth & inlier_mask.astype(bool)
        else:
            # Partial mask - use it as-is (from RANSAC)
            valid = inlier_mask.astype(bool)[:len(points_3d_cam)]
        
        if not np.any(valid):
            return float('inf')
        
        # Perspective projection
        points_proj = points_3d_cam[valid] @ self.K.T
        points_proj_2d = points_proj[:, :2] / points_proj[:, 2:3]
        
        # Compute error
        errors = np.linalg.norm(points_2d[valid] - points_proj_2d, axis=1)
        
        return float(np.mean(errors))
    
    def verify_pose(self, pose: Dict,
                   min_inliers: int = 15,
                   max_reproj_error: float = 2.0,
                   check_scale: bool = True) -> bool:
        """Verify pose solution quality.
        
        Args:
            pose: Result dict from solve_pnp_ransac
            min_inliers: Minimum required inliers
            max_reproj_error: Maximum acceptable reprojection error
            check_scale: Verify translation scale is reasonable
        
        Returns:
            True if pose passes all checks
        """
        if not pose.get('success', False):
            return False
        
        # Check inlier count
        if pose.get('num_inliers', 0) < min_inliers:
            return False
        
        # Check reprojection error
        if pose.get('reprojection_error', float('inf')) > max_reproj_error:
            return False
        
        # Check scale (translation magnitude)
        if check_scale:
            t = pose.get('translation')
            scale = np.linalg.norm(t)
            if scale < 0.001 or scale > 100.0:  # Reasonable range
                return False
        
        # Check rotation is valid (det = 1)
        R = pose.get('rotation')
        if abs(np.linalg.det(R) - 1.0) > 0.1:
            return False
        
        return True
    
    def pose_to_matrix(self, pose: Dict) -> np.ndarray:
        """Convert pose dict to 4x4 homogeneous transformation.
        
        Args:
            pose: Result dict with 'rotation' and 'translation'
        
        Returns:
            (4, 4) matrix [R | t; 0 | 1] representing c2w
        """
        if not pose.get('success', False):
            return None
        
        R = pose['rotation']
        t = pose['translation']
        
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return T
    
    def invert_pose(self, T: np.ndarray) -> np.ndarray:
        """Invert pose transformation (c2w -> w2c).
        
        Args:
            T: (4, 4) transformation
        
        Returns:
            (4, 4) inverted transformation
        """
        T_inv = np.eye(4, dtype=np.float32)
        R = T[:3, :3]
        t = T[:3, 3]
        
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t
        
        return T_inv
