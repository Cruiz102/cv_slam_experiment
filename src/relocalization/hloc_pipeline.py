"""
HLoc Pipeline: Main orchestrator for hierarchical visual localization.

Workflow:
1. Global descriptor retrieval (place recognition)
2. Local feature matching (geometric verification)
3. PnP + RANSAC (6-DoF pose recovery)
4. Validation and confidence scoring
"""

import time
from typing import Dict, Optional

import cv2
import numpy as np

from .map_manager import RelocalizationMap
from .global_descriptor import GlobalDescriptorExtractor
from .local_matcher import LocalMatcher
from .pose_solver import PoseSolver


class HLocPipeline:
    """Main HLoc relocalization pipeline.
    
    Orchestrates:
    - Retrieval stage: global descriptor matching to find candidate keyframes
    - Verification stage: local feature matching for geometric confirmation
    - Pose solving stage: PnP-RANSAC to compute precise camera pose
    - Validation stage: confidence scoring and sanity checks
    """
    
    def __init__(self, map_dir: str,
                 device: str = "cuda",
                 global_descriptor_type: str = "mixvpr",
                 config: Optional[Dict] = None):
        """Initialize HLoc pipeline.
        
        Args:
            map_dir: Directory containing relocalization map
            device: "cuda" or "cpu"
            global_descriptor_type: "mixvpr" or "dinov2"
            config: Optional config dict to override defaults
        """
        self.map_dir = map_dir
        self.device = device
        
        # Load map
        self.map = RelocalizationMap(map_dir)
        try:
            self.map.load()
            print(f"Loaded map: {len(self.map)} keyframes, "
                  f"{len(self.map.points_3d) if self.map.points_3d is not None else 0} 3D points")
        except Exception as e:
            print(f"Warning: Could not load map: {e}")
        
        # Initialize components
        self.global_desc = GlobalDescriptorExtractor(
            model_type=global_descriptor_type,
            device=device
        )
        self.local_matcher = LocalMatcher(device=device)
        
        if self.map.K is not None:
            self.pose_solver = PoseSolver(self.map.K)
        else:
            print("Warning: Map has no intrinsics set. PnP will fail.")
            self.pose_solver = None
        
        # Configuration
        self.config = config or self._default_config()
    
    @staticmethod
    def _default_config() -> Dict:
        """Default HLoc pipeline configuration."""
        return {
            # Retrieval stage
            'top_k_retrieval': 15,
            'retrieval_threshold': 0.0,  # No threshold, use top-k
            
            # Local matching stage
            'min_matches_for_pose': 10,
            'match_threshold': 0.8,  # Lowe's ratio test
            'mutual_check': True,
            
            # PnP-RANSAC stage
            'ransac_iterations': 1000,
            'ransac_confidence': 0.99,
            'ransac_inlier_threshold': 8.0,  # pixels
            
            # Pose validation
            'min_inliers': 15,
            'max_reproj_error': 2.0,
            'check_scale': True,
        }
    
    def relocalize(self, image_rgb: np.ndarray,
                  prior_R: Optional[np.ndarray] = None,
                  prior_p: Optional[np.ndarray] = None,
                  return_debug: bool = False) -> Dict:
        """Full HLoc relocalization pipeline.
        
        Args:
            image_rgb: (H, W, 3) uint8 query image in RGB
            prior_R: (3, 3) optional rotation prior from IMU
            prior_p: (3,) optional position prior
            return_debug: Return debug visualization
        
        Returns:
            {
                'success': bool - did relocalization succeed?
                'pose_c2w': (4, 4) camera-to-world transformation if success
                'pose_w2c': (4, 4) world-to-camera transformation if success
                'num_inliers': int number of inliers in final pose
                'matched_keyframe_id': int ID of best matching keyframe
                'confidence': float (0-1) confidence score
                'retrieval_time': float ms
                'matching_time': float ms
                'pose_time': float ms
                'total_time': float ms
                'stages': {stage: timing} detailed timing
                'debug_matches': ndarray or None (visualization)
                'reason': str (if failure)
            }
        """
        total_t0 = time.perf_counter()
        stages = {}
        
        # ===== Stage 1: Global Retrieval =====
        t0 = time.perf_counter()
        try:
            query_desc_global = self.global_desc.extract(image_rgb)
            candidates = self.map.get_top_k_candidates(
                query_desc_global,
                k=self.config['top_k_retrieval']
            )
            stages['retrieval'] = (time.perf_counter() - t0) * 1000
        except Exception as e:
            return {
                'success': False,
                'reason': f'Retrieval failed: {e}',
                'stages': stages,
                'total_time': (time.perf_counter() - total_t0) * 1000,
            }
        
        if not candidates:
            return {
                'success': False,
                'reason': 'No candidate keyframes found',
                'stages': stages,
                'total_time': (time.perf_counter() - total_t0) * 1000,
            }
        
        # ===== Stage 2: Extract Query Features =====
        try:
            gray_query = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            features_query = self.local_matcher.extract_features(gray_query)
        except Exception as e:
            return {
                'success': False,
                'reason': f'Query feature extraction failed: {e}',
                'stages': stages,
                'total_time': (time.perf_counter() - total_t0) * 1000,
            }
        
        # ===== Stage 3: Matching & Pose Solving =====
        t0 = time.perf_counter()
        best_result = None
        best_inliers = 0
        matched_keyframe_id = -1
        all_matches_debug = []
        
        for cand_idx, candidate_kf in enumerate(candidates):
            # Extract candidate features
            try:
                features_kf = {
                    'keypoints': candidate_kf.keypoints,
                    'descriptors': candidate_kf.descriptors_local,
                    'image_shape': (candidate_kf.gray_image.shape[0],
                                    candidate_kf.gray_image.shape[1])
                    if candidate_kf.gray_image is not None else (480, 640)
                }
            except Exception as e:
                continue
            
            # Match features
            try:
                matches = self.local_matcher.match_pairs(
                    features_query, features_kf,
                    match_threshold=self.config['match_threshold'],
                    mutual_check=self.config['mutual_check']
                )
            except Exception as e:
                continue
            
            if len(matches['matches']) < self.config['min_matches_for_pose']:
                continue
            
            # Map 2D matches to 3D points
            try:
                pts2d = matches['matches_mkpts0']
                match_indices = matches['matches'][:, 1]  # Indices into keyframe keypoints
                
                # Get 3D point indices
                point_ids = candidate_kf.point_ids[match_indices]
                
                # Filter out invalid point IDs
                valid_mask = point_ids != -1
                pts2d = pts2d[valid_mask]
                point_ids = point_ids[valid_mask]
                
                if len(pts2d) < self.config['min_matches_for_pose']:
                    continue
                
                pts3d = self.map.points_3d[point_ids.astype(int)]
            except Exception as e:
                continue
            
            # Solve PnP
            try:
                pose_result = self.pose_solver.solve_pnp_ransac(
                    pts2d, pts3d,
                    iterations=self.config['ransac_iterations']
                )
            except Exception as e:
                continue
            
            if not pose_result.get('success', False):
                continue
            
            # Verify pose quality
            num_inliers = pose_result.get('num_inliers', 0)
            if num_inliers > best_inliers:
                best_result = pose_result
                best_inliers = num_inliers
                matched_keyframe_id = candidate_kf.id
                all_matches_debug.append({
                    'kf_id': candidate_kf.id,
                    'inliers': num_inliers,
                    'matches': matches,
                    'pose': pose_result
                })
        
        stages['matching'] = (time.perf_counter() - t0) * 1000
        
        if best_result is None:
            return {
                'success': False,
                'reason': 'No valid pose found from any candidate',
                'stages': stages,
                'total_time': (time.perf_counter() - total_t0) * 1000,
            }
        
        # ===== Stage 4: Validation =====
        t0 = time.perf_counter()
        valid = self.pose_solver.verify_pose(
            best_result,
            min_inliers=self.config['min_inliers'],
            max_reproj_error=self.config['max_reproj_error'],
            check_scale=self.config['check_scale']
        )
        stages['validation'] = (time.perf_counter() - t0) * 1000
        
        if not valid:
            return {
                'success': False,
                'reason': 'Pose failed validation checks',
                'num_inliers': best_inliers,
                'stages': stages,
                'total_time': (time.perf_counter() - total_t0) * 1000,
            }
        
        # ===== Assemble Success Result =====
        pose_c2w = self.pose_solver.pose_to_matrix(best_result)
        pose_w2c = self.pose_solver.invert_pose(pose_c2w)
        
        # Confidence: normalize inlier count
        confidence = min(best_inliers / 50.0, 1.0)
        
        result = {
            'success': True,
            'pose_c2w': pose_c2w,
            'pose_w2c': pose_w2c,
            'num_inliers': best_inliers,
            'matched_keyframe_id': matched_keyframe_id,
            'confidence': confidence,
            'reproj_error': best_result.get('reprojection_error', -1),
            'stages': stages,
            'total_time': (time.perf_counter() - total_t0) * 1000,
        }
        
        # Optionally add debug visualization
        if return_debug and len(all_matches_debug) > 0:
            result['debug'] = all_matches_debug[0]  # Best match
        
        return result
    
    def benchmark_stages(self, image_rgb: np.ndarray, num_trials: int = 3) -> Dict:
        """Benchmark each pipeline stage.
        
        Args:
            image_rgb: Test image
            num_trials: Number of times to run each stage
        
        Returns:
            Timing dict
        """
        timings = {}
        
        # Retrieval
        t0 = time.perf_counter()
        for _ in range(num_trials):
            query_desc = self.global_desc.extract(image_rgb)
            candidates = self.map.get_top_k_candidates(query_desc, k=15)
        timings['retrieval'] = (time.perf_counter() - t0) * 1000 / num_trials
        
        # Feature extraction
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        t0 = time.perf_counter()
        for _ in range(num_trials):
            features = self.local_matcher.extract_features(gray)
        timings['feature_extraction'] = (time.perf_counter() - t0) * 1000 / num_trials
        
        return timings
    
    def summary(self) -> str:
        """Print pipeline summary."""
        summary = (
            f"\nHLoc Pipeline Summary:\n"
            f"  Map: {len(self.map)} keyframes, "
            f"{len(self.map.points_3d) if self.map.points_3d is not None else 0} 3D points\n"
            f"  Camera intrinsics: {self.map.K.shape if self.map.K is not None else 'Not set'}\n"
            f"  Device: {self.device}\n"
            f"  Global descriptor: {self.global_desc.model_type}\n"
            f"  Local matcher: {'SuperPoint+LightGlue' if self.local_matcher.superpoint else 'ORB+BFMatcher (fallback)'}\n"
        )
        return summary
