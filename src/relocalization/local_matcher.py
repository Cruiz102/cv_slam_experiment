"""
Local Matcher: SuperPoint + LightGlue for robust geometric verification.

Provides:
- SuperPoint keypoint detection and description
- LightGlue feature matching
- Fallback to ORB+BFMatcher if models unavailable
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class LocalMatcher:
    """SuperPoint + LightGlue for robust local feature matching.
    
    Workflow:
    1. Extract SuperPoint features from both images
    2. Match features using LightGlue
    3. Return 2D-2D correspondences
    
    Falls back to ORB+BFMatcher if deep models unavailable.
    """
    
    def __init__(self, device: str = "cuda",
                 use_superpoint: bool = True,
                 use_lightglue: bool = True):
        """Initialize local matcher.
        
        Args:
            device: "cuda" or "cpu"
            use_superpoint: Try to load SuperPoint. Fall back to ORB if unavailable.
            use_lightglue: Try to load LightGlue. Fall back to BFMatcher if unavailable.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_superpoint = use_superpoint
        self.use_lightglue = use_lightglue
        
        self.superpoint = None
        self.lightglue = None
        self.orb = None
        self.bf_matcher = None
        
        if use_superpoint:
            self._try_load_superpoint()
        
        if use_lightglue:
            self._try_load_lightglue()
        
        # Always have fallback ready
        if self.superpoint is None or self.orb is None:
            self._setup_orb_fallback()
    
    def _try_load_superpoint(self) -> bool:
        """Try to load SuperPoint model."""
        try:
            # Import DL2 SuperPoint implementation
            # (This is a placeholder; actual implementation requires model weights)
            print("[LocalMatcher] SuperPoint model loading not yet implemented.")
            print("[LocalMatcher] Falling back to ORB detector.")
            return False
        except Exception as e:
            print(f"[LocalMatcher] Failed to load SuperPoint: {e}")
            return False
    
    def _try_load_lightglue(self) -> bool:
        """Try to load LightGlue matcher."""
        try:
            # Import DL2 LightGlue implementation
            print("[LocalMatcher] LightGlue model loading not yet implemented.")
            print("[LocalMatcher] Falling back to BFMatcher.")
            return False
        except Exception as e:
            print(f"[LocalMatcher] Failed to load LightGlue: {e}")
            return False
    
    def _setup_orb_fallback(self) -> None:
        """Setup ORB + BFMatcher as fallback."""
        self.orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def extract_features(self, image_gray: np.ndarray) -> Dict:
        """Extract local features from image.
        
        Args:
            image_gray: (H, W) uint8 grayscale image
        
        Returns:
            {
                'keypoints': (K, 2) normalized pixel coords [0, 1]
                'descriptors': (K, D) descriptor vectors
                'scores': (K,) detection confidence scores
                'image_shape': (H, W)
            }
        """
        if image_gray.dtype != np.uint8:
            image_gray = np.uint8(image_gray * 255.0)
        
        H, W = image_gray.shape
        
        if self.superpoint is not None:
            return self._extract_superpoint(image_gray)
        else:
            return self._extract_orb(image_gray)
    
    def _extract_superpoint(self, image_gray: np.ndarray) -> Dict:
        """Extract SuperPoint features.
        
        (Placeholder: implement when model available)
        """
        # Would implement actual SuperPoint extraction here
        # For now, falls back to ORB
        return self._extract_orb(image_gray)
    
    def _extract_orb(self, image_gray: np.ndarray) -> Dict:
        """Extract ORB features (fallback).
        
        Args:
            image_gray: (H, W) uint8
        
        Returns:
            Feature dict
        """
        H, W = image_gray.shape
        
        # Detect keypoints and descriptors
        kpts, descs = self.orb.detectAndCompute(image_gray, None)
        
        if descs is None or len(kpts) == 0:
            # No features found
            return {
                'keypoints': np.zeros((0, 2), dtype=np.float32),
                'descriptors': np.zeros((0, 32), dtype=np.uint8),
                'scores': np.zeros((0,), dtype=np.float32),
                'image_shape': (H, W),
            }
        
        # Normalize keypoint coordinates to [0, 1]
        keypoints = np.array([[kpt.pt[0] / W, kpt.pt[1] / H] for kpt in kpts],
                             dtype=np.float32)
        scores = np.array([kpt.response for kpt in kpts], dtype=np.float32)
        
        return {
            'keypoints': keypoints,
            'descriptors': descs.astype(np.uint8),
            'scores': scores,
            'image_shape': (H, W),
        }
    
    def match_pairs(self, features_query: Dict,
                    features_keyframe: Dict,
                    match_threshold: float = 0.7,
                    mutual_check: bool = True) -> Dict:
        """Match features between query and keyframe.
        
        Args:
            features_query: Query image features dict
            features_keyframe: Keyframe image features dict
            match_threshold: Confidence threshold for matches
            mutual_check: Apply mutual nearest-neighbor check
        
        Returns:
            {
                'matches': (M, 2) pairs of keypoint indices [idx_q, idx_kf]
                'scores': (M,) match confidence scores
                'matches_mkpts0': (M, 2) query image pixel coords (denormalized)
                'matches_mkpts1': (M, 2) keyframe pixel coords (denormalized)
            }
        """
        kpts_q = features_query['keypoints']
        descs_q = features_query['descriptors']
        H_q, W_q = features_query['image_shape']
        
        kpts_kf = features_keyframe['keypoints']
        descs_kf = features_keyframe['descriptors']
        H_kf, W_kf = features_keyframe['image_shape']
        
        if len(kpts_q) == 0 or len(kpts_kf) == 0:
            return {
                'matches': np.zeros((0, 2), dtype=np.int32),
                'scores': np.zeros((0,), dtype=np.float32),
                'matches_mkpts0': np.zeros((0, 2), dtype=np.float32),
                'matches_mkpts1': np.zeros((0, 2), dtype=np.float32),
            }
        
        if self.lightglue is not None:
            return self._match_lightglue(features_query, features_keyframe,
                                        match_threshold)
        else:
            return self._match_bfmatcher(features_query, features_keyframe,
                                         match_threshold, mutual_check)
    
    def _match_lightglue(self, features_query: Dict,
                        features_keyframe: Dict,
                        match_threshold: float) -> Dict:
        """Match using LightGlue (placeholder).
        
        Actual LightGlue implementation pending model setup.
        """
        # Falls back to BFMatcher for now
        return self._match_bfmatcher(features_query, features_keyframe,
                                     match_threshold, mutual_check=True)
    
    def _match_bfmatcher(self, features_query: Dict,
                        features_keyframe: Dict,
                        match_threshold: float,
                        mutual_check: bool = True) -> Dict:
        """Match using BFMatcher (fallback).
        
        Args:
            features_query, features_keyframe: Feature dicts
            match_threshold: Accepted ratio threshold (Lowe's ratio test)
            mutual_check: Apply bidirectional matching check
        
        Returns:
            Matches dict
        """
        descs_q = features_query['descriptors']
        descs_kf = features_keyframe['descriptors']
        
        kpts_q = features_query['keypoints']
        kpts_kf = features_keyframe['keypoints']
        H_q, W_q = features_query['image_shape']
        H_kf, W_kf = features_keyframe['image_shape']
        
        # Query -> Keyframe
        matches_qk = self.bf_matcher.knnMatch(descs_q, descs_kf, k=2)
        
        # Filter by Lowe's ratio test
        good_matches = []
        for match_pair in matches_qk:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < match_threshold * n.distance:
                    good_matches.append(m)
        
        # Optional: bidirectional check
        if mutual_check and len(good_matches) > 0:
            matches_kq = self.bf_matcher.knnMatch(descs_kf, descs_q, k=2)
            mutual_matches = set()
            
            for match_pair in matches_kq:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < match_threshold * n.distance:
                        # m.queryIdx = kf idx, m.trainIdx = q idx
                        # Check if this is reciprocal
                        for m_qk in good_matches:
                            if m_qk.queryIdx == m.trainIdx and m_qk.trainIdx == m.queryIdx:
                                mutual_matches.add((m_qk.queryIdx, m_qk.trainIdx))
            
            good_matches = [(q_idx, kf_idx) for q_idx, kf_idx in mutual_matches]
        else:
            good_matches = [(m.queryIdx, m.trainIdx) for m in good_matches]
        
        if len(good_matches) == 0:
            return {
                'matches': np.zeros((0, 2), dtype=np.int32),
                'scores': np.zeros((0,), dtype=np.float32),
                'matches_mkpts0': np.zeros((0, 2), dtype=np.float32),
                'matches_mkpts1': np.zeros((0, 2), dtype=np.float32),
            }
        
        # Convert to indices and pixel coordinates
        matches = np.array(good_matches, dtype=np.int32)
        
        # Denormalize pixel coordinates
        matches_mkpts0 = kpts_q[matches[:, 0]] * np.array([W_q, H_q], dtype=np.float32)
        matches_mkpts1 = kpts_kf[matches[:, 1]] * np.array([W_kf, H_kf], dtype=np.float32)
        
        # Scores as uniform (all matched)
        scores = np.ones(len(matches), dtype=np.float32)
        
        return {
            'matches': matches,
            'scores': scores,
            'matches_mkpts0': matches_mkpts0,
            'matches_mkpts1': matches_mkpts1,
        }
