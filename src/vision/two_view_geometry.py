from __future__ import annotations

import cv2
import numpy as np


class TwoViewEstimator:
    def __init__(self, K: np.ndarray, ransac_thresh: float = 1.0, ransac_prob: float = 0.999) -> None:
        self.K = K.astype(np.float64)
        self.ransac_thresh = ransac_thresh
        self.ransac_prob = ransac_prob

    def estimate_relative_pose(self, pts0: np.ndarray, pts1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(pts0) < 8 or len(pts1) < 8:
            raise ValueError("Need at least 8 correspondences")

        E, inlier_mask = cv2.findEssentialMat(
            pts0,
            pts1,
            self.K,
            method=cv2.RANSAC,
            prob=self.ransac_prob,
            threshold=self.ransac_thresh,
        )
        if E is None:
            raise ValueError("Essential matrix estimation failed")

        _, R, t, pose_mask = cv2.recoverPose(E, pts0, pts1, self.K)
        if pose_mask is None:
            pose_mask = np.zeros((len(pts0), 1), dtype=np.uint8)

        mask = pose_mask.reshape(-1).astype(bool)
        return R.astype(np.float64), t.reshape(3).astype(np.float64), mask
