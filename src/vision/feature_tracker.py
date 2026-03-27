from __future__ import annotations

import cv2
import numpy as np


class FeatureTracker:
    def __init__(
        self,
        max_corners: int = 600,
        quality_level: float = 0.01,
        min_distance: float = 8.0,
        lk_win_size: int = 21,
        lk_max_level: int = 3,
    ) -> None:
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.lk_params = dict(
            winSize=(lk_win_size, lk_win_size),
            maxLevel=lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def detect(self, gray: np.ndarray) -> np.ndarray:
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
        )
        if pts is None:
            return np.empty((0, 2), dtype=np.float32)
        return pts.reshape(-1, 2).astype(np.float32)

    def track(self, prev_gray: np.ndarray, curr_gray: np.ndarray, prev_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if prev_pts.size == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        p0 = prev_pts.reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **self.lk_params)
        if p1 is None or st is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        st = st.reshape(-1).astype(bool)
        prev_in = prev_pts[st]
        curr_in = p1.reshape(-1, 2)[st]
        return prev_in.astype(np.float32), curr_in.astype(np.float32)
