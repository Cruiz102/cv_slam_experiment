from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.config.calibration import default_frame_conventions, load_euroc_cam_calibration
from src.eval.trajectory_metrics import write_trajectory_csv
from src.fusion.ekf_loose_vio import LooseVioFusion
from src.inertial.imu_propagator import ImuPropagator
from src.inertial.imu_state import ImuState
from src.io.euroc_loader import imu_between, load_euroc_sequence, load_gray_image
from src.vision.feature_tracker import FeatureTracker
from src.vision.two_view_geometry import TwoViewEstimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal monocular+IMU baseline on EuRoC-like sequence")
    parser.add_argument("--seq", type=Path, required=True, help="Path to EuRoC sequence root")
    parser.add_argument(
        "--cam-yaml",
        type=Path,
        default=None,
        help="Path to cam0 sensor.yaml (defaults to seq/mav0/cam0/sensor.yaml)",
    )
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--out", type=Path, default=Path("outputs/trajectory_estimate.csv"))
    return parser.parse_args()


def undistort(gray: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    return cv2.undistort(gray, K, dist)


def main() -> None:
    args = parse_args()
    seq = load_euroc_sequence(args.seq)

    cam_yaml = args.cam_yaml or (args.seq / "mav0" / "cam0" / "sensor.yaml")
    cam_calib = load_euroc_cam_calibration(cam_yaml)
    K = cam_calib.K
    dist = cam_calib.dist

    tracker = FeatureTracker()
    two_view = TwoViewEstimator(K)

    conv = default_frame_conventions()
    imu_state = ImuState()
    propagator = ImuPropagator(gravity_world=conv.gravity_world)
    fusion = LooseVioFusion()

    n = min(args.max_frames, len(seq.frames))
    if n < 2:
        raise ValueError("Need at least 2 camera frames")

    timestamps: list[float] = []
    positions: list[np.ndarray] = []

    frame0 = seq.frames[0]
    prev_t = frame0.timestamp_s
    prev_gray = undistort(load_gray_image(frame0.image_path), K, dist)
    prev_pts = tracker.detect(prev_gray)

    timestamps.append(prev_t)
    positions.append(imu_state.p_wb.copy())

    for i in range(1, n):
        frame = seq.frames[i]
        curr_t = frame.timestamp_s
        curr_gray = undistort(load_gray_image(frame.image_path), K, dist)

        # 1) IMU prediction between camera timestamps.
        imu_seq = imu_between(seq.imu, prev_t, curr_t)
        if len(imu_seq) >= 2:
            imu_state = propagator.propagate(imu_state, imu_seq)

        # 2) Visual tracking and relative motion from camera.
        pts0, pts1 = tracker.track(prev_gray, curr_gray, prev_pts)
        if len(pts0) >= 8:
            try:
                dR_vis, dt_vis, inlier = two_view.estimate_relative_pose(pts0, pts1)
                imu_state = fusion.update_with_visual_delta(imu_state, dR_vis, dt_vis)
                pts1 = pts1[inlier]
            except ValueError:
                pass

        if len(pts1) < 80:
            prev_pts = tracker.detect(curr_gray)
        else:
            prev_pts = pts1

        timestamps.append(curr_t)
        positions.append(imu_state.p_wb.copy())

        prev_gray = curr_gray
        prev_t = curr_t

    write_trajectory_csv(args.out, timestamps, positions)
    print(f"Wrote trajectory: {args.out}")
    print(f"Frames processed: {n}")


if __name__ == "__main__":
    main()
