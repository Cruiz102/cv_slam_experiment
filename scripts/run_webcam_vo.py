from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.vision.feature_tracker import FeatureTracker
from src.vision.two_view_geometry import TwoViewEstimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run monocular visual odometry from webcam and visualize trajectory")
    parser.add_argument("--camera-id", type=int, default=0, help="OpenCV webcam index")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--step-scale", type=float, default=0.05, help="Scale used to integrate monocular translation direction")
    parser.add_argument("--min-inliers", type=int, default=20)
    parser.add_argument("--plot-every", type=int, default=2, help="Update 3D trajectory plot every N frames")
    return parser.parse_args()


def make_default_intrinsics(width: int, height: int) -> np.ndarray:
    # Fallback intrinsics for webcam without calibration.
    fx = 0.9 * width
    fy = 0.9 * width
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def setup_3d_plot() -> tuple[plt.Figure, plt.Axes]:
    plt.ion()
    fig = plt.figure("VO 3D Trajectory", figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Webcam Visual Odometry Trajectory")
    return fig, ax


def update_3d_plot(fig: plt.Figure, ax: plt.Axes, trajectory: list[np.ndarray]) -> None:
    if not trajectory:
        return

    pts = np.asarray(trajectory, dtype=np.float64)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    ax.cla()
    ax.plot3D(x, y, z, color="lime", linewidth=2.0)
    ax.scatter(x[-1], y[-1], z[-1], color="red", s=30)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Webcam Visual Odometry Trajectory")

    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    span = np.max(maxs - mins)
    span = max(span, 1.0)
    center = (mins + maxs) / 2.0
    half = span / 2.0
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    fig.canvas.draw_idle()
    plt.pause(0.001)


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam")

    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Unable to read first webcam frame")

    gray_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray_prev.shape
    K = make_default_intrinsics(w, h)

    tracker = FeatureTracker(max_corners=900)
    estimator = TwoViewEstimator(K)

    pts_prev = tracker.detect(gray_prev)

    R_w = np.eye(3, dtype=np.float64)
    p_w = np.zeros(3, dtype=np.float64)
    trajectory = [p_w.copy()]
    fig, ax = setup_3d_plot()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pts0, pts1 = tracker.track(gray_prev, gray, pts_prev)

        inliers_count = 0
        if len(pts0) >= 8:
            try:
                dR, dt_dir, inlier_mask = estimator.estimate_relative_pose(pts0, pts1)
                inliers_count = int(np.count_nonzero(inlier_mask))

                if inliers_count >= args.min_inliers:
                    R_w = R_w @ dR
                    p_w = p_w + R_w @ (dt_dir * args.step_scale)
                    trajectory.append(p_w.copy())

                pts1 = pts1[inlier_mask]
            except ValueError:
                pass

        if len(pts1) < 120:
            pts_prev = tracker.detect(gray)
        else:
            pts_prev = pts1

        for p in pts_prev[:200]:
            cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 255, 255), -1, cv2.LINE_AA)

        frame_idx += 1
        if frame_idx % max(args.plot_every, 1) == 0:
            update_3d_plot(fig, ax, trajectory)

        cv2.putText(frame, f"tracks: {len(pts_prev)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 240, 50), 2)
        cv2.putText(frame, f"inliers: {inliers_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 240, 50), 2)
        cv2.putText(
            frame,
            f"pos (x,y,z): {p_w[0]:.2f}, {p_w[1]:.2f}, {p_w[2]:.2f}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (50, 240, 50),
            2,
        )

        cv2.imshow("Webcam VO", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            R_w = np.eye(3, dtype=np.float64)
            p_w = np.zeros(3, dtype=np.float64)
            trajectory = [p_w.copy()]
            update_3d_plot(fig, ax, trajectory)

        gray_prev = gray

    cap.release()
    cv2.destroyAllWindows()
    plt.close(fig)


if __name__ == "__main__":
    main()
