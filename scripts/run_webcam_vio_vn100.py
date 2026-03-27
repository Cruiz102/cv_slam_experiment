from __future__ import annotations

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import serial

if "QT_QPA_FONTDIR" not in os.environ and os.path.isdir("/usr/share/fonts/truetype/dejavu"):
    os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"

import cv2

from src.inertial.madgwick import MadgwickAHRS
from src.inertial.vn100_serial import VN100Serial
from src.vision.feature_tracker import FeatureTracker
from src.vision.two_view_geometry import TwoViewEstimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run webcam visual odometry fused with VN100 IMU orientation")
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--camera-path", type=str, default="", help="Optional camera device path, e.g. /dev/video0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--step-scale", type=float, default=0.05, help="Monocular translation scale factor")
    parser.add_argument("--min-inliers", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0.08, help="Madgwick filter gain")
    parser.add_argument("--imu-orient-weight", type=float, default=0.8, help="Weight of IMU orientation in [0,1]")
    parser.add_argument("--plot-every", type=int, default=2)
    parser.add_argument("--configure-vnymr", action="store_true")
    return parser.parse_args()


def make_default_intrinsics(width: int, height: int) -> np.ndarray:
    fx = 0.9 * width
    fy = 0.9 * width
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def setup_3d_plot() -> tuple[plt.Figure, plt.Axes]:
    plt.ion()
    fig = plt.figure("VIO 3D Trajectory", figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Webcam + VN100 Fused Trajectory")
    return fig, ax


def update_3d_plot(fig: plt.Figure, ax: plt.Axes, trajectory: list[np.ndarray]) -> None:
    if not trajectory:
        return

    pts = np.asarray(trajectory, dtype=np.float64)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    ax.cla()
    ax.plot3D(x, y, z, color="cyan", linewidth=2.0)
    ax.scatter(x[-1], y[-1], z[-1], color="red", s=30)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Webcam + VN100 Fused Trajectory")

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


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def project_to_so3(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(R)
    out = u @ vt
    if np.linalg.det(out) < 0.0:
        u[:, -1] *= -1.0
        out = u @ vt
    return out


def configure_vnymr(sensor: VN100Serial) -> None:
    sensor.send_ascii_command("VNWRG,06,14")
    sensor.send_ascii_command("VNWRG,07,1")


def open_camera(camera_id: int, camera_path: str, width: int, height: int) -> tuple[cv2.VideoCapture, str]:
    candidates: list[object] = []
    if camera_path:
        candidates.append(camera_path)
    candidates.append(camera_id)
    candidates.append(f"/dev/video{camera_id}")

    for cand in candidates:
        cap = cv2.VideoCapture(cand, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, str(cand)
        cap.release()

    raise RuntimeError("Unable to open webcam with provided id/path")


def main() -> None:
    args = parse_args()

    imu_weight = float(np.clip(args.imu_orient_weight, 0.0, 1.0))

    sensor = VN100Serial(port=args.port, baudrate=args.baud)
    if args.configure_vnymr:
        configure_vnymr(sensor)
        time.sleep(0.2)

    cap, opened_cam = open_camera(args.camera_id, args.camera_path, args.width, args.height)
    print(f"Opened camera source: {opened_cam}")

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        sensor.close()
        raise RuntimeError("Unable to read first webcam frame")

    gray_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray_prev.shape
    K = make_default_intrinsics(w, h)

    tracker = FeatureTracker(max_corners=900)
    estimator = TwoViewEstimator(K)
    pts_prev = tracker.detect(gray_prev)

    madgwick = MadgwickAHRS(beta=args.beta)
    t_imu_prev = time.monotonic()

    R_vis = np.eye(3, dtype=np.float64)
    R_fused = np.eye(3, dtype=np.float64)
    p_fused = np.zeros(3, dtype=np.float64)
    trajectory = [p_fused.copy()]

    fig, ax = setup_3d_plot()
    frame_idx = 0

    try:
        while True:
            try:
                pkt = sensor.read_vnymr()
            except serial.SerialException:
                pkt = None
            if pkt is not None:
                t_imu = time.monotonic()
                dt = t_imu - t_imu_prev
                t_imu_prev = t_imu
                if dt <= 0.0 or dt > 0.5:
                    dt = 0.01
                madgwick.update(
                    gyro_rad_s=np.radians(pkt.gyro_xyz_deg_s),
                    accel=pkt.accel_xyz,
                    mag=pkt.mag_xyz,
                    dt=dt,
                )

            ok, frame = cap.read()
            if not ok or frame is None:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pts0, pts1 = tracker.track(gray_prev, gray, pts_prev)

            inliers_count = 0
            if len(pts0) >= 8:
                try:
                    dR_vis, dt_dir, inlier_mask = estimator.estimate_relative_pose(pts0, pts1)
                    inliers_count = int(np.count_nonzero(inlier_mask))

                    if inliers_count >= args.min_inliers:
                        R_vis = R_vis @ dR_vis
                        R_imu = quat_to_rotmat(madgwick.q)
                        R_blend = (1.0 - imu_weight) * R_vis + imu_weight * R_imu
                        R_fused = project_to_so3(R_blend)
                        p_fused = p_fused + R_fused @ (dt_dir * args.step_scale)
                        trajectory.append(p_fused.copy())

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

            rpy = madgwick.euler_deg()
            cv2.putText(frame, f"tracks: {len(pts_prev)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 240, 50), 2)
            cv2.putText(frame, f"inliers: {inliers_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 240, 50), 2)
            cv2.putText(frame, f"imu rpy: {rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 240, 50), 2)
            cv2.putText(frame, f"pos xyz: {p_fused[0]:.2f}, {p_fused[1]:.2f}, {p_fused[2]:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 240, 50), 2)

            cv2.imshow("Webcam + VN100 VIO", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                R_vis = np.eye(3, dtype=np.float64)
                R_fused = np.eye(3, dtype=np.float64)
                p_fused = np.zeros(3, dtype=np.float64)
                trajectory = [p_fused.copy()]
                update_3d_plot(fig, ax, trajectory)

            gray_prev = gray

    finally:
        cap.release()
        sensor.close()
        cv2.destroyAllWindows()
        plt.close(fig)


if __name__ == "__main__":
    main()
