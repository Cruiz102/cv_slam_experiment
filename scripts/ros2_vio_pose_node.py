#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
try:
    import rclpy
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from nav_msgs.msg import Odometry, Path
    from rclpy.node import Node
    from sensor_msgs.msg import Imu
    from std_msgs.msg import Float32, Int32
    from tf2_ros import TransformBroadcaster
except ModuleNotFoundError as exc:
    if exc.name in {"rclpy", "geometry_msgs", "nav_msgs", "tf2_ros", "sensor_msgs", "std_msgs"}:
        suggested_setup = "/opt/ros/jazzy/setup.bash"
        if not os.path.exists(suggested_setup):
            suggested_setup = "/opt/ros/<distro>/setup.bash"
        msg = (
            "ROS 2 Python packages are not available in this interpreter.\n"
            "Use one of these options:\n"
            "  1) Source ROS 2 before system Python, or\n"
            "  2) (Recommended for this project) Activate .venv then source ROS 2 so OpenCV/Numpy versions stay compatible.\n"
            "Recommended command sequence:\n"
            "  source .venv/bin/activate\n"
            f"  source {suggested_setup}\n"
            "  cd /home/cesar/cv_slam_experiment\n"
            "  export PYTHONPATH=.:$PYTHONPATH\n"
            "  python3 scripts/ros2_vio_pose_node.py --ros-args -p port:=/dev/ttyUSB0 -p camera_path:=/dev/video0 -p configure_vnymr:=true\n"
        )
        print(msg, file=sys.stderr)
        raise SystemExit(2)
    raise

from src.inertial.madgwick import MadgwickAHRS
from src.inertial.vn100_serial import VN100Serial
from src.vision.feature_tracker import FeatureTracker
from src.vision.two_view_geometry import TwoViewEstimator


if "QT_QPA_FONTDIR" not in os.environ and os.path.isdir("/usr/share/fonts/truetype/dejavu"):
    os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"


def make_default_intrinsics(width: int, height: int) -> np.ndarray:
    fx = 0.9 * width
    fy = 0.9 * width
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def project_to_so3(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(R)
    out = u @ vt
    if np.linalg.det(out) < 0.0:
        u[:, -1] *= -1.0
        out = u @ vt
    return out


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


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    return q


def slerp_quat(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical interpolation between two unit quaternions."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    q0 = q0 / max(np.linalg.norm(q0), 1e-12)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return q / max(np.linalg.norm(q), 1e-12)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s0 = np.cos(theta) - dot * sin_theta / max(sin_theta_0, 1e-12)
    s1 = sin_theta / max(sin_theta_0, 1e-12)
    q = s0 * q0 + s1 * q1
    return q / max(np.linalg.norm(q), 1e-12)


def open_camera(camera_id: int, camera_path: str, width: int, height: int) -> tuple[cv2.VideoCapture, str]:
    candidates: list[object] = []
    if camera_path:
        candidates.append(camera_path)
    candidates.append(camera_id)
    candidates.append(f"/dev/video{camera_id}")

    # Try explicit V4L2 first, then generic backend fallback.
    backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    last_err = ""
    for cand in candidates:
        for backend in backends:
            cap = cv2.VideoCapture(cand, backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    return cap, f"{cand} (backend={backend})"
                last_err = f"opened but no frames from {cand} with backend {backend}"
            else:
                last_err = f"cannot open {cand} with backend {backend}"
            cap.release()

    raise RuntimeError(f"Unable to open webcam with provided id/path. Last error: {last_err}")


class VioPoseNode(Node):
    def __init__(self) -> None:
        super().__init__("vio_pose_node")

        self.declare_parameter("use_vectornav_ros_imu", True)
        self.declare_parameter("imu_topic", "/vectornav/imu")
        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("baud", 115200)
        self.declare_parameter("camera_id", 0)
        self.declare_parameter("camera_path", "")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("step_scale", 0.05)
        self.declare_parameter("min_inliers", 20)
        self.declare_parameter("beta", 0.08)
        self.declare_parameter("imu_orient_weight", 0.98)
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("configure_vnymr", False)
        self.declare_parameter("max_path_length", 2000)
        self.declare_parameter("show_debug_window", False)
        self.declare_parameter("publish_tracking_status", True)
        self.declare_parameter("enable_relocalization_correction", True)
        self.declare_parameter("relocalization_pose_topic", "vio/relocalized_pose")
        self.declare_parameter("relocalization_correction_alpha", 0.5)
        self.declare_parameter("relocalization_min_inliers", 10)

        self.use_vectornav_ros_imu = bool(self.get_parameter("use_vectornav_ros_imu").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        port = self.get_parameter("port").value
        baud = int(self.get_parameter("baud").value)
        camera_id = int(self.get_parameter("camera_id").value)
        camera_path = str(self.get_parameter("camera_path").value)
        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        self.step_scale = float(self.get_parameter("step_scale").value)
        self.min_inliers = int(self.get_parameter("min_inliers").value)
        self.beta = float(self.get_parameter("beta").value)
        self.imu_weight = float(np.clip(self.get_parameter("imu_orient_weight").value, 0.0, 1.0))
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.child_frame_id = str(self.get_parameter("child_frame_id").value)
        self.configure_vnymr = bool(self.get_parameter("configure_vnymr").value)
        self.max_path_length = int(self.get_parameter("max_path_length").value)
        self.show_debug_window = bool(self.get_parameter("show_debug_window").value)
        self.publish_tracking_status = bool(self.get_parameter("publish_tracking_status").value)
        self.enable_relocalization_correction = bool(self.get_parameter("enable_relocalization_correction").value)
        self.relocalization_pose_topic = str(self.get_parameter("relocalization_pose_topic").value)
        self.relocalization_correction_alpha = float(
            np.clip(self.get_parameter("relocalization_correction_alpha").value, 0.0, 1.0)
        )
        self.relocalization_min_inliers = int(self.get_parameter("relocalization_min_inliers").value)

        self.sensor: Optional[VN100Serial] = None
        self.last_imu_msg: Optional[Imu] = None
        if self.use_vectornav_ros_imu:
            self.create_subscription(Imu, self.imu_topic, self._on_imu, 50)
            self.get_logger().info(f"Using ROS IMU topic: {self.imu_topic}")
        else:
            self.sensor = VN100Serial(port=port, baudrate=baud, timeout_s=0.01)
            if self.configure_vnymr:
                self.sensor.send_ascii_command("VNWRG,06,14")
                self.sensor.send_ascii_command("VNWRG,07,1")
                time.sleep(0.2)
            self.get_logger().info("Using direct serial IMU mode")

        self.cap, opened_src = open_camera(camera_id, camera_path, width, height)
        self.get_logger().info(f"Opened camera source: {opened_src}")

        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Unable to read first webcam frame")

        self.gray_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = self.gray_prev.shape
        K = make_default_intrinsics(w, h)

        self.tracker = FeatureTracker(max_corners=900)
        self.estimator = TwoViewEstimator(K)
        self.pts_prev = self.tracker.detect(self.gray_prev)

        self.madgwick = MadgwickAHRS(beta=self.beta)
        self.t_imu_prev = time.monotonic()

        self.R_vis = np.eye(3, dtype=np.float64)
        self.R_fused = np.eye(3, dtype=np.float64)
        self.R_imu = np.eye(3, dtype=np.float64)
        self.p_fused = np.zeros(3, dtype=np.float64)

        self.path_msg = Path()
        self.path_msg.header.frame_id = self.frame_id
        self.path_deque: deque[PoseStamped] = deque(maxlen=self.max_path_length)

        self.pose_pub = self.create_publisher(PoseStamped, "vio/pose", 10)
        self.odom_pub = self.create_publisher(Odometry, "vio/odom", 10)
        self.path_pub = self.create_publisher(Path, "vio/path", 10)
        self.inlier_pub = self.create_publisher(Int32, "vio/inlier_count", 10)
        self.tracking_quality_pub = self.create_publisher(Float32, "vio/tracking_quality", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        if self.enable_relocalization_correction:
            self.create_subscription(
                PoseStamped,
                self.relocalization_pose_topic,
                self._on_relocalized_pose,
                10,
            )
            self.get_logger().info(
                f"Relocalization correction enabled (topic={self.relocalization_pose_topic}, alpha={self.relocalization_correction_alpha:.2f})"
            )

        self.timer = self.create_timer(1.0 / 30.0, self._tick)

    def _on_imu(self, msg: Imu) -> None:
        self.last_imu_msg = msg

    def _tick(self) -> None:
        pkt = None
        if self.use_vectornav_ros_imu:
            if self.last_imu_msg is not None:
                q = self.last_imu_msg.orientation
                qn = np.array([q.w, q.x, q.y, q.z], dtype=np.float64)
                if np.linalg.norm(qn) > 1e-8:
                    qn = qn / np.linalg.norm(qn)
                    self.R_imu = quat_to_rotmat(qn)
        else:
            try:
                if self.sensor is not None:
                    pkt = self.sensor.read_vnymr()
            except Exception:
                pkt = None

            if pkt is not None:
                t_imu = time.monotonic()
                dt = t_imu - self.t_imu_prev
                self.t_imu_prev = t_imu
                if dt <= 0.0 or dt > 0.5:
                    dt = 0.01
                self.madgwick.update(
                    gyro_rad_s=np.radians(pkt.gyro_xyz_deg_s),
                    accel=pkt.accel_xyz,
                    mag=pkt.mag_xyz,
                    dt=dt,
                )
                self.R_imu = quat_to_rotmat(self.madgwick.q)

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.get_logger().warning("Webcam frame read failed")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pts0, pts1 = self.tracker.track(self.gray_prev, gray, self.pts_prev)

        inliers_count = 0
        if len(pts0) >= 8:
            try:
                dR_vis, dt_dir, inlier_mask = self.estimator.estimate_relative_pose(pts0, pts1)
                inliers_count = int(np.count_nonzero(inlier_mask))

                if inliers_count >= self.min_inliers:
                    self.R_vis = self.R_vis @ dR_vis
                    self.p_fused = self.p_fused + self.R_fused @ (dt_dir * self.step_scale)

                pts1 = pts1[inlier_mask]
            except ValueError:
                pass

        # Fuse IMU/vision orientation each tick, not only on successful visual updates.
        R_blend = (1.0 - self.imu_weight) * self.R_vis + self.imu_weight * self.R_imu
        self.R_fused = project_to_so3(R_blend)

        if len(pts1) < 120:
            self.pts_prev = self.tracker.detect(gray)
        else:
            self.pts_prev = pts1

        self.gray_prev = gray

        if self.show_debug_window:
            for p in self.pts_prev[:200]:
                cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, f"tracks: {len(self.pts_prev)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 240, 50), 2)
            cv2.putText(frame, f"inliers: {inliers_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 240, 50), 2)
            if self.use_vectornav_ros_imu and self.last_imu_msg is not None:
                q = self.last_imu_msg.orientation
                cv2.putText(frame, f"imu q(wxyz): {q.w:.2f} {q.x:.2f} {q.y:.2f} {q.z:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 240, 50), 2)
            else:
                rpy = self.madgwick.euler_deg()
                cv2.putText(frame, f"imu rpy: {rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 240, 50), 2)
            cv2.putText(frame, f"imu_w: {self.imu_weight:.2f}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 240, 50), 2)
            cv2.imshow("VIO Debug", frame)
            cv2.waitKey(1)

        self._publish_ros(inliers_count)

    def _on_relocalized_pose(self, msg: PoseStamped) -> None:
        """Blend relocalization correction into current VIO estimate."""
        q_obs = np.array(
            [
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
            ],
            dtype=np.float64,
        )
        if np.linalg.norm(q_obs) < 1e-8:
            self.get_logger().warning("Received relocalized pose with invalid quaternion")
            return

        q_est = rotmat_to_quat(self.R_fused)
        q_new = slerp_quat(q_est, q_obs, self.relocalization_correction_alpha)
        self.R_fused = quat_to_rotmat(q_new)
        self.R_vis = self.R_fused.copy()

        p_obs = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float64,
        )
        a = self.relocalization_correction_alpha
        self.p_fused = (1.0 - a) * self.p_fused + a * p_obs
        self.get_logger().info("Applied relocalization correction to VIO state")

    def _publish_ros(self, inliers_count: int) -> None:
        stamp = self.get_clock().now().to_msg()
        q = rotmat_to_quat(self.R_fused)

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = float(self.p_fused[0])
        pose.pose.position.y = float(self.p_fused[1])
        pose.pose.position.z = float(self.p_fused[2])
        pose.pose.orientation.w = float(q[0])
        pose.pose.orientation.x = float(q[1])
        pose.pose.orientation.y = float(q[2])
        pose.pose.orientation.z = float(q[3])
        self.pose_pub.publish(pose)

        odom = Odometry()
        odom.header = pose.header
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose = pose.pose
        self.odom_pub.publish(odom)

        self.path_deque.append(pose)
        self.path_msg.header.stamp = stamp
        self.path_msg.header.frame_id = self.frame_id
        self.path_msg.poses = list(self.path_deque)
        self.path_pub.publish(self.path_msg)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = self.frame_id
        tf_msg.child_frame_id = self.child_frame_id
        tf_msg.transform.translation.x = pose.pose.position.x
        tf_msg.transform.translation.y = pose.pose.position.y
        tf_msg.transform.translation.z = pose.pose.position.z
        tf_msg.transform.rotation = pose.pose.orientation
        self.tf_broadcaster.sendTransform(tf_msg)

        if self.publish_tracking_status:
            inlier_msg = Int32()
            inlier_msg.data = int(inliers_count)
            self.inlier_pub.publish(inlier_msg)

            quality_msg = Float32()
            quality_msg.data = float(inliers_count) / float(max(self.relocalization_min_inliers, 1))
            self.tracking_quality_pub.publish(quality_msg)

    def destroy_node(self) -> bool:
        try:
            if self.cap is not None:
                self.cap.release()
            if self.sensor is not None:
                self.sensor.close()
            cv2.destroyAllWindows()
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node: Optional[VioPoseNode] = None
    try:
        node = VioPoseNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
