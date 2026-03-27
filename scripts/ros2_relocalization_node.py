#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
from typing import Optional

import cv2
import numpy as np
try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float32, Int32
    from std_srvs.srv import Trigger
except ModuleNotFoundError as exc:
    if exc.name in {"rclpy", "geometry_msgs", "sensor_msgs", "std_msgs", "std_srvs"}:
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
            "  python3 scripts/ros2_relocalization_node.py\n"
        )
        print(msg, file=sys.stderr)
        raise SystemExit(2)
    raise

from src.relocalization import HLocPipeline


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


class RelocalizationNode(Node):
    """ROS 2 node for online relocalization with manual and auto triggering."""

    def __init__(self) -> None:
        super().__init__("relocalization_node")

        self.declare_parameter("enabled", True)
        self.declare_parameter("map_dir", "data/relocalization_map")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("global_descriptor_type", "dinov2")
        self.declare_parameter("top_k_retrieval", 10)
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("vio_pose_topic", "vio/pose")
        self.declare_parameter("vio_inlier_topic", "vio/inlier_count")
        self.declare_parameter("vio_tracking_quality_topic", "vio/tracking_quality")
        self.declare_parameter("relocalized_pose_topic", "vio/relocalized_pose")
        self.declare_parameter("auto_trigger_on_inlier_drop", True)
        self.declare_parameter("min_inlier_threshold", 10)
        self.declare_parameter("min_tracking_quality", 0.5)
        self.declare_parameter("trigger_cooldown_s", 2.0)

        self.enabled = bool(self.get_parameter("enabled").value)
        self.map_dir = str(self.get_parameter("map_dir").value)
        self.device = str(self.get_parameter("device").value)
        self.global_descriptor_type = str(self.get_parameter("global_descriptor_type").value)
        self.top_k_retrieval = int(self.get_parameter("top_k_retrieval").value)
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.vio_pose_topic = str(self.get_parameter("vio_pose_topic").value)
        self.vio_inlier_topic = str(self.get_parameter("vio_inlier_topic").value)
        self.vio_tracking_quality_topic = str(self.get_parameter("vio_tracking_quality_topic").value)
        self.relocalized_pose_topic = str(self.get_parameter("relocalized_pose_topic").value)
        self.auto_trigger_on_inlier_drop = bool(self.get_parameter("auto_trigger_on_inlier_drop").value)
        self.min_inlier_threshold = int(self.get_parameter("min_inlier_threshold").value)
        self.min_tracking_quality = float(self.get_parameter("min_tracking_quality").value)
        self.trigger_cooldown_s = float(self.get_parameter("trigger_cooldown_s").value)

        self.hloc: Optional[HLocPipeline] = None
        self.map_ready: bool = False
        self.latest_image_rgb: Optional[np.ndarray] = None
        self.latest_image_stamp = None
        self.latest_vio_pose: Optional[PoseStamped] = None
        self.last_inlier_count: int = 0
        self.last_tracking_quality: float = 0.0
        self.last_trigger_time = 0.0

        if self.enabled:
            self._init_pipeline()
        else:
            self.get_logger().warning("Relocalization node started with enabled=false")

        self.create_subscription(Image, self.image_topic, self._on_image, 10)
        self.create_subscription(PoseStamped, self.vio_pose_topic, self._on_vio_pose, 10)
        self.create_subscription(Int32, self.vio_inlier_topic, self._on_inlier_count, 10)
        self.create_subscription(Float32, self.vio_tracking_quality_topic, self._on_tracking_quality, 10)

        self.relocalized_pose_pub = self.create_publisher(PoseStamped, self.relocalized_pose_topic, 10)
        self.trigger_srv = self.create_service(Trigger, "vio/relocalize_trigger", self._on_trigger_relocalize)

        self.get_logger().info(
            "Relocalization node ready: "
            f"image_topic={self.image_topic}, map_dir={self.map_dir}, auto_trigger={self.auto_trigger_on_inlier_drop}"
        )

    def _init_pipeline(self) -> None:
        try:
            self.hloc = HLocPipeline(
                self.map_dir,
                device=self.device,
                global_descriptor_type=self.global_descriptor_type,
                config={"top_k_retrieval": self.top_k_retrieval},
            )

            num_kf = len(self.hloc.map) if self.hloc is not None else 0
            has_k = bool(self.hloc is not None and self.hloc.map.K is not None)
            has_pts = bool(self.hloc is not None and self.hloc.map.points_3d is not None)
            self.map_ready = bool(num_kf > 0 and has_k and has_pts)

            if not self.map_ready:
                self.hloc = None
                self.get_logger().error(
                    "Map is not ready for relocalization. "
                    f"map_dir={self.map_dir}, keyframes={num_kf}, "
                    f"intrinsics_set={has_k}, points3d_set={has_pts}. "
                    "Build/populate the map first (including intrinsics.npy)."
                )
            else:
                self.get_logger().info(
                    f"HLoc pipeline initialized (keyframes={num_kf}, intrinsics=ok, points3d=ok)"
                )
        except Exception as exc:
            self.hloc = None
            self.map_ready = False
            self.get_logger().error(f"Failed to initialize HLoc pipeline: {exc}")

    def _on_image(self, msg: Image) -> None:
        img = self._ros_image_to_rgb(msg)
        if img is None:
            return
        self.latest_image_rgb = img
        self.latest_image_stamp = msg.header.stamp

    def _on_vio_pose(self, msg: PoseStamped) -> None:
        self.latest_vio_pose = msg

    def _on_inlier_count(self, msg: Int32) -> None:
        self.last_inlier_count = int(msg.data)
        if self.auto_trigger_on_inlier_drop and self.last_inlier_count < self.min_inlier_threshold:
            self._try_auto_trigger(f"low inliers ({self.last_inlier_count})")

    def _on_tracking_quality(self, msg: Float32) -> None:
        self.last_tracking_quality = float(msg.data)
        if self.auto_trigger_on_inlier_drop and self.last_tracking_quality < self.min_tracking_quality:
            self._try_auto_trigger(f"low tracking quality ({self.last_tracking_quality:.2f})")

    def _try_auto_trigger(self, reason: str) -> None:
        if self.hloc is None:
            return
        now = time.monotonic()
        if now - self.last_trigger_time < self.trigger_cooldown_s:
            return
        self.last_trigger_time = now
        self.get_logger().warning(f"Auto relocalization trigger: {reason}")
        ok, msg = self._attempt_relocalization(source="auto")
        if not ok:
            self.get_logger().warning(msg)

    def _on_trigger_relocalize(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        ok, reason = self._attempt_relocalization(source="manual")
        response.success = ok
        response.message = reason
        return response

    def _attempt_relocalization(self, source: str) -> tuple[bool, str]:
        if not self.enabled:
            return False, "Node disabled"
        if self.hloc is None:
            return False, "HLoc pipeline/map not initialized"
        if self.latest_image_rgb is None:
            return False, "No image available yet"

        prior_R = None
        prior_p = None
        if self.latest_vio_pose is not None:
            q = self.latest_vio_pose.pose.orientation
            qn = np.array([q.w, q.x, q.y, q.z], dtype=np.float64)
            if np.linalg.norm(qn) > 1e-8:
                qn = qn / np.linalg.norm(qn)
                prior_R = quat_to_rotmat(qn)
            p = self.latest_vio_pose.pose.position
            prior_p = np.array([p.x, p.y, p.z], dtype=np.float64)

        try:
            result = self.hloc.relocalize(
                self.latest_image_rgb,
                prior_R=prior_R,
                prior_p=prior_p,
                return_debug=False,
            )
        except Exception as exc:
            return False, f"Relocalization pipeline exception: {exc}"

        if not result.get("success", False):
            return False, f"Relocalization failed: {result.get('reason', 'unknown')}"

        pose_w2c = result.get("pose_w2c")
        if pose_w2c is None:
            return False, "Relocalization returned no pose_w2c"

        # Convert world-to-camera pose to camera-in-world (for publishing map-frame pose).
        try:
            pose_c2w = np.linalg.inv(pose_w2c)
        except np.linalg.LinAlgError:
            return False, "Pose inversion failed"

        R = pose_c2w[:3, :3]
        t = pose_c2w[:3, 3]
        qwxyz = self._rotmat_to_quat(R)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = float(t[0])
        msg.pose.position.y = float(t[1])
        msg.pose.position.z = float(t[2])
        msg.pose.orientation.w = float(qwxyz[0])
        msg.pose.orientation.x = float(qwxyz[1])
        msg.pose.orientation.y = float(qwxyz[2])
        msg.pose.orientation.z = float(qwxyz[3])
        self.relocalized_pose_pub.publish(msg)

        inliers = int(result.get("num_inliers", 0))
        conf = float(result.get("confidence", 0.0))
        total_ms = float(result.get("total_time", 0.0))
        info = (
            f"{source} relocalization success: inliers={inliers}, confidence={conf:.2f}, "
            f"latency={total_ms:.1f} ms"
        )
        self.get_logger().info(info)
        return True, info

    @staticmethod
    def _ros_image_to_rgb(msg: Image) -> Optional[np.ndarray]:
        if msg.height <= 0 or msg.width <= 0:
            return None

        h = int(msg.height)
        w = int(msg.width)
        enc = msg.encoding.lower()

        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)

            if enc in {"bgr8", "rgb8"}:
                expected = h * w * 3
                if data.size < expected:
                    return None
                img = data[:expected].reshape((h, w, 3))
                if enc == "bgr8":
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img

            if enc == "mono8":
                expected = h * w
                if data.size < expected:
                    return None
                gray = data[:expected].reshape((h, w))
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        except Exception:
            return None

        return None

    @staticmethod
    def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
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
        return q / max(np.linalg.norm(q), 1e-12)


def main() -> None:
    rclpy.init()
    node: Optional[RelocalizationNode] = None
    try:
        node = RelocalizationNode()
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
