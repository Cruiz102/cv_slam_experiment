#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path as FilePath
from typing import Optional

import cv2
import numpy as np
import rclpy
import torch
from depth_anything_3.api import DepthAnything3
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as NavPath
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


if "QT_QPA_FONTDIR" not in os.environ and os.path.isdir("/usr/share/fonts/truetype/dejavu"):
    os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"


def make_default_intrinsics(width: int, height: int) -> np.ndarray:
    fx = 0.9 * width
    fy = 0.9 * width
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


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
    q /= max(np.linalg.norm(q), 1e-12)
    return q


def image_msg_from_depth(depth: np.ndarray, frame_id: str, stamp) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(depth.shape[0])
    msg.width = int(depth.shape[1])
    msg.encoding = "32FC1"
    msg.is_bigendian = 0
    msg.step = int(depth.shape[1] * 4)
    msg.data = depth.astype(np.float32).tobytes()
    return msg


def pointcloud_from_depth(depth: np.ndarray, K: np.ndarray, frame_id: str, stamp, stride: int = 4) -> PointCloud2:
    h, w = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[0:h:stride, 0:w:stride]

    valid = np.isfinite(z) & (z > 0.0)
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = z[valid].astype(np.float32)

    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy

    points = np.column_stack((x, y, z)).astype(np.float32)
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return point_cloud2.create_cloud_xyz32(header, points.tolist())


class DA3PoseDepthNode(Node):
    def __init__(self) -> None:
        super().__init__("da3_pose_depth_node")

        self.declare_parameter("camera_id", 0)
        self.declare_parameter("camera_path", "/dev/video0")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("model_id", "depth-anything/DA3-LARGE-1.1")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("camera_frame_id", "da3_camera")
        self.declare_parameter("infer_fps", 2.0)
        self.declare_parameter("pc_stride", 4)
        self.declare_parameter("path_max_poses", 2000)
        self.declare_parameter("min_sharpness", 80.0)
        self.declare_parameter("min_feature_count", 120)
        self.declare_parameter("min_keyframe_motion_px", 12.0)
        self.declare_parameter("min_keyframe_interval_s", 0.5)
        self.declare_parameter("reference_dir", "data/reference_memory")
        self.declare_parameter("reference_poll_s", 0.5)
        self.declare_parameter("use_manual_reference_only", True)
        self.declare_parameter("force_cpu", False)
        self.declare_parameter("show_debug_window", False)

        camera_id = int(self.get_parameter("camera_id").value)
        camera_path = str(self.get_parameter("camera_path").value)
        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        self.model_id = str(self.get_parameter("model_id").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.camera_frame_id = str(self.get_parameter("camera_frame_id").value)
        infer_fps = float(self.get_parameter("infer_fps").value)
        self.pc_stride = int(self.get_parameter("pc_stride").value)
        self.path_max_poses = int(self.get_parameter("path_max_poses").value)
        self.min_sharpness = float(self.get_parameter("min_sharpness").value)
        self.min_feature_count = int(self.get_parameter("min_feature_count").value)
        self.min_keyframe_motion_px = float(self.get_parameter("min_keyframe_motion_px").value)
        self.min_keyframe_interval_s = float(self.get_parameter("min_keyframe_interval_s").value)
        self.reference_dir = FilePath(str(self.get_parameter("reference_dir").value))
        self.reference_poll_s = float(self.get_parameter("reference_poll_s").value)
        self.use_manual_reference_only = bool(self.get_parameter("use_manual_reference_only").value)
        self.force_cpu = bool(self.get_parameter("force_cpu").value)
        self.show_debug = bool(self.get_parameter("show_debug_window").value)

        self.depth_pub = self.create_publisher(Image, "da3/depth", 10)
        self.pose_pub = self.create_publisher(PoseStamped, "da3/pose", 10)
        self.path_pub = self.create_publisher(NavPath, "da3/path", 10)
        self.pc_pub = self.create_publisher(PointCloud2, "da3/pointcloud", 10)
        self.path_msg = NavPath()
        self.path_msg.header.frame_id = self.frame_id

        self.cap = self._open_camera(camera_id, camera_path, width, height)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Unable to read first webcam frame")

        h, w = frame.shape[:2]
        self.K = make_default_intrinsics(w, h)
        self.frame_buffer: deque[np.ndarray] = deque(maxlen=2)
        self.frame_buffer.append(frame)
        self.frame_h = int(h)
        self.frame_w = int(w)
        self.keyframe_bgr = frame.copy()
        self.keyframe_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.last_keyframe_time = time.monotonic()
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        self.last_reference_scan = 0.0
        self.reference_images_bgr: list[np.ndarray] = []
        self.reference_state_sig: tuple[tuple[str, int], ...] = tuple()
        self._load_manual_reference_if_available(force=True)

        self.device = torch.device("cpu" if self.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.get_logger().info(f"Loading DA3 model {self.model_id} on {self.device} ...")
        self.model = DepthAnything3.from_pretrained(self.model_id).to(device=self.device)
        self.get_logger().info("DA3 model loaded")

        self.timer = self.create_timer(1.0 / max(infer_fps, 0.2), self._tick)

    def _frame_quality(self, gray: np.ndarray) -> tuple[float, int]:
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=600, qualityLevel=0.01, minDistance=8)
        feature_count = 0 if pts is None else int(len(pts))
        return sharpness, feature_count

    def _estimate_motion_px(self, ref_gray: np.ndarray, cur_gray: np.ndarray) -> float:
        pts = cv2.goodFeaturesToTrack(ref_gray, maxCorners=500, qualityLevel=0.01, minDistance=8)
        if pts is None or len(pts) < 20:
            return 0.0

        nxt, st, _ = cv2.calcOpticalFlowPyrLK(ref_gray, cur_gray, pts, None)
        if nxt is None or st is None:
            return 0.0

        good = st.reshape(-1) == 1
        if int(np.count_nonzero(good)) < 20:
            return 0.0

        d = nxt[good].reshape(-1, 2) - pts[good].reshape(-1, 2)
        return float(np.median(np.linalg.norm(d, axis=1)))

    def _maybe_update_keyframe(self, frame: np.ndarray, gray: np.ndarray) -> None:
        sharpness, feature_count = self._frame_quality(gray)
        if sharpness < self.min_sharpness or feature_count < self.min_feature_count:
            return

        elapsed = time.monotonic() - self.last_keyframe_time
        if elapsed < self.min_keyframe_interval_s:
            return

        motion_px = self._estimate_motion_px(self.keyframe_gray, gray)
        if motion_px < self.min_keyframe_motion_px:
            return

        self.keyframe_bgr = frame.copy()
        self.keyframe_gray = gray.copy()
        self.last_keyframe_time = time.monotonic()
        self.get_logger().debug(
            f"Updated keyframe: sharp={sharpness:.1f}, feats={feature_count}, motion={motion_px:.2f}px"
        )

    def _load_manual_reference_if_available(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self.last_reference_scan) < self.reference_poll_s:
            return
        self.last_reference_scan = now

        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        files: list[FilePath] = []
        for ext in exts:
            files.extend(self.reference_dir.glob(ext))
        if not files:
            self.reference_images_bgr = []
            self.reference_state_sig = tuple()
            return

        files = sorted(files, key=lambda p: p.stat().st_mtime)
        state_sig = tuple((str(p), int(p.stat().st_mtime_ns)) for p in files)
        if not force and state_sig == self.reference_state_sig:
            return

        loaded: list[np.ndarray] = []
        for p in files:
            ref = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if ref is None:
                self.get_logger().warning(f"Failed to load reference image: {p}")
                continue
            if ref.shape[0] != self.frame_h or ref.shape[1] != self.frame_w:
                ref = cv2.resize(ref, (self.frame_w, self.frame_h), interpolation=cv2.INTER_AREA)
            loaded.append(ref)

        if loaded:
            # Keep backward compatibility for code paths that still reference a keyframe image.
            self.keyframe_bgr = loaded[-1]
            self.keyframe_gray = cv2.cvtColor(self.keyframe_bgr, cv2.COLOR_BGR2GRAY)
            self.last_keyframe_time = now

        self.reference_images_bgr = loaded
        self.reference_state_sig = state_sig
        self.get_logger().info(f"Loaded {len(loaded)} manual reference image(s) from {self.reference_dir}")

    def _open_camera(self, camera_id: int, camera_path: str, width: int, height: int) -> cv2.VideoCapture:
        candidates: list[object] = [camera_path, camera_id, f"/dev/video{camera_id}"]
        for cand in candidates:
            cap = cv2.VideoCapture(cand, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    self.get_logger().info(f"Opened camera source: {cand}")
                    return cap
            cap.release()
        raise RuntimeError("Unable to open camera")

    def _switch_to_cpu(self) -> None:
        if self.device.type == "cpu":
            return
        self.get_logger().warning("Switching DA3 model to CPU after CUDA OOM")
        self.device = torch.device("cpu")
        self.model = self.model.to(device=self.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _tick(self) -> None:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.get_logger().warning("Camera read failed")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._load_manual_reference_if_available()
        if not self.use_manual_reference_only:
            self._maybe_update_keyframe(frame, gray)

        # Consume all manual references (if available) and always append the latest frame.
        if self.reference_images_bgr:
            images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in self.reference_images_bgr]
            images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            images = [
                cv2.cvtColor(self.keyframe_bgr, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            ]

        try:
            pred = self.model.inference(images)
        except Exception as exc:
            if "CUDA out of memory" in str(exc):
                self._switch_to_cpu()
                try:
                    pred = self.model.inference(images)
                except Exception as retry_exc:
                    self.get_logger().error(f"DA3 inference failed after CPU fallback: {retry_exc}")
                    return
            else:
                self.get_logger().error(f"DA3 inference failed: {exc}")
                return

        depth = pred.depth[-1].astype(np.float32)

        if pred.extrinsics is not None and len(pred.extrinsics) > 0:
            ext = pred.extrinsics[-1]
            if ext.shape == (3, 4):
                ext4 = np.eye(4, dtype=np.float64)
                ext4[:3, :4] = ext
            else:
                ext4 = ext.astype(np.float64)
        else:
            ext4 = np.eye(4, dtype=np.float64)

        R_wc = ext4[:3, :3].T
        t_wc = -R_wc @ ext4[:3, 3]
        q = rotmat_to_quat(R_wc)

        stamp = self.get_clock().now().to_msg()

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = float(t_wc[0])
        pose.pose.position.y = float(t_wc[1])
        pose.pose.position.z = float(t_wc[2])
        pose.pose.orientation.w = float(q[0])
        pose.pose.orientation.x = float(q[1])
        pose.pose.orientation.y = float(q[2])
        pose.pose.orientation.z = float(q[3])
        self.pose_pub.publish(pose)

        self.path_msg.header.stamp = stamp
        self.path_msg.poses.append(pose)
        if self.path_max_poses > 0 and len(self.path_msg.poses) > self.path_max_poses:
            self.path_msg.poses = self.path_msg.poses[-self.path_max_poses :]
        self.path_pub.publish(self.path_msg)

        depth_msg = image_msg_from_depth(depth, self.camera_frame_id, stamp)
        self.depth_pub.publish(depth_msg)

        pc_msg = pointcloud_from_depth(depth, self.K, self.camera_frame_id, stamp, stride=self.pc_stride)
        self.pc_pub.publish(pc_msg)

        if self.show_debug:
            d = depth.copy()
            d = d - np.nanmin(d)
            d = d / max(float(np.nanmax(d)), 1e-6)
            dvis = (255.0 * d).astype(np.uint8)
            dvis = cv2.applyColorMap(dvis, cv2.COLORMAP_INFERNO)
            cv2.imshow("DA3 Depth", dvis)
            cv2.waitKey(1)

    def destroy_node(self) -> bool:
        try:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node: Optional[DA3PoseDepthNode] = None
    try:
        node = DA3PoseDepthNode()
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
