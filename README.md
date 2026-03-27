# cv_slam_experiment

Monocular visual-inertial SLAM experimentation in Python with OpenCV, ROS 2, and HLoc-style relocalization.

This repository now includes:
- Baseline VO/VIO scripts
- ROS 2 VIO pose publishing
- Offline relocalization map building
- Online ROS 2 relocalization triggering and correction
- Utility apps for map/reference image capture

## Implemented Components

### Core vision/inertial
- EuRoC-style loader (`cam0` + `imu0`)
- Sparse visual front-end (Shi-Tomasi + LK optical flow)
- Two-view geometry (`findEssentialMat`, `recoverPose`)
- IMU orientation integration (Madgwick)
- Loose visual-inertial fusion baseline

### Relocalization (HLoc-style)
- Global descriptor extraction (`dinov2`, `mixvpr` interface)
- Local features/matching (ORB + BF fallback, SuperPoint/LightGlue interfaces)
- Pose solve with PnP-RANSAC
- Persistent map format (keyframes, 3D points, intrinsics)
- Offline map builder and quality tools
- ROS 2 relocalization node with manual and auto triggers

### ROS 2 integration
- `scripts/ros2_vio_pose_node.py` publishes pose/odom/path/tf
- `scripts/ros2_relocalization_node.py` performs online relocalization
- VIO publishes tracking health (`/vio/inlier_count`, `/vio/tracking_quality`)
- VIO consumes corrected pose (`/vio/relocalized_pose`) and blends correction

## Repository Scripts (Most Used)

- `scripts/ros2_vio_pose_node.py` - Main ROS 2 VIO node
- `scripts/ros2_relocalization_node.py` - ROS 2 online relocalization node
- `scripts/build_map_offline.py` - Build relocalization map from images
- `scripts/capture_relocalization_map_images.py` - OpenCV map image capture app
- `scripts/test_relocalization.py` - Phase 1 module tests
- `scripts/test_map_building_v2.py` - Phase 2 workflow test

## Environment Setup

## 1) Python venv

```bash
cd /home/cesar/cv_slam_experiment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) ROS 2 shell setup (for ROS commands)

```bash
source /opt/ros/jazzy/setup.bash
```

In most runs you want both:

```bash
source /home/cesar/cv_slam_experiment/.venv/bin/activate
source /opt/ros/jazzy/setup.bash
export PYTHONPATH=/home/cesar/cv_slam_experiment:$PYTHONPATH
```

## Quick Validation Commands

```bash
cd /home/cesar/cv_slam_experiment
source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
python3 scripts/test_relocalization.py
python3 scripts/test_map_building_v2.py
```

## Phase 2: Build Relocalization Map

## A) Capture map images (new OpenCV app)

```bash
cd /home/cesar/cv_slam_experiment
source .venv/bin/activate
python3 scripts/capture_relocalization_map_images.py \
  --camera-path /dev/video0 \
  --output-dir data/mapping_images \
  --start-auto \
  --auto-interval 1.0 \
  --show-grid
```

App hotkeys:
- `s`: save one frame
- `a`: toggle auto-capture
- `g`: toggle grid
- `+` / `-`: adjust auto interval
- `q`: quit

## B) Build map

```bash
cd /home/cesar/cv_slam_experiment
source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
python3 scripts/build_map_offline.py \
  --image_dir data/mapping_images \
  --output_map_dir data/relocalization_map \
  --global_descriptor dinov2 \
  --device cuda
```

## C) Verify map files exist

```bash
ls -la data/relocalization_map
```

Expected files:
- `keyframes.pkl`
- `points_3d.npy`
- `intrinsics.npy`
- `metadata.json`

## Phase 3: ROS 2 VIO + Relocalization Bring-up

Use separate terminals.

## Terminal 1: VectorNav ROS driver (IMU publisher)

```bash
source /opt/ros/jazzy/setup.bash
cd /home/cesar/cv_slam_experiment/vendor/vectornav
source install/setup.bash
ros2 launch vectornav vectornav.launch.py
```

## Terminal 2: VIO node

```bash
cd /home/cesar/cv_slam_experiment
source .venv/bin/activate
source /opt/ros/jazzy/setup.bash
source /home/cesar/cv_slam_experiment/vendor/vectornav/install/setup.bash
export PYTHONPATH=$PWD:$PYTHONPATH
python3 scripts/ros2_vio_pose_node.py --ros-args \
  -p use_vectornav_ros_imu:=true \
  -p imu_topic:=/vectornav/imu \
  -p camera_path:=/dev/video0 \
  -p publish_tracking_status:=true \
  -p enable_relocalization_correction:=true \
  -p relocalization_pose_topic:=vio/relocalized_pose
```

## Terminal 3: Relocalization node

Set `image_topic` to a real camera topic in your system.
If `/camera/image_raw` is missing, use one from `ros2 topic list` (for example `/hydrus/rgb_camera/image_color`).

```bash
cd /home/cesar/cv_slam_experiment
source .venv/bin/activate
source /opt/ros/jazzy/setup.bash
export PYTHONPATH=$PWD:$PYTHONPATH
python3 scripts/ros2_relocalization_node.py --ros-args \
  -p map_dir:=data/relocalization_map \
  -p global_descriptor_type:=dinov2 \
  -p image_topic:=/camera/image_raw \
  -p auto_trigger_on_inlier_drop:=true \
  -p min_inlier_threshold:=10 \
  -p min_tracking_quality:=0.5
```

## Terminal 4: Monitoring and manual trigger

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic list
ros2 topic echo /vectornav/imu --once
ros2 topic echo /vio/inlier_count
ros2 topic echo /vio/tracking_quality
ros2 service call /vio/relocalize_trigger std_srvs/srv/Trigger "{}"
```

## Main ROS Topics

Published by VIO:
- `/vio/pose` (`geometry_msgs/PoseStamped`)
- `/vio/odom` (`nav_msgs/Odometry`)
- `/vio/path` (`nav_msgs/Path`)
- `/vio/inlier_count` (`std_msgs/Int32`)
- `/vio/tracking_quality` (`std_msgs/Float32`)
- `/tf` map -> base_link

Published by Relocalization:
- `/vio/relocalized_pose` (`geometry_msgs/PoseStamped`)

Service:
- `/vio/relocalize_trigger` (`std_srvs/srv/Trigger`)

## Common Failure Modes (Important)

## 1) IMU not being used

Symptom:
- VIO subscribes to `/vectornav/imu` but topic has no publisher.

Check:
```bash
source /opt/ros/jazzy/setup.bash
ros2 topic info /vectornav/imu -v
```

Fix:
- Start VectorNav launch first (Terminal 1 above).

## 2) "Map has no intrinsics set. PnP will fail."

Symptom:
- Relocalization node starts but cannot solve pose.

Root cause:
- Empty or incomplete map directory.

Fix:
- Rebuild map with `scripts/build_map_offline.py`.
- Confirm `intrinsics.npy` exists in `data/relocalization_map`.

## 3) Relocalization auto-trigger spam

Symptom:
- Repeated warnings: low inliers every cooldown interval.

Fix options:
- Ensure map is valid and non-empty.
- Ensure `image_topic` is correct and receiving images.
- Temporarily disable auto trigger for debugging:
```bash
-p auto_trigger_on_inlier_drop:=false
```
- Lower thresholds while testing:
```bash
-p min_inlier_threshold:=5 -p min_tracking_quality:=0.3
```

## 4) Wrong image topic

Symptom:
- Relocalization never succeeds, no useful matching.

Check available topics:
```bash
source /opt/ros/jazzy/setup.bash
ros2 topic list
```

Then set `-p image_topic:=<your_actual_topic>`.

## Optional: Depth Anything 3 ROS 2 Node

Install (already vendored):

```bash
cd /home/cesar/cv_slam_experiment
source .venv/bin/activate
pip install -e external/depth-anything-3 --no-deps
```

Run:

```bash
cd /home/cesar/cv_slam_experiment
source .venv/bin/activate
source /opt/ros/jazzy/setup.bash
export PYTHONPATH=$PWD:$PYTHONPATH
python3 scripts/ros2_da3_pose_depth_node.py --ros-args \
  -p camera_path:=/dev/video0 \
  -p model_id:=depth-anything/DA3-SMALL \
  -p infer_fps:=2.0
```

Publishes:
- `/da3/pose` (`geometry_msgs/PoseStamped`)
- `/da3/depth` (`sensor_msgs/Image`, `32FC1`)
- `/da3/pointcloud` (`sensor_msgs/PointCloud2`)

## Notes

- Monocular translation from `recoverPose` is direction-only without metric scale constraints.
- Keep descriptor type consistent between map build and relocalization runtime (`dinov2` recommended).
- For robust relocalization, collect diverse map views: different angles, distances, and slight illumination variation.
