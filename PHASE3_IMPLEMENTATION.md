# PHASE 3 IMPLEMENTATION: Online Relocalization + ROS 2 Integration

Status: COMPLETE

## What was implemented

1. VIO node integration updates in scripts/ros2_vio_pose_node.py
- Added tracking health publishers:
  - vio/inlier_count (std_msgs/Int32)
  - vio/tracking_quality (std_msgs/Float32)
- Added relocalization correction subscriber:
  - vio/relocalized_pose (geometry_msgs/PoseStamped)
- Added correction blending into active VIO estimate:
  - Quaternion SLERP for orientation
  - Linear blend for position
- Added new parameters:
  - publish_tracking_status (bool)
  - enable_relocalization_correction (bool)
  - relocalization_pose_topic (string)
  - relocalization_correction_alpha (float 0..1)
  - relocalization_min_inliers (int)

2. New relocalization ROS 2 node in scripts/ros2_relocalization_node.py
- Subscribes to:
  - camera image topic (/camera/image_raw by default)
  - vio pose (vio/pose)
  - inlier status (vio/inlier_count)
  - tracking quality (vio/tracking_quality)
- Publishes:
  - corrected pose (vio/relocalized_pose)
- Provides manual trigger service:
  - /vio/relocalize_trigger (std_srvs/srv/Trigger)
- Supports automatic trigger on degradation:
  - low inlier count threshold
  - low tracking quality threshold
  - cooldown to avoid trigger spam
- Uses Phase 1/2 HLoc pipeline directly:
  - map loading from map_dir
  - relocalize(latest_image, prior_R, prior_p)

3. Runtime docs update in README.md
- Added complete Phase 3 launch instructions
- Added manual service trigger command
- Added topic list and behavior summary

## Verification performed

1. Static diagnostics
- get_errors run on:
  - scripts/ros2_vio_pose_node.py
  - scripts/ros2_relocalization_node.py
- Result: no editor/language errors

2. Syntax validation
- Command:
  source .venv/bin/activate && python3 -m py_compile scripts/ros2_vio_pose_node.py scripts/ros2_relocalization_node.py
- Result: success

## How to run

Terminal 1 (VIO):
source .venv/bin/activate
source /opt/ros/jazzy/setup.bash
export PYTHONPATH=.:$PYTHONPATH
python3 scripts/ros2_vio_pose_node.py --ros-args \
  -p camera_path:=/dev/video0 \
  -p publish_tracking_status:=true \
  -p enable_relocalization_correction:=true

Terminal 2 (Relocalization):
source .venv/bin/activate
source /opt/ros/jazzy/setup.bash
export PYTHONPATH=.:$PYTHONPATH
python3 scripts/ros2_relocalization_node.py --ros-args \
  -p map_dir:=data/relocalization_map \
  -p global_descriptor_type:=dinov2 \
  -p auto_trigger_on_inlier_drop:=true \
  -p min_inlier_threshold:=10 \
  -p min_tracking_quality:=0.5

Manual trigger:
source /opt/ros/jazzy/setup.bash
ros2 service call /vio/relocalize_trigger std_srvs/srv/Trigger "{}"

## Notes

- For best compatibility, use the same descriptor type in map building and relocalization node (dinov2 recommended in current setup).
- This Phase 3 integration avoids custom ROS message/service generation by using standard message types and one Trigger service.
- Auto-triggering is enabled by default but can be disabled with auto_trigger_on_inlier_drop:=false.
