# Map Building Guide: Step-by-Step

## Overview

This guide walks through the complete process of collecting mapping data and building a relocalization map for your VIO system.

**Time to complete**: 20-30 minutes
**Difficulty**: Beginner to Intermediate
**Prerequisites**: Phase 1 implementation (HLoc modules)

## 1. Data Collection

### 1.1 Prepare Your Environment

Choose a test environment:
- **Best**: Textured indoor room with varied lighting (office, hallway)
- **Good**: Outdoor area with consistent lighting, some texture
- **Avoid**: Featureless walls, extreme lighting changes, motion blur

### 1.2 Collect Images

**Option A: Manual Mobile Phone**
```bash
# On your phone, take 30-50 images
# Requirements:
# - Pan horizontally and vertically
# - Move forwards and backwards slightly
# - Vary distance from features (0.5m to 5m)
# - Keep motion smooth
# - Avoid fast motion (causes blur)
# - Vary lighting conditions if possible

# Transfer to computer
mkdir -p data/my_mapping_images
# Copy images: data/my_mapping_images/
```

**Option B: Robot/Vehicle with Camera**
```bash
# If using your robotics platform:
ros2 run image_view image_saver --ros-args -r /image:=/camera/image_raw

# Or extract from rosbag:
ros2 bag play my_data.bag --topics /camera/image_raw
# With separate terminal:
ros2 run image_view image_saver --ros-args -r /image:=/camera/image_raw

# Create directory
mkdir -p data/my_mapping_images
# Images will be saved to ~/.ros/
cp ~/.ros/frame*.jpg data/my_mapping_images/
```

**Option C: Video File**
```bash
# Extract frames from video at 2 FPS
ffmpeg -i video.mp4 -vf "fps=2" data/my_mapping_images/frame_%06d.jpg

# Or extract every 10th frame for sparse mapping
ffmpeg -i video.mp4 -vf "select=not(mod(n\,10)),scale=640:480" \
       -vsync drop data/my_mapping_images/frame_%06d.jpg
```

### 1.3 Verify Data

```bash
# Check how many images collected
ls data/my_mapping_images/ | wc -l

# Check image dimensions
identify data/my_mapping_images/image_001.jpg

# Quick preview
feh data/my_mapping_images/  # or any image viewer
```

**Minimum requirements**:
- At least 20-30 images
- Images should have DIFFERENT viewpoints (not identical frames)
- Images should be well-lit and contain texture

## 2. Camera Calibration

If using your own camera, you should provide accurate intrinsics (K matrix):

### 2.1 Get Intrinsics from Your Camera

**If you have calibration:**
```python
# From camera calibration file (e.g., OpenCV YAML format)
import cv2
fs = cv2.FileStorage('camera_calibration.yaml', cv2.FILE_STORAGE_READ)
K = fs.getNode('camera_matrix').mat()
```

**If you don't have calibration:**
```python
# Use simple estimate based on image size
# Assume 35mm sensor, typical focal length ~35mm

from pathlib import Path
import cv2
import numpy as np

# Read first image
img = cv2.imread('data/my_mapping_images/image_001.jpg')
H, W = img.shape[:2]

# Estimate intrinsics (for typical phone/camera)
fx = W  # or adjust based on your camera's FoV
fy = W
cx = W / 2
cy = H / 2

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)

print(f"Estimated K matrix for {W}x{H} image:")
print(K)
```

### 2.2 (Optional) Run OpenCV Camera Calibration

For more accurate results:

```bash
# 1. Print checkerboard pattern (data/calibration_checker.pdf)
# 2. Take 15-20 images of checkerboard in different positions
# 3. Run OpenCV calibration:

python3 <<'EOF'
import cv2
import numpy as np
from glob import glob

# Define checkerboard size (usually 9x6 corners for A4 paper)
pattern_size = (9, 6)
square_size = 0.025  # 2.5 cm in meters

# Prepare calibration images
images = sorted(glob('calibration_images/*.jpg'))

objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        print(f"✓ {fname}")
    else:
        print(f"✗ {fname}")

# Calibrate camera
if objpoints:
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    print(f"\nCalibration successful!")
    print(f"K =\n{K}")
    print(f"Distortion: {dist.ravel()}")
else:
    print("No valid calibration images found")
EOF
```

## 3. Build Map

### 3.1 Simple Map Building

```bash
cd /home/cesar/cv_slam_experiment

# Activate environment
source .venv/bin/activate

# Build map with default settings
python3 scripts/build_map_offline.py \
    --image_dir data/my_mapping_images \
    --output_map_dir data/relocalization_map \
    --global_descriptor dinov2 \
    --device cuda
```

Expected output:
```
[MapBuilder] Output: /home/cesar/cv_slam_experiment/data/relocalization_map
[MapBuilder] Device: cuda
[MapBuilder] Global descriptor: dinov2
[MapBuilder] Using COLMAP: False
[MapBuilder] Loading images from data/my_mapping_images...
[MapBuilder] Found 32 images
[MapBuilder] Loaded 32 images successfully
[MapBuilder] Extracting features...
Features: 100%|██████████| 32/32 [00:45<00:00,  1.41s/image]
[MapBuilder] Extracted features for 32 keyframes
[MapBuilder] Estimating camera poses...
[MapBuilder] Using fallback pose estimation (assumes forward motion)...
[MapBuilder] Generated 32 poses (fallback mode)
[MapBuilder] Camera intrinsics:
[[576. 0. 320.]
 [0. 576. 240.]
 [0. 0. 1.]]
[MapBuilder] Triangulating 3D points...
[MapBuilder] Triangulated 23490 3D points
============================================================
Building Relocalization Map
============================================================
Map built successfully!
Summary:
  Keyframes: 32
  3D Points: 23490
  ...
✓ Map built successfully!
✓ Map saved to: /home/cesar/cv_slam_experiment/data/relocalization_map
```

### 3.2 Advanced Options

```bash
# For larger scenes, use fewer but selected keyframes
python3 scripts/build_map_offline.py \
    --image_dir data/my_mapping_images \
    --output_map_dir data/relocalization_map \
    --global_descriptor dinov2 \
    --resize 480 640 \
    --max_images 100

# If COLMAP available: use for accurate SfM
python3 scripts/build_map_offline.py \
    --image_dir data/my_mapping_images \
    --output_map_dir data/relocalization_map \
    --use_colmap \
    --global_descriptor dinov2
```

## 4. Validate Map Quality

### 4.1 Run Quality Assessment

```python
from pathlib import Path
from src.relocalization import RelocalizationMap
from src.relocalization.map_builder_utils import MapQualityAssessment
import numpy as np

# Load built map
map_obj = RelocalizationMap('data/relocalization_map').load()

# Run quality assessment
metrics = MapQualityAssessment.full_assessment(
    np.array([kf.pose_w2c for kf in map_obj.keyframes]),
    map_obj.points_3d,
    [kf.descriptor_global for kf in map_obj.keyframes]
)

# Print results
print("\n=== Map Quality Report ===")
print(f"Keyframes:           {metrics['num_keyframes']}")
print(f"3D Points:           {metrics['num_3d_points']}")
print(f"Spatial Coverage:    {metrics['spatial_coverage']:.1%}")
print(f"Point Density:       {metrics['point_density']:.2f} pts/m³")
print(f"Descriptor Diversity:{metrics['descriptor_diversity']:.2f}")
print(f"Quality Score:       {metrics['quality_score']:.2f}/1.00")

# Interpretation
if metrics['quality_score'] > 0.7:
    print("\n✓ EXCELLENT map quality - ready for deployment")
elif metrics['quality_score'] > 0.5:
    print("\n~ ACCEPTABLE map quality - consider more data for improvements")
else:
    print("\n✗ POOR map quality - collect more/better images")
```

### 4.2 Interpret Results

| Metric | Target | Action if Low |
|--------|--------|---------------|
| `quality_score` | > 0.7 | Collect more diverse images |
| `num_keyframes` | > 20 | Collect longer sequence |
| `spatial_coverage` | > 0.5 | Map larger area or add images |
| `point_density` | > 1.0 | Ensure textured environments |
| `descriptor_diversity` | > 0.8 | Vary viewing angles, lighting |

## 5. Test Relocalization on Built Map

### 5.1 Quick Test with Synthetic Query

```python
from src.relocalization import HLocPipeline, RelocalizationMap
import cv2

# Load map
map_obj = RelocalizationMap('data/relocalization_map').load()

# Initialize pipeline
pipeline = HLocPipeline(map_obj, top_k_retrieval=5)

# Test on query image (from your dataset)
query_img = cv2.imread('data/my_mapping_images/image_015.jpg')
query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

result = pipeline.relocalize(query_img)

if result['success']:
    print("✓ Relocalization successful!")
    print(f"  Pose confidence: {result['confidence']:.2f}")
    print(f"  Inliers: {result['num_inliers']}")
    print(f"  Reprojection error: {result['reprojection_error']:.2f} px")
    print(f"  Pose:\n{result['pose']}")
else:
    print("✗ Relocalization failed")
```

### 5.2 Test on Unseen Images

For true validation, test on images NOT used in mapping:

```bash
# Capture few new images from same location (different angle/lighting)
mkdir -p data/query_images
# Collect 5-10 images

# Run test
python3 <<'EOF'
import cv2
from src.relocalization import HLocPipeline, RelocalizationMap
from glob import glob
import numpy as np

map_obj = RelocalizationMap('data/relocalization_map').load()
pipeline = HLocPipeline(map_obj, top_k_retrieval=5)

successes = 0
total = 0

for img_path in sorted(glob('data/query_images/*.jpg')):
    query_img = cv2.imread(img_path)
    if query_img is None:
        continue
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    
    result = pipeline.relocalize(query_img)
    total += 1
    
    if result['success']:
        successes += 1
        print(f"✓ {img_path}: {result['confidence']:.2f} confidence")
    else:
        print(f"✗ {img_path}: Failed")

print(f"\nSuccess rate: {successes}/{total} ({100*successes/max(total,1):.0f}%)")
EOF
```

## 6. Integration Steps

### 6.1 For ROS 2 Integration

**Save your map path for ROS nodes:**

```bash
# Save map path to config
cat > config/map_config.yaml <<EOF
map_path: "/home/cesar/cv_slam_experiment/data/relocalization_map"
top_k_retrieval: 5
device: "cuda"
min_inliers: 15
max_reprojection_error: 2.0
EOF
```

### 6.2 Use in Relocalization Service

```python
from src.relocalization import RelocalizationMap, HLocPipeline

# Load once at startup
MAP = RelocalizationMap('data/relocalization_map').load()
PIPELINE = HLocPipeline(MAP, top_k_retrieval=5)

# In service handler:
def handle_relocalize(self, image):
    result = PIPELINE.relocalize(image)
    return result
```

## 7. Troubleshooting

### Problem: "No images found"

```bash
# Check path
ls data/my_mapping_images/
ls data/my_mapping_images/*.jpg | head -5

# Fix: Rename if using uppercase extensions
for f in data/my_mapping_images/*.JPG; do mv "$f" "${f%.JPG}.jpg"; done
```

### Problem: "Low quality score"

**Symptoms**: quality_score < 0.5

**Causes**:
1. Too few images (need ≥20)
2. Featureless environment (white wall, sky)
3. Poor lighting (too dark, backlit)
4. Excessive motion blur
5. All images taken from same location

**Solutions**:
```bash
# 1. Collect more images
ffmpeg -i video.mp4 -vf "fps=1" data/my_mapping_images/frame_%06d.jpg  # 1 FPS = more frames

# 2. Choose better environment
# - Move to textured area: office with posters, nature with trees
# - Improve lighting: outdoor daytime, or add lights indoors

# 3. Move slower
# - Reduce camera motion speed
# - Increase image capture rate
```

### Problem: Feature extraction fails

```bash
# Check if DINOv2 is properly installed
python3 -c "from src.relocalization import GlobalDescriptorExtractor; print('OK')"

# If fails, reinstall:
cd /home/cesar/cv_slam_experiment
source .venv/bin/activate
pip install --upgrade -r requirements_relocalization.txt

# Test feature extraction
python3 <<'EOF'
from src.relocalization import GlobalDescriptorExtractor
import cv2
extractor = GlobalDescriptorExtractor("dinov2", device="cuda")
img = cv2.imread('data/my_mapping_images/image_001.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
desc = extractor.extract(img)
print(f"Descriptor shape: {desc.shape}")
print(f"Descriptor norm: {np.linalg.norm(desc):.3f}")
EOF
```

### Problem: COLMAP not found

```bash
# Install COLMAP (Ubuntu)
sudo apt-get install colmap

# Or build from source:
# https://colmap.github.io/installation.html

# Verify installation
which colmap
colmap --version

# For now, use fallback (less accurate but works)
python3 scripts/build_map_offline.py \
    --image_dir data/my_mapping_images \
    --output_map_dir data/relocalization_map
    # --use_colmap false (default)
```

## 8. Next Steps

1. **Test with live camera**: Integrate with your VIO node
   ```bash
   ros2 launch cv_slam_experiment vio_with_relocalization.launch.py
   ```

2. **Optimize for your environment**: Tune keyframe selection, descriptor type
   ```python
   config = KeyframeSelectionConfig(
       min_translation_distance=0.2,   # Tighter spacing
       min_rotation_angle=5.0,          # More keyframes
   )
   ```

3. **Deploy to robot**: Follow Phase 3 (ROS 2 Integration)

4. **Monitor and iterate**: Check success rates, refine map if needed

## Appendix: Checklist

- [ ] Environment selected ✓
- [ ] 30+ images collected ✓
- [ ] Images copied to `data/my_mapping_images/` ✓
- [ ] Camera intrinsics available (or estimated) ✓
- [ ] Map built: `python3 scripts/build_map_offline.py ...` ✓
- [ ] Quality checked: quality_score > 0.5 ✓
- [ ] Tested on unseen images: > 50% success ✓
- [ ] Integrated with ROS 2 ✓
- [ ] Tested on live camera ✓
- [ ] Ready for deployment ✓
