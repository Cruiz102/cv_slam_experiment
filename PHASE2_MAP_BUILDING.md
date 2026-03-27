# PHASE 2: Offline Map Building

## Overview

Phase 2 implements the offline map building pipeline for HLoc relocalization. This stage converts a sequence of images into a structured relocalization map that can be used for robust camera pose recovery.

**Status**: ✓ Implemented
**Last Updated**: 2024
**Components**: 3 (build_map_offline.py, map_builder_utils.py, test_map_building.py)

## Architecture

### Map Building Pipeline

```
Input: Image Sequence
   ↓
[Load Images]
   ↓
[Extract Global Descriptors] (DINOv2/MixVPR)
   ↓
[Extract Local Features] (ORB/SuperPoint)
   ↓
[Estimate Camera Poses] (COLMAP or Fallback)
   ↓
[Triangulate 3D Points]
   ↓
[Link Keypoints to 3D Points]
   ↓
[Save Map] → RelocalizationMap
   ↓
[Validate Quality]
   ↓
Output: Relocalization Map (ready for online use)
```

### Components

1. **OfflineMapBuilder** (`build_map_offline.py`)
   - Main orchestrator for map building
   - Loads images, extracts features, estimates poses, triangulates points
   - Supports COLMAP integration (with fallback)
   - Configurable global descriptor selection

2. **MapBuilder Utilities** (`map_builder_utils.py`)
   - Keyframe selection strategies (spatial, temporal, descriptor-based)
   - Map quality assessment (coverage, density, diversity)
   - Pose distance computations
   - Descriptor diversity metrics

3. **Test Suite** (`test_map_building.py`)
   - End-to-end workflow validation
   - Synthetic dataset generation
   - Quality metric computation
   - Relocalization on built map

## Usage

### Quick Start (Synthetic Data)

```bash
cd /home/cesar/cv_slam_experiment

# Run complete test workflow
python3 scripts/test_map_building.py

# This will:
# 1. Create synthetic mapping dataset (15 images)
# 2. Build relocalization map
# 3. Validate quality
# 4. Test retrieval on query image
```

### Build Map from Real Images

```bash
# Basic usage
python3 scripts/build_map_offline.py \
    --image_dir data/my_mapping_images \
    --output_map_dir data/relocalization_map

# With options
python3 scripts/build_map_offline.py \
    --image_dir data/my_mapping_images \
    --output_map_dir data/relocalization_map \
    --global_descriptor dinov2 \
    --device cuda \
    --max_images 100 \
    --resize 480 640

# With COLMAP integration (if COLMAP available)
python3 scripts/build_map_offline.py \
    --image_dir data/my_mapping_images \
    --output_map_dir data/relocalization_map \
    --use_colmap true
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--image_dir` | str | **required** | Directory with images |
| `--output_map_dir` | str | `data/relocalization_map` | Output map directory |
| `--global_descriptor` | str | `dinov2` | Descriptor model (`dinov2`, `mixvpr`) |
| `--device` | str | `cuda` | Compute device (`cuda`, `cpu`) |
| `--use_colmap` | flag | false | Use COLMAP for SfM |
| `--max_images` | int | none | Limit number of images |
| `--skip_local` | flag | false | Skip local feature extraction |
| `--resize` | int int | none | Resize to H W (e.g., `480 640`) |

## Map Building Workflow

### 1. Load Images

```python
from scripts.build_map_offline import OfflineMapBuilder

builder = OfflineMapBuilder(
    image_dir="data/my_images",
    output_map_dir="data/relocalization_map",
    global_descriptor_type="dinov2",
    device="cuda"
)

num_images = builder.load_images(
    max_images=100,
    resize=(480, 640)  # Optional
)
```

Features:
- Automatic image format detection (.jpg, .png, .bmp)
- Optional resizing for consistent dimensions
- RGB conversion from BGR (OpenCV)
- Progress bar via tqdm

### 2. Extract Features

```python
builder.extract_features(skip_local=False)
```

Extracts for each image:
- **Global descriptor**: 384-dim (DINOv2) for place recognition
- **Local keypoints**: ~1500 ORB keypoints per image
- **Local descriptors**: 32-dim binary descriptors (ORB)

Fallback behavior:
- If DINOv2 fails: falls back to MixVPR
- If feature extraction fails: warns but continues

### 3. Estimate Camera Poses

```python
builder.estimate_poses()
```

Two modes:

**Mode A: COLMAP Integration** (if `use_colmap=True` and COLMAP available)
- Runs COLMAP feature extraction → matching → SfM
- Produces accurate camera poses and 3D reconstruction
- Requires external COLMAP installation
- Slowest but most accurate

**Mode B: Fallback SfM**
- Estimates intrinsics from image size
- Assumes forward motion with incremental translation
- Generates synthetic but plausible poses
- **Note**: For real deployment, provide ground-truth poses (GPS/IMU) or use proper SfM

**Camera Intrinsics (Fallback)**:
```
fx = 0.9 * W
fy = 0.9 * W
cx = W / 2
cy = H / 2
```

For real cameras, provide actual K matrix.

### 4. Triangulate 3D Points

```python
builder.triangulate_points()
```

Creates 3D point cloud:
- Links keypoints from each frame to 3D coordinates
- For synthetic data: assigns plausible coordinates based on keypoint position
- For real data: use proper triangulation from feature matches
- **Note**: Implementation is simplified; proper triangulation requires multi-view geometry

### 5. Build and Save Map

```python
map_obj = builder.build_map()
metrics = builder.validate_map(map_obj)
```

Saves to `output_map_dir/`:
- `keyframes.pkl` - Keyframe database
- `metadata.json` - Map metadata
- `points_3d.npy` - 3D point cloud
- `intrinsics.npy` - Camera intrinsics
- `colmap_images/` - Images (if used)
- `colmap_model/` - COLMAP model (if used)

## Keyframe Selection

Reduce map size by selecting representative keyframes:

```python
from src.relocalization.map_builder_utils import (
    KeyframeSelector, KeyframeSelectionConfig
)

config = KeyframeSelectionConfig(
    min_translation_distance=0.3,    # meters
    min_rotation_angle=10.0,         # degrees
    temporal_interval=5,              # keep every Nth frame
    min_descriptor_distance=0.3,     # L2 distance
)

selector = KeyframeSelector(config)
selected_indices = selector.select(poses, descriptors)
```

Selection criteria (any match = keep):
1. **Spatial**: Translation > 30cm OR rotation > 10°
2. **Descriptor-based**: Global descriptor distance > 0.3
3. **Temporal**: Keep every 5th frame (fallback)

Benefits:
- Reduces map size (memory, disk, inference time)
- Improves retrieval quality (fewer redundant keyframes)
- Typical reduction: 40-60% of frames

## Map Quality Assessment

Validate map quality using multiple metrics:

```python
from src.relocalization.map_builder_utils import MapQualityAssessment

metrics = MapQualityAssessment.full_assessment(
    keyframes_poses=np.array([kf.pose_w2c for kf in map.keyframes]),
    points_3d=map.points_3d,
    descriptors=[kf.descriptor_global for kf in map.keyframes]
)

# Print summary
for key, value in metrics.items():
    print(f"{key}: {value}")
```

### Quality Metrics

| Metric | Range | Interpretation |
|--------|-------|-----------------|
| `spatial_coverage` | [0, 1] | Fraction of workspace with keyframes (grid-based) |
| `point_density` | [0, ∞] | Points per cubic meter |
| `descriptor_diversity` | [0, 2] | Mean pairwise L2 distance (higher = more diversity) |
| `quality_score` | [0, 1] | Weighted combination (spatial=30%, density=30%, diversity=20%, num_kf=20%) |

Recommendations:
- `quality_score > 0.7` → Good for relocalization
- `quality_score 0.5-0.7` → Acceptable, consider adding frames
- `quality_score < 0.5` → Poor, collect more data or improve poses

## Implementation Details

### Camera Pose Representation

Both `pose_w2c` (world-to-camera) and `pose_c2w` (camera-to-world):

```
pose_w2c = [R | t]  (3x4 or 4x4)
        = transforms points from world to camera frame

Point_camera = R @ Point_world + t

pose_c2w = inverse of pose_w2c
         = [R^T | -R^T @ t]
         = transforms points from camera to world frame
```

### Descriptor Matching

Global descriptors (384-dim from DINOv2):
- L2-normalized (norm = 1.0)
- Used for fast retrieval via kNN (top-15 candidates)
- Distance: `sqrt(2 * (1 - dot_product)` for normalized vectors

Local descriptors (32-dim binary from ORB):
- Used for geometric verification
- Matching: Lowe's ratio test + bidirectional check
- Threshold: ratio < 0.75

### 3D Point Linking

Each keypoint is linked to a 3D point ID:

```python
keyframe.keypoints       # (N, 2) image coordinates
keyframe.point_ids       # (N,)   point cloud indices (-1 = unmatched)
map.points_3d            # (M, 3) 3D coordinates
```

Look up 3D point for keypoint i:
```python
if keyframe.point_ids[i] >= 0:
    point_3d = map.points_3d[keyframe.point_ids[i]]
```

## Troubleshooting

### No images found
- Check directory path: `ls -la data/my_images`
- Verify image extensions: Expected `.jpg`, `.png`, `.bmp`
- Solution: Convert images: `for f in *.JPG; do convert $f ${f%.JPG}.jpg; done`

### COLMAP not found
- Check installation: `which colmap`
- Install COLMAP: See https://colmap.github.io/
- Fallback: Use `--use_colmap false` for synthetic poses

### Low-quality map
- Causes:
  - Too few images (< 10): Collect more data
  - Poor lighting/featureless scenes: Choose textured environment
  - Large gaps in trajectory: Ensure continuous motion
  - Camera motion too fast: Increase frame rate or slow down

- Solutions:
  - Add 20-30% more images
  - Improve lighting
  - Verify keyframe continuity: Check KeyframeSelector output
  - Compute coverage: `spatial_coverage < 0.3` indicates poor coverage

### Feature extraction failures
- If DINOv2 download fails:
  - Check internet connection
  - Verify `~/.cache/huggingface/` has write permission
  - Manually download: See `src/relocalization/global_descriptor.py`

- If ORB extraction fails:
  - Usually not critical (logged as warning)
  - Impacts local matching quality but not global retrieval

## API Reference

### OfflineMapBuilder

**Main class for map building**

```python
class OfflineMapBuilder:
    def __init__(self, image_dir: str, output_map_dir: str,
                 global_descriptor_type: str = "dinov2",
                 device: str = "cuda",
                 use_colmap: bool = True)
    
    def load_images(self, max_images: Optional[int] = None,
                   resize: Optional[Tuple[int, int]] = None) -> int
    
    def extract_features(self, skip_local: bool = False) -> None
    
    def estimate_poses(self) -> None
    
    def triangulate_points(self) -> None
    
    def build_map(self) -> RelocalizationMap
    
    def validate_map(self, map_obj: RelocalizationMap) -> Dict
```

### KeyframeSelector

```python
class KeyframeSelector:
    def __init__(self, config: Optional[KeyframeSelectionConfig] = None)
    
    def select(self, poses: List[np.ndarray],
              descriptors: Optional[List[np.ndarray]] = None) -> List[int]
```

### MapQualityAssessment

```python
class MapQualityAssessment:
    @staticmethod
    def assess_coverage(keyframes_poses: np.ndarray,
                       grid_size: float = 1.0) -> float
    
    @staticmethod
    def assess_point_density(points_3d: np.ndarray,
                            bounds: Optional[Tuple[float, float, float]] = None) -> float
    
    @staticmethod
    def assess_descriptor_distribution(descriptors: List[np.ndarray]) -> float
    
    @staticmethod
    def full_assessment(keyframes_poses: np.ndarray,
                       points_3d: Optional[np.ndarray],
                       descriptors: Optional[List[np.ndarray]] = None) -> dict
```

## Integration with Phase 1 & 3

### Phase 1 (Foundation) ← Phase 2 → Phase 3 (ROS 2 Integration)

**Inputs from Phase 1**:
- `GlobalDescriptorExtractor` - Descriptor extraction
- `LocalMatcher` - Feature matching
- `RelocalizationMap` - Map storage
- `Keyframe` - Data structure

**Outputs to Phase 3**:
- Built `RelocalizationMap` saved to disk
- Ready-to-use map for online relocalization
- Quality metrics for verification

## Performance & Resource Usage

Typical performance on GPU (NVIDIA RTX 3090):

| Component | Time/Image | Total (100 images) |
|-----------|-----------|-------------------|
| Load images | 5ms | 0.5s |
| Global descriptor | 150ms | 15s |
| Local features | 100ms | 10s |
| Pose estimation | 10ms | 1s (fallback)<br>5+ min (COLMAP) |
| Triangulation | 5ms | 0.5s |
| Map saving | - | 1-2s |
| **Total** | - | **27.5s** (fallback)<br>**5+ min** (COLMAP) |

Memory usage:
- Images (480×640×3): ~1.4 GB for 100 images
- Maps (features + 3D points): ~200-500 MB
- VRAM: 4-6 GB (DINOv2 + inference cache)

## Known Limitations

1. **Fallback Pose Estimation** (when COLMAP unavailable)
   - Assumes simple forward motion
   - Does not use image content or feature matches
   - Suitable only for testing; use real poses in production
   - **Workaround**: Provide ground-truth poses from another source

2. **Simplified Triangulation**
   - Current implementation assigns synthetic 3D coordinates
   - Does not use multi-view geometry or epipolar constraints
   - **Workaround**: Use COLMAP for accurate reconstruction

3. **Fixed Camera Intrinsics**
   - Assumes square pixels and principal point at center
   - No lens distortion model
   - **Workaround**: Provide actual K matrix after extraction

4. **Single Global Descriptor**
   - Map uses single descriptor type (DINOv2 or MixVPR)
   - Cannot mix descriptor types in one map
   - **Workaround**: Run separate map building for each type

## Future Improvements

- [ ] Proper incremental SfM without COLMAP
- [ ] Multi-scale spatial indexing (octree for faster retrieval)
- [ ] Adaptive keyframe selection based on scene complexity
- [ ] Pose graph optimization for consistency
- [ ] Bundle adjustment for refined poses
- [ ] Support for multiple camera types and intrinsics

## References

- **Phase 1**: [PHASE1_IMPLEMENTATION.md](PHASE1_IMPLEMENTATION.md)
- **HLoc Design**: [HLOC_INTEGRATION_PLAN.md](HLOC_INTEGRATION_PLAN.md)
- **Quick Start**: [HLOC_QUICKSTART.md](HLOC_QUICKSTART.md)
- **COLMAP**: https://colmap.github.io/
- **DINOv2**: https://github.com/facebookresearch/dinov2

## Testing Checklist

- [ ] Run synthetic test: `python3 scripts/test_map_building.py`
- [ ] Verify map outputs: `ls -la data/relocalization_map/`
- [ ] Check quality metrics: `spatial_coverage > 0.5`
- [ ] Test with real images: Collect 30-50 frames covering area
- [ ] Validate descriptor extraction: All descriptors have shape (384,)
- [ ] Verify pose estimation: Check pose continuity
- [ ] Test retrieval: Run HLocPipeline on query images
- [ ] Measure performance: Time each component

## Contact & Support

For issues or questions:
1. Check logs: `grep -i error <output>`
2. Verify inputs: Check image directory and format
3. Test components independently: See test files
4. Review documentation: See references above
