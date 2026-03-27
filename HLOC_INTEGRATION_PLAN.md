# HLoc-Style Relocalization Integration Plan

**Date**: March 26, 2026  
**Project**: CV SLAM Experiment (Optical Flow + IMU EKF VIO)  
**Goal**: Add robust relocalization via hierarchical visual localization (HLoc)

---

## 1. System Architecture Overview

### Current System
```
┌─────────────────────────────────────────────┐
│        Optical Flow + IMU EKF VIO           │
├─────────────────────────────────────────────┤
│  Fast Loop (30 Hz):                         │
│  ├─ Webcam frame (640×480)                  │
│  ├─ Feature tracking (optical flow)         │
│  ├─ Two-view geometry (relative pose)       │
│  ├─ IMU fusion (Madgwick + orientation)     │
│  ├─ EKF update (R_fused, p_fused)           │
│  └─ Publish: PoseStamped, Path, Odom       │
│                                              │
│  Slow Sensors:                              │
│  ├─ IMU (VN-100 serial or ROS topic)        │
│  └─ Reference benchmark frames              │
└─────────────────────────────────────────────┘
```

### Target: Hybrid Relocalization Stack
```
┌────────────────────────────────────────────────────────┐
│  Optical Flow + IMU EKF VIO (unchanged)                │
│  ├─ Fast tracking loop                                 │
│  ├─ IMU fusion                                         │
│  └─ Publishes pose/path/odom                          │
└──────────────────────────┬─────────────────────────────┘
                           │
                    ┌──────▼────────┐
                    │ Loss Detection │
                    │ (inlier drop)  │
                    └──────┬────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼─────┐    ┌──────▼────────┐   ┌────▼─────┐
   │ Retrieval │    │ Local Matching│   │   PnP    │
   │ (MixVPR)  │───▶│(SuperPoint +  │──▶│ RANSAC   │
   │           │    │ LightGlue)    │   │          │
   └───────────┘    └───────────────┘   └────┬─────┘
                                             │
                                        ┌────▼────────┐
                                        │   Geometry  │
                                        │ Verification│
                                        └────┬────────┘
                                             │
                                        ┌────▼──────────┐
                                        │ EKF Correction│
                                        │ (reinit pose) │
                                        └───────────────┘
```

---

## 2. Technical Stack & Dependencies

### Core Libraries
| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| **Global Descriptor** | MixVPR | Latest | Place recognition / retrieval |
| **Local Features** | SuperPoint | `superpoint.onnx` | Keypoint detector |
| **Feature Matching** | LightGlue | `lightglue.onnx` | Robust matching |
| **Pose Recovery** | OpenCV | 4.5+ | PnP-RANSAC |
| **Map Storage** | COLMAP (or .pkl) | Any | SfM 3D points / keyframes |
| **Geometry** | NumPy/SciPy | Latest | Quaternion, transform ops |
| **ROS 2** | rclpy | Jazz | Service/client comms |

### Installation
```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python scipy numpy
pip install transformers  # For Vision Foundation models if using DINOv2 fallback

# Download pretrained models
# - MixVPR encoder weights (huggingface model)
# - SuperPoint weights
# - LightGlue weights

# Optional: Install colmap tools (for offline map building)
# sudo apt install colmap
```

---

## 3. File Structure

### Proposed Directory Layout
```
/home/cesar/cv_slam_experiment/
├── scripts/
│   ├── ros2_vio_pose_node.py                    (existing)
│   ├── ros2_relocalization_node.py              (NEW - main relocalization service)
│   ├── build_map_offline.py                     (NEW - map builder utility)
│   ├── test_relocalization.py                   (NEW - debug/test script)
│   └── data_collection_for_mapping.py           (NEW - keyframe collection)
│
├── src/
│   ├── relocalization/                          (NEW - relocalization module)
│   │   ├── __init__.py
│   │   ├── hloc_pipeline.py                     (Main relocalization logic)
│   │   ├── map_manager.py                       (Map storage/retrieval)
│   │   ├── global_descriptor.py                 (MixVPR wrapper)
│   │   ├── local_matcher.py                     (SuperPoint + LightGlue)
│   │   ├── pose_solver.py                       (PnP-RANSAC + geometry checks)
│   │   └── relocalization_filter.py             (EKF correction/validation)
│   │
│   ├── fusion/                                  (existing - may extend)
│   │   └── ekf_corrector.py                     (NEW - ingest relocalization poses)
│   │
│   └── vision/ (existing)
│
├── data/
│   ├── reference_memory/                        (existing - benchmark frames)
│   ├── relocalization_map/                      (NEW - offline mapped keyframes)
│   │   ├── keyframes.pkl                        (keyframe metadata + descriptors)
│   │   ├── descriptors_global/                  (MixVPR embeddings)
│   │   ├── descriptors_local/                   (SuperPoint keypoints + desc)
│   │   ├── 3d_points.pkl                        (SfM reconstruction points)
│   │   ├── camera_intrinsics.json               (K matrix for this camera)
│   │   └── metadata.json                        (map build time, scale, coverage)
│   │
│   └── models/                                  (NEW - pretrained model weights)
│       ├── mixvpr_encoder.pkl
│       ├── superpoint.onnx
│       └── lightglue.onnx
│
├── config/
│   ├── relocalization_config.yaml               (NEW - HLoc pipeline parameters)
│   └── calibration.py                           (existing)
│
├── HLOC_INTEGRATION_PLAN.md                     (this file)
└── requirements_relocalization.txt              (NEW - pip dependencies)
```

---

## 4. Component Design (Detailed)

### 4.1 Map Manager (`src/relocalization/map_manager.py`)

**Purpose**: Persistent storage and fast retrieval of map data

```python
class RelocalizationMap:
    """Store and manage keyframes, 3D points, and descriptors."""
    
    def __init__(self, map_dir: str):
        self.map_dir = map_dir
        self.keyframes: List[Keyframe] = []
        self.points_3d: np.ndarray = None  # (N, 3)
        self.K: np.ndarray = None  # (3, 3) intrinsics
        self.metadata: dict = {}
    
    class Keyframe:
        id: int
        image_path: str
        timestamp: float
        pose_w2c: np.ndarray  # (4, 4)
        gray_image: np.ndarray  # (H, W) for visualization
        descriptor_global: np.ndarray  # (1024,) MixVPR
        keypoints: np.ndarray  # (K, 2)
        descriptors_local: np.ndarray  # (K, 256) SuperPoint
        point_ids: np.ndarray  # (K,) indices into points_3d, -1 if no 3D
    
    def add_keyframe(self, kf: Keyframe) -> None:
        """Insert new keyframe into map."""
    
    def get_top_k_candidates(self, query_desc_global: np.ndarray, 
                             k: int = 20) -> List[Keyframe]:
        """Retrieve top-K keyframes by global descriptor similarity."""
    
    def save(self) -> None:
        """Serialize map to disk (pickle + .json)."""
    
    def load(self) -> None:
        """Deserialize map from disk."""
```

### 4.2 Global Descriptor (`src/relocalization/global_descriptor.py`)

**Purpose**: Fast place recognition via learned image-level embeddings

```python
class GlobalDescriptorExtractor:
    """MixVPR: Modern pooled descriptor for place recognition."""
    
    def __init__(self, model_name: str = "MixVPR-G-65536",
                 device: str = "cuda"):
        # Load MixVPR from HuggingFace or local weights
        self.model = load_mixvpr_model(model_name, device)
        self.device = device
        self.embedding_dim = 65536  # or 131072 for larger
    
    def extract(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Extract global descriptor (appearance signature).
        
        Args:
            image_rgb: (H, W, 3) uint8
        
        Returns:
            descriptor: (D,) float32, L2-normalized
        """
        # Normalize image
        # Pass through model backbone (e.g., ResNet50)
        # Apply MixVPR aggregation (weighted channel pooling)
        # L2 normalize
        # Return
    
    def find_top_k_similar(self, query_desc: np.ndarray,
                           corpus_descs: List[np.ndarray],
                           k: int = 20) -> List[float]:
        """
        Fast kNN via cosine similarity.
        Returns indices of top-k most similar keyframes.
        """
```

### 4.3 Local Matcher (`src/relocalization/local_matcher.py`)

**Purpose**: Geometric verification via robust 2D-3D matching

```python
class LocalMatcher:
    """SuperPoint + LightGlue for robust local feature matching."""
    
    def __init__(self, device: str = "cuda"):
        self.superpoint = load_superpoint_model(device)
        self.lightglue = load_lightglue_model(device)
        self.device = device
    
    def extract_features(self, image_gray: np.ndarray) -> dict:
        """
        Extract SuperPoint keypoints + descriptors.
        
        Returns:
            {
                'keypoints': (K, 2) normalized coords,
                'descriptors': (K, 256) float32,
                'scores': (K,) detection scores
            }
        """
    
    def match_pairs(self, features_query: dict,
                    features_keyframe: dict,
                    threshold: float = 0.7) -> dict:
        """
        Match query image to candidate keyframe.
        LightGlue handles adaptive matching.
        
        Returns:
            {
                'matches': (M, 2) pairs of keypoint indices,
                'scores': (M,) match confidence,
                'matches_mkpts0': (M, 2) image pixel coords,
                'matches_mkpts1': (M, 2) keyframe pixel coords
            }
        """
```

### 4.4 Pose Solver (`src/relocalization/pose_solver.py`)

**Purpose**: Compute camera pose from 2D-3D correspondences

```python
class PoseSolver:
    """PnP-RANSAC with geometric verification."""
    
    def __init__(self, K: np.ndarray, inlier_threshold: float = 8.0):
        self.K = K
        self.inlier_threshold = inlier_threshold
    
    def solve_pnp_ransac(self, 
                         points_2d: np.ndarray,      # (M, 2) image coords
                         points_3d: np.ndarray,      # (M, 3) world coords
                         confidence: float = 0.99,
                         iterations: int = 1000) -> dict:
        """
        Estimate pose via RANSAC.
        
        Returns:
            {
                'success': bool,
                'rotation': (3, 3) R_c2w,
                'translation': (3,) t_c2w,
                'inlier_mask': (M,) boolean,
                'reprojection_error': float,
                'num_inliers': int
            }
        """
    
    def verify_pose(self, pose: dict, 
                    min_inliers: int = 15,
                    max_reproj_error: float = 2.0) -> bool:
        """Geometric sanity checks on recovered pose."""
```

### 4.5 HLoc Pipeline (`src/relocalization/hloc_pipeline.py`)

**Purpose**: Orchestrate full relocalization

```python
class HLocPipeline:
    """Main orchestrator: retrieval → matching → pose solving."""
    
    def __init__(self, map_dir: str, device: str = "cuda",
                 config: dict = None):
        self.map = RelocalizationMap(map_dir)
        self.map.load()
        
        self.global_desc = GlobalDescriptorExtractor(device=device)
        self.local_matcher = LocalMatcher(device=device)
        self.pose_solver = PoseSolver(self.map.K)
        
        self.config = config or self._default_config()
    
    def relocalize(self, image_rgb: np.ndarray,
                   prior_R: np.ndarray = None,
                   prior_p: np.ndarray = None) -> dict:
        """
        Full relocalization pipeline.
        
        Args:
            image_rgb: (H, W, 3) current frame
            prior_R: (3, 3) optional rotation prior from IMU
            prior_p: (3,) optional position prior
        
        Returns:
            {
                'success': bool,
                'pose_w2c': (4, 4) if success else None,
                'pose_c2w': (4, 4) if success else None,
                'num_inliers': int,
                'matched_keyframe_id': int,
                'confidence': float (0-1),
                'retrieval_time': float,
                'matching_time': float,
                'pose_time': float,
                'debug_image': np.ndarray (optional)
            }
        """
        times = {}
        
        # Stage 1: Global retrieval
        t0 = time.perf_counter()
        query_desc = self.global_desc.extract(image_rgb)
        candidates = self.map.get_top_k_candidates(query_desc, k=15)
        times['retrieval'] = time.perf_counter() - t0
        
        if not candidates:
            return {'success': False, 'reason': 'No candidates found'}
        
        # Stage 2: Local feature extraction
        gray_query = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        features_query = self.local_matcher.extract_features(gray_query)
        
        # Stage 3: Matching against candidates
        t0 = time.perf_counter()
        best_pose = None
        best_keyframe = None
        best_inliers = 0
        
        for candidate_kf in candidates:
            features_kf = {
                'keypoints': candidate_kf.keypoints,
                'descriptors': candidate_kf.descriptors_local
            }
            
            matches = self.local_matcher.match_pairs(
                features_query, features_kf
            )
            
            if len(matches['matches']) < 10:
                continue
            
            # 2D → 3D: map surface keypoints to 3D points
            pts2d = matches['matches_mkpts0']
            pts3d = self.map.points_3d[candidate_kf.point_ids[matches['matches'][:, 1]]]
            
            # Stage 4: PnP + RANSAC
            t0_pnp = time.perf_counter()
            result = self.pose_solver.solve_pnp_ransac(pts2d, pts3d)
            times['pose'] = time.perf_counter() - t0_pnp
            
            if result['success'] and result['num_inliers'] > best_inliers:
                best_pose = result
                best_keyframe = candidate_kf
                best_inliers = result['num_inliers']
        
        times['matching'] = time.perf_counter() - t0
        
        if best_pose is None:
            return {'success': False, 'reason': 'No valid matches'}
        
        # Assemble output
        return {
            'success': True,
            'pose_w2c': np.eye(4),  # Construct from R/t
            'pose_c2w': np.eye(4),  # Invert
            'num_inliers': best_inliers,
            'matched_keyframe_id': best_keyframe.id,
            'confidence': min(best_inliers / 50.0, 1.0),
            'retrieval_time': times['retrieval'],
            'matching_time': times['matching'],
            'pose_time': times['pose'],
        }
    
    @staticmethod
    def _default_config() -> dict:
        return {
            'top_k_retrieval': 15,
            'min_matches_for_pose': 10,
            'min_inliers_pnp': 15,
            'max_reproj_error': 2.0,
            'ransac_iterations': 1000,
            'confidence': 0.99,
        }
```

### 4.6 EKF Corrector (`src/fusion/ekf_corrector.py`)

**Purpose**: Safely integrate relocalization pose into ongoing VIO

```python
class EKFRelocalizationCorrector:
    """
    Fuse externally-recovered pose back into VIO state.
    
    Key: only correct when relocalization is confident.
    Do not destabilize the filter with bad poses.
    """
    
    def __init__(self, initial_R: np.ndarray,
                 initial_p: np.ndarray,
                 process_noise: float = 1e-3,
                 measurement_noise: float = 0.1):
        self.R_est = initial_R
        self.p_est = initial_p
        
        self.Q = process_noise * np.eye(6)  # motion model noise
        self.R_meas = measurement_noise * np.eye(6)  # relocalization noise
    
    def correct_pose(self, 
                     observed_pose_c2w: np.ndarray,  # (4, 4)
                     confidence: float,
                     num_inliers: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply Bayesian pose correction.
        
        If confidence high and inliers sufficient:
            → blend observed pose into estimate
        Else:
            → keep current estimate, small weight
        
        Returns:
            (R_corrected, p_corrected)
        """
        
        # Extract pose components
        R_obs = observed_pose_c2w[:3, :3]
        p_obs = observed_pose_c2w[:3, 3]
        
        # Confidence blending
        alpha = min(confidence, num_inliers / 50.0)  # 0 = trust current, 1 = trust obs
        
        # Rotation: SLERP or direct blend via quaternions
        q_est = rotmat_to_quat(self.R_est)
        q_obs = rotmat_to_quat(R_obs)
        q_corrected = slerp(q_est, q_obs, alpha)
        R_corrected = quat_to_rotmat(q_corrected)
        
        # Translation: linear blend
        p_corrected = (1 - alpha) * self.p_est + alpha * p_obs
        
        # Update internal state
        self.R_est = R_corrected
        self.p_est = p_corrected
        
        return R_corrected, p_corrected
    
    def get_confidence_factor(self, num_inliers: int,
                              reproj_error: float) -> float:
        """
        Compute overall confidence [0, 1].
        
        Factors:
        - inlier count (target ~ 30-50)
        - reprojection error (target ~ 1-2 px)
        - temporal consistency
        """
```

---

## 5. ROS 2 Integration

### 5.1 Relocalization Service Node

**File**: `scripts/ros2_relocalization_node.py`

```python
class RelocalizationVioNode(Node):
    """
    ROS 2 node providing:
    1. Relocalization service (on-demand pose recovery)
    2. Subscription to VIO pose for context
    3. Optional continuous mode (triggered by tracking loss)
    """
    
    def __init__(self):
        super().__init__('relocalization_vio_node')
        
        # Parameters
        self.declare_parameter('map_dir', '/path/to/relocalization_map')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('enabled', True)
        self.declare_parameter('auto_trigger_on_inlier_drop', True)
        self.declare_parameter('min_inlier_threshold', 10)  # trigger if inliers < this
        
        # Load config
        self.map_dir = self.get_parameter('map_dir').value
        self.hloc = HLocPipeline(self.map_dir)
        
        # Service: trigger relocalization on current frame
        self.srv_relocalize = self.create_service(
            Trigger, 'vio/relocalize_trigger',
            self._srv_relocalize_callback
        )
        
        # Subscription: listen to VIO pose + inlier count
        # (Extend PoseStamped message or use custom VIO status topic)
        self.sub_vio_status = self.create_subscription(
            VioStatus, 'vio/status',
            self._on_vio_status, 10
        )
        
        # Publisher: emit corrected pose when relocalization succeeds
        self.pub_reloc_pose = self.create_publisher(
            PoseStamped, 'vio/relocalized_pose', 10
        )
        
        # Client: call back to VIO node to apply correction
        self.cli_correct_vio = self.create_client(
            CorrectVioPose, 'vio/correct_pose'
        )
        
        self.current_frame = None
        self.vio_inlier_count = 0
```

### 5.2 Updated VIO Node

**Update**: `scripts/ros2_vio_pose_node.py` + service for correction

Add to the VIO node:

```python
# In VioPoseNode.__init__():
self.create_service(
    CorrectVioPose, 'vio/correct_pose',
    self._srv_correct_pose_callback
)

# New publisher for VIO status
self.status_pub = self.create_publisher(VioStatus, 'vio/status', 10)

# In _tick(), after estimating pose:
status_msg = VioStatus()
status_msg.header.stamp = self.get_clock().now().to_msg()
status_msg.R_fused = self.R_fused.flatten()  # (9,)
status_msg.p_fused = self.p_fused  # (3,)
status_msg.inlier_count = inliers_count
status_msg.tracking_quality = float(inliers_count) / self.min_inliers
self.status_pub.publish(status_msg)

def _srv_correct_pose_callback(self, request, response):
    """Apply relocalization correction to VIO state."""
    R_corr = request.rotation  # (3, 3) flattened
    p_corr = request.translation  # (3,)
    
    # Blend correction into current estimate
    alpha = min(0.5, request.confidence)  # Conservative blend
    q_est = rotmat_to_quat(self.R_fused)
    q_obs = rotmat_to_quat(R_corr)
    q_corrected = slerp(q_est, q_obs, alpha)
    
    self.R_fused = quat_to_rotmat(q_corrected)
    self.p_fused = (1 - alpha) * self.p_fused + alpha * p_corr
    
    response.success = True
    return response
```

---

## 6. Offline Map Building

### 6.1 Map Builder Script

**File**: `scripts/build_map_offline.py`

```python
class OfflineMapBuilder:
    """
    Build relocalization map offline from sequence of images.
    
    Workflow:
    1. Collect keyframe images (walking around scene)
    2. Extract global + local descriptors
    3. Run SfM (COLMAP) to get 3D points + poses
    4. Build kNN index for retrieval
    5. Save map to disk
    """
    
    def __init__(self, image_dir: str, output_map_dir: str):
        self.image_dir = image_dir
        self.output_dir = output_map_dir
        self.hloc = HLocPipeline(output_map_dir)
    
    def collect_keyframes(self, image_list: List[str]) -> List[Keyframe]:
        """
        Extract features for all images.
        
        Returns list of Keyframe objects.
        """
    
    def run_sfm(self) -> Tuple[List[np.ndarray], Dict]:
        """
        Run COLMAP sparse reconstruction.
        Output: 3D points, keyframe poses, intrinsics.
        """
    
    def build_map(self) -> None:
        """Full pipeline executable."""
        keyframes = self.collect_keyframes(...)
        points_3d, poses = self.run_sfm()
        
        # Populate map
        map_obj = RelocalizationMap(self.output_dir)
        # ... add keyframes, points, build indices
        map_obj.save()
        
        print(f"Map built: {len(keyframes)} keyframes, {len(points_3d)} 3D points")
```

**Usage**:
```bash
python3 scripts/build_map_offline.py \
    --image_dir /path/to/captured/images \
    --output_map_dir /home/cesar/cv_slam_experiment/data/relocalization_map
```

---

## 7. Loss Detection & Triggering

### Automatic Triggering Logic

In the relocalization node:

```python
def _on_vio_status(self, msg: VioStatus) -> None:
    """Monitor VIO quality and trigger relocalization on degradation."""
    self.vio_inlier_count = msg.inlier_count
    
    # Trigger if:
    # 1. Inlier count drops below threshold
    if msg.inlier_count < self.min_inlier_threshold:
        self.get_logger().warn(f"VIO tracking degraded (inliers={msg.inlier_count}), triggering relocalization")
        self._trigger_relocalization_async()
    
    # 2. Tracking quality too low
    if msg.tracking_quality < 0.5:
        self._trigger_relocalization_async()
    
    # 3. (Optional) Chain rule: if previous relocalization was recent but pose diverged
```

---

## 8. Integration Checklist

### Phase 1: Foundation (Weeks 1-2)
- [ ] Install MixVPR, SuperPoint, LightGlue models
- [ ] Implement `map_manager.py` (data structure + I/O)
- [ ] Implement `global_descriptor.py` (MixVPR wrapper)
- [ ] Implement `local_matcher.py` (SuperPoint + LightGlue)
- [ ] Implement `pose_solver.py` (PnP-RANSAC)
- [ ] **Test offline**: `python3 test_relocalization.py` on sample images

### Phase 2: Map Building (Weeks 2-3)
- [ ] Implement `build_map_offline.py` (COLMAP integration)
- [ ] Collect reference sequence (walk around scene, capture ~50-100 keyframes)
- [ ] Build and validate map (check 3D reconstruction quality)
- [ ] Save map to `/data/relocalization_map/`

### Phase 3: ROS 2 Integration (Weeks 3-4)
- [ ] Implement `ros2_relocalization_node.py`
- [ ] Add VioStatus message type to project
- [ ] Update `ros2_vio_pose_node.py` with correction service
- [ ] Test service calls in isolation
- [ ] Integration test with both nodes running

### Phase 4: Validation (Week 4)
- [ ] Indoor testing: lose tracking deliberately, verify relocalization recovers pose
- [ ] Benchmark: measure latency (goal: ~500-1000ms per relocalization)
- [ ] Robustness: test under appearance change, motion blur, occlusion
- [ ] EKF stability: ensure corrections don't cause oscillation

---

## 9. Custom ROS 2 Message Types

Create `src/msg/VioStatus.msg`:
```
std_msgs/Header header
geometry_msgs/PoseStamped current_pose
float32[] R_fused                  # Flattened 3×3 rotation matrix
geometry_msgs/Vector3 p_fused      # Position
int32 inlier_count                 # Number of good optical flow matches
float32 tracking_quality           # (0-1) inlier_count / min_inliers
```

Create `src/srv/CorrectVioPose.srv`:
```
# Request
float32[9] rotation                # Flattened 3×3 rotation matrix
geometry_msgs/Vector3 translation
float32 confidence                 # (0-1) how confident is this pose?

# Response
bool success
string message
```

---

## 10. Performance Targets

| Metric | Target | Note |
|--------|--------|------|
| Retrieval latency | < 200 ms | MixVPR on 1 GPU |
| Matching latency | < 300 ms | SuperPoint + LightGlue |
| PnP-RANSAC latency | < 100 ms | Standard OpenCV |
| **Total reloc cycle** | < 1 sec | Given 30 Hz VIO frame rate |
| Success rate | > 80% | Indoors, lit, moderate viewpoint change |
| Pose accuracy | ±10 cm / 5° | Compared to ground truth |
| EKF stability | No oscillation | Max 0.5 sec settling after correction injection |

---

## 11. Debugging & Validation

### Test Script: `scripts/test_relocalization.py`

```python
def test_on_single_frame():
    """Quick validation: can we relocalize a single test frame?"""
    test_frame = cv2.imread('test_frame.png')
    result = hloc.relocalize(test_frame)
    assert result['success']
    assert result['num_inliers'] > 15
    print(f"✓ Relocalization successful: {result['num_inliers']} inliers")

def benchmark_timing():
    """Profile each stage of the pipeline."""
    
def visualize_matches():
    """Draw matched keypoints for debugging."""
```

### Diagnostic Plot
```
┌─────────────────────────────────────┐
│ HLoc Diagnostics                    │
├─────────────────────────────────────┤
│ Retrieval:  [████████░░] 150 ms     │
│ Matching:   [███████░░░] 280 ms     │
│ PnP+RANSAC: [██░░░░░░░░] 95 ms      │
│                                     │
│ Inliers:    42 / 50 (84%)           │
│ Reproj err: 1.2 px ✓                │
│ Pose conf:  0.92 ✓                  │
│                                     │
│ Match visual:                       │
│ [Query]        [Keyframe]           │
│ o o o      o o │ o o o              │
│  o      o  o   │  o   o             │
│    o o    o    │  o o               │
└─────────────────────────────────────┘
```

---

## 12. Known Challenges & Solutions

| Challenge | Root Cause | Solution |
|-----------|-----------|----------|
| **Slow retrieval** | MixVPR CPU bottleneck | Use GPU, batch queries, caching |
| **Flickering poses** | Bad matches accepted | Raise match confidence threshold, post-filter outliers |
| **EKF oscillation** | Over-correcting poses | Lower `alpha` blending factor, add damping |
| **4 DOF ambiguity** | Monocular depth scale | Use IMU prior OR include known scale landmarks |
| **Memory bloat** | Large descriptor corpus | Hierarchical retrieval, limit keyframe count |
| **Appearance change** | Illumination, season | Use DINOv2 / AnyLoc instead of task-specific CNN |

---

## 13. Immediate Next Steps (Action Items)

1. **Create config file**: `config/relocalization_config.yaml`
   ```yaml
   hloc:
     global_descriptor: mixvpr
     retrieval_top_k: 15
     local_features: superpoint
     matcher: lightglue
     min_inliers_pnp: 15
     max_reproj_error: 2.0
     
   device: cuda
   map_dir: /home/cesar/cv_slam_experiment/data/relocalization_map
   ```

2. **Download models**: Create `scripts/download_models.sh`
   ```bash
   #!/bin/bash
   mkdir -p data/models
   # Download MixVPR weights from HuggingFace
   # Download SuperPoint + LightGlue from GitHub releases
   ```

3. **Reserve time for**:
   - Map collection (30-60 min: walk scene with camera)
   - SfM parameter tuning (varies with environment)
   - ROS 2 message/service testing (1-2 hours)

---

**End of Plan**

---

### Questions to Answer Before Starting

1. **Scope of relocalization**: Are you mapping a single room, building, or outdoor area?
2. **Expected map size**: How many keyframes? (Target: 50-200 for typical indoor room)
3. **Appearance change**: Will lighting/season/clutter change significantly?
4. **Real-time constraint**: Must corrections happen within 1 frame (~33 ms)? Or slower okay?
5. **Fallback**: If relocalization fails, should VIO continue dead-reckoning or pause?

---

**Integration Plan v1.0 — Ready to implement!**
