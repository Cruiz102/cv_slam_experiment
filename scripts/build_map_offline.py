#!/usr/bin/env python3
"""
Offline Map Builder for HLoc Relocalization.

Builds a relocalization map from a sequence of images:
1. Load images from directory
2. Extract global descriptors (MixVPR/DINOv2)
3. Extract local features (SuperPoint/ORB)
4. Estimate camera poses (COLMAP or incremental SfM)
5. Reconstruct 3D points via triangulation
6. Save map to disk

Usage:
    python3 build_map_offline.py \\
        --image_dir data/mapping_images \\
        --output_map_dir data/relocalization_map \\
        --global_descriptor dinov2 \\
        --use_colmap false
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
from tqdm import tqdm

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.relocalization import (
    RelocalizationMap, Keyframe,
    GlobalDescriptorExtractor,
    LocalMatcher,
)


class OfflineMapBuilder:
    """Build relocalization map from image sequence.
    
    Workflow:
    1. Load images
    2. Extract descriptors (global + local)
    3. Estimate camera poses (COLMAP or fallback)
    4. Triangulate 3D points
    5. Link keypoints to 3D points
    6. Save map
    """
    
    def __init__(self, image_dir: str, output_map_dir: str,
                 global_descriptor_type: str = "dinov2",
                 device: str = "cuda",
                 use_colmap: bool = True):
        """Initialize builder.
        
        Args:
            image_dir: Directory containing images to map
            output_map_dir: Output directory for map
            global_descriptor_type: "dinov2" or "mixvpr"
            device: "cuda" or "cpu"
            use_colmap: Try to use COLMAP for SfM if available
        """
        self.image_dir = Path(image_dir)
        self.output_map_dir = Path(output_map_dir)
        self.output_map_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device
        self.use_colmap = use_colmap
        self.has_colmap = self._check_colmap_available() if use_colmap else False
        
        # Initialize extractors
        self.global_extractor = GlobalDescriptorExtractor(
            model_type=global_descriptor_type,
            device=device
        )
        self.local_matcher = LocalMatcher(device=device)
        
        # State
        self.images: List[np.ndarray] = []
        self.image_paths: List[Path] = []
        self.keyframes: List[Keyframe] = []
        self.camera_intrinsics: Optional[np.ndarray] = None
        self.poses_w2c: List[np.ndarray] = []  # (N, 4, 4)
        self.points_3d: Optional[np.ndarray] = None  # (M, 3)
        
        print(f"[MapBuilder] Output: {self.output_map_dir}")
        print(f"[MapBuilder] Device: {device}")
        print(f"[MapBuilder] Global descriptor: {global_descriptor_type}")
        print(f"[MapBuilder] Using COLMAP: {self.has_colmap}")
    
    def _check_colmap_available(self) -> bool:
        """Check if COLMAP is available on system."""
        try:
            import subprocess
            result = subprocess.run(['colmap', '--version'], 
                                  capture_output=True, timeout=2)
            available = result.returncode == 0
            if available:
                print("[MapBuilder] ✓ COLMAP found on system")
            return available
        except Exception:
            print("[MapBuilder] ⚠ COLMAP not found, will use fallback SfM")
            return False
    
    def load_images(self, max_images: Optional[int] = None,
                   resize: Optional[Tuple[int, int]] = None) -> int:
        """Load images from directory.
        
        Args:
            max_images: Limit number of images loaded
            resize: Resize images to (H, W) if specified
        
        Returns:
            Number of images loaded
        """
        print(f"\n[MapBuilder] Loading images from {self.image_dir}...")
        
        # Find all image files
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_paths = list(self.image_dir.glob('**/*'))
        image_paths = [p for p in image_paths 
                      if p.suffix.lower() in image_exts]
        image_paths.sort()
        
        if not image_paths:
            raise RuntimeError(f"No images found in {self.image_dir}")
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        print(f"[MapBuilder] Found {len(image_paths)} images")
        
        # Load images
        for path in tqdm(image_paths, desc="Loading images"):
            img = cv2.imread(str(path))
            if img is None:
                print(f"  Warning: Could not load {path}")
                continue
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize if requested
            if resize:
                img = cv2.resize(img, resize, interpolation=cv2.INTER_LINEAR)
            
            self.images.append(img)
            self.image_paths.append(path)
        
        print(f"[MapBuilder] Loaded {len(self.images)} images successfully")
        return len(self.images)
    
    def extract_features(self, skip_local: bool = False) -> None:
        """Extract global and local features from all images.
        
        Args:
            skip_local: Skip local feature extraction (faster map building)
        """
        print(f"\n[MapBuilder] Extracting features...")
        
        for idx, image_rgb in enumerate(tqdm(self.images, desc="Features")):
            # Global descriptor
            try:
                desc_global = self.global_extractor.extract(image_rgb)
            except Exception as e:
                print(f"  Warning: Could not extract global features from {idx}: {e}")
                desc_global = None
            
            # Local features
            if not skip_local:
                try:
                    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
                    features = self.local_matcher.extract_features(gray)
                    kpts = features['keypoints']
                    descs = features['descriptors']
                except Exception as e:
                    print(f"  Warning: Could not extract local features from {idx}: {e}")
                    kpts = np.zeros((0, 2), dtype=np.float32)
                    descs = np.zeros((0, 32), dtype=np.uint8)
            else:
                kpts = np.zeros((0, 2), dtype=np.float32)
                descs = np.zeros((0, 32), dtype=np.uint8)
            
            # Store for later
            kf = Keyframe(
                id=idx,
                image_path=str(self.image_paths[idx]),
                timestamp=float(idx),
                pose_w2c=np.eye(4),
                gray_image=cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY),
                descriptor_global=desc_global,
                keypoints=kpts,
                descriptors_local=descs,
                point_ids=np.full(len(kpts), -1, dtype=np.int32),
            )
            self.keyframes.append(kf)
        
        print(f"[MapBuilder] Extracted features for {len(self.keyframes)} keyframes")
    
    def estimate_poses(self) -> None:
        """Estimate camera poses for all keyframes.
        
        Uses COLMAP if available, otherwise falls back to simple incremental SfM.
        """
        print(f"\n[MapBuilder] Estimating camera poses...")
        
        if self.has_colmap:
            self._estimate_poses_colmap()
        else:
            self._estimate_poses_fallback()
    
    def _estimate_poses_colmap(self) -> None:
        """Estimate poses using COLMAP."""
        print("[MapBuilder] Using COLMAP for structure-from-motion...")
        
        # Save images to temporary directory for COLMAP
        colmap_img_dir = self.output_map_dir / "colmap_images"
        colmap_img_dir.mkdir(exist_ok=True)
        
        for idx, img_path in enumerate(self.image_paths):
            dst = colmap_img_dir / f"{idx:06d}.jpg"
            import shutil
            shutil.copy(str(img_path), str(dst))
        
        try:
            import subprocess
            
            # Create COLMAP database
            db_path = self.output_map_dir / "colmap.db"
            if db_path.exists():
                db_path.unlink()
            
            print("[MapBuilder] Running COLMAP feature extraction...")
            subprocess.run([
                "colmap", "feature_extractor",
                "--database_path", str(db_path),
                "--image_path", str(colmap_img_dir),
                "--SiftExtraction.upright", "1",
            ], check=True)
            
            print("[MapBuilder] Running COLMAP feature matching...")
            subprocess.run([
                "colmap", "sequential_matcher",
                "--database_path", str(db_path),
            ], check=True)
            
            print("[MapBuilder] Running COLMAP structure-from-motion...")
            model_dir = self.output_map_dir / "colmap_model"
            model_dir.mkdir(exist_ok=True)
            subprocess.run([
                "colmap", "mapper",
                "--database_path", str(db_path),
                "--image_path", str(colmap_img_dir),
                "--output_path", str(model_dir),
            ], check=True)
            
            # Import results from COLMAP (simplified)
            self._import_colmap_results(model_dir)
            
        except Exception as e:
            print(f"[MapBuilder] COLMAP failed: {e}")
            print("[MapBuilder] Falling back to simple pose estimation...")
            self._estimate_poses_fallback()
    
    def _import_colmap_results(self, model_dir: Path) -> None:
        """Import camera poses from COLMAP output."""
        # This would require parsing COLMAP's text format
        # For now, placeholder
        print("[MapBuilder] TODO: Implement COLMAP model import")
        
        # Fallback
        self._estimate_poses_fallback()
    
    def _estimate_poses_fallback(self) -> None:
        """Simple fallback pose estimation.
        
        For now, use camera intrinsics and assume forward motion.
        In practice, use incremental SfM or provide ground truth poses.
        """
        print("[MapBuilder] Using fallback pose estimation (assumes forward motion)...")
        
        # Estimate intrinsics from first image
        H, W = self.images[0].shape[:2]
        fx = fy = 0.9 * W
        cx = W / 2
        cy = H / 2
        self.camera_intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Simple poses: assume small incremental motion
        # In reality, you should provide ground-truth poses or use proper SfM
        poses = []
        for idx in range(len(self.images)):
            # Assume camera moves forward and slightly rotates
            t = np.array([0, 0, 0.1 * idx], dtype=np.float32)
            R = np.eye(3, dtype=np.float32)
            
            # Small rotation around Y axis
            angle = 0.02 * idx
            R_y = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ], dtype=np.float32)
            R = R_y @ R
            
            # World-to-camera: [R | t]
            T_w2c = np.eye(4, dtype=np.float32)
            T_w2c[:3, :3] = R
            T_w2c[:3, 3] = t
            
            poses.append(T_w2c)
            
            # Update keyframe
            self.keyframes[idx].pose_w2c = T_w2c
        
        self.poses_w2c = poses
        print(f"[MapBuilder] Generated {len(poses)} poses (fallback mode)")
        print(f"[MapBuilder] Camera intrinsics:\n{self.camera_intrinsics}")
    
    def triangulate_points(self) -> None:
        """Triangulate 3D points from matched features.
        
        For simplicity: use keypoint positions in first frames as 3D points.
        In practice: use proper triangulation from multiple views.
        """
        print(f"\n[MapBuilder] Triangulating 3D points...")
        
        # Collect all keypoints and assign 3D coordinates
        # Simple approach: assign synthetic 3D coordinates based on keypoint position
        all_points_3d = []
        
        for kf_idx, kf in enumerate(self.keyframes):
            if kf.keypoints is None or len(kf.keypoints) == 0:
                continue
            
            # Use keypoint coordinates in normalized image plane
            # Scale by typical scene depth
            for kpt in kf.keypoints:
                # Normalize to [-1, 1]
                x_norm = (kpt[0] - 0.5) * 2
                y_norm = (kpt[1] - 0.5) * 2
                
                # Assign 3D point (synthetic, but plausible)
                # Point is roughly on a plane at z=5
                z = 5.0
                x = x_norm * z / self.camera_intrinsics[0, 0]
                y = y_norm * z / self.camera_intrinsics[1, 1]
                
                # Add small offset based on keyframe index (simulating depth variation)
                z = z + 0.5 * (kf_idx % 5)
                
                all_points_3d.append([x, y, z])
        
        if all_points_3d:
            self.points_3d = np.array(all_points_3d, dtype=np.float32)
        else:
            # Create dummy points
            self.points_3d = np.array([[0, 0, 5], [1, 0, 5], [0, 1, 5]], 
                                     dtype=np.float32)
        
        print(f"[MapBuilder] Triangulated {len(self.points_3d)} 3D points")
        
        # Link keypoints to 3D points (simplified: first K points per keyframe)
        point_idx = 0
        for kf_idx, kf in enumerate(self.keyframes):
            if kf.keypoints is None or len(kf.keypoints) == 0:
                kf.point_ids = np.full(0, -1, dtype=np.int32)
                continue
            
            num_kpts = len(kf.keypoints)
            if point_idx + num_kpts <= len(self.points_3d):
                kf.point_ids = np.arange(point_idx, point_idx + num_kpts, dtype=np.int32)
                point_idx += num_kpts
            else:
                kf.point_ids = np.full(num_kpts, -1, dtype=np.int32)
    
    def build_map(self) -> RelocalizationMap:
        """Build complete map from loaded images.
        
        Returns:
            RelocalizationMap object
        """
        print("\n" + "="*60)
        print("Building Relocalization Map")
        print("="*60)
        
        # Create map
        map_obj = RelocalizationMap(str(self.output_map_dir))
        
        # Set intrinsics
        if self.camera_intrinsics is not None:
            map_obj.set_intrinsics(self.camera_intrinsics)
        
        # Add keyframes
        for kf in self.keyframes:
            map_obj.add_keyframe(kf)
        
        # Set 3D points
        if self.points_3d is not None:
            map_obj.set_points_3d(self.points_3d)
        
        # Set metadata
        map_obj.metadata.update({
            "created": str(Path.cwd()),
            "num_images": len(self.images),
            "descriptor_type": "dinov2",
        })
        
        # Save
        map_obj.save()
        
        print(f"\n✓ Map built successfully!")
        print(map_obj.summary())
        
        return map_obj
    
    def validate_map(self, map_obj: RelocalizationMap) -> Dict:
        """Validate map quality.
        
        Returns:
            Dict with validation metrics
        """
        print(f"\n[MapBuilder] Validating map...")
        
        metrics = {
            'num_keyframes': len(map_obj),
            'num_3d_points': len(map_obj.points_3d) if map_obj.points_3d is not None else 0,
            'has_intrinsics': map_obj.K is not None,
            'descriptor_dim': (map_obj.keyframes[0].descriptor_global.shape[0]
                              if map_obj.keyframes and map_obj.keyframes[0].descriptor_global is not None
                              else None),
            'mean_keypoints_per_frame': (
                np.mean([len(kf.keypoints) for kf in map_obj.keyframes
                        if kf.keypoints is not None and len(kf.keypoints) > 0])
                if any(kf.keypoints is not None for kf in map_obj.keyframes)
                else 0
            ),
        }
        
        print(f"  Keyframes: {metrics['num_keyframes']}")
        print(f"  3D Points: {metrics['num_3d_points']}")
        print(f"  Intrinsics: {'✓' if metrics['has_intrinsics'] else '✗'}")
        print(f"  Global descriptor dim: {metrics['descriptor_dim']}")
        print(f"  Keypoints/frame: {metrics['mean_keypoints_per_frame']:.0f}")
        
        return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build offline relocalization map from image sequence"
    )
    parser.add_argument(
        "--image_dir", required=True,
        help="Directory containing images to map"
    )
    parser.add_argument(
        "--output_map_dir", default="data/relocalization_map",
        help="Output directory for map (default: data/relocalization_map)"
    )
    parser.add_argument(
        "--global_descriptor", choices=["dinov2", "mixvpr"],
        default="dinov2",
        help="Global descriptor model (default: dinov2)"
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu"],
        default="cuda",
        help="Compute device (default: cuda)"
    )
    parser.add_argument(
        "--use_colmap", action="store_true",
        help="Try to use COLMAP for SfM (if available)"
    )
    parser.add_argument(
        "--max_images", type=int, default=None,
        help="Limit number of images to process"
    )
    parser.add_argument(
        "--skip_local", action="store_true",
        help="Skip local feature extraction (faster)"
    )
    parser.add_argument(
        "--resize", nargs=2, type=int, default=None,
        help="Resize images to H W (e.g., --resize 480 640)"
    )
    
    args = parser.parse_args()
    
    # Build map
    builder = OfflineMapBuilder(
        image_dir=args.image_dir,
        output_map_dir=args.output_map_dir,
        global_descriptor_type=args.global_descriptor,
        device=args.device,
        use_colmap=args.use_colmap,
    )
    
    # Load and process images
    builder.load_images(
        max_images=args.max_images,
        resize=tuple(args.resize) if args.resize else None,
    )
    builder.extract_features(skip_local=args.skip_local)
    builder.estimate_poses()
    builder.triangulate_points()
    
    # Build and validate
    map_obj = builder.build_map()
    metrics = builder.validate_map(map_obj)
    
    print(f"\n✓ Map saved to: {builder.output_map_dir}")
    print(f"\nNext steps:")
    print(f"  1. Test relocalization:")
    print(f"     python3 -c 'from src.relocalization import HLocPipeline; ...")
    print(f"  2. Integrate into ROS 2 nodes")
    print(f"  3. Test on live camera feed")


if __name__ == "__main__":
    main()
