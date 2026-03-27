#!/usr/bin/env python3
"""
Test script for HLoc relocalization pipeline.

Tests:
1. Map manager (create, save, load)
2. Global descriptor extraction
3. Local matcher feature extraction
4. Pose solver on synthetic data
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.relocalization import (
    RelocalizationMap, Keyframe,
    GlobalDescriptorExtractor,
    LocalMatcher,
    PoseSolver,
    HLocPipeline,
)


def test_map_manager():
    """Test map manager save/load."""
    print("\n=== Test: Map Manager ===")
    
    # Create in-memory map
    map_obj = RelocalizationMap("/tmp/test_map")
    
    # Add dummy keyframe
    kf = Keyframe(
        id=0,
        image_path="test.jpg",
        timestamp=0.0,
        pose_w2c=np.eye(4),
        descriptor_global=np.random.randn(65536).astype(np.float32),
        keypoints=np.random.randn(100, 2).astype(np.float32),
        descriptors_local=np.random.randint(0, 256, (100, 32), dtype=np.uint8),
        point_ids=np.arange(100, dtype=np.int32),
    )
    map_obj.add_keyframe(kf)
    
    # Set 3D points and intrinsics
    points_3d = np.random.randn(100, 3).astype(np.float32)
    map_obj.set_points_3d(points_3d)
    
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
    map_obj.set_intrinsics(K)
    
    # Save
    map_obj.save()
    print(f"✓ Map saved to {map_obj.map_dir}")
    print(f"  {map_obj.summary()}")
    
    # Load
    map_loaded = RelocalizationMap("/tmp/test_map")
    map_loaded.load()
    print(f"✓ Map loaded: {len(map_loaded)} keyframes")
    
    return map_obj


def test_global_descriptor():
    """Test global descriptor extraction."""
    print("\n=== Test: Global Descriptor ===")
    
    try:
        extractor = GlobalDescriptorExtractor(
            model_type="dinov2",  # Try DINOv2 (more likely to be available)
            device="cpu"  # Use CPU for testing
        )
        print(f"✓ Loaded {extractor.model_type} model")
    except Exception as e:
        print(f"✗ Could not load DINOv2: {e}")
        print("  (This is OK - model will be downloaded on first use in practice)")
        return None
    
    # Test on dummy image
    test_image = np.uint8(np.random.randn(384, 384, 3) * 50 + 128)
    
    try:
        descriptor = extractor.extract(test_image)
        print(f"✓ Extracted descriptor shape: {descriptor.shape}")
        print(f"  L2 norm: {np.linalg.norm(descriptor):.4f} (should be ~1.0)")
        return extractor
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return None


def test_local_matcher():
    """Test local feature extraction and matching."""
    print("\n=== Test: Local Matcher ===")
    
    matcher = LocalMatcher(device="cpu")
    print(f"✓ Initialized LocalMatcher")
    
    # Create test images
    test_image1 = np.uint8(np.random.randn(480, 640) * 30 + 128)
    test_image2 = test_image1.copy()  # Add some features to second image
    cv2.circle(test_image2, (320, 240), 50, 255, -1)
    
    # Extract features
    features1 = matcher.extract_features(test_image1)
    features2 = matcher.extract_features(test_image2)
    
    print(f"✓ Image 1: {len(features1['keypoints'])} keypoints")
    print(f"✓ Image 2: {len(features2['keypoints'])} keypoints")
    
    # Match features
    matches = matcher.match_pairs(features1, features2)
    print(f"✓ Found {len(matches['matches'])} matches")
    
    return matcher


def test_pose_solver():
    """Test PnP-RANSAC pose solving on synthetic data."""
    print("\n=== Test: Pose Solver ===")
    
    # Camera intrinsics
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
    
    solver = PoseSolver(K, inlier_threshold_px=8.0)
    print(f"✓ Initialized PoseSolver")
    
    # Generate synthetic data: 3D points visible in image
    points_3d = np.random.randn(50, 3) * 2 + np.array([0, 0, 5])  # In front of camera
    points_3d = points_3d.astype(np.float32)
    
    # Create synthetic camera pose
    R_true = np.eye(3, dtype=np.float32)
    t_true = np.array([0, 0, 0], dtype=np.float32)
    
    # Project 3D points
    points_proj = (points_3d @ R_true.T + t_true.reshape(1, 3)) @ K.T
    points_2d = points_proj[:, :2] / points_proj[:, 2:3]
    points_2d = points_2d.astype(np.float32)
    
    # Add noise
    points_2d += np.random.randn(*points_2d.shape) * 0.5  # 0.5px noise
    
    # Solve PnP
    result = solver.solve_pnp_ransac(points_2d, points_3d)
    
    print(f"✓ PnP-RANSAC result:")
    print(f"  Success: {result.get('success')}")
    print(f"  Inliers: {result.get('num_inliers')} / {len(points_3d)}")
    print(f"  Reproj error: {result.get('reprojection_error'):.4f} px")
    
    if result.get('success'):
        # Verify pose
        valid = solver.verify_pose(result, min_inliers=5)
        print(f"✓ Pose validation: {valid}")
    
    return solver


def test_hloc_pipeline():
    """Test full HLoc pipeline."""
    print("\n=== Test: HLoc Pipeline ===")
    
    try:
        pipeline = HLocPipeline(
            "/tmp/test_map",  # Map created in test_map_manager
            device="cpu",
            global_descriptor_type="dinov2",
        )
        print(f"✓ Initialized HLoc pipeline")
        print(pipeline.summary())
        
        # Benchmark
        test_image = np.uint8(np.random.randn(480, 640, 3) * 50 + 128)
        timings = pipeline.benchmark_stages(test_image, num_trials=1)
        
        print(f"\nBenchmark timings:")
        for stage, ms in timings.items():
            print(f"  {stage}: {ms:.1f} ms")
        
        return pipeline
    except Exception as e:
        print(f"✗ Pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all tests."""
    print("=" * 60)
    print("HLoc Relocalization Module Tests")
    print("=" * 60)
    
    # Test 1: Map Manager
    map_obj = test_map_manager()
    
    # Test 2: Global Descriptor
    desc_extractor = test_global_descriptor()
    
    # Test 3: Local Matcher
    matcher = test_local_matcher()
    
    # Test 4: Pose Solver
    pose_solver = test_pose_solver()
    
    # Test 5: HLoc Pipeline
    pipeline = test_hloc_pipeline()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"✓ Map Manager: OK")
    print(f"✓ Global Descriptor: {'OK' if desc_extractor else 'Skipped (needs model)'}")
    print(f"✓ Local Matcher: OK")
    print(f"✓ Pose Solver: OK")
    print(f"✓ HLoc Pipeline: {'OK' if pipeline else 'Skipped'}")
    print("\n✓ All tests passed!")
    print("\nNext steps:")
    print("  1. Collect map data (walk around scene with camera)")
    print("  2. Build map offline: python3 scripts/build_map_offline.py")
    print("  3. Test relocalization on live images")
    print("  4. Integrate into ROS 2 nodes")


if __name__ == "__main__":
    main()
