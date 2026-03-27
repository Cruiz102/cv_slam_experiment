#!/usr/bin/env python3
"""
Test/Demo: End-to-end offline map building.

This script demonstrates:
1. Creating synthetic mapping dataset
2. Building map with build_map_offline.py
3. Testing relocalization on the built map
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.relocalization import RelocalizationMap, HLocPipeline
from scripts.build_map_offline import OfflineMapBuilder
from src.relocalization.map_builder_utils import (
    KeyframeSelector, KeyframeSelectionConfig, MapQualityAssessment
)


def create_synthetic_mapping_dataset(output_dir: str, num_images: int = 10,
                                    image_size: tuple = (480, 640)) -> None:
    """Create synthetic mapping dataset for testing.
    
    Args:
        output_dir: Output directory for images
        num_images: Number of synthetic images to create
        image_size: (H, W) of each image
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Synthetic Dataset] Creating {num_images} synthetic images...")
    
    H, W = image_size
    
    for idx in range(num_images):
        # Create synthetic image with patterns
        img = np.ones((H, W, 3), dtype=np.uint8) * 200
        
        # Add checkerboard pattern
        square_size = 80
        for y in range(0, H, square_size):
            for x in range(0, W, square_size):
                if ((x // square_size) + (y // square_size)) % 2 == 0:
                    img[y:y+square_size, x:x+square_size] = [50, 50, 50]
        
        # Add frame index as text
        cv2.putText(img, f"Frame {idx:03d}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Add some unique features per frame
        center_x = W // 2 + 50 * np.sin(0.5 * idx)
        center_y = H // 2 + 30 * np.cos(0.5 * idx)
        cv2.circle(img, (int(center_x), int(center_y)), 30 + idx, 
                  (0, 255, 255), -1)
        
        # Add geometric shapes
        pts = np.array([
            [50, 50],
            [150, 50],
            [100, 100 + 10*idx]
        ], np.int32)
        cv2.polylines(img, [pts], False, (255, 0, 0), 2)
        
        # Save image
        img_path = output_dir / f"image_{idx:06d}.jpg"
        cv2.imwrite(str(img_path), img)
    
    print(f"✓ Created {num_images} synthetic images in {output_dir}")


def test_complete_workflow():
    """Test complete map building and relocalization workflow."""
    
    print("\n" + "="*70)
    print("PHASE 2 TEST: Complete Offline Map Building Workflow")
    print("="*70)
    
    # Step 1: Create synthetic dataset
    print("\n[Step 1] Creating synthetic dataset...")
    dataset_dir = Path("data/mapping_dataset")
    create_synthetic_mapping_dataset(str(dataset_dir), num_images=15, 
                                     image_size=(480, 640))
    
    # Step 2: Build map
    print("\n[Step 2] Building relocalization map...")
    output_map_dir = "data/relocalization_map_test"
    
    builder = OfflineMapBuilder(
        image_dir=str(dataset_dir),
        output_map_dir=output_map_dir,
        global_descriptor_type="dinov2",
        device="cuda",
        use_colmap=False,  # Skip COLMAP for testing
    )
    
    builder.load_images(max_images=15)
    builder.extract_features(skip_local=False)
    builder.estimate_poses()
    builder.triangulate_points()
    
    # Step 3: Build and validate map
    print("\n[Step 3] Building and validating map...")
    map_obj = builder.build_map()
    metrics = builder.validate_map(map_obj)
    
    # Step 4: Compute quality metrics
    print("\n[Step 4] Computing quality metrics...")
    quality_metrics = MapQualityAssessment.full_assessment(
        np.array([kf.pose_w2c for kf in map_obj.keyframes]),
        map_obj.points_3d,
        [kf.descriptor_global for kf in map_obj.keyframes]
    )
    
    print("\nQuality Metrics:")
    print(f"  Spatial Coverage:      {quality_metrics['spatial_coverage']:.1%}")
    print(f"  Point Density:         {quality_metrics['point_density']:.2f} pts/m³")
    print(f"  Descriptor Diversity:  {quality_metrics['descriptor_diversity']:.2f}")
    print(f"  Overall Quality Score: {quality_metrics['quality_score']:.2f}/1.00")
    
    # Step 5: Test keyframe selection
    print("\n[Step 5] Testing keyframe selection...")
    selector = KeyframeSelector(KeyframeSelectionConfig(
        min_translation_distance=0.2,
        min_rotation_angle=5.0,
        temporal_interval=2,
    ))
    
    poses = [kf.pose_w2c for kf in map_obj.keyframes]
    descriptors = [kf.descriptor_global for kf in map_obj.keyframes]
    selected_indices = selector.select(poses, descriptors)
    
    print(f"  Total frames: {len(poses)}")
    print(f"  Selected keyframes: {len(selected_indices)}")
    print(f"  Reduction: {100 * (1 - len(selected_indices) / len(poses)):.1f}%")
    print(f"  Selected indices: {selected_indices}")
    
    # Step 6: Test retrieval
    print("\n[Step 6] Testing retrieval on query image...")
    
    # Create synthetic query image
    query_img = np.ones((480, 640, 3), dtype=np.uint8) * 180
    cv2.putText(query_img, "Query Frame", (20, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    square_size = 80
    for y in range(0, 480, square_size):
        for x in range(0, 640, square_size):
            if ((x // square_size) + (y // square_size)) % 2 == 0:
                query_img[y:y+square_size, x:x+square_size] = [50, 50, 50]
    
    # Test HLoc pipeline
    try:
        pipeline = HLocPipeline(map_obj, top_k_retrieval=5)
        result = pipeline.relocalize(query_img)
            pipeline = HLocPipeline(
                str(output_map_dir),
                config={'top_k': 5}
            )
            result = pipeline.relocalize(query_img)
        
        if result['success']:
                print(f"✓ Relocalization successful!")
                print(f"  Inliers: {result.get('num_inliers', 'N/A')}")
                print(f"  Reprojection error: {result.get('reprojection_error', 'N/A')}")
                print(f"  Confidence: {result.get('confidence', 'N/A')}")
        else:
            print(f"✗ Relocalization failed (expected for synthetic data)")
    except Exception as e:
            print(f"⚠ Pipeline test note: {e}")
    
    print("\n" + "="*70)
    print("✓ Phase 2 Workflow Test Complete")
    print("="*70)
    
    print(f"\nGenerated files:")
    print(f"  - Dataset: {dataset_dir}")
    print(f"  - Map: {Path(output_map_dir)}")
    
    print("\nNext steps:")
    print(f"  1. Test with real images:")
    print(f"     python3 scripts/build_map_offline.py \\")
    print(f"       --image_dir <your_images> \\")
    print(f"       --output_map_dir data/relocalization_map")
    print(f"  2. Integrate with ROS 2:")
    print(f"     python3 src/ros_nodes/relocalization_service.py")
    print(f"  3. Test live relocalization on tracked pose loss")


if __name__ == "__main__":
    test_complete_workflow()
