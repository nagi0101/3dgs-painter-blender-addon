"""
Test script for npr_core module (standalone, no Blender dependency)

Run this to verify Phase 1 completion:
    python -m pytest tests/test_npr_core.py

Or run directly:
    cd src
    python -c "from npr_core import NPRCoreAPI; api = NPRCoreAPI(); api.create_circular_brush(); print('SUCCESS!')"
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

import numpy as np


def test_gaussian_creation():
    """Test Gaussian2D creation"""
    from npr_core.gaussian import Gaussian2D, create_test_gaussian

    g = create_test_gaussian(1.0, 2.0)
    assert g.position[0] == 1.0
    assert g.position[1] == 2.0
    assert g.position[2] == 0.0  # 2D constraint
    print("✓ Gaussian creation test passed")


def test_scene_data():
    """Test SceneData operations"""
    from npr_core.scene_data import SceneData
    from npr_core.gaussian import create_test_gaussian

    scene = SceneData()
    assert len(scene) == 0

    # Add some gaussians
    gaussians = [create_test_gaussian(i, i) for i in range(10)]
    scene.extend(gaussians)
    assert len(scene) == 10

    # Clear
    scene.clear()
    assert len(scene) == 0

    print("✓ SceneData test passed")


def test_brush_creation():
    """Test brush creation (no Blender dependency)"""
    from npr_core.brush import BrushStamp

    brush = BrushStamp()
    brush.create_circular_pattern(num_gaussians=20, radius=0.5)

    assert len(brush.gaussians) == 20
    assert len(brush.base_gaussians) == 20

    # Test parameter application
    brush.apply_parameters(
        color=np.array([1.0, 0.0, 0.0]), size_multiplier=2.0, global_opacity=0.5
    )

    assert np.allclose(brush.current_color, [1.0, 0.0, 0.0])
    assert brush.current_size_multiplier == 2.0

    print("✓ Brush creation test passed")


def test_api():
    """Test NPRCoreAPI (synchronous API)"""
    from npr_core.api import NPRCoreAPI

    api = NPRCoreAPI()

    # Create brush
    success = api.create_circular_brush(num_gaussians=15, radius=0.2)
    assert success

    # Set parameters
    success = api.set_brush_parameters(
        color=np.array([0.0, 1.0, 0.0]), size_multiplier=1.5
    )
    assert success

    # Check scene is empty
    assert api.get_gaussian_count() == 0

    # Note: Stroke operations require spline module which may have scipy dependency
    # We'll test those in integration tests

    print("✓ API test passed")


def test_gpu_context():
    """Test GPU context (doesn't require CUDA to be available)"""
    from npr_core.gpu_context import get_gpu_context

    ctx = get_gpu_context()

    # This should not crash even without PyTorch/CUDA
    try:
        ctx.initialize()
        print(f"  GPU available: {ctx.is_cuda_available()}")
        print(f"  PyTorch available: {ctx.is_torch_available()}")
    except Exception as e:
        print(f"  GPU initialization skipped: {e}")

    # Memory usage should return dict even if CUDA unavailable
    mem = ctx.get_memory_usage()
    assert isinstance(mem, dict)
    assert "allocated" in mem

    print("✓ GPU context test passed")


if __name__ == "__main__":
    print("\n=== Testing npr_core module (Phase 1) ===\n")

    test_gaussian_creation()
    test_scene_data()
    test_brush_creation()
    test_api()
    test_gpu_context()

    print("\n=== All tests passed! ===")
    print("\nPhase 1 Core Refactoring: ✓ COMPLETE")
    print("\nNext steps:")
    print("  - Phase 2: Dependency management")
    print("  - Phase 3: Viewport rendering (GLSL)")
    print("  - Phase 4: Painting interaction")
