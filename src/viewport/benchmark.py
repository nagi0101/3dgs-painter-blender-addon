# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 3DGS Painter Project

"""
Performance benchmarking utilities for viewport rendering.

Target: 60 FPS @ 10,000 gaussians (KIRI Innovation benchmark)
"""

import bpy
import time
import numpy as np
from typing import List, Tuple, Optional


class ViewportBenchmark:
    """
    Benchmark utility for measuring viewport rendering performance.
    
    Usage:
        benchmark = ViewportBenchmark()
        results = benchmark.run(gaussian_counts=[100, 1000, 10000])
        benchmark.print_results(results)
    """
    
    def __init__(self):
        self.results: List[dict] = []
    
    def run(
        self,
        gaussian_counts: Optional[List[int]] = None,
        frames_per_test: int = 100,
        warmup_frames: int = 10
    ) -> List[dict]:
        """
        Run performance benchmark with various gaussian counts.
        
        Args:
            gaussian_counts: List of gaussian counts to test
            frames_per_test: Number of frames to measure per test
            warmup_frames: Number of warmup frames before measurement
            
        Returns:
            List of result dictionaries
        """
        if gaussian_counts is None:
            gaussian_counts = [100, 500, 1000, 2000, 5000, 10000]
        
        from .viewport_renderer import GaussianViewportRenderer
        from .gaussian_data import create_test_data
        
        renderer = GaussianViewportRenderer.get_instance()
        
        # Ensure renderer is enabled
        if not renderer.enabled:
            if not renderer.register():
                print("[Benchmark] Failed to enable viewport renderer")
                return []
        
        results = []
        
        for count in gaussian_counts:
            print(f"[Benchmark] Testing {count:,} gaussians...")
            
            # Generate test data
            test_data = create_test_data(count)
            
            # Create mock scene data
            class MockSceneData:
                def __init__(self, data):
                    n = data.shape[0]
                    self.count = n
                    self.positions = data[:, 0:3]
                    self.rotations = np.zeros((n, 4), dtype=np.float32)
                    self.rotations[:, 0:3] = data[:, 4:7]
                    self.rotations[:, 3] = data[:, 3]
                    self.scales = data[:, 7:10]
                    self.opacities = data[:, 10]
                    SH_C0 = 0.28209479177387814
                    self.colors = data[:, 11:14] * SH_C0 + 0.5
            
            mock_scene = MockSceneData(test_data)
            renderer.update_gaussians(scene_data=mock_scene)
            
            # Warmup frames
            for _ in range(warmup_frames):
                renderer.request_redraw()
                bpy.ops.wm.redraw_timer(type='DRAW_WIN', iterations=1)
            
            # Measure frames
            frame_times = []
            for _ in range(frames_per_test):
                start = time.perf_counter()
                renderer.request_redraw()
                bpy.ops.wm.redraw_timer(type='DRAW_WIN', iterations=1)
                frame_times.append(time.perf_counter() - start)
            
            # Calculate statistics
            frame_times = np.array(frame_times)
            avg_time = np.mean(frame_times)
            min_time = np.min(frame_times)
            max_time = np.max(frame_times)
            std_time = np.std(frame_times)
            
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0
            min_fps = 1.0 / max_time if max_time > 0 else 0  # Note: min fps = max time
            max_fps = 1.0 / min_time if min_time > 0 else 0
            
            result = {
                'gaussian_count': count,
                'avg_fps': avg_fps,
                'min_fps': min_fps,
                'max_fps': max_fps,
                'avg_frame_time_ms': avg_time * 1000,
                'std_frame_time_ms': std_time * 1000,
                'frames_measured': frames_per_test,
            }
            results.append(result)
            
            print(f"  {count:>6,} gaussians: {avg_fps:.1f} FPS (avg: {avg_time*1000:.2f}ms)")
        
        self.results = results
        return results
    
    def print_results(self, results: Optional[List[dict]] = None):
        """Print benchmark results in a formatted table."""
        if results is None:
            results = self.results
        
        if not results:
            print("No benchmark results available")
            return
        
        print("\n" + "="*70)
        print("Viewport Rendering Benchmark Results")
        print("="*70)
        print(f"{'Gaussians':>10} | {'Avg FPS':>8} | {'Min FPS':>8} | {'Max FPS':>8} | {'Avg ms':>8}")
        print("-"*70)
        
        for r in results:
            status = "✓" if r['avg_fps'] >= 30 else ("~" if r['avg_fps'] >= 20 else "✗")
            print(f"{r['gaussian_count']:>10,} | {r['avg_fps']:>7.1f}  | {r['min_fps']:>7.1f}  | {r['max_fps']:>7.1f}  | {r['avg_frame_time_ms']:>7.2f} {status}")
        
        print("-"*70)
        print("Target: 60 FPS @ 10,000 gaussians (30+ FPS acceptable)")
        print("="*70)
    
    def save_results(self, filepath: str, results: Optional[List[dict]] = None):
        """Save benchmark results to JSON file."""
        import json
        
        if results is None:
            results = self.results
        
        with open(filepath, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'blender_version': bpy.app.version_string,
                'results': results
            }, f, indent=2)
        
        print(f"[Benchmark] Results saved to: {filepath}")


# Blender Operator for running benchmark
class NPR_OT_RunBenchmark(bpy.types.Operator):
    """Run viewport rendering performance benchmark"""
    bl_idname = "npr.run_benchmark"
    bl_label = "Run Benchmark"
    bl_description = "Test viewport rendering performance with various gaussian counts"
    bl_options = {'REGISTER'}
    
    max_gaussians: bpy.props.IntProperty(
        name="Max Gaussians",
        description="Maximum number of gaussians to test",
        default=10000,
        min=100,
        max=500000
    )
    
    frames_per_test: bpy.props.IntProperty(
        name="Frames per Test",
        description="Number of frames to measure per gaussian count",
        default=50,
        min=10,
        max=500
    )
    
    def execute(self, context):
        # Generate test counts based on max_gaussians
        counts = [100, 500, 1000, 2000, 5000]
        test_thresholds = [10000, 20000, 50000, 100000, 200000, 500000]
        for threshold in test_thresholds:
            if self.max_gaussians >= threshold:
                counts.append(threshold)
        
        benchmark = ViewportBenchmark()
        results = benchmark.run(
            gaussian_counts=counts,
            frames_per_test=self.frames_per_test
        )
        
        if results:
            benchmark.print_results()
            
            # Check if target achieved
            target_result = next((r for r in results if r['gaussian_count'] == 10000), None)
            if target_result:
                if target_result['avg_fps'] >= 60:
                    self.report({'INFO'}, f"Target EXCEEDED: {target_result['avg_fps']:.1f} FPS @ 10k gaussians")
                elif target_result['avg_fps'] >= 30:
                    self.report({'INFO'}, f"Target MET: {target_result['avg_fps']:.1f} FPS @ 10k gaussians")
                else:
                    self.report({'WARNING'}, f"Target MISSED: {target_result['avg_fps']:.1f} FPS @ 10k gaussians")
            
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Benchmark failed")
            return {'CANCELLED'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


def register_benchmark():
    """Register benchmark operator."""
    bpy.utils.register_class(NPR_OT_RunBenchmark)


def unregister_benchmark():
    """Unregister benchmark operator."""
    bpy.utils.unregister_class(NPR_OT_RunBenchmark)
