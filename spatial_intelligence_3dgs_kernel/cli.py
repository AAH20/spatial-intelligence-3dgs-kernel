"""
Command-line Interface for Spatial Intelligence & 3D Gaussian Splatting Kernel.
Pure Python standard library (argparse, json, time).
"""

from __future__ import annotations
import argparse
import json
import sys
import time

from spatial_intelligence_3dgs_kernel.engine import SpatialIntelligenceEngine


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="spatial-3dgs-kernel",
        description="Spatial Intelligence & 3D Gaussian Splatting Submodular Optimization Kernel",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: benchmark-all
    bench_parser = subparsers.add_parser("benchmark-all", help="Execute complete suite benchmark across all 5 spatial solvers")
    bench_parser.add_argument("--gaussians", type=int, default=5000, help="Number of synthetic 3D Gaussians (default: 5000)")
    bench_parser.add_argument("--json", action="store_true", help="Output pure JSON format")

    # Command: prune
    prune_parser = subparsers.add_parser("prune", help="Run submodular knapsack pruning on 3DGS scene")
    prune_parser.add_argument("--count", type=int, default=10000, help="Candidate count")
    prune_parser.add_argument("--target", type=int, default=4000, help="Target budget count")

    # Command: optimize-poses
    pose_parser = subparsers.add_parser("optimize-poses", help="Run SE(3) pose graph bundle adjustment with outlier rejection")
    pose_parser.add_argument("--poses", type=int, default=30, help="Number of trajectory camera nodes")

    # Command: cull
    cull_parser = subparsers.add_parser("cull", help="Run frustum & hierarchical occlusion culling test")
    cull_parser.add_argument("--gaussians", type=int, default=5000, help="Gaussian scene count")

    # Command: plan-nbv
    nbv_parser = subparsers.add_parser("plan-nbv", help="Run Shannon entropy Next-Best-View exploration planner")
    nbv_parser.add_argument("--frontiers", type=int, default=500, help="Frontier voxel count")

    # Command: decimate-mesh
    mesh_parser = subparsers.add_parser("decimate-mesh", help="Run QEM topological edge collapse mesh decimation")
    mesh_parser.add_argument("--reduction", type=float, default=50.0, help="Target reduction percentage (default: 50.0)")

    args = parser.parse_args()
    engine = SpatialIntelligenceEngine()

    if args.command == "benchmark-all" or args.command is None:
        gaussians_count = getattr(args, "gaussians", 5000)
        report = engine.run_full_pipeline_benchmark(gaussian_count=gaussians_count)

        if getattr(args, "json", False):
            print(json.dumps(report.__dict__, indent=2))
            return 0

        print("=" * 76)
        print("  SPATIAL INTELLIGENCE & 3D GAUSSIAN SPLATTING BENCHMARK REPORT")
        print("=" * 76)
        print(f" Timestamp:                 {report.timestamp}")
        print(f" Total Suite Latency:       {report.total_runtime_ms:.2f} ms")
        print("-" * 76)
        print(" 1. Submodular Gaussian Pruner:")
        for k, v in report.gaussian_pruning_summary.items():
            print(f"    - {k:<25}: {v}")
        print(" 2. Robust SE(3) Pose Graph Bundle Adjustment:")
        for k, v in report.pose_graph_summary.items():
            print(f"    - {k:<25}: {v}")
        print(" 3. Real-Time Frustum & Occlusion Culler:")
        for k, v in report.frustum_culling_summary.items():
            print(f"    - {k:<25}: {v}")
        print(" 4. Shannon Entropy Next-Best-View Planner:")
        for k, v in report.nbv_planning_summary.items():
            print(f"    - {k:<25}: {v}")
        print(" 5. QEM Topological Mesh Decimator:")
        for k, v in report.mesh_decimation_summary.items():
            print(f"    - {k:<25}: {v}")
        print("=" * 76)
        return 0

    elif args.command == "prune":
        gaussians, _, _, _, _, _, _ = engine.generate_synthetic_scene(gaussian_count=args.count)
        res = engine.pruner.prune_scene(gaussians, target_count=args.target)
        print(f"Pruned {res.original_count} -> {res.retained_count} Gaussians ({res.compression_ratio_pct:.1f}% reduction) in {res.pruning_time_ms:.2f} ms | PSNR Est: {res.psnr_retention_score:.4f}")
        return 0

    elif args.command == "optimize-poses":
        _, poses, constraints, _, _, _, _ = engine.generate_synthetic_scene(pose_count=args.poses)
        _, rep = engine.pose_optimizer.optimize_graph(poses, constraints)
        print(f"Pose Graph: {rep.total_nodes} nodes, {rep.total_edges} edges | Rejected {rep.outlier_loops_rejected} false-positive loop closures | Error reduced {rep.error_reduction_pct:.1f}% in {rep.optimization_latency_ms:.2f} ms")
        return 0

    elif args.command == "cull":
        gaussians, _, _, frustum, _, _, _ = engine.generate_synthetic_scene(gaussian_count=args.gaussians)
        res = engine.culler.cull(frustum, gaussians)
        print(f"Culling: {res.total_input_gaussians} input -> {len(res.visible_gaussians)} visible ({res.frustum_culled_count} frustum culled, {res.occlusion_culled_count} occlusion culled) | Speedup: {res.render_speedup_factor}x in {res.culling_latency_us:.1f} us")
        return 0

    elif args.command == "plan-nbv":
        _, poses, _, _, frontiers, _, _ = engine.generate_synthetic_scene()
        res = engine.nbv_planner.evaluate_next_view(poses[0], [poses[1], poses[5], poses[10]], frontiers)
        print(f"Next-Best-View: Selected {res.recommended_pose.pose_id} | Expected Info Gain: {res.expected_information_gain_voxels} voxels ({res.frontier_coverage_pct:.1f}%) | Path: {res.motion_cost_meters:.2f} m in {res.planning_latency_ms:.2f} ms")
        return 0

    elif args.command == "decimate-mesh":
        _, _, _, _, _, v, f = engine.generate_synthetic_scene()
        _, _, rep = engine.mesh_decimator.decimate(v, f, target_reduction_pct=args.reduction)
        print(f"Decimation: {rep.original_faces} -> {rep.decimated_faces} faces ({rep.reduction_pct:.1f}% reduction) | Euler Invariant Preserved: {rep.euler_characteristic_preserved} | Hausdorff Error: {rep.hausdorff_error_est:.4f} in {rep.decimation_time_ms:.2f} ms")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
