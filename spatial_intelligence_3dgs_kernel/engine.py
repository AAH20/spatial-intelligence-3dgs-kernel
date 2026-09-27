"""
Spatial Intelligence & 3D Gaussian Splatting Core Engine.
Coordinates submodular knapsack pruning, robust SE(3) pose graph bundle adjustment,
real-time frustum/occlusion culling, active next-best-view planning, and QEM topological mesh decimation.
Zero external pip dependencies. Pure Python 3.10+.
"""

from __future__ import annotations
import datetime
import math
import random
import time
from typing import Dict, List, Tuple, Set, Optional

from spatial_intelligence_3dgs_kernel.core.models import (
    Gaussian3D,
    PrunedPointCloud,
    Pose3D,
    RelativePoseConstraint,
    PoseGraphOptimizationReport,
    CameraFrustum,
    FrustumCullResult,
    NextBestViewDecision,
    MeshFace,
    DecimatedMeshReport,
    SpatialIntelligenceBenchmarkReport,
)
from spatial_intelligence_3dgs_kernel.core.gaussian_submodular_pruner import (
    GaussianSubmodularPruner,
)
from spatial_intelligence_3dgs_kernel.core.pose_graph_bundle_adjustment import (
    RobustPoseGraphOptimizer,
)
from spatial_intelligence_3dgs_kernel.core.frustum_voxel_occlusion_culler import (
    FrustumVoxelOcclusionCuller,
)
from spatial_intelligence_3dgs_kernel.core.next_best_view_art_gallery import (
    NextBestViewArtGalleryPlanner,
)
from spatial_intelligence_3dgs_kernel.core.topological_mesh_decimator import (
    TopologicalMeshDecimator,
)


def _look_at_quat(camera_pos: Tuple[float, float, float], target_pos: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
    dx = target_pos[0] - camera_pos[0]
    dy = target_pos[1] - camera_pos[1]
    dz = target_pos[2] - camera_pos[2]
    d_len = math.sqrt(dx * dx + dy * dy + dz * dz)
    if d_len < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    vx, vy, vz = dx / d_len, dy / d_len, dz / d_len
    dot = vz
    if dot < -0.999999:
        return (0.0, 1.0, 0.0, 0.0)
    w = 1.0 + dot
    rx, ry, rz = -vy, vx, 0.0
    q_len = math.sqrt(w * w + rx * rx + ry * ry + rz * rz)
    return (round(w / q_len, 4), round(rx / q_len, 4), round(ry / q_len, 4), round(rz / q_len, 4))


class SpatialIntelligenceEngine:
    """
    High-performance, pure-Python engine orchestrating spatial intelligence,
    volumetric SLAM, and 3D Gaussian Splatting optimization pipelines.
    """

    def __init__(
        self,
        pruner_voxel_size_m: float = 0.05,
        huber_delta: float = 0.5,
        culler_grid_size: int = 32,
        nbv_fov_deg: float = 75.0,
    ):
        self.pruner = GaussianSubmodularPruner(voxel_size_m=pruner_voxel_size_m)
        self.pose_optimizer = RobustPoseGraphOptimizer(huber_delta=huber_delta)
        self.culler = FrustumVoxelOcclusionCuller(angular_grid_size=culler_grid_size)
        self.nbv_planner = NextBestViewArtGalleryPlanner(fov_deg=nbv_fov_deg)
        self.mesh_decimator = TopologicalMeshDecimator()

    @staticmethod
    def generate_synthetic_scene(
        gaussian_count: int = 5000,
        pose_count: int = 25,
        seed: int = 42,
    ) -> Tuple[
        List[Gaussian3D],
        List[Pose3D],
        List[RelativePoseConstraint],
        CameraFrustum,
        Set[Tuple[float, float, float]],
        List[Tuple[float, float, float]],
        List[MeshFace],
    ]:
        """
        Generates realistic synthetic 3D environment data:
        - 3D Gaussian cloud distributed on room boundaries & geometric objects
        - Camera trajectory loop with noisy loop closures
        - Active camera frustum
        - Unexplored frontier voxel coordinates
        - 2-manifold surface mesh (triangulated sphere/torus)
        """
        rng = random.Random(seed)

        # 1. Generate Gaussians
        gaussians: List[Gaussian3D] = []
        for i in range(gaussian_count):
            # Cluster around room walls and central object
            if i % 3 == 0:
                # Central cluster
                x = rng.gauss(0.0, 1.2)
                y = rng.gauss(0.0, 1.2)
                z = rng.uniform(0.1, 2.5)
            else:
                # Surrounding walls
                theta = rng.uniform(0, 2.0 * math.pi)
                rad = rng.uniform(3.0, 6.0)
                x = rad * math.cos(theta) + rng.gauss(0.0, 0.2)
                y = rad * math.sin(theta) + rng.gauss(0.0, 0.2)
                z = rng.uniform(0.0, 3.0)

            scale_val = rng.uniform(0.02, 0.15)
            opacity = rng.uniform(0.05, 0.98) if rng.random() > 0.08 else rng.uniform(0.001, 0.04)  # some floaters
            gaussians.append(
                Gaussian3D(
                    gaussian_id=f"g_{i:06d}",
                    mean=(round(x, 4), round(y, 4), round(z, 4)),
                    scale=(round(scale_val, 4), round(scale_val * rng.uniform(0.8, 1.2), 4), round(scale_val * rng.uniform(0.8, 1.2), 4)),
                    rotation_quat=(1.0, 0.0, 0.0, 0.0),
                    opacity=round(opacity, 4),
                    color_rgb=(round(rng.random(), 3), round(rng.random(), 3), round(rng.random(), 3)),
                    importance_score=round(rng.uniform(0.5, 1.5), 3),
                )
            )

        # 2. Generate Camera Trajectory (circular loop)
        poses: List[Pose3D] = []
        constraints: List[RelativePoseConstraint] = []
        loop_radius = 4.0
        for i in range(pose_count):
            angle = (2.0 * math.pi * i) / pose_count
            cx = loop_radius * math.cos(angle)
            cy = loop_radius * math.sin(angle)
            cz = 1.2 + 0.2 * math.sin(angle * 2.0)
            c_quat = _look_at_quat((cx, cy, cz), (0.0, 0.0, 1.2))
            poses.append(
                Pose3D(
                    pose_id=f"cam_{i:03d}",
                    translation=(round(cx, 4), round(cy, 4), round(cz, 4)),
                    quaternion=c_quat,
                    timestamp_ns=i * 100_000_000,
                )
            )

        # Sequential odometry constraints
        for i in range(pose_count - 1):
            p1 = poses[i].translation
            p2 = poses[i + 1].translation
            rel_t = (round(p2[0] - p1[0] + rng.gauss(0.0, 0.01), 4),
                     round(p2[1] - p1[1] + rng.gauss(0.0, 0.01), 4),
                     round(p2[2] - p1[2] + rng.gauss(0.0, 0.01), 4))
            constraints.append(
                RelativePoseConstraint(
                    from_pose_id=poses[i].pose_id,
                    to_pose_id=poses[i + 1].pose_id,
                    rel_translation=rel_t,
                    rel_rotation=(1.0, 0.0, 0.0, 0.0),
                    information_weight=1.0,
                    is_loop_closure=False,
                )
            )

        # Valid Loop Closure (closing the circle)
        p_last = poses[-1].translation
        p_first = poses[0].translation
        rel_t = (round(p_first[0] - p_last[0], 4),
                 round(p_first[1] - p_last[1], 4),
                 round(p_first[2] - p_last[2], 4))
        constraints.append(
            RelativePoseConstraint(
                from_pose_id=poses[-1].pose_id,
                to_pose_id=poses[0].pose_id,
                rel_translation=rel_t,
                rel_rotation=(1.0, 0.0, 0.0, 0.0),
                information_weight=1.0,
                is_loop_closure=True,
            )
        )

        # False-positive / Outlier Loop Closure (should be rejected by Huber loss)
        constraints.append(
            RelativePoseConstraint(
                from_pose_id=poses[3].pose_id,
                to_pose_id=poses[18].pose_id,
                rel_translation=(10.0, -15.0, 8.0),  # Grossly erroneous constraint
                rel_rotation=(1.0, 0.0, 0.0, 0.0),
                information_weight=1.0,
                is_loop_closure=True,
            )
        )

        # 3. Camera Frustum for Culling
        frustum = CameraFrustum(
            camera_pos=(loop_radius, 0.0, 1.2),
            look_dir=(-1.0, 0.0, 0.0),
            up_dir=(0.0, 0.0, 1.0),
            fov_deg=65.0,
            near_clip=0.2,
            far_clip=12.0,
            aspect_ratio=1.333,
        )

        # 4. Frontier Voxels for Next-Best-View planning
        frontier_voxels: Set[Tuple[float, float, float]] = set()
        for _ in range(300):
            fx = round(rng.uniform(-2.0, 2.0), 2)
            fy = round(rng.uniform(-2.0, 2.0), 2)
            fz = round(rng.uniform(0.5, 2.5), 2)
            frontier_voxels.add((fx, fy, fz))

        # 5. Triangulated 2-Manifold Mesh (Icosahedron / UV sphere subdivision)
        # 12 vertices of regular icosahedron
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        raw_v = [
            (-1.0,  phi, 0.0),
            ( 1.0,  phi, 0.0),
            (-1.0, -phi, 0.0),
            ( 1.0, -phi, 0.0),
            (0.0, -1.0,  phi),
            (0.0,  1.0,  phi),
            (0.0, -1.0, -phi),
            (0.0,  1.0, -phi),
            ( phi, 0.0, -1.0),
            ( phi, 0.0,  1.0),
            (-phi, 0.0, -1.0),
            (-phi, 0.0,  1.0),
        ]
        # Normalize to unit sphere
        mesh_vertices = []
        for vx, vy, vz in raw_v:
            mag = math.sqrt(vx * vx + vy * vy + vz * vz)
            mesh_vertices.append((round(vx / mag, 4), round(vy / mag, 4), round(vz / mag, 4)))

        # 20 triangular faces
        mesh_faces = [
            MeshFace(0, 11, 5), MeshFace(0, 5, 1), MeshFace(0, 1, 7), MeshFace(0, 7, 10), MeshFace(0, 10, 11),
            MeshFace(1, 5, 9), MeshFace(5, 11, 4), MeshFace(11, 10, 2), MeshFace(10, 7, 6), MeshFace(7, 1, 8),
            MeshFace(3, 9, 4), MeshFace(3, 4, 2), MeshFace(3, 2, 6), MeshFace(3, 6, 8), MeshFace(3, 8, 9),
            MeshFace(4, 9, 5), MeshFace(2, 4, 11), MeshFace(6, 2, 10), MeshFace(8, 6, 7), MeshFace(9, 8, 1),
        ]

        return gaussians, poses, constraints, frustum, frontier_voxels, mesh_vertices, mesh_faces

    def run_full_pipeline_benchmark(self, gaussian_count: int = 5000) -> SpatialIntelligenceBenchmarkReport:
        """
        Runs comprehensive benchmark spanning all 5 spatial optimization algorithms.
        """
        total_start = time.perf_counter()

        (
            gaussians,
            poses,
            constraints,
            frustum,
            frontiers,
            mesh_vertices,
            mesh_faces,
        ) = self.generate_synthetic_scene(gaussian_count=gaussian_count)

        # 1. 3D Gaussian Submodular Pruning
        pruned_result = self.pruner.prune_scene(gaussians, target_count=int(gaussian_count * 0.4))

        # 2. Robust Pose Graph Optimization
        optimized_poses, pose_report = self.pose_optimizer.optimize_graph(poses, constraints)

        # 3. Real-Time Frustum & Occlusion Culler
        cull_result = self.culler.cull(frustum, pruned_result.retained_gaussians)

        # 4. Next-Best-View Art Gallery Planner
        candidate_views = [poses[0], poses[5], poses[12], poses[18]]
        nbv_result = self.nbv_planner.evaluate_next_view(
            current_pose=poses[0],
            candidate_poses=candidate_views,
            frontier_voxels=frontiers,
        )

        # 5. Topological Mesh Decimator
        dec_v, dec_f, mesh_report = self.mesh_decimator.decimate(
            vertices=mesh_vertices,
            faces=mesh_faces,
            target_reduction_pct=40.0,
        )

        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0

        return SpatialIntelligenceBenchmarkReport(
            total_runtime_ms=round(total_elapsed_ms, 2),
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            gaussian_pruning_summary={
                "original_count": float(pruned_result.original_count),
                "retained_count": float(pruned_result.retained_count),
                "compression_ratio_pct": round(pruned_result.compression_ratio_pct, 2),
                "psnr_retention_score": round(pruned_result.psnr_retention_score, 4),
                "latency_ms": round(pruned_result.pruning_time_ms, 3),
            },
            pose_graph_summary={
                "nodes": float(pose_report.total_nodes),
                "edges": float(pose_report.total_edges),
                "outlier_loops_rejected": float(pose_report.outlier_loops_rejected),
                "error_reduction_pct": round(pose_report.error_reduction_pct, 2),
                "latency_ms": round(pose_report.optimization_latency_ms, 3),
            },
            frustum_culling_summary={
                "input_gaussians": float(cull_result.total_input_gaussians),
                "visible_gaussians": float(len(cull_result.visible_gaussians)),
                "frustum_culled": float(cull_result.frustum_culled_count),
                "occlusion_culled": float(cull_result.occlusion_culled_count),
                "speedup_factor": cull_result.render_speedup_factor,
                "latency_us": cull_result.culling_latency_us,
            },
            nbv_planning_summary={
                "recommended_pose": float(candidate_views.index(nbv_result.recommended_pose)),
                "info_gain_voxels": float(nbv_result.expected_information_gain_voxels),
                "motion_cost_m": nbv_result.motion_cost_meters,
                "frontier_coverage_pct": nbv_result.frontier_coverage_pct,
                "latency_ms": nbv_result.planning_latency_ms,
            },
            mesh_decimation_summary={
                "original_faces": float(mesh_report.original_faces),
                "decimated_faces": float(mesh_report.decimated_faces),
                "reduction_pct": mesh_report.reduction_pct,
                "euler_preserved": 1.0 if mesh_report.euler_characteristic_preserved else 0.0,
                "hausdorff_err": mesh_report.hausdorff_error_est,
                "latency_ms": mesh_report.decimation_time_ms,
            },
        )
