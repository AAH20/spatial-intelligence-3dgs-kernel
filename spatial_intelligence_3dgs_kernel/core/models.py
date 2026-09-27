"""
Data models and typed structures for Spatial Intelligence & 3D Gaussian Splatting Kernel.
Zero external pip dependencies. Strict Python 3.10+ typing.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple


@dataclass(slots=True)
class Gaussian3D:
    gaussian_id: str
    mean: Tuple[float, float, float]  # (x, y, z) in meters
    scale: Tuple[float, float, float] # (sx, sy, sz)
    rotation_quat: Tuple[float, float, float, float]  # (w, x, y, z)
    opacity: float  # [0, 1]
    color_rgb: Tuple[float, float, float]  # (r, g, b) in [0, 1]
    sh_degree: int = 0
    importance_score: float = 1.0


@dataclass(slots=True)
class PrunedPointCloud:
    retained_gaussians: List[Gaussian3D]
    original_count: int
    retained_count: int
    compression_ratio_pct: float
    psnr_retention_score: float
    pruning_time_ms: float


@dataclass(slots=True)
class Pose3D:
    pose_id: str
    translation: Tuple[float, float, float]  # (x, y, z)
    quaternion: Tuple[float, float, float, float]  # (w, x, y, z)
    timestamp_ns: int = 0


@dataclass(slots=True)
class RelativePoseConstraint:
    from_pose_id: str
    to_pose_id: str
    rel_translation: Tuple[float, float, float]
    rel_rotation: Tuple[float, float, float, float]
    information_weight: float = 1.0
    is_loop_closure: bool = False


@dataclass(slots=True)
class PoseGraphOptimizationReport:
    total_nodes: int
    total_edges: int
    outlier_loops_rejected: int
    initial_error: float
    optimized_error: float
    error_reduction_pct: float
    optimization_latency_ms: float


@dataclass(slots=True)
class CameraFrustum:
    camera_pos: Tuple[float, float, float]
    look_dir: Tuple[float, float, float]  # normalized direction
    up_dir: Tuple[float, float, float]
    fov_deg: float = 60.0
    near_clip: float = 0.1
    far_clip: float = 50.0
    aspect_ratio: float = 1.333


@dataclass(slots=True)
class FrustumCullResult:
    total_input_gaussians: int
    visible_gaussians: List[Gaussian3D]
    frustum_culled_count: int
    occlusion_culled_count: int
    render_speedup_factor: float
    culling_latency_us: float


@dataclass(slots=True)
class NextBestViewDecision:
    recommended_pose: Pose3D
    expected_information_gain_voxels: int
    motion_cost_meters: float
    frontier_coverage_pct: float
    planning_latency_ms: float


@dataclass(slots=True)
class MeshFace:
    v0: int
    v1: int
    v2: int


@dataclass(slots=True)
class DecimatedMeshReport:
    original_vertices: int
    original_faces: int
    decimated_vertices: int
    decimated_faces: int
    euler_characteristic_preserved: bool
    reduction_pct: float
    hausdorff_error_est: float
    decimation_time_ms: float


@dataclass(slots=True)
class SpatialIntelligenceBenchmarkReport:
    total_runtime_ms: float
    timestamp: str
    gaussian_pruning_summary: Dict[str, float]
    pose_graph_summary: Dict[str, float]
    frustum_culling_summary: Dict[str, float]
    nbv_planning_summary: Dict[str, float]
    mesh_decimation_summary: Dict[str, float]
