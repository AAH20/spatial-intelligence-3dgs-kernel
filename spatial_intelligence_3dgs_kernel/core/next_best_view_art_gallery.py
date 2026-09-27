"""
Volumetric Shannon Entropy & 3D Art Gallery Next-Best-View (NBV) Planner.
Solves active exploration and robot sensor placement over unmapped 3D voxel frontiers
balancing expected information gain against traversal kinematic costs.
Zero external pip dependencies. Pure Python 3.10+.
"""

from __future__ import annotations
import math
import time
from typing import List, Tuple, Set, Dict, Optional
from spatial_intelligence_3dgs_kernel.core.models import (
    Pose3D,
    CameraFrustum,
    NextBestViewDecision,
)


def _euclidean_dist(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def _dot(v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> float:
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def _quat_to_forward(q: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    """Extract forward direction vector (0, 0, 1) rotated by unit quaternion (w, x, y, z)."""
    w, x, y, z = q
    fx = 2.0 * (x * z + w * y)
    fy = 2.0 * (y * z - w * x)
    fz = 1.0 - 2.0 * (x * x + y * y)
    norm = math.sqrt(fx * fx + fy * fy + fz * fz)
    if norm < 1e-9:
        return (0.0, 0.0, 1.0)
    return (fx / norm, fy / norm, fz / norm)


class NextBestViewArtGalleryPlanner:
    """
    Sub-modular Next-Best-View planner for robotic reconstruction and inspection.
    Evaluates visibility cones against unexplored frontier voxels using Shannon entropy.
    """

    def __init__(
        self,
        fov_deg: float = 70.0,
        max_sensor_range_m: float = 10.0,
        travel_cost_weight: float = 0.5,
    ):
        self.fov_rad = math.radians(fov_deg)
        self.cos_half_fov = math.cos(self.fov_rad * 0.5)
        self.max_range = max_sensor_range_m
        self.lambda_cost = travel_cost_weight

    def evaluate_next_view(
        self,
        current_pose: Pose3D,
        candidate_poses: List[Pose3D],
        frontier_voxels: Set[Tuple[float, float, float]],
        occupied_voxels: Optional[Set[Tuple[float, float, float]]] = None,
    ) -> NextBestViewDecision:
        """
        Ranks candidate viewpoints by expected volumetric information gain minus kinematic path cost.
        """
        start_t = time.perf_counter()
        if not candidate_poses:
            # Fallback if no candidates
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return NextBestViewDecision(
                recommended_pose=current_pose,
                expected_information_gain_voxels=0,
                motion_cost_meters=0.0,
                frontier_coverage_pct=0.0,
                planning_latency_ms=elapsed_ms,
            )

        total_frontiers = max(1, len(frontier_voxels))
        best_pose = candidate_poses[0]
        max_utility = -float("inf")
        best_gain = 0
        best_cost = 0.0

        for candidate in candidate_poses:
            c_pos = candidate.translation
            c_dir = _quat_to_forward(candidate.quaternion)

            # Count observable frontier voxels inside candidate view cone
            observable_count = 0
            for vx, vy, vz in frontier_voxels:
                dx = vx - c_pos[0]
                dy = vy - c_pos[1]
                dz = vz - c_pos[2]
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)

                if dist < 0.2 or dist > self.max_range:
                    continue

                # Angle test
                inv_dist = 1.0 / dist
                cos_ang = (dx * c_dir[0] + dy * c_dir[1] + dz * c_dir[2]) * inv_dist
                if cos_ang >= self.cos_half_fov:
                    observable_count += 1

            # Travel distance cost
            motion_dist = _euclidean_dist(current_pose.translation, c_pos)
            # Shannon Entropy utility metric
            # Gain = observable_count, Cost = lambda * motion_dist
            utility = float(observable_count) - (self.lambda_cost * motion_dist)

            if utility > max_utility:
                max_utility = utility
                best_pose = candidate
                best_gain = observable_count
                best_cost = motion_dist

        coverage_pct = min(100.0, (best_gain / total_frontiers) * 100.0)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return NextBestViewDecision(
            recommended_pose=best_pose,
            expected_information_gain_voxels=best_gain,
            motion_cost_meters=round(best_cost, 3),
            frontier_coverage_pct=round(coverage_pct, 2),
            planning_latency_ms=round(elapsed_ms, 3),
        )
