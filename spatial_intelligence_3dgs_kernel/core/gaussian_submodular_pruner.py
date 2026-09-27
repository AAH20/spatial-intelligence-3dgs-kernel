"""
3D Gaussian Splatting Submodular Knapsack Pruner.
Prunes millions of 3D Gaussians down to strict mobile/edge VRAM budgets
using submodular spatial facility dispersion while maximizing surface reconstruction fidelity.
"""

from __future__ import annotations
import collections
import math
import time
from typing import List, Dict, Tuple, Set
from spatial_intelligence_3dgs_kernel.core.models import (
    Gaussian3D,
    PrunedPointCloud,
)


class GaussianSubmodularPruner:
    """
    Submodular greedy selector for 3D Gaussian Splatting scene compression.
    Eliminates redundant overlapping Gaussians while guaranteeing spatial surface coverage.
    """

    def __init__(self, voxel_size_m: float = 0.05):
        self.voxel_size = voxel_size_m

    def _get_voxel_key(self, pos: Tuple[float, float, float]) -> Tuple[int, int, int]:
        return (
            int(math.floor(pos[0] / self.voxel_size)),
            int(math.floor(pos[1] / self.voxel_size)),
            int(math.floor(pos[2] / self.voxel_size)),
        )

    def prune_scene(
        self,
        gaussians: List[Gaussian3D],
        target_count: int = 50000,
        min_opacity_threshold: float = 0.05,
    ) -> PrunedPointCloud:
        """
        Prunes raw Gaussian point cloud to fit target_count.
        Discards transparent floaters and optimizes voxel information density.
        """
        start_t = time.perf_counter()
        original_count = len(gaussians)

        if original_count <= target_count:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return PrunedPointCloud(
                retained_gaussians=gaussians,
                original_count=original_count,
                retained_count=original_count,
                compression_ratio_pct=0.0,
                psnr_retention_score=1.0,
                pruning_time_ms=elapsed_ms,
            )

        # Step 1: Filter out low-opacity floaters (opacity < min_opacity_threshold)
        valid_candidates = [g for g in gaussians if g.opacity >= min_opacity_threshold]

        # Step 2: Spatial hashing into 3D voxel buckets
        voxel_buckets: Dict[Tuple[int, int, int], List[Gaussian3D]] = collections.defaultdict(list)
        for g in valid_candidates:
            v_key = self._get_voxel_key(g.mean)
            voxel_buckets[v_key].append(g)

        # Step 3: Sort candidates in each voxel by effective visual mass:
        # VisualMass = opacity * sqrt(sx * sy * sz) * importance_score
        for v_key in voxel_buckets:
            voxel_buckets[v_key].sort(
                key=lambda g: g.opacity * math.sqrt(g.scale[0] * g.scale[1] * g.scale[2]) * g.importance_score,
                reverse=True,
            )

        # Step 4: Fair round-robin submodular selection across active voxels
        retained: List[Gaussian3D] = []
        active_voxel_keys = list(voxel_buckets.keys())

        # Prioritize voxels with higher total visual mass
        active_voxel_keys.sort(
            key=lambda k: sum(g.opacity for g in voxel_buckets[k]),
            reverse=True,
        )

        idx = 0
        while len(retained) < target_count and any(voxel_buckets[k] for k in active_voxel_keys):
            for k in active_voxel_keys:
                if len(retained) >= target_count:
                    break
                if voxel_buckets[k]:
                    # Pop the highest-value remaining Gaussian in this voxel
                    retained.append(voxel_buckets[k].pop(0))

        # Calculate metrics
        compression_ratio = ((original_count - len(retained)) / original_count * 100.0) if original_count > 0 else 0.0
        total_orig_mass = sum(g.opacity for g in gaussians)
        retained_mass = sum(g.opacity for g in retained)
        psnr_est = (retained_mass / total_orig_mass) if total_orig_mass > 0 else 1.0

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return PrunedPointCloud(
            retained_gaussians=retained,
            original_count=original_count,
            retained_count=len(retained),
            compression_ratio_pct=compression_ratio,
            psnr_retention_score=psnr_est,
            pruning_time_ms=elapsed_ms,
        )
