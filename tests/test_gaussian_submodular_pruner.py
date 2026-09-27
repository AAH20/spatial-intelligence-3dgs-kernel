"""
Tests for Gaussian Submodular Knapsack Pruner.
"""

import unittest
from spatial_intelligence_3dgs_kernel.core.models import Gaussian3D
from spatial_intelligence_3dgs_kernel.core.gaussian_submodular_pruner import GaussianSubmodularPruner


class TestGaussianSubmodularPruner(unittest.TestCase):
    def setUp(self):
        self.pruner = GaussianSubmodularPruner(voxel_size_m=0.1)

    def test_pruning_budget_respect(self):
        gaussians = [
            Gaussian3D(
                gaussian_id=f"g_{i}",
                mean=(float(i % 10) * 0.05, float((i // 10) % 10) * 0.05, 0.0),
                scale=(0.02, 0.02, 0.02),
                rotation_quat=(1.0, 0.0, 0.0, 0.0),
                opacity=0.8,
                color_rgb=(0.5, 0.5, 0.5),
            )
            for i in range(200)
        ]
        res = self.pruner.prune_scene(gaussians, target_count=50)
        self.assertEqual(res.retained_count, 50)
        self.assertGreater(res.compression_ratio_pct, 50.0)
        self.assertGreater(res.psnr_retention_score, 0.0)

    def test_floater_rejection(self):
        # Floaters with opacity < 0.05
        gaussians = [
            Gaussian3D(
                gaussian_id=f"floater_{i}",
                mean=(0.0, 0.0, 0.0),
                scale=(0.01, 0.01, 0.01),
                rotation_quat=(1.0, 0.0, 0.0, 0.0),
                opacity=0.01,
                color_rgb=(0.1, 0.1, 0.1),
            )
            for i in range(10)
        ] + [
            Gaussian3D(
                gaussian_id="solid_1",
                mean=(1.0, 1.0, 1.0),
                scale=(0.05, 0.05, 0.05),
                rotation_quat=(1.0, 0.0, 0.0, 0.0),
                opacity=0.9,
                color_rgb=(0.9, 0.9, 0.9),
            )
        ]
        res = self.pruner.prune_scene(gaussians, target_count=5)
        self.assertEqual(res.retained_count, 1)
        self.assertEqual(res.retained_gaussians[0].gaussian_id, "solid_1")


if __name__ == "__main__":
    unittest.main()
