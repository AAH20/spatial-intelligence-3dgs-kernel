"""
Tests for Spatial Intelligence Engine.
"""

import unittest
from spatial_intelligence_3dgs_kernel.engine import SpatialIntelligenceEngine


class TestSpatialIntelligenceEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SpatialIntelligenceEngine()

    def test_full_pipeline_benchmark(self):
        report = self.engine.run_full_pipeline_benchmark(gaussian_count=1000)
        self.assertGreater(report.total_runtime_ms, 0.0)
        self.assertLess(report.total_runtime_ms, 1500.0)  # fast execution
        self.assertEqual(report.gaussian_pruning_summary["original_count"], 1000.0)
        self.assertGreater(report.pose_graph_summary["outlier_loops_rejected"], 0.0)
        self.assertGreater(report.frustum_culling_summary["speedup_factor"], 1.0)
        self.assertGreater(report.nbv_planning_summary["info_gain_voxels"], 0.0)
        self.assertEqual(report.mesh_decimation_summary["euler_preserved"], 1.0)


if __name__ == "__main__":
    unittest.main()
