"""
Tests for Shannon Entropy Next-Best-View Art Gallery Planner.
"""

import unittest
from spatial_intelligence_3dgs_kernel.core.models import Pose3D
from spatial_intelligence_3dgs_kernel.core.next_best_view_art_gallery import NextBestViewArtGalleryPlanner


class TestNextBestViewPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = NextBestViewArtGalleryPlanner(fov_deg=80.0, max_sensor_range_m=15.0)

    def test_next_view_selection_maximizes_gain(self):
        curr_pose = Pose3D("curr", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        # Candidate 1: close, facing away from cluster
        # Candidate 2: close, facing directly towards cluster of 50 voxels at (5, 0, 0)
        c1 = Pose3D("c1", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))  # facing backwards (-Z)
        c2 = Pose3D("c2", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))  # facing forwards (+Z)

        # Dense frontier voxels clustered at (1.0, 0.0, 5.0) -> in front of c2 (+Z)
        frontiers = {(1.0 + float(i)*0.01, 0.0, 4.0 + float(i)*0.01) for i in range(40)}

        decision = self.planner.evaluate_next_view(curr_pose, [c1, c2], frontiers)
        self.assertEqual(decision.recommended_pose.pose_id, "c2")
        self.assertGreater(decision.expected_information_gain_voxels, 0)
        self.assertGreater(decision.frontier_coverage_pct, 50.0)


if __name__ == "__main__":
    unittest.main()
