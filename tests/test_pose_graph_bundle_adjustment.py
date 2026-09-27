"""
Tests for Robust SE(3) Pose Graph Bundle Adjustment Optimizer.
"""

import unittest
from spatial_intelligence_3dgs_kernel.core.models import Pose3D, RelativePoseConstraint
from spatial_intelligence_3dgs_kernel.core.pose_graph_bundle_adjustment import RobustPoseGraphOptimizer


class TestPoseGraphOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = RobustPoseGraphOptimizer(huber_delta=0.5, max_iterations=20)

    def test_odometry_relaxation_and_loop_closure(self):
        # 4 poses in a square: (0,0,0) -> (1,0,0) -> (1,1,0) -> (0,1,0)
        poses = [
            Pose3D("p0", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            Pose3D("p1", (1.05, 0.02, 0.0), (1.0, 0.0, 0.0, 0.0)),  # slight noise
            Pose3D("p2", (1.02, 0.98, 0.0), (1.0, 0.0, 0.0, 0.0)),
            Pose3D("p3", (-0.03, 1.04, 0.0), (1.0, 0.0, 0.0, 0.0)),
        ]
        constraints = [
            RelativePoseConstraint("p0", "p1", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            RelativePoseConstraint("p1", "p2", (0.0, 1.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            RelativePoseConstraint("p2", "p3", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            RelativePoseConstraint("p3", "p0", (0.0, -1.0, 0.0), (1.0, 0.0, 0.0, 0.0), is_loop_closure=True),
        ]

        opt_poses, report = self.optimizer.optimize_graph(poses, constraints)
        self.assertEqual(report.total_nodes, 4)
        self.assertEqual(report.outlier_loops_rejected, 0)
        self.assertLess(report.optimized_error, report.initial_error)

    def test_outlier_rejection(self):
        # 3 poses with a wildly corrupted loop closure edge
        poses = [
            Pose3D("p0", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            Pose3D("p1", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            Pose3D("p2", (2.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        ]
        constraints = [
            RelativePoseConstraint("p0", "p1", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            RelativePoseConstraint("p1", "p2", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            # Outlier loop closure claiming p2 is at (-100, 50, 20) relative to p0
            RelativePoseConstraint("p0", "p2", (-100.0, 50.0, 20.0), (1.0, 0.0, 0.0, 0.0), is_loop_closure=True),
        ]
        _, report = self.optimizer.optimize_graph(poses, constraints)
        self.assertEqual(report.outlier_loops_rejected, 1)


if __name__ == "__main__":
    unittest.main()
