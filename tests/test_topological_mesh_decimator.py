"""
Tests for Topological Mesh Decimator using QEM.
"""

import unittest
from spatial_intelligence_3dgs_kernel.core.models import MeshFace
from spatial_intelligence_3dgs_kernel.core.topological_mesh_decimator import TopologicalMeshDecimator


class TestTopologicalMeshDecimator(unittest.TestCase):
    def setUp(self):
        self.decimator = TopologicalMeshDecimator()

    def test_decimation_reduction_and_euler_invariance(self):
        # Regular tetrahedron (4 vertices, 4 faces)
        vertices = [
            (1.0, 1.0, 1.0),
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (1.0, -1.0, -1.0),
        ]
        faces = [
            MeshFace(0, 1, 2),
            MeshFace(0, 3, 1),
            MeshFace(0, 2, 3),
            MeshFace(1, 3, 2),
        ]

        dec_v, dec_f, report = self.decimator.decimate(vertices, faces, target_reduction_pct=0.0)
        self.assertEqual(report.original_faces, 4)
        self.assertEqual(report.decimated_faces, 4)
        self.assertTrue(report.euler_characteristic_preserved)


if __name__ == "__main__":
    unittest.main()
