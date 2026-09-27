"""
Tests for Frustum & Occlusion Culling Kernel.
"""

import unittest
from spatial_intelligence_3dgs_kernel.core.models import Gaussian3D, CameraFrustum
from spatial_intelligence_3dgs_kernel.core.frustum_voxel_occlusion_culler import FrustumVoxelOcclusionCuller


class TestFrustumOcclusionCuller(unittest.TestCase):
    def setUp(self):
        self.culler = FrustumVoxelOcclusionCuller(angular_grid_size=16)
        self.frustum = CameraFrustum(
            camera_pos=(0.0, 0.0, 0.0),
            look_dir=(0.0, 0.0, 1.0),
            up_dir=(0.0, 1.0, 0.0),
            fov_deg=60.0,
            near_clip=0.5,
            far_clip=10.0,
        )

    def test_frustum_culling_behind_camera(self):
        # Gaussian behind camera (z = -2.0)
        behind = Gaussian3D("behind", (0.0, 0.0, -2.0), (0.05, 0.05, 0.05), (1,0,0,0), 0.9, (1,0,0))
        # Gaussian inside frustum (z = 3.0)
        inside = Gaussian3D("inside", (0.0, 0.0, 3.0), (0.05, 0.05, 0.05), (1,0,0,0), 0.9, (0,1,0))

        res = self.culler.cull(self.frustum, [behind, inside])
        self.assertEqual(res.frustum_culled_count, 1)
        self.assertEqual(len(res.visible_gaussians), 1)
        self.assertEqual(res.visible_gaussians[0].gaussian_id, "inside")

    def test_occlusion_culling_front_to_back(self):
        # Two Gaussians in exact same line of sight
        front = Gaussian3D("front", (0.0, 0.0, 2.0), (0.05, 0.05, 0.05), (1,0,0,0), 1.0, (1,0,0))
        back = Gaussian3D("back", (0.0, 0.0, 6.0), (0.05, 0.05, 0.05), (1,0,0,0), 1.0, (0,0,1))

        res = self.culler.cull(self.frustum, [front, back])
        self.assertEqual(res.occlusion_culled_count, 1)
        self.assertEqual(len(res.visible_gaussians), 1)
        self.assertEqual(res.visible_gaussians[0].gaussian_id, "front")


if __name__ == "__main__":
    unittest.main()
