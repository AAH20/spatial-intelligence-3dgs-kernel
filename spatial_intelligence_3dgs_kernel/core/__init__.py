"""
Core algorithmic modules for Spatial Intelligence & 3D Gaussian Splatting Kernel.
"""

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

__all__ = [
    "Gaussian3D",
    "PrunedPointCloud",
    "Pose3D",
    "RelativePoseConstraint",
    "PoseGraphOptimizationReport",
    "CameraFrustum",
    "FrustumCullResult",
    "NextBestViewDecision",
    "MeshFace",
    "DecimatedMeshReport",
    "SpatialIntelligenceBenchmarkReport",
    "GaussianSubmodularPruner",
    "RobustPoseGraphOptimizer",
    "FrustumVoxelOcclusionCuller",
    "NextBestViewArtGalleryPlanner",
    "TopologicalMeshDecimator",
]
