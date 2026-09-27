"""
Spatial Intelligence & 3D Gaussian Splatting Optimization Kernel.
Zero external dependencies. Pure Python 3.10+.
"""

from spatial_intelligence_3dgs_kernel.core import (
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
    GaussianSubmodularPruner,
    RobustPoseGraphOptimizer,
    FrustumVoxelOcclusionCuller,
    NextBestViewArtGalleryPlanner,
    TopologicalMeshDecimator,
)
from spatial_intelligence_3dgs_kernel.engine import SpatialIntelligenceEngine

__version__ = "0.1.0"

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
    "SpatialIntelligenceEngine",
]
