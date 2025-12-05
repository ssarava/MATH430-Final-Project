"""
Projective Geometry Computer Vision Project
============================================

This package provides implementations of projective geometry concepts
applied to computer vision tasks, including:

- Homography computation and warping
- Perspective correction
- Image stitching

The code is designed to be educational, with detailed comments explaining
the mathematical concepts behind each operation.

Modules:
--------
homography : Core homography computation functions
perspective_correction : Interactive perspective correction tool
image_stitching : Image stitching and panorama creation
visualization : Visualization utilities

Example Usage:
--------------
    from src.homography import compute_homography, warp_image
    from src.perspective_correction import PerspectiveCorrectionTool
    from src.image_stitching import ImageStitcher
"""

__version__ = "1.0.0"
__author__ = "Student"
__course__ = "MATH 430: Euclidean and Non-Euclidean Geometries"

# Make key classes and functions available at package level
from .homography import (
    compute_homography,
    compute_homography_dlt,
    warp_image,
    apply_homography_to_point,
    HomographyMatrix
)

from .perspective_correction import PerspectiveCorrectionTool

from .image_stitching import ImageStitcher

from .visualization import (
    display_images_side_by_side,
    draw_matches,
    draw_points_on_image,
    create_comparison_figure
)
