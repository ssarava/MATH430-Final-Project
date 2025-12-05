"""
Homography Module
=================

This module implements the core mathematical operations for computing and
applying homography transformations, which are central to projective geometry
and computer vision.

Mathematical Background:
------------------------
A homography (also called a projective transformation or collineation) is a
transformation between two projective planes that preserves collinearity.
In computer vision, homographies model the transformation between:
- Two views of a planar surface
- Two views from a rotating camera (no translation)

The homography is represented by a 3x3 matrix H that transforms points
in homogeneous coordinates:
    
    [x']   [h11 h12 h13] [x]
    [y'] = [h21 h22 h23] [y]
    [w']   [h31 h32 h33] [w]

where (x, y, w) is the source point and (x', y', w') is the destination
point in homogeneous coordinates. The actual 2D coordinates are:
    (x/w, y/w) -> (x'/w', y'/w')

The matrix H has 9 elements but only 8 degrees of freedom (scale doesn't
matter), so we need at least 4 point correspondences to compute it.

References:
-----------
- Birchfield, S. "An Introduction to Projective Geometry (for computer vision)"
- Hartley & Zisserman, "Multiple View Geometry in Computer Vision", Ch. 4
"""

import numpy as np
from typing import Tuple, List, Optional
import cv2


class HomographyMatrix:
    """
    A class representing a 3x3 homography matrix with associated methods.
    
    This class encapsulates the homography matrix and provides methods for
    common operations like transformation, decomposition, and analysis.
    
    Attributes:
    -----------
    matrix : np.ndarray
        The 3x3 homography matrix
    source_points : np.ndarray or None
        The source points used to compute this homography (if available)
    dest_points : np.ndarray or None
        The destination points used to compute this homography (if available)
    
    Mathematical Note:
    ------------------
    The homography matrix has 8 degrees of freedom because overall scale
    doesn't matter in homogeneous coordinates. The 9 elements satisfy one
    constraint (typically h33 = 1 or ||H||_F = 1).
    """
    
    def __init__(self, matrix: np.ndarray, 
                 source_points: Optional[np.ndarray] = None,
                 dest_points: Optional[np.ndarray] = None):
        """
        Initialize a HomographyMatrix object.
        
        Parameters:
        -----------
        matrix : np.ndarray
            A 3x3 numpy array representing the homography transformation
        source_points : np.ndarray, optional
            The source points used to compute this homography
        dest_points : np.ndarray, optional
            The destination points used to compute this homography
        
        Raises:
        -------
        ValueError
            If the matrix is not 3x3 or is singular
        """
        # Validate matrix dimensions
        if matrix.shape != (3, 3):
            raise ValueError(f"Homography matrix must be 3x3, got {matrix.shape}")
        
        # Normalize so that h33 = 1 (standard convention) or ||H|| = 1 if h33 ≈ 0
        # This ensures a canonical form for the homography
        if abs(matrix[2, 2]) > 1e-10:
            self.matrix = matrix / matrix[2, 2]
        else:
            # If h33 is near zero, normalize by Frobenius norm instead
            self.matrix = matrix / np.linalg.norm(matrix, 'fro')
        
        # Store the points used to compute this homography
        self.source_points = source_points
        self.dest_points = dest_points
    
    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """
        Transform a single 2D point using this homography.
        
        Parameters:
        -----------
        point : np.ndarray
            A 2D point as (x, y) array
        
        Returns:
        --------
        np.ndarray
            The transformed point as (x', y') array
        
        Example:
        --------
        >>> H = HomographyMatrix(np.eye(3))
        >>> H.transform_point(np.array([1.0, 2.0]))
        array([1., 2.])
        """
        # Convert to homogeneous coordinates: (x, y) -> (x, y, 1)
        homogeneous_point = np.array([point[0], point[1], 1.0])
        
        # Apply homography transformation: p' = H @ p
        transformed = self.matrix @ homogeneous_point
        
        # Convert back to Cartesian: (x', y', w') -> (x'/w', y'/w')
        # Guard against division by zero (point at infinity)
        if abs(transformed[2]) < 1e-10:
            return np.array([np.inf, np.inf])
        
        return transformed[:2] / transformed[2]
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform multiple 2D points using this homography.
        
        Parameters:
        -----------
        points : np.ndarray
            Array of shape (N, 2) containing N 2D points
        
        Returns:
        --------
        np.ndarray
            Array of shape (N, 2) containing the transformed points
        """
        # Number of points
        n_points = points.shape[0]
        
        # Convert all points to homogeneous coordinates
        # Shape: (N, 2) -> (N, 3) by appending column of ones
        homogeneous = np.hstack([points, np.ones((n_points, 1))])
        
        # Apply homography to all points at once: (N, 3) @ (3, 3).T = (N, 3)
        transformed = homogeneous @ self.matrix.T
        
        # Convert back to Cartesian coordinates
        # Divide x and y by w for each point
        result = transformed[:, :2] / transformed[:, 2:3]
        
        return result
    
    def inverse(self) -> 'HomographyMatrix':
        """
        Compute the inverse homography.
        
        Returns:
        --------
        HomographyMatrix
            The inverse transformation
        
        Note:
        -----
        The inverse homography transforms points in the opposite direction.
        If H maps image A to image B, then H^(-1) maps image B to image A.
        """
        inverse_matrix = np.linalg.inv(self.matrix)
        return HomographyMatrix(inverse_matrix, self.dest_points, self.source_points)
    
    def compose(self, other: 'HomographyMatrix') -> 'HomographyMatrix':
        """
        Compose this homography with another.
        
        Parameters:
        -----------
        other : HomographyMatrix
            Another homography to compose with
        
        Returns:
        --------
        HomographyMatrix
            The composed homography (self followed by other)
        
        Note:
        -----
        Composition is matrix multiplication: if H1 maps A->B and H2 maps B->C,
        then H2 @ H1 maps A->C directly.
        """
        composed_matrix = other.matrix @ self.matrix
        return HomographyMatrix(composed_matrix)
    
    def get_decomposition(self) -> dict:
        """
        Decompose the homography into interpretable components.
        
        Returns:
        --------
        dict
            Dictionary containing:
            - 'scale': Overall scale factor
            - 'rotation': Approximate rotation angle (radians)
            - 'translation': Translation component
            - 'perspective': Perspective distortion terms
        
        Note:
        -----
        This is an approximate decomposition that helps understand what
        geometric transformation the homography represents.
        """
        H = self.matrix
        
        # Extract the approximate rotation angle from the upper-left 2x2 block
        # This assumes the transformation is close to a similarity transform
        rotation_angle = np.arctan2(H[1, 0], H[0, 0])
        
        # Approximate scale from the upper-left 2x2 block
        scale = np.sqrt(H[0, 0]**2 + H[1, 0]**2)
        
        # Translation components (normalized)
        translation = np.array([H[0, 2], H[1, 2]]) / H[2, 2] if abs(H[2, 2]) > 1e-10 else np.array([H[0, 2], H[1, 2]])
        
        # Perspective components (bottom row excluding h33)
        perspective = np.array([H[2, 0], H[2, 1]])
        
        return {
            'scale': scale,
            'rotation': rotation_angle,
            'rotation_degrees': np.degrees(rotation_angle),
            'translation': translation,
            'perspective': perspective
        }
    
    def __repr__(self) -> str:
        """String representation of the homography matrix."""
        return f"HomographyMatrix(\n{self.matrix}\n)"


def compute_homography_dlt(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """
    Compute homography using the Direct Linear Transform (DLT) algorithm.
    
    This is a fundamental algorithm in computer vision that computes the
    homography matrix given point correspondences between two images.
    
    Parameters:
    -----------
    src_points : np.ndarray
        Source points of shape (N, 2) where N >= 4
    dst_points : np.ndarray
        Destination points of shape (N, 2) where N >= 4
    
    Returns:
    --------
    np.ndarray
        The 3x3 homography matrix H such that dst ≈ H @ src (in homogeneous coords)
    
    Mathematical Background:
    ------------------------
    For each point correspondence (x, y) -> (x', y'), we have:
        
        x' = (h11*x + h12*y + h13) / (h31*x + h32*y + h33)
        y' = (h21*x + h22*y + h23) / (h31*x + h32*y + h33)
    
    Rearranging to eliminate the denominator:
        
        x'*(h31*x + h32*y + h33) = h11*x + h12*y + h13
        y'*(h31*x + h32*y + h33) = h21*x + h22*y + h23
    
    This gives us 2 linear equations per point correspondence. With 4 points,
    we get 8 equations for 8 unknowns (since h33 = 1 by convention).
    
    We solve this as a homogeneous system Ah = 0, where h is the vector of
    homography elements, using SVD to find the null space.
    
    References:
    -----------
    Hartley & Zisserman, "Multiple View Geometry", Algorithm 4.1
    """
    # Validate inputs
    assert src_points.shape[0] >= 4, "Need at least 4 point correspondences"
    assert src_points.shape == dst_points.shape, "Point arrays must have same shape"
    
    n_points = src_points.shape[0]
    
    # Normalization: Improve numerical stability by normalizing coordinates
    # This is crucial for accurate computation with real-world data
    src_normalized, T_src = _normalize_points(src_points)
    dst_normalized, T_dst = _normalize_points(dst_points)
    
    # Build the coefficient matrix A for the linear system Ah = 0
    # Each point correspondence gives 2 rows
    A = np.zeros((2 * n_points, 9))
    
    for i in range(n_points):
        # Source point (x, y) and destination point (x', y')
        x, y = src_normalized[i]
        xp, yp = dst_normalized[i]
        
        # First equation: x'*(h31*x + h32*y + h33) - (h11*x + h12*y + h13) = 0
        # Rearranged: -x, -y, -1, 0, 0, 0, x'*x, x'*y, x'
        A[2*i] = [-x, -y, -1, 0, 0, 0, xp*x, xp*y, xp]
        
        # Second equation: y'*(h31*x + h32*y + h33) - (h21*x + h22*y + h23) = 0
        # Rearranged: 0, 0, 0, -x, -y, -1, y'*x, y'*y, y'
        A[2*i + 1] = [0, 0, 0, -x, -y, -1, yp*x, yp*y, yp]
    
    # Solve using SVD: the homography is the right singular vector
    # corresponding to the smallest singular value
    _, _, Vh = np.linalg.svd(A)
    
    # The solution is the last row of V^T (or last column of V)
    # This corresponds to the null space of A
    h = Vh[-1]
    
    # Reshape the 9-element vector into a 3x3 matrix
    H_normalized = h.reshape(3, 3)
    
    # Denormalize: H = T_dst^(-1) @ H_normalized @ T_src
    # This transforms the homography back to the original coordinate system
    H = np.linalg.inv(T_dst) @ H_normalized @ T_src
    
    # Normalize so that h33 = 1 (standard convention)
    if abs(H[2, 2]) > 1e-10:
        H = H / H[2, 2]
    
    return H


def _normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize points for numerical stability in DLT algorithm.
    
    The normalization transforms points so that:
    - The centroid is at the origin
    - The average distance from the origin is sqrt(2)
    
    Parameters:
    -----------
    points : np.ndarray
        Points of shape (N, 2)
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        - Normalized points of shape (N, 2)
        - The 3x3 normalization matrix T
    
    Mathematical Background:
    ------------------------
    Normalization significantly improves the numerical conditioning of
    the DLT algorithm. Without normalization, results can be very poor
    when point coordinates have very different magnitudes.
    
    References:
    -----------
    Hartley, "In Defense of the Eight-Point Algorithm", PAMI 1997
    """
    # Compute centroid (mean of all points)
    centroid = np.mean(points, axis=0)
    
    # Translate points so centroid is at origin
    shifted = points - centroid
    
    # Compute average distance from origin
    distances = np.sqrt(np.sum(shifted**2, axis=1))
    avg_distance = np.mean(distances)
    
    # Scale factor to make average distance = sqrt(2)
    scale = np.sqrt(2) / (avg_distance + 1e-10)  # Add epsilon to avoid division by zero
    
    # Build the normalization matrix
    #     [scale    0    -scale*cx]
    # T = [  0    scale  -scale*cy]
    #     [  0      0         1   ]
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])
    
    # Apply normalization
    normalized_points = shifted * scale
    
    return normalized_points, T


def compute_homography(src_points: np.ndarray, dst_points: np.ndarray, 
                       method: str = 'dlt') -> HomographyMatrix:
    """
    Compute homography between two sets of corresponding points.
    
    This is a convenience function that wraps the DLT algorithm and returns
    a HomographyMatrix object.
    
    Parameters:
    -----------
    src_points : np.ndarray
        Source points of shape (N, 2) where N >= 4
    dst_points : np.ndarray
        Destination points of shape (N, 2) where N >= 4
    method : str
        Method to use: 'dlt' for Direct Linear Transform (default)
        or 'opencv' to use OpenCV's implementation
    
    Returns:
    --------
    HomographyMatrix
        The computed homography transformation
    
    Example:
    --------
    >>> src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    >>> dst = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float32)
    >>> H = compute_homography(src, dst)
    >>> H.transform_point(np.array([0.5, 0.5]))
    array([1., 1.])
    """
    if method == 'dlt':
        matrix = compute_homography_dlt(src_points, dst_points)
    elif method == 'opencv':
        # Use OpenCV's implementation with RANSAC for robustness
        matrix, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'dlt' or 'opencv'.")
    
    return HomographyMatrix(matrix, src_points, dst_points)


def compute_homography_ransac(src_points: np.ndarray, dst_points: np.ndarray,
                              n_iterations: int = 1000,
                              threshold: float = 3.0) -> Tuple[HomographyMatrix, np.ndarray]:
    """
    Compute homography using RANSAC for robustness to outliers.
    
    RANSAC (RANdom SAmple Consensus) is an iterative algorithm that:
    1. Randomly selects 4 point correspondences
    2. Computes homography from these 4 points
    3. Counts how many other points agree (inliers)
    4. Keeps the homography with the most inliers
    
    Parameters:
    -----------
    src_points : np.ndarray
        Source points of shape (N, 2) where N >= 4
    dst_points : np.ndarray
        Destination points of shape (N, 2) where N >= 4
    n_iterations : int
        Number of RANSAC iterations (default: 1000)
    threshold : float
        Distance threshold for considering a point an inlier (default: 3.0 pixels)
    
    Returns:
    --------
    Tuple[HomographyMatrix, np.ndarray]
        - The best homography found
        - Boolean mask indicating inliers
    
    Note:
    -----
    This is a simplified implementation. OpenCV's findHomography with
    cv2.RANSAC is more sophisticated and usually preferred in practice.
    """
    n_points = src_points.shape[0]
    best_H = None
    best_inlier_mask = None
    best_n_inliers = 0
    
    for _ in range(n_iterations):
        # Randomly select 4 points
        indices = np.random.choice(n_points, 4, replace=False)
        
        try:
            # Compute homography from these 4 points
            H = compute_homography_dlt(src_points[indices], dst_points[indices])
            
            # Transform all source points
            H_obj = HomographyMatrix(H)
            transformed = H_obj.transform_points(src_points)
            
            # Compute distances to actual destination points
            distances = np.sqrt(np.sum((transformed - dst_points)**2, axis=1))
            
            # Count inliers
            inlier_mask = distances < threshold
            n_inliers = np.sum(inlier_mask)
            
            # Update best if this is better
            if n_inliers > best_n_inliers:
                best_n_inliers = n_inliers
                best_inlier_mask = inlier_mask
                best_H = H
        except:
            # Degenerate configuration, skip
            continue
    
    if best_H is None:
        raise ValueError("RANSAC failed to find a valid homography")
    
    # Optionally: Recompute homography using all inliers for better accuracy
    if np.sum(best_inlier_mask) >= 4:
        best_H = compute_homography_dlt(
            src_points[best_inlier_mask],
            dst_points[best_inlier_mask]
        )
    
    return HomographyMatrix(best_H, src_points, dst_points), best_inlier_mask


def warp_image(image: np.ndarray, H: HomographyMatrix, 
               output_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Warp an image using a homography transformation.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image (grayscale or color)
    H : HomographyMatrix
        The homography transformation to apply
    output_size : Tuple[int, int], optional
        Size of output image as (width, height). If None, uses input size.
    
    Returns:
    --------
    np.ndarray
        The warped image
    
    Note:
    -----
    This function uses OpenCV's warpPerspective for efficient implementation.
    The mathematical operation being performed is:
        For each pixel (x', y') in the output, find the corresponding
        pixel (x, y) in the input using H^(-1), then copy/interpolate.
    """
    if output_size is None:
        output_size = (image.shape[1], image.shape[0])  # (width, height)
    
    # OpenCV's warpPerspective takes the forward transformation
    warped = cv2.warpPerspective(image, H.matrix, output_size)
    
    return warped


def apply_homography_to_point(H: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Apply a homography matrix to a single point.
    
    This is a simple utility function for applying a raw 3x3 matrix
    to a point without creating a HomographyMatrix object.
    
    Parameters:
    -----------
    H : np.ndarray
        A 3x3 homography matrix
    point : np.ndarray
        A 2D point as (x, y)
    
    Returns:
    --------
    np.ndarray
        The transformed point as (x', y')
    """
    # Convert to homogeneous coordinates
    p_homogeneous = np.array([point[0], point[1], 1.0])
    
    # Apply transformation
    p_transformed = H @ p_homogeneous
    
    # Convert back to Cartesian
    return p_transformed[:2] / p_transformed[2]


def create_perspective_transform(src_quad: np.ndarray, 
                                  dst_quad: np.ndarray) -> HomographyMatrix:
    """
    Create a perspective transform from a source quadrilateral to a destination.
    
    This is useful for perspective correction, where you want to transform
    a quadrilateral (e.g., a tilted document) to a rectangle.
    
    Parameters:
    -----------
    src_quad : np.ndarray
        Four corners of source quadrilateral, shape (4, 2)
        Order: top-left, top-right, bottom-right, bottom-left
    dst_quad : np.ndarray
        Four corners of destination quadrilateral, shape (4, 2)
    
    Returns:
    --------
    HomographyMatrix
        The perspective transformation
    
    Example:
    --------
    >>> # Transform a tilted rectangle to a proper rectangle
    >>> src = np.array([[100, 50], [400, 80], [420, 320], [80, 300]], dtype=np.float32)
    >>> dst = np.array([[0, 0], [300, 0], [300, 200], [0, 200]], dtype=np.float32)
    >>> H = create_perspective_transform(src, dst)
    """
    return compute_homography(src_quad.astype(np.float32), dst_quad.astype(np.float32))


# ============================================================================
# Mathematical Constants and Special Homographies
# ============================================================================

def identity_homography() -> HomographyMatrix:
    """
    Create the identity homography (no transformation).
    
    Returns:
    --------
    HomographyMatrix
        The 3x3 identity matrix as a homography
    """
    return HomographyMatrix(np.eye(3))


def translation_homography(tx: float, ty: float) -> HomographyMatrix:
    """
    Create a homography that translates points by (tx, ty).
    
    Parameters:
    -----------
    tx : float
        Translation in x direction
    ty : float
        Translation in y direction
    
    Returns:
    --------
    HomographyMatrix
        A translation homography
    
    Note:
    -----
    Translation is a special case of projective transformation:
        [1  0  tx]
    H = [0  1  ty]
        [0  0  1 ]
    """
    H = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float64)
    return HomographyMatrix(H)


def rotation_homography(angle: float, center: Tuple[float, float] = (0, 0)) -> HomographyMatrix:
    """
    Create a homography that rotates points around a center.
    
    Parameters:
    -----------
    angle : float
        Rotation angle in radians (counterclockwise positive)
    center : Tuple[float, float]
        Center of rotation (default: origin)
    
    Returns:
    --------
    HomographyMatrix
        A rotation homography
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    cx, cy = center
    
    # Rotation around (cx, cy): translate to origin, rotate, translate back
    H = np.array([
        [cos_a, -sin_a, cx - cos_a*cx + sin_a*cy],
        [sin_a, cos_a, cy - sin_a*cx - cos_a*cy],
        [0, 0, 1]
    ], dtype=np.float64)
    return HomographyMatrix(H)


def scale_homography(sx: float, sy: Optional[float] = None) -> HomographyMatrix:
    """
    Create a homography that scales points.
    
    Parameters:
    -----------
    sx : float
        Scale factor in x direction
    sy : float, optional
        Scale factor in y direction (default: same as sx)
    
    Returns:
    --------
    HomographyMatrix
        A scaling homography
    """
    if sy is None:
        sy = sx
    
    H = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ], dtype=np.float64)
    return HomographyMatrix(H)


if __name__ == "__main__":
    # Demo: Test the homography computation
    print("=" * 60)
    print("Homography Module Demo")
    print("=" * 60)
    
    # Define 4 source points (a unit square)
    src = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ], dtype=np.float32)
    
    # Define destination points (a scaled and translated square)
    dst = np.array([
        [100, 100],
        [300, 100],
        [300, 300],
        [100, 300]
    ], dtype=np.float32)
    
    # Compute homography
    H = compute_homography(src, dst)
    
    print("\nSource points:")
    print(src)
    print("\nDestination points:")
    print(dst)
    print("\nComputed homography matrix:")
    print(H)
    
    # Test transformation
    print("\nTransforming source points:")
    transformed = H.transform_points(src)
    print("Transformed points:")
    print(transformed)
    print("Expected (destination):")
    print(dst)
    print("Max error:", np.max(np.abs(transformed - dst)))
    
    # Test decomposition
    print("\nHomography decomposition:")
    decomp = H.get_decomposition()
    for key, value in decomp.items():
        print(f"  {key}: {value}")
