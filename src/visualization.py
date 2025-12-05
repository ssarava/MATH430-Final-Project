"""
Visualization Module
====================

This module provides utilities for visualizing images, point correspondences,
feature matches, and the results of projective transformations.

These visualizations are essential for:
- Understanding how homographies transform images
- Debugging feature matching in image stitching
- Presenting results in papers and presentations
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def display_images_side_by_side(images: List[np.ndarray], 
                                 titles: Optional[List[str]] = None,
                                 figsize: Tuple[int, int] = (15, 5),
                                 cmap: str = 'gray') -> Figure:
    """
    Display multiple images side by side using matplotlib.
    
    Parameters:
    -----------
    images : List[np.ndarray]
        List of images to display
    titles : List[str], optional
        Titles for each image
    figsize : Tuple[int, int]
        Figure size in inches
    cmap : str
        Colormap for grayscale images (default: 'gray')
    
    Returns:
    --------
    Figure
        The matplotlib figure object
    
    Example:
    --------
    >>> img1 = cv2.imread('image1.jpg')
    >>> img2 = cv2.imread('image2.jpg')
    >>> fig = display_images_side_by_side([img1, img2], ['Original', 'Transformed'])
    >>> plt.show()
    """
    n_images = len(images)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    
    # Handle single image case
    if n_images == 1:
        axes = [axes]
    
    for i, (ax, img) in enumerate(zip(axes, images)):
        # Convert BGR to RGB if it's a color image (OpenCV uses BGR)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_display)
        else:
            ax.imshow(img, cmap=cmap)
        
        # Set title if provided
        if titles is not None and i < len(titles):
            ax.set_title(titles[i])
        
        # Remove axis ticks for cleaner look
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def draw_points_on_image(image: np.ndarray, 
                         points: np.ndarray,
                         color: Tuple[int, int, int] = (0, 255, 0),
                         radius: int = 5,
                         thickness: int = -1,
                         labels: Optional[List[str]] = None) -> np.ndarray:
    """
    Draw points on an image with optional labels.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image (will be modified in place if not copied)
    points : np.ndarray
        Array of points with shape (N, 2)
    color : Tuple[int, int, int]
        BGR color for the points
    radius : int
        Radius of the circles
    thickness : int
        Thickness of circle outline (-1 for filled)
    labels : List[str], optional
        Labels to display next to each point
    
    Returns:
    --------
    np.ndarray
        Image with points drawn
    """
    # Create a copy to avoid modifying the original
    output = image.copy()
    
    # Ensure image is color
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    
    for i, point in enumerate(points):
        # Convert to integer coordinates
        x, y = int(point[0]), int(point[1])
        
        # Draw circle
        cv2.circle(output, (x, y), radius, color, thickness)
        
        # Add label if provided
        if labels is not None and i < len(labels):
            cv2.putText(output, labels[i], (x + radius + 2, y + radius),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return output


def draw_quadrilateral(image: np.ndarray,
                       points: np.ndarray,
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2,
                       closed: bool = True) -> np.ndarray:
    """
    Draw a quadrilateral (or polygon) on an image.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    points : np.ndarray
        Array of 4 corner points with shape (4, 2)
    color : Tuple[int, int, int]
        BGR color for the lines
    thickness : int
        Line thickness
    closed : bool
        Whether to close the polygon
    
    Returns:
    --------
    np.ndarray
        Image with quadrilateral drawn
    """
    output = image.copy()
    
    # Ensure image is color
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    
    # Convert points to proper format for polylines
    pts = points.reshape((-1, 1, 2)).astype(np.int32)
    
    # Draw the polygon
    cv2.polylines(output, [pts], closed, color, thickness)
    
    return output


def draw_matches(img1: np.ndarray, 
                 img2: np.ndarray,
                 pts1: np.ndarray, 
                 pts2: np.ndarray,
                 inlier_mask: Optional[np.ndarray] = None,
                 max_matches: int = 50) -> np.ndarray:
    """
    Draw matching points between two images.
    
    Creates a side-by-side visualization showing corresponding points
    connected by lines.
    
    Parameters:
    -----------
    img1 : np.ndarray
        First image
    img2 : np.ndarray
        Second image
    pts1 : np.ndarray
        Points in first image, shape (N, 2)
    pts2 : np.ndarray
        Corresponding points in second image, shape (N, 2)
    inlier_mask : np.ndarray, optional
        Boolean array indicating which matches are inliers
        Inliers are drawn in green, outliers in red
    max_matches : int
        Maximum number of matches to draw (for clarity)
    
    Returns:
    --------
    np.ndarray
        Combined image showing matches
    """
    # Convert images to color if needed
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    # Get image dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create output image (side by side)
    height = max(h1, h2)
    width = w1 + w2
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[:h1, :w1] = img1
    output[:h2, w1:w1+w2] = img2
    
    # Limit number of matches to draw
    n_matches = min(len(pts1), max_matches)
    indices = np.random.choice(len(pts1), n_matches, replace=False) if len(pts1) > max_matches else range(len(pts1))
    
    for i in indices:
        # Get points
        pt1 = tuple(map(int, pts1[i]))
        pt2 = (int(pts2[i][0]) + w1, int(pts2[i][1]))  # Offset for second image
        
        # Determine color based on inlier status
        if inlier_mask is not None:
            color = (0, 255, 0) if inlier_mask[i] else (0, 0, 255)  # Green for inlier, red for outlier
        else:
            color = (255, 255, 0)  # Cyan for unknown status
        
        # Draw line connecting the points
        cv2.line(output, pt1, pt2, color, 1)
        
        # Draw circles at the points
        cv2.circle(output, pt1, 3, color, -1)
        cv2.circle(output, pt2, 3, color, -1)
    
    return output


def create_comparison_figure(original: np.ndarray,
                             transformed: np.ndarray,
                             title: str = "Projective Transformation",
                             save_path: Optional[str] = None) -> Figure:
    """
    Create a publication-quality comparison figure.
    
    Parameters:
    -----------
    original : np.ndarray
        Original image
    transformed : np.ndarray
        Transformed image
    title : str
        Figure title
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    Figure
        The matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    if len(original.shape) == 3:
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Transformed image
    if len(transformed.shape) == 3:
        axes[1].imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
    else:
        axes[1].imshow(transformed, cmap='gray')
    axes[1].set_title("Transformed")
    axes[1].axis('off')
    
    # Overall title
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_homography_grid(image: np.ndarray,
                               H: np.ndarray,
                               grid_size: Tuple[int, int] = (10, 10)) -> np.ndarray:
    """
    Visualize how a homography transforms a grid overlaid on the image.
    
    This is helpful for understanding the geometric nature of the transformation.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    H : np.ndarray
        3x3 homography matrix
    grid_size : Tuple[int, int]
        Number of grid lines in (horizontal, vertical) directions
    
    Returns:
    --------
    np.ndarray
        Image with transformed grid overlay
    """
    output = image.copy()
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    
    h, w = image.shape[:2]
    
    # Create grid points
    x_lines = np.linspace(0, w, grid_size[0])
    y_lines = np.linspace(0, h, grid_size[1])
    
    # Draw transformed horizontal lines
    for y in y_lines:
        pts = np.array([[x, y] for x in np.linspace(0, w, 100)], dtype=np.float32)
        # Transform points
        pts_h = np.column_stack([pts, np.ones(len(pts))])
        transformed = (H @ pts_h.T).T
        transformed = transformed[:, :2] / transformed[:, 2:3]
        
        # Draw line segments
        pts_int = transformed.astype(np.int32)
        for i in range(len(pts_int) - 1):
            cv2.line(output, tuple(pts_int[i]), tuple(pts_int[i+1]), (0, 255, 255), 1)
    
    # Draw transformed vertical lines
    for x in x_lines:
        pts = np.array([[x, y] for y in np.linspace(0, h, 100)], dtype=np.float32)
        # Transform points
        pts_h = np.column_stack([pts, np.ones(len(pts))])
        transformed = (H @ pts_h.T).T
        transformed = transformed[:, :2] / transformed[:, 2:3]
        
        # Draw line segments
        pts_int = transformed.astype(np.int32)
        for i in range(len(pts_int) - 1):
            cv2.line(output, tuple(pts_int[i]), tuple(pts_int[i+1]), (255, 0, 255), 1)
    
    return output


def create_demo_image(width: int = 400, height: int = 300) -> np.ndarray:
    """
    Create a simple demo image for testing.
    
    The image contains a checkerboard pattern with colored regions,
    useful for visualizing how transformations affect the image.
    
    Parameters:
    -----------
    width : int
        Image width
    height : int
        Image height
    
    Returns:
    --------
    np.ndarray
        A color test image
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create checkerboard pattern
    block_size = 40
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            if (i // block_size + j // block_size) % 2 == 0:
                image[i:i+block_size, j:j+block_size] = [200, 200, 200]  # Light gray
            else:
                image[i:i+block_size, j:j+block_size] = [100, 100, 100]  # Dark gray
    
    # Add colored corners for reference
    corner_size = 60
    image[:corner_size, :corner_size] = [0, 0, 255]  # Red - top left
    image[:corner_size, -corner_size:] = [0, 255, 0]  # Green - top right
    image[-corner_size:, -corner_size:] = [255, 0, 0]  # Blue - bottom right
    image[-corner_size:, :corner_size] = [0, 255, 255]  # Yellow - bottom left
    
    # Add text
    cv2.putText(image, "TEST IMAGE", (width//2 - 60, height//2),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return image


def add_corner_labels(image: np.ndarray, 
                      corners: np.ndarray,
                      labels: List[str] = ["TL", "TR", "BR", "BL"]) -> np.ndarray:
    """
    Add labels to corner points on an image.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    corners : np.ndarray
        Array of 4 corner points, shape (4, 2)
    labels : List[str]
        Labels for each corner (default: TL, TR, BR, BL)
    
    Returns:
    --------
    np.ndarray
        Image with labeled corners
    """
    output = image.copy()
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # R, G, B, Y
    
    for i, (corner, label) in enumerate(zip(corners, labels)):
        x, y = int(corner[0]), int(corner[1])
        
        # Draw circle at corner
        cv2.circle(output, (x, y), 8, colors[i], -1)
        cv2.circle(output, (x, y), 8, (255, 255, 255), 2)
        
        # Add label
        cv2.putText(output, label, (x + 12, y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
    
    return output


if __name__ == "__main__":
    # Demo: Test visualization functions
    print("=" * 60)
    print("Visualization Module Demo")
    print("=" * 60)
    
    # Create a test image
    test_img = create_demo_image(400, 300)
    
    # Add some points
    points = np.array([
        [50, 50],
        [350, 50],
        [350, 250],
        [50, 250]
    ])
    
    # Draw points and quadrilateral
    img_with_points = draw_points_on_image(test_img, points, labels=['1', '2', '3', '4'])
    img_with_quad = draw_quadrilateral(img_with_points, points, color=(0, 255, 0))
    
    # Display
    print("Creating visualization...")
    fig = display_images_side_by_side(
        [test_img, img_with_quad],
        ['Original', 'With Annotations']
    )
    
    # Save figure
    fig.savefig('test_images/output/visualization_demo.png', 
                dpi=150, bbox_inches='tight')
    print("Saved to output/visualization_demo.png")
    
    plt.close(fig)
