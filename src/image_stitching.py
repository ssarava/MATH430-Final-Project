"""
Image Stitching Module
======================

This module implements image stitching (panorama creation) using homography
transformations. This is one of the most impressive applications of projective
geometry in computer vision.

The Process:
------------
1. Feature Detection: Find distinctive points in both images (SIFT, ORB, etc.)
2. Feature Matching: Find corresponding points between images
3. Homography Estimation: Compute the projective transformation relating the images
4. Warping: Transform one image to align with the other
5. Blending: Combine the images seamlessly

Mathematical Connection to Projective Geometry:
-----------------------------------------------
Image stitching works because:
- For a rotating camera (pure rotation, no translation), views of the world
  are related by a projective transformation (homography)
- The same is true for any planar scene (even with camera translation)

The homography matrix H (3x3) maps points from one image to another:
    p2 = H @ p1 (in homogeneous coordinates)

This is the same projective transformation discussed in Birchfield's article!

References:
-----------
- Birchfield, S. "An Introduction to Projective Geometry (for computer vision)"
- Hartley & Zisserman, "Multiple View Geometry", Chapter 13
- Szeliski, R. "Computer Vision: Algorithms and Applications", Chapter 9
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from .homography import HomographyMatrix, warp_image
from .visualization import draw_matches, display_images_side_by_side, create_demo_image


class ImageStitcher:
    """
    A class for stitching images together using homography transformations.
    
    This class implements the complete image stitching pipeline, from feature
    detection to final blending, demonstrating how projective geometry enables
    the creation of panoramic images.
    
    Attributes:
    -----------
    feature_detector : str
        The feature detection method to use ('orb', 'sift', 'akaze')
    matcher_type : str
        The feature matching method ('bf' for brute force, 'flann')
    min_matches : int
        Minimum number of matches required for valid stitching
    ransac_threshold : float
        RANSAC threshold for homography estimation
    
    Example:
    --------
    >>> stitcher = ImageStitcher(feature_detector='orb')
    >>> result = stitcher.stitch([img1, img2])
    """
    
    def __init__(self, 
                 feature_detector: str = 'orb',
                 matcher_type: str = 'bf',
                 min_matches: int = 10,
                 ransac_threshold: float = 5.0):
        """
        Initialize the image stitcher.
        
        Parameters:
        -----------
        feature_detector : str
            Feature detection algorithm: 'orb', 'sift', or 'akaze'
            ORB is fastest, SIFT is most accurate but requires opencv-contrib
        matcher_type : str
            Matching algorithm: 'bf' (brute force) or 'flann'
        min_matches : int
            Minimum number of good matches required
        ransac_threshold : float
            RANSAC reprojection threshold in pixels
        """
        self.feature_detector_name = feature_detector.lower()
        self.matcher_type = matcher_type.lower()
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        
        # Initialize the feature detector
        self.detector = self._create_detector()
        
        # Initialize the matcher
        self.matcher = self._create_matcher()
        
        # Storage for intermediate results (useful for visualization)
        self.keypoints1: Optional[List] = None
        self.keypoints2: Optional[List] = None
        self.matches: Optional[List] = None
        self.good_matches: Optional[List] = None
        self.homography: Optional[HomographyMatrix] = None
        self.inlier_mask: Optional[np.ndarray] = None
    
    def _create_detector(self):
        """
        Create the feature detector based on configuration.
        
        Returns:
        --------
        The OpenCV feature detector object
        
        Note:
        -----
        Different detectors have different characteristics:
        - ORB: Fast, good for real-time applications, rotation invariant
        - SIFT: Very robust, scale and rotation invariant, but patented (use opencv-contrib)
        - AKAZE: Good balance of speed and accuracy, no patent issues
        """
        if self.feature_detector_name == 'orb':
            # ORB: Oriented FAST and Rotated BRIEF
            # Fast and efficient, works well for most cases
            return cv2.ORB_create(nfeatures=2000)
        
        elif self.feature_detector_name == 'sift':
            # SIFT: Scale-Invariant Feature Transform
            # Most robust but requires opencv-contrib-python
            try:
                return cv2.SIFT_create()
            except AttributeError:
                print("SIFT not available. Falling back to ORB.")
                print("Install opencv-contrib-python for SIFT support.")
                return cv2.ORB_create(nfeatures=2000)
        
        elif self.feature_detector_name == 'akaze':
            # AKAZE: Accelerated-KAZE
            # Good balance of speed and accuracy
            return cv2.AKAZE_create()
        
        else:
            raise ValueError(f"Unknown detector: {self.feature_detector_name}")
    
    def _create_matcher(self):
        """
        Create the feature matcher based on configuration.
        
        Returns:
        --------
        The OpenCV matcher object
        """
        if self.matcher_type == 'bf':
            # Brute Force matcher
            # For ORB, use Hamming distance (binary descriptors)
            # For SIFT/AKAZE, use L2 distance
            if self.feature_detector_name == 'orb':
                return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            else:
                return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        elif self.matcher_type == 'flann':
            # FLANN: Fast Library for Approximate Nearest Neighbors
            # Faster for large numbers of features
            if self.feature_detector_name == 'orb':
                # FLANN parameters for binary descriptors
                index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                                   table_number=6,
                                   key_size=12,
                                   multi_probe_level=1)
            else:
                # FLANN parameters for floating point descriptors
                index_params = dict(algorithm=1,  # FLANN_INDEX_KDTREE
                                   trees=5)
            
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        
        else:
            raise ValueError(f"Unknown matcher: {self.matcher_type}")
    
    def detect_and_match(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect features in both images and find matches.
        
        Parameters:
        -----------
        img1 : np.ndarray
            First image (grayscale or color)
        img2 : np.ndarray
            Second image (grayscale or color)
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Arrays of matched points (pts1, pts2) each of shape (N, 2)
        
        Process:
        --------
        1. Convert images to grayscale if needed
        2. Detect keypoints and compute descriptors
        3. Match descriptors between images
        4. Filter matches using ratio test
        """
        # Convert to grayscale for feature detection
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        # Detect keypoints and compute descriptors
        print("Detecting features...")
        self.keypoints1, descriptors1 = self.detector.detectAndCompute(gray1, None)
        self.keypoints2, descriptors2 = self.detector.detectAndCompute(gray2, None)
        
        print(f"  Image 1: {len(self.keypoints1)} keypoints")
        print(f"  Image 2: {len(self.keypoints2)} keypoints")
        
        if descriptors1 is None or descriptors2 is None:
            raise ValueError("Could not compute descriptors. Images may be too simple.")
        
        # Match features using k-nearest neighbors
        print("Matching features...")
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        
        # Apply Lowe's ratio test to filter good matches
        # A match is good if the best match is significantly better than the second best
        self.good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                # Ratio test: keep match if best match distance < 0.75 * second best distance
                if m.distance < 0.75 * n.distance:
                    self.good_matches.append(m)
        
        print(f"  Good matches after ratio test: {len(self.good_matches)}")
        
        if len(self.good_matches) < self.min_matches:
            raise ValueError(f"Not enough good matches: {len(self.good_matches)} < {self.min_matches}")
        
        # Extract matched point coordinates
        pts1 = np.float32([self.keypoints1[m.queryIdx].pt for m in self.good_matches])
        pts2 = np.float32([self.keypoints2[m.trainIdx].pt for m in self.good_matches])
        
        return pts1, pts2
    
    def compute_homography(self, pts1: np.ndarray, pts2: np.ndarray) -> HomographyMatrix:
        """
        Compute the homography between two sets of matched points using RANSAC.
        
        Parameters:
        -----------
        pts1 : np.ndarray
            Points in first image, shape (N, 2)
        pts2 : np.ndarray
            Corresponding points in second image, shape (N, 2)
        
        Returns:
        --------
        HomographyMatrix
            The computed homography transformation
        
        Mathematical Background:
        ------------------------
        RANSAC (Random Sample Consensus) is used to robustly estimate the homography
        even when some matches are incorrect (outliers).
        
        The homography H satisfies: pts2 ≈ H @ pts1 (in homogeneous coordinates)
        
        This is the projective transformation from Birchfield's article!
        """
        print("Computing homography with RANSAC...")
        
        # Use OpenCV's findHomography with RANSAC
        H_matrix, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, self.ransac_threshold)
        
        if H_matrix is None:
            raise ValueError("Could not compute homography")
        
        # Store the inlier mask for visualization
        self.inlier_mask = mask.ravel().astype(bool) if mask is not None else None
        
        # Create HomographyMatrix object
        self.homography = HomographyMatrix(H_matrix, pts1, pts2)
        
        # Report results
        n_inliers = np.sum(self.inlier_mask) if self.inlier_mask is not None else len(pts1)
        print(f"  Homography computed")
        print(f"  Inliers: {n_inliers}/{len(pts1)} ({100*n_inliers/len(pts1):.1f}%)")
        
        # Print the homography matrix for educational purposes
        print(f"\n  Homography Matrix H:")
        print(f"  {self.homography.matrix}")
        
        return self.homography
    
    def warp_and_blend(self, img1: np.ndarray, img2: np.ndarray, 
                       H: HomographyMatrix) -> np.ndarray:
        """
        Warp the second image and blend it with the first to create a panorama.
        
        Parameters:
        -----------
        img1 : np.ndarray
            First image (reference/base image)
        img2 : np.ndarray
            Second image (to be warped)
        H : HomographyMatrix
            Homography from image 2 to image 1
        
        Returns:
        --------
        np.ndarray
            The stitched panorama image
        
        Process:
        --------
        1. Compute the size of the output panorama
        2. Compute translation to keep all pixels positive
        3. Warp both images into the output canvas
        4. Blend the overlapping regions
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Compute corners of img2 after transformation
        corners2 = np.float32([
            [0, 0],
            [w2, 0],
            [w2, h2],
            [0, h2]
        ]).reshape(-1, 1, 2)
        
        # Transform corners of img2
        corners2_transformed = cv2.perspectiveTransform(corners2, H.matrix)
        
        # Corners of img1
        corners1 = np.float32([
            [0, 0],
            [w1, 0],
            [w1, h1],
            [0, h1]
        ]).reshape(-1, 1, 2)
        
        # Combine all corners to find the bounding box
        all_corners = np.concatenate([corners1, corners2_transformed], axis=0)
        
        # Find min and max coordinates
        x_min = int(np.floor(all_corners[:, :, 0].min()))
        x_max = int(np.ceil(all_corners[:, :, 0].max()))
        y_min = int(np.floor(all_corners[:, :, 1].min()))
        y_max = int(np.ceil(all_corners[:, :, 1].max()))
        
        # Translation matrix to shift everything to positive coordinates
        # This creates a larger canvas that fits both images
        translation = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Output size
        output_width = x_max - x_min
        output_height = y_max - y_min
        output_size = (output_width, output_height)
        
        print(f"\nWarping images...")
        print(f"  Output panorama size: {output_width} x {output_height}")
        
        # Warp img2 using H combined with translation
        H_combined = translation @ H.matrix
        warped2 = cv2.warpPerspective(img2, H_combined, output_size)
        
        # Warp img1 using just translation (to place it in the output canvas)
        warped1 = cv2.warpPerspective(img1, translation, output_size)
        
        # Simple blending: use pixels from warped2 where warped1 is black
        # More sophisticated blending (multiband, feathering) would look better
        result = self._simple_blend(warped1, warped2)
        
        return result
    
    def _simple_blend(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Simple blending of two images.
        
        For overlapping regions, take the average.
        For non-overlapping regions, take whichever image has content.
        
        Parameters:
        -----------
        img1 : np.ndarray
            First warped image
        img2 : np.ndarray
            Second warped image
        
        Returns:
        --------
        np.ndarray
            Blended result
        """
        # Create masks for non-zero regions
        if len(img1.shape) == 3:
            mask1 = np.any(img1 > 0, axis=2).astype(np.float32)
            mask2 = np.any(img2 > 0, axis=2).astype(np.float32)
        else:
            mask1 = (img1 > 0).astype(np.float32)
            mask2 = (img2 > 0).astype(np.float32)
        
        # Overlap region
        overlap = mask1 * mask2
        
        # Compute blending weights
        # In overlap region, use average; otherwise use the image that has content
        weight1 = np.where(overlap > 0, 0.5, mask1)
        weight2 = np.where(overlap > 0, 0.5, mask2)
        
        # Normalize weights
        weight_sum = weight1 + weight2
        weight_sum[weight_sum == 0] = 1  # Avoid division by zero
        weight1 = weight1 / weight_sum
        weight2 = weight2 / weight_sum
        
        # Expand dimensions for color images
        if len(img1.shape) == 3:
            weight1 = weight1[:, :, np.newaxis]
            weight2 = weight2[:, :, np.newaxis]
        
        # Blend
        result = (img1.astype(np.float32) * weight1 + 
                  img2.astype(np.float32) * weight2).astype(np.uint8)
        
        return result
    
    def stitch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Stitch a list of images into a panorama.
        
        Parameters:
        -----------
        images : List[np.ndarray]
            List of images to stitch (currently supports 2 images)
        
        Returns:
        --------
        np.ndarray
            The stitched panorama
        
        Example:
        --------
        >>> stitcher = ImageStitcher()
        >>> panorama = stitcher.stitch([left_image, right_image])
        """
        if len(images) < 2:
            raise ValueError("Need at least 2 images to stitch")
        
        print("\n" + "=" * 50)
        print("Image Stitching Pipeline")
        print("=" * 50)
        
        # For now, we handle 2 images
        # Extension to multiple images would use sequential or bundle adjustment
        img1, img2 = images[0], images[1]
        
        print(f"\nImage 1: {img1.shape}")
        print(f"Image 2: {img2.shape}")
        
        # Step 1: Detect features and find matches
        pts1, pts2 = self.detect_and_match(img1, img2)
        
        # Step 2: Compute homography
        H = self.compute_homography(pts1, pts2)
        
        # Step 3: Warp and blend
        result = self.warp_and_blend(img1, img2, H)
        
        print("\nStitching complete!")
        print("=" * 50)
        
        return result
    
    def visualize_matches(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Create a visualization of the feature matches.
        
        Parameters:
        -----------
        img1 : np.ndarray
            First image
        img2 : np.ndarray
            Second image
        
        Returns:
        --------
        np.ndarray
            Image showing matches between the two images
        """
        if self.keypoints1 is None or self.good_matches is None:
            raise ValueError("No matches computed yet. Call stitch() first.")
        
        # Extract matched points
        pts1 = np.float32([self.keypoints1[m.queryIdx].pt for m in self.good_matches])
        pts2 = np.float32([self.keypoints2[m.trainIdx].pt for m in self.good_matches])
        
        # Draw matches
        return draw_matches(img1, img2, pts1, pts2, self.inlier_mask)


def create_test_images() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create two test images with overlapping content for stitching demo.
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Two images that can be stitched together
    """
    # Create a larger base image
    base = np.zeros((400, 800, 3), dtype=np.uint8)
    
    # Add some recognizable features (shapes, patterns)
    # Colored rectangles
    cv2.rectangle(base, (50, 50), (150, 150), (255, 0, 0), -1)  # Blue
    cv2.rectangle(base, (200, 100), (350, 250), (0, 255, 0), -1)  # Green
    cv2.rectangle(base, (400, 50), (550, 200), (0, 0, 255), -1)  # Red
    cv2.rectangle(base, (600, 150), (750, 350), (255, 255, 0), -1)  # Cyan
    
    # Circles
    cv2.circle(base, (100, 300), 50, (255, 0, 255), -1)  # Magenta
    cv2.circle(base, (300, 300), 40, (0, 255, 255), -1)  # Yellow
    cv2.circle(base, (500, 300), 60, (128, 128, 255), -1)  # Light red
    cv2.circle(base, (700, 250), 45, (128, 255, 128), -1)  # Light green
    
    # Add some texture (lines)
    for i in range(0, 800, 30):
        cv2.line(base, (i, 0), (i, 400), (50, 50, 50), 1)
    for i in range(0, 400, 30):
        cv2.line(base, (0, i), (800, i), (50, 50, 50), 1)
    
    # Add text
    cv2.putText(base, "LEFT", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(base, "CENTER", (350, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(base, "RIGHT", (600, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Extract two overlapping regions
    # Image 1: left portion with some overlap
    img1 = base[:, 0:500].copy()
    
    # Image 2: right portion with some overlap
    img2 = base[:, 300:800].copy()
    
    return img1, img2


def run_demo():
    """
    Run a demonstration of image stitching.
    
    This function creates test images and demonstrates the complete
    stitching pipeline, showing how projective geometry enables
    panorama creation.
    """
    print("\n" + "=" * 60)
    print("Image Stitching Demo")
    print("=" * 60)
    print("\nThis demo shows how projective geometry enables image stitching.")
    print("Two overlapping images are combined by:")
    print("  1. Finding corresponding points (features)")
    print("  2. Computing the homography relating the images")
    print("  3. Warping one image to align with the other")
    print("  4. Blending the images together")
    print("=" * 60)
    
    # Create test images
    print("\nCreating test images with overlapping content...")
    img1, img2 = create_test_images()
    
    # Save test images
    cv2.imwrite('test_images/output/stitch_left.png', img1)
    cv2.imwrite('test_images/output/stitch_right.png', img2)
    print(f"  Saved test images to output/")
    
    # Create stitcher
    stitcher = ImageStitcher(feature_detector='orb')
    
    try:
        # Perform stitching
        result = stitcher.stitch([img1, img2])
        
        # Save result
        cv2.imwrite('test_images/output/stitch_result.png', result)
        print("\nSaved panorama to output/stitch_result.png")
        
        # Create and save match visualization
        match_viz = stitcher.visualize_matches(img1, img2)
        cv2.imwrite('test_images/output/stitch_matches.png', match_viz)
        print("Saved match visualization to output/stitch_matches.png")
        
        # Display results
        print("\nDisplaying results...")
        
        # Show input images
        cv2.imshow("Image 1 (Left)", img1)
        cv2.imshow("Image 2 (Right)", img2)
        cv2.imshow("Feature Matches", match_viz)
        cv2.imshow("Stitched Panorama", result)
        
        print("\nPress any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return result
        
    except Exception as e:
        print(f"\nError during stitching: {e}")
        print("This can happen if there aren't enough matching features.")
        return None


if __name__ == "__main__":
    run_demo()
