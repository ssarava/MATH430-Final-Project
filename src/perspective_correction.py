"""
Perspective Correction Tool
============================

This module provides an interactive tool for perspective correction using
homography transformations - a direct application of projective geometry.

The tool allows users to:
1. Select four corners of a quadrilateral in an image (e.g., a tilted document)
2. Compute the homography that maps it to a rectangle
3. Apply the transformation to obtain a "frontal" view

Mathematical Background:
------------------------
When viewing a planar surface (like a document) from an angle, the image
undergoes a projective transformation. The corners of a rectangle appear
as a general quadrilateral in the image.

To correct this, we:
1. Identify 4 corners of the quadrilateral in the image
2. Specify where these corners should map to (a rectangle)
3. Compute the homography H such that H maps the quadrilateral to the rectangle
4. Apply H to the entire image

This is a fundamental application of projective geometry in computer vision.

References:
-----------
- Birchfield, S. "An Introduction to Projective Geometry (for computer vision)"
- OpenCV documentation on perspective transforms
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Callable
from .homography import HomographyMatrix, compute_homography, warp_image
from .visualization import draw_quadrilateral, add_corner_labels, create_demo_image


class PerspectiveCorrectionTool:
    """
    An interactive tool for perspective correction using projective geometry.
    
    This tool demonstrates how homography transformations can be used to
    correct perspective distortion in images - a practical application of
    the projective geometry concepts discussed in the paper.
    
    Attributes:
    -----------
    image : np.ndarray
        The input image to be corrected
    corners : List[Tuple[int, int]]
        List of selected corner points (up to 4)
    output_size : Tuple[int, int]
        Size of the output corrected image (width, height)
    window_name : str
        Name of the OpenCV window
    
    Usage:
    ------
    >>> tool = PerspectiveCorrectionTool()
    >>> tool.load_image("document.jpg")
    >>> tool.run()  # Interactive mode - click 4 corners
    >>> corrected = tool.get_corrected_image()
    """
    
    def __init__(self, output_size: Tuple[int, int] = (400, 500)):
        """
        Initialize the perspective correction tool.
        
        Parameters:
        -----------
        output_size : Tuple[int, int]
            Size of the corrected output image (width, height)
            Default is 400x500, suitable for A4-like documents
        """
        # The output size for the corrected (rectangular) image
        self.output_size = output_size
        
        # Initialize state variables
        self.image: Optional[np.ndarray] = None  # Original image
        self.display_image: Optional[np.ndarray] = None  # Image shown in window
        self.corners: List[Tuple[int, int]] = []  # Selected corner points
        self.homography: Optional[HomographyMatrix] = None  # Computed homography
        self.corrected_image: Optional[np.ndarray] = None  # Result
        
        # Window settings
        self.window_name = "Perspective Correction Tool"
        
        # Corner labels following standard order: top-left, top-right, bottom-right, bottom-left
        self.corner_labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
    
    def load_image(self, image_or_path) -> bool:
        """
        Load an image from file or from a numpy array.
        
        Parameters:
        -----------
        image_or_path : str or np.ndarray
            Either a path to an image file or a numpy array
        
        Returns:
        --------
        bool
            True if image was loaded successfully
        
        Example:
        --------
        >>> tool = PerspectiveCorrectionTool()
        >>> tool.load_image("my_document.jpg")
        True
        """
        # Clear any previous state
        self.corners = []
        self.homography = None
        self.corrected_image = None
        
        # Load image
        if isinstance(image_or_path, str):
            # Load from file path
            self.image = cv2.imread(image_or_path)
            if self.image is None:
                print(f"Error: Could not load image from {image_or_path}")
                return False
        elif isinstance(image_or_path, np.ndarray):
            # Use provided array
            self.image = image_or_path.copy()
        else:
            print("Error: Expected file path or numpy array")
            return False
        
        # Create display copy
        self.display_image = self.image.copy()
        return True
    
    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """
        Handle mouse click events for corner selection.
        
        This is the callback function for OpenCV's setMouseCallback.
        It records clicked points and updates the display.
        
        Parameters:
        -----------
        event : int
            OpenCV mouse event type
        x, y : int
            Mouse coordinates
        flags : int
            OpenCV flags
        param : any
            Optional parameter (not used)
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Only accept up to 4 corners
            if len(self.corners) < 4:
                # Record the clicked point
                self.corners.append((x, y))
                print(f"Selected corner {len(self.corners)}: ({x}, {y}) - {self.corner_labels[len(self.corners)-1]}")
                
                # Update the display
                self._update_display()
                
                # If we have all 4 corners, compute the homography
                if len(self.corners) == 4:
                    self._compute_correction()
    
    def _update_display(self) -> None:
        """
        Update the display image with selected corners and connecting lines.
        
        This method redraws the image with:
        - Circles at selected corner points
        - Lines connecting the corners
        - Labels for each corner
        """
        # Start with a fresh copy of the original
        self.display_image = self.image.copy()
        
        if len(self.corners) > 0:
            # Convert corners to numpy array for drawing
            corners_array = np.array(self.corners)
            
            # Draw the corners and labels
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # BGR colors
            
            for i, (corner, label) in enumerate(zip(self.corners, self.corner_labels)):
                # Draw filled circle at corner
                cv2.circle(self.display_image, corner, 8, colors[i], -1)
                # Draw white border
                cv2.circle(self.display_image, corner, 8, (255, 255, 255), 2)
                # Add label
                label_pos = (corner[0] + 15, corner[1] + 5)
                cv2.putText(self.display_image, f"{i+1}: {label}", label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
            
            # Draw lines connecting corners if we have more than one
            if len(self.corners) > 1:
                for i in range(len(self.corners)):
                    pt1 = self.corners[i]
                    pt2 = self.corners[(i + 1) % len(self.corners)] if len(self.corners) > i + 1 else self.corners[0]
                    if i < len(self.corners) - 1 or len(self.corners) == 4:
                        cv2.line(self.display_image, pt1, pt2, (0, 255, 0), 2)
        
        # Add instruction text
        remaining = 4 - len(self.corners)
        if remaining > 0:
            instruction = f"Click {remaining} more corner(s). Order: TL -> TR -> BR -> BL"
        else:
            instruction = "Press SPACE to save, R to reset, ESC to exit"
        
        cv2.putText(self.display_image, instruction, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.display_image, instruction, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    def _compute_correction(self) -> None:
        """
        Compute the homography and apply perspective correction.
        
        This is the core mathematical operation:
        1. Define source points (the 4 clicked corners)
        2. Define destination points (corners of output rectangle)
        3. Compute the homography matrix
        4. Warp the image using the homography
        """
        if len(self.corners) != 4:
            print("Error: Need exactly 4 corners")
            return
        
        # Source points: the 4 corners selected by user
        src_points = np.array(self.corners, dtype=np.float32)
        
        # Destination points: corners of the output rectangle
        # Order matches the corner selection: TL, TR, BR, BL
        w, h = self.output_size
        dst_points = np.array([
            [0, 0],          # Top-left
            [w - 1, 0],      # Top-right
            [w - 1, h - 1],  # Bottom-right
            [0, h - 1]       # Bottom-left
        ], dtype=np.float32)
        
        print("\n" + "=" * 50)
        print("Computing Homography")
        print("=" * 50)
        print(f"\nSource points (selected corners):")
        for i, (pt, label) in enumerate(zip(src_points, self.corner_labels)):
            print(f"  {label}: ({pt[0]:.1f}, {pt[1]:.1f})")
        
        print(f"\nDestination points (rectangle corners):")
        for i, (pt, label) in enumerate(zip(dst_points, self.corner_labels)):
            print(f"  {label}: ({pt[0]:.1f}, {pt[1]:.1f})")
        
        # Compute the homography using our implementation
        self.homography = compute_homography(src_points, dst_points)
        
        print(f"\nComputed Homography Matrix:")
        print(self.homography.matrix)
        
        # Decompose for educational purposes
        decomp = self.homography.get_decomposition()
        print(f"\nHomography decomposition:")
        print(f"  Approximate scale: {decomp['scale']:.3f}")
        print(f"  Approximate rotation: {decomp['rotation_degrees']:.1f} degrees")
        print(f"  Translation: ({decomp['translation'][0]:.1f}, {decomp['translation'][1]:.1f})")
        print(f"  Perspective terms: ({decomp['perspective'][0]:.6f}, {decomp['perspective'][1]:.6f})")
        
        # Apply the transformation
        self.corrected_image = warp_image(self.image, self.homography, self.output_size)
        
        print("\nPerspective correction complete!")
        print("=" * 50)
        
        # Show the result in a new window
        cv2.imshow("Corrected Image", self.corrected_image)
        
        print("\nPress any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def run(self) -> Optional[np.ndarray]:
        """
        Run the interactive perspective correction tool.
        
        Opens a window where the user can click on 4 corners of a
        quadrilateral to correct its perspective.
        
        Returns:
        --------
        np.ndarray or None
            The corrected image, or None if cancelled
        
        Controls:
        ---------
        - Left click: Select corner (in order: TL, TR, BR, BL)
        - R: Reset all corners
        - SPACE/ENTER: Save result and exit
        - ESC: Cancel and exit
        """
        if self.image is None:
            print("Error: No image loaded. Call load_image() first.")
            return None
        
        # Create window and set up mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        # Initial display update
        self._update_display()
        
        print("\n" + "=" * 50)
        print("Interactive Perspective Correction Tool")
        print("=" * 50)
        print("\nInstructions:")
        print("  1. Click on the 4 corners of the quadrilateral")
        print("  2. Click in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left")
        print("  3. The tool will automatically compute the perspective correction")
        print("\nControls:")
        print("  - Left Click: Select a corner")
        print("  - R: Reset all corners")
        print("  - SPACE or ENTER: Save and exit")
        print("  - ESC: Cancel and exit")
        print("=" * 50 + "\n")
        
        while True:
            cv2.imshow(self.window_name, self.display_image)
            
            # Wait for key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC - cancel
                print("Cancelled.")
                cv2.destroyAllWindows()
                return None
            
            elif key == ord('r') or key == ord('R'):  # Reset
                print("Resetting corners...")
                self.corners = []
                self.homography = None
                self.corrected_image = None
                cv2.destroyWindow("Corrected Image")
                self._update_display()
            
            elif key == ord(' ') or key == 13:  # SPACE or ENTER - save and exit
                if self.corrected_image is not None:
                    print("Saving and exiting...")
                    cv2.destroyAllWindows()
                    return self.corrected_image
                else:
                    print("Please select all 4 corners first.")
        
        cv2.destroyAllWindows()
        return self.corrected_image
    
    def get_corrected_image(self) -> Optional[np.ndarray]:
        """
        Get the corrected image without running the interactive tool.
        
        Returns:
        --------
        np.ndarray or None
            The corrected image if available
        """
        return self.corrected_image
    
    def get_homography(self) -> Optional[HomographyMatrix]:
        """
        Get the computed homography matrix.
        
        Returns:
        --------
        HomographyMatrix or None
            The homography if computed
        """
        return self.homography
    
    def correct_with_corners(self, corners: List[Tuple[int, int]]) -> np.ndarray:
        """
        Perform perspective correction with pre-specified corners.
        
        This method allows programmatic use without the interactive tool.
        
        Parameters:
        -----------
        corners : List[Tuple[int, int]]
            Four corner points in order: TL, TR, BR, BL
        
        Returns:
        --------
        np.ndarray
            The corrected image
        
        Example:
        --------
        >>> tool = PerspectiveCorrectionTool()
        >>> tool.load_image("document.jpg")
        >>> corners = [(100, 50), (400, 80), (420, 500), (80, 480)]
        >>> corrected = tool.correct_with_corners(corners)
        """
        if self.image is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        if len(corners) != 4:
            raise ValueError("Exactly 4 corners required")
        
        self.corners = list(corners)
        self._compute_correction()
        return self.corrected_image


def run_demo():
    """
    Run a demonstration of the perspective correction tool.
    
    This function creates a demo image with a visible quadrilateral
    and shows how the tool can correct its perspective.
    """
    print("\n" + "=" * 60)
    print("Perspective Correction Demo")
    print("=" * 60)
    print("\nThis demo shows how projective geometry enables perspective correction.")
    print("A quadrilateral in an image can be transformed to a rectangle using")
    print("a homography - a 3x3 matrix representing a projective transformation.")
    print("=" * 60 + "\n")
    
    # Create a demo image with a visible quadrilateral
    demo_image = create_demo_image(600, 400)
    
    # Draw a quadrilateral on the image to simulate a tilted document
    quad_pts = np.array([
        [150, 80],   # Top-left (shifted)
        [450, 100],  # Top-right (shifted differently)
        [480, 350],  # Bottom-right
        [120, 320]   # Bottom-left
    ], dtype=np.int32)
    
    # Fill the quadrilateral with a "document" appearance
    mask = np.zeros_like(demo_image)
    cv2.fillPoly(mask, [quad_pts], (230, 230, 230))  # Light gray fill
    cv2.polylines(mask, [quad_pts], True, (0, 0, 150), 3)  # Dark red border
    
    # Add "text" lines inside the document
    for i in range(4):
        y = 120 + i * 50
        cv2.line(mask, (180, y), (440, y + 5), (100, 100, 100), 2)
    
    # Blend with original
    demo_image = cv2.addWeighted(demo_image, 0.5, mask, 0.5, 0)
    
    # Add the border back
    cv2.polylines(demo_image, [quad_pts], True, (0, 0, 200), 2)
    
    # Create tool and load the demo image
    tool = PerspectiveCorrectionTool(output_size=(300, 400))
    tool.load_image(demo_image)
    
    print("Demo image created with a tilted 'document' quadrilateral.")
    print("Running interactive mode - select the 4 corners of the document.")
    print("\nTip: Click corners in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left\n")
    
    # Run the interactive tool
    result = tool.run()
    
    if result is not None:
        # Save the results
        cv2.imwrite('test_images/output/perspective_original.png', demo_image)
        cv2.imwrite('test_images/output/perspective_corrected.png', result)
        print("\nResults saved to output/ directory")
    
    return result


if __name__ == "__main__":
    run_demo()
