#!/usr/bin/env python3
"""
Projective Geometry in Computer Vision - Main Application
==========================================================

This is the main entry point for the Projective Geometry Computer Vision project.
It provides an interactive menu to run various demonstrations of how projective
geometry concepts are applied in computer vision.

Project Overview:
-----------------
This project accompanies a final paper for MATH 430 (Euclidean and Non-Euclidean
Geometries) exploring the connections between projective geometry and computer vision.

Demonstrations included:
1. Perspective Correction - Transform a quadrilateral to a rectangle
2. Image Stitching - Combine overlapping images into a panorama
3. Homography Basics - Interactive exploration of homography transformations

Mathematical Foundation:
------------------------
All demonstrations are based on the homography matrix, a 3x3 projective
transformation that relates two views of a planar surface. Key concepts:

- Homogeneous Coordinates: (x, y) -> (x, y, 1) allowing linear representation
- Projective Transformation: p' = H @ p maps points between images
- Cross Ratio Invariance: The key invariant preserved under projection

References:
-----------
- Birchfield, S. "An Introduction to Projective Geometry (for computer vision)"
- Hartley & Zisserman, "Multiple View Geometry in Computer Vision"

Usage:
------
    python main.py              # Run with interactive menu
    python main.py --demo 1     # Run specific demo (1=perspective, 2=stitching, 3=basics)
    python main.py --help       # Show help

Author: Student
Course: MATH 430 - Euclidean and Non-Euclidean Geometries
"""

import sys
import os
import argparse
import numpy as np
import cv2

# Add the project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our modules
from src.homography import (
    HomographyMatrix, 
    compute_homography, 
    compute_homography_dlt,
    identity_homography,
    translation_homography,
    rotation_homography,
    scale_homography
)
from src.perspective_correction import PerspectiveCorrectionTool, run_demo as perspective_demo
from src.image_stitching import ImageStitcher, run_demo as stitching_demo, create_test_images
from src.visualization import create_demo_image, display_images_side_by_side


def print_header():
    """Print the application header."""
    header = """
╔══════════════════════════════════════════════════════════════════════════════╗
║        PROJECTIVE GEOMETRY IN COMPUTER VISION - INTERACTIVE DEMOS            ║
║                                                                              ║
║         Final Project for MATH 430: Euclidean and Non-Euclidean Geometries   ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(header)


def print_menu():
    """Print the main menu."""
    menu = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MAIN MENU                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Perspective Correction Demo                                              │
│     - Click 4 corners to correct perspective distortion                      │
│     - Shows how homography transforms quadrilaterals to rectangles           │
│                                                                              │
│  2. Image Stitching Demo                                                     │
│     - Combine overlapping images into a panorama                             │
│     - Demonstrates feature matching and homography estimation                │
│                                                                              │
│  3. Homography Basics Demo                                                   │
│     - Interactive exploration of projective transformations                  │
│     - See how different homographies affect images                           │
│                                                                              │
│  4. Run All Demos                                                            │
│     - Execute all demonstrations in sequence                                 │
│                                                                              │
│  5. About This Project                                                       │
│     - Learn about the mathematical background                                │
│                                                                              │
│  0. Exit                                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
    """
    print(menu)


def demo_homography_basics():
    """
    Interactive demonstration of basic homography transformations.
    
    This demo shows how different types of transformations (translation,
    rotation, scaling, and perspective) affect an image, helping build
    intuition for projective geometry.
    """
    print("\n" + "=" * 60)
    print("Homography Basics Demo")
    print("=" * 60)
    print("""
This demo shows how different transformations affect an image.
We'll demonstrate:
1. Translation (shifting)
2. Rotation
3. Scaling
4. Perspective transformation (the general case)

Each transformation is represented by a 3x3 homography matrix!
    """)
    
    # Create a test image
    img = create_demo_image(400, 300)
    
    # Create output directory if needed
    os.makedirs('test_images/output', exist_ok=True)
    
    print("\nDemonstrating transformations...")
    
    # 1. Identity (no change)
    H_identity = identity_homography()
    img_identity = cv2.warpPerspective(img, H_identity.matrix, (500, 400))
    print(f"\n1. Identity Matrix (no transformation):")
    print(H_identity.matrix)
    
    # 2. Translation
    H_translate = translation_homography(50, 30)  # Shift right 50, down 30
    img_translated = cv2.warpPerspective(img, H_translate.matrix, (500, 400))
    print(f"\n2. Translation (50 right, 30 down):")
    print(H_translate.matrix)
    
    # 3. Rotation
    # Rotate 15 degrees around the center
    center = (img.shape[1] // 2, img.shape[0] // 2)
    H_rotate = rotation_homography(np.radians(15), center)
    img_rotated = cv2.warpPerspective(img, H_rotate.matrix, (500, 400))
    print(f"\n3. Rotation (15° counterclockwise):")
    print(H_rotate.matrix)
    
    # 4. Scaling
    H_scale = scale_homography(1.2, 0.8)  # Scale x by 1.2, y by 0.8
    img_scaled = cv2.warpPerspective(img, H_scale.matrix, (500, 400))
    print(f"\n4. Non-uniform Scaling (1.2x horizontal, 0.8x vertical):")
    print(H_scale.matrix)
    
    # 5. Perspective (the general projective transformation)
    # This is what makes projective geometry special!
    src_pts = np.float32([[0, 0], [399, 0], [399, 299], [0, 299]])
    dst_pts = np.float32([[50, 20], [350, 50], [380, 280], [20, 260]])
    H_perspective = compute_homography(src_pts, dst_pts)
    img_perspective = cv2.warpPerspective(img, H_perspective.matrix, (500, 400))
    print(f"\n5. Perspective Transformation:")
    print(H_perspective.matrix)
    decomp = H_perspective.get_decomposition()
    print(f"   Approximate scale: {decomp['scale']:.3f}")
    print(f"   Approximate rotation: {decomp['rotation_degrees']:.1f}°")
    print(f"   Perspective terms: {decomp['perspective']}")
    
    # Save all results
    cv2.imwrite('test_images/output/basics_original.png', img)
    cv2.imwrite('test_images/output/basics_translated.png', img_translated)
    cv2.imwrite('test_images/output/basics_rotated.png', img_rotated)
    cv2.imwrite('test_images/output/basics_scaled.png', img_scaled)
    cv2.imwrite('test_images/output/basics_perspective.png', img_perspective)
    
    print("\n\nSaved all transformation examples to output/ directory")
    
    # Display in windows
    print("\nDisplaying transformations (press any key to continue)...")
    
    cv2.imshow("Original", img)
    cv2.imshow("Translated", img_translated)
    cv2.imshow("Rotated", img_rotated)
    cv2.imshow("Scaled", img_scaled)
    cv2.imshow("Perspective", img_perspective)
    
    print("""
Key Insight:
- Translation, rotation, and scaling are SPECIAL CASES of projective transformations
- The perspective transformation is the GENERAL case
- All can be represented by 3x3 homography matrices
- This is the power of projective geometry!
    """)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("=" * 60)


def demo_cross_ratio():
    """
    Demonstrate the cross-ratio invariant of projective geometry.
    
    The cross ratio is the fundamental invariant preserved under
    projective transformations.
    """
    print("\n" + "=" * 60)
    print("Cross Ratio Demonstration")
    print("=" * 60)
    print("""
The cross ratio is a fundamental concept in projective geometry.
It's the ONLY measure preserved under projective transformations!

For four collinear points A, B, C, D, the cross ratio is:
    CR(A,B,C,D) = (|AC| * |BD|) / (|AD| * |BC|)

Let's verify this is preserved under a projective transformation.
    """)
    
    # Define 4 collinear points on a line
    points = np.array([
        [100, 200],  # A
        [200, 200],  # B
        [350, 200],  # C
        [450, 200]   # D
    ], dtype=np.float32)
    
    # Compute cross ratio before transformation
    def compute_cross_ratio(pts):
        """Compute cross ratio of 4 collinear points."""
        A, B, C, D = pts
        AC = np.linalg.norm(C - A)
        BD = np.linalg.norm(D - B)
        AD = np.linalg.norm(D - A)
        BC = np.linalg.norm(C - B)
        return (AC * BD) / (AD * BC)
    
    cr_before = compute_cross_ratio(points)
    print(f"\nOriginal points:")
    for i, (label, pt) in enumerate(zip(['A', 'B', 'C', 'D'], points)):
        print(f"  {label}: ({pt[0]:.1f}, {pt[1]:.1f})")
    print(f"\nCross Ratio before transformation: {cr_before:.6f}")
    
    # Apply a perspective transformation
    src = np.float32([[0, 0], [500, 0], [500, 400], [0, 400]])
    dst = np.float32([[50, 30], [450, 60], [480, 380], [20, 350]])
    H = compute_homography(src, dst)
    
    print(f"\nApplying perspective transformation:")
    print(H.matrix)
    
    # Transform the points
    transformed = H.transform_points(points)
    
    print(f"\nTransformed points:")
    for i, (label, pt) in enumerate(zip(['A', 'B', 'C', 'D'], transformed)):
        print(f"  {label}: ({pt[0]:.1f}, {pt[1]:.1f})")
    
    cr_after = compute_cross_ratio(transformed)
    print(f"\nCross Ratio after transformation: {cr_after:.6f}")
    
    print(f"\nDifference: {abs(cr_before - cr_after):.10f}")
    print(f"(Should be nearly zero - differences are due to numerical precision)")
    
    print("""
This demonstrates the key theorem of projective geometry:
The cross ratio is INVARIANT under projective transformations!

This is why projective geometry is so powerful in computer vision -
even when cameras distort lengths and angles, the cross ratio is preserved.
    """)
    print("=" * 60)


def show_about():
    """Display information about the project."""
    about = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                           ABOUT THIS PROJECT                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Course: MATH 430 - Euclidean and Non-Euclidean Geometries                   ║
║  Topic: Projective Geometry and Computer Vision                              ║
║                                                                              ║
║  ════════════════════════════════════════════════════════════                ║
║                                                                              ║
║  Mathematical Background:                                                    ║
║  ────────────────────────                                                    ║
║  This project explores how projective geometry provides the mathematical     ║
║  foundation for modern computer vision. Key concepts include:                ║
║                                                                              ║
║  • Homogeneous Coordinates: Represent points as (x, y, 1) allowing linear    ║
║    representation of projective transformations                              ║
║                                                                              ║
║  • The Projective Plane P²: The Euclidean plane augmented with points at    ║
║    infinity, where parallel lines meet                                       ║
║                                                                              ║
║  • Homography: A 3×3 matrix representing a projective transformation         ║
║    - Maps points: p' = H @ p (in homogeneous coordinates)                    ║
║    - 8 degrees of freedom (scale doesn't matter)                             ║
║    - Preserves: collinearity, incidence, cross ratio                         ║
║                                                                              ║
║  • Cross Ratio: The fundamental invariant of projective geometry             ║
║    - Preserved under ALL projective transformations                          ║
║    - Key to understanding perspective in images                              ║
║                                                                              ║
║  Applications:                                                               ║
║  ─────────────                                                               ║
║  • Perspective Correction: Transform tilted documents to frontal view        ║
║  • Image Stitching: Create panoramas by aligning overlapping images          ║
║  • Camera Calibration: Determine camera parameters                           ║
║  • Augmented Reality: Place virtual objects in real scenes                   ║
║  • 3D Reconstruction: Recover 3D structure from multiple images              ║
║                                                                              ║
║  Key References:                                                             ║
║  ───────────────                                                             ║
║  1. Birchfield, S. "An Introduction to Projective Geometry (for CV)"        ║
║  2. Hartley & Zisserman, "Multiple View Geometry in Computer Vision"         ║
║  3. Faugeras, O. "Three-Dimensional Computer Vision"                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(about)
    input("\nPress Enter to return to menu...")


def run_all_demos():
    """Run all demonstrations in sequence."""
    print("\n" + "=" * 60)
    print("Running All Demos")
    print("=" * 60)
    
    # Demo 1: Homography Basics
    print("\n>>> Demo 1: Homography Basics")
    demo_homography_basics()
    
    input("\nPress Enter to continue to Image Stitching demo...")
    
    # Demo 2: Image Stitching
    print("\n>>> Demo 2: Image Stitching")
    stitching_demo()
    
    input("\nPress Enter to continue to Perspective Correction demo...")
    
    # Demo 3: Perspective Correction
    print("\n>>> Demo 3: Perspective Correction")
    perspective_demo()
    
    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)


def main():
    """Main function with interactive menu."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Projective Geometry in Computer Vision - Demo Application',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              # Run with interactive menu
  python main.py --demo 1     # Run perspective correction demo
  python main.py --demo 2     # Run image stitching demo
  python main.py --demo 3     # Run homography basics demo
        """
    )
    parser.add_argument('--demo', type=int, choices=[1, 2, 3, 4],
                        help='Run specific demo: 1=perspective, 2=stitching, 3=basics, 4=all')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs('test_images/output', exist_ok=True)
    
    # If a specific demo was requested, run it and exit
    if args.demo:
        print_header()
        if args.demo == 1:
            perspective_demo()
        elif args.demo == 2:
            stitching_demo()
        elif args.demo == 3:
            demo_homography_basics()
        elif args.demo == 4:
            run_all_demos()
        return
    
    # Interactive menu mode
    while True:
        print_header()
        print_menu()
        
        try:
            choice = input("Enter your choice (0-5): ").strip()
            
            if choice == '0':
                print("\nThank you for exploring projective geometry!")
                print("Check out the output/ directory for saved results.")
                break
            
            elif choice == '1':
                perspective_demo()
                input("\nPress Enter to return to menu...")
            
            elif choice == '2':
                stitching_demo()
                input("\nPress Enter to return to menu...")
            
            elif choice == '3':
                demo_homography_basics()
                input("\nPress Enter to return to menu...")
            
            elif choice == '4':
                run_all_demos()
                input("\nPress Enter to return to menu...")
            
            elif choice == '5':
                show_about()
            
            else:
                print("\nInvalid choice. Please enter 0-5.")
                input("Press Enter to continue...")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
