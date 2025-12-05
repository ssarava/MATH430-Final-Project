# Projective Geometry and Computer Vision: From Theory to Application

## Final Project for MATH 430: Euclidean and Non-Euclidean Geometries

---

## Table of Contents

1. [Paper Structure](#paper-structure)
2. [Application Implementation](#application-implementation)
3. [Presentation Structure](#presentation-structure)
4. [Key References](#key-references)
5. [Project Files](#project-files)

---

## Paper Structure

### Suggested Title
**"Projective Geometry in Computer Vision: The Mathematics Behind Image Transformations"**

### Abstract (150-200 words)
The abstract should:
- Introduce projective geometry as a mathematical framework extending Euclidean geometry
- State the paper's purpose: connecting projective geometry to computer vision algorithms
- Briefly mention the key applications covered (homography, camera calibration, image stitching)
- Note the accompanying Python demonstration

**Sample Abstract:**
> Projective geometry provides a powerful mathematical framework for understanding the imaging process of cameras and forms the theoretical foundation of modern computer vision. Unlike Euclidean geometry, which preserves lengths and angles, projective geometry preserves incidence and a measure called the cross ratio—properties essential for modeling how three-dimensional scenes are projected onto two-dimensional images. This paper explores the fundamental concepts of projective geometry, including homogeneous coordinates, the projective plane P², and projective transformations, drawing primarily from Birchfield's accessible introduction to the subject. We then demonstrate how these mathematical concepts directly enable key computer vision algorithms: homography estimation for image alignment, the fundamental matrix for stereo vision, and camera calibration for 3D reconstruction. An accompanying Python implementation provides an interactive demonstration of projective transformations, allowing users to perform perspective correction and image stitching—concrete applications of the abstract mathematics discussed.

---

### Introduction (1-1.5 pages)

#### Opening Hook
Begin with a relatable scenario: "Anyone who has taken a photograph knows that parallel lines in the real world—like railroad tracks or the edges of a building—appear to converge in the image. This seemingly simple observation reveals a fundamental truth: the camera does not preserve Euclidean geometry."

#### Motivation
Explain why projective geometry matters:
- Euclidean geometry fails to model the imaging process (lengths, angles, parallelism not preserved)
- Projective geometry provides the natural mathematical language for cameras
- This connection has practical importance: computer vision is a multi-billion dollar industry

#### Scope Statement
Clearly state what the paper will cover:
1. Mathematical foundations from Birchfield's article
2. Key concepts: homogeneous coordinates, projective transformations, the fundamental matrix
3. Applications: homography, image stitching, perspective correction
4. Python demonstration

#### Brief Background
Mention the hierarchy of geometries (Euclidean ⊂ Similarity ⊂ Affine ⊂ Projective) and what each preserves.

---

### Body Section 1: Mathematical Foundations (2-2.5 pages)

#### 1.1 Homogeneous Coordinates
- Definition: A point (x, y) in Euclidean plane becomes (x, y, 1) in projective plane
- Equivalence: (X, Y, W) = λ(X, Y, W) for any λ ≠ 0
- Points and lines have the same representation in P²
- Incidence: point p lies on line u iff p^T u = 0

**Key Figure:** Include the figure showing the four geometries and their invariants (from Birchfield, Figure 1)

#### 1.2 The Projective Plane P²
Four models (explain at least two in detail):
1. **Homogeneous coordinates** (primary model)
2. **Ray space** (visualizing a point as a line through the origin in R³)
3. The unit sphere model
4. Augmented affine plane

**Definition:** P² is the affine plane augmented by an ideal line and ideal points (points at infinity), where ideal entities are not distinguishable from regular ones.

#### 1.3 Points at Infinity and Ideal Lines
- Ideal points: (X, Y, 0) — associated with each direction
- Ideal line: (0, 0, 1) — where parallel lines meet
- Why this elegantly handles parallel lines intersecting

#### 1.4 Projective Transformations (Collineations)
- General form: p' = T p where T is a 3×3 matrix
- 8 degrees of freedom (9 elements, scale doesn't matter)
- Preserves: collinearity, incidence, cross ratio

**Include:** The matrix forms for different transformations (Euclidean, Similarity, Affine, Projective)

---

### Body Section 2: From Geometry to Computer Vision (2-2.5 pages)

#### 2.1 The Camera as a Projective Device
- Image formation: projection from P³ to P²
- The projection equations in homogeneous coordinates (linear!)
- Camera matrix P = K[R|t] decomposition

**Key Equation:**
```
p' = T_perspective · p
where T = [f 0 0 0; 0 f 0 0; 0 0 1 0]
```

#### 2.2 The Homography Matrix
- Definition: A projective transformation between two views of a planar surface
- 3×3 matrix with 8 degrees of freedom
- When homography applies:
  - Planar scenes
  - Pure camera rotation (no translation)
  
**Applications:**
- Perspective correction
- Image mosaicing/stitching
- Augmented reality (virtual billboards)

#### 2.3 The Fundamental Matrix
- Encodes the epipolar geometry between two uncalibrated cameras
- Constraint: m₂^T F m₁ = 0 for corresponding points
- 7 degrees of freedom
- Derivation sketch (from Birchfield Section 4.2)

#### 2.4 The Essential Matrix
- For calibrated cameras: E = [t]ₓR
- Contains rotation and translation between cameras
- 5 degrees of freedom

---

### Body Section 3: Application — Image Homography (1-1.5 pages)

#### 3.1 Computing a Homography
- Need 4 point correspondences (minimum)
- Direct Linear Transform (DLT) algorithm overview
- RANSAC for robust estimation with outliers

#### 3.2 Image Stitching Pipeline
1. Feature detection (SIFT, ORB)
2. Feature matching between images
3. Homography estimation via RANSAC
4. Warping with cv2.warpPerspective
5. Blending the warped images

#### 3.3 Perspective Correction
- Selecting 4 corners of a document/object
- Computing homography to rectangle
- Obtaining a "frontal" view

**Reference the Python demo:** "The accompanying implementation demonstrates these concepts interactively."

---

### Conclusion (0.5-1 page)

#### Summary
- Projective geometry extends Euclidean geometry by adding points at infinity
- This mathematical framework naturally models camera imaging
- The homography and fundamental matrices are practical computational tools

#### Broader Impact
- Computer vision applications: autonomous vehicles, medical imaging, augmented reality
- The mathematical elegance of projective geometry enables these technologies

#### Personal Reflection (Optional)
What you found most interesting or surprising about the connection between pure geometry and practical applications.

---

## Application Implementation

### Overview
The Python application provides an interactive demonstration of projective geometry concepts through two main features:
1. **Perspective Correction Tool:** Select four corners of a document/planar object and warp it to a frontal view
2. **Image Stitching Demo:** Combine overlapping images using homography estimation

### Directory Structure
```
projective_geometry_cv_project/
├── PROJECT_OVERVIEW.md      # This file
├── src/
│   ├── __init__.py
│   ├── homography.py        # Core homography computation functions
│   ├── perspective_correction.py  # Interactive perspective correction tool
│   ├── image_stitching.py   # Image stitching implementation
│   └── visualization.py     # Visualization utilities
├── data/
│   └── (sample images)
├── output/
│   └── (generated results)
├── requirements.txt
└── main.py                  # Main entry point with menu
```

### Dependencies
```
numpy>=1.20.0
opencv-python>=4.5.0
matplotlib>=3.4.0
```

### Running the Application
```bash
# Install dependencies
pip install -r requirements.txt

# Run the main application
python main.py

# Or run individual modules
python -m src.perspective_correction  # Interactive perspective correction
python -m src.image_stitching         # Image stitching demo
```

### Code Documentation
All code includes:
- Module-level docstrings explaining purpose
- Class docstrings with attributes
- Function docstrings with parameters, returns, and examples
- Inline comments explaining non-obvious logic
- Type hints for better code clarity

---

## Presentation Structure

### Slide 1: Title Slide
**Title:** Projective Geometry in Computer Vision: The Mathematics Behind Image Transformations

**Content:**
- Your name
- Course: MATH 430 — Euclidean and Non-Euclidean Geometries
- Date

### Slide 2: Motivation — Why Projective Geometry?
**Content:**
- Show an image with converging parallel lines (e.g., railroad tracks)
- Question: "Why don't cameras preserve parallel lines?"
- Answer: Because cameras perform projective transformations, not Euclidean ones

**Explain in more detail:** The fundamental observation that cameras don't preserve Euclidean properties

### Slide 3: The Hierarchy of Geometries
**Content:**
- Diagram showing: Euclidean ⊂ Similarity ⊂ Affine ⊂ Projective
- Table of what each preserves (adapted from Birchfield Figure 1)

**Explain in less detail:** Just establish the hierarchy; details come later

**Suggested Figure:** Recreation of Birchfield's Figure 1

### Slide 4: Homogeneous Coordinates — The Key Idea
**Content:**
- (x, y) → (x, y, 1) in homogeneous form
- Equivalence: (X, Y, W) ≡ (λX, λY, λW)
- Points and lines have the same representation!

**Explain in more detail:** This is a central concept; spend time here

### Slide 5: Points at Infinity
**Content:**
- Points with W = 0 are "points at infinity"
- Each direction has its own point at infinity
- This is where parallel lines meet!

**Suggested Figure:** Visual showing parallel lines meeting at a point at infinity

### Slide 6: The Projective Plane P²
**Content:**
- Ray space visualization (point = line through origin)
- Unit sphere model
- Definition: Affine plane + ideal line + ideal points

**Explain in less detail:** Give intuition, don't get lost in details

### Slide 7: Projective Transformations
**Content:**
- p' = T · p where T is 3×3
- 8 degrees of freedom
- Preserves: incidence, collinearity, cross ratio

**Explain in more detail:** This directly leads to applications

### Slide 8: The Camera as a Projective Device
**Content:**
- Image formation equation in homogeneous coordinates
- Show how nonlinear perspective equations become linear
- Camera matrix P = K[R|t]

**Suggested Figure:** Diagram of pinhole camera model

### Slide 9: The Homography Matrix
**Content:**
- Transformation between two views of a plane
- When it applies (planar scene, pure rotation)
- Key applications: perspective correction, stitching, AR

**Explain in more detail:** This is what the demo shows

### Slide 10: Computing Homography
**Content:**
- Need 4 point correspondences
- Brief mention of DLT algorithm
- RANSAC for robustness

**Explain in less detail:** Focus on the concept, not algorithm details

### Slide 11: The Fundamental Matrix (Brief)
**Content:**
- Encodes epipolar geometry
- m₂ᵀ F m₁ = 0
- Enables stereo vision and 3D reconstruction

**Explain in less detail:** Mention it exists and why it matters, but don't go deep

### Slide 12: Live Demo — Perspective Correction
**Content:**
- Run the interactive perspective correction tool
- Show selecting corners of a book/document
- Show the corrected "frontal" view

**LIVE DEMO:** Run `python -m src.perspective_correction`

### Slide 13: Live Demo — Image Stitching
**Content:**
- Run the image stitching demonstration
- Show feature matching visualization
- Show the final panorama

**LIVE DEMO:** Run `python -m src.image_stitching`

### Slide 14: Summary and Connections
**Content:**
- Projective geometry provides the mathematical foundation for computer vision
- Key concepts: homogeneous coordinates, points at infinity, projective transformations
- Practical tools: homography matrix, fundamental matrix

### Slide 15: References and Questions
**Content:**
- Key references (Birchfield, Hartley & Zisserman)
- "Questions?"

---

### Presentation Tips

**Concepts to Explain in More Detail:**
1. Why Euclidean geometry fails for cameras (motivation)
2. Homogeneous coordinates (central to everything)
3. How homography enables the demo applications

**Concepts to Explain in Less Detail:**
1. Mathematical derivations (show the results, not all steps)
2. Algorithm specifics (DLT, RANSAC) — just convey the idea
3. The fundamental matrix — acknowledge it, but don't derive it

**Demo Preparation:**
- Test the demos before the presentation
- Have backup images ready in case of issues
- Consider recording a video backup

---

## Key References

### Primary Reference
1. **Birchfield, S.** (1998). "An Introduction to Projective Geometry (for computer vision)." Stanford University. Available at: http://robotics.stanford.edu/~birch/projective/
   - *This is the foundational article recommended by your professor. It provides an accessible introduction to projective geometry with computer vision applications.*

### Secondary References

2. **Hartley, R. & Zisserman, A.** (2004). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press. ISBN: 978-0521540513
   - *The definitive textbook on the mathematics of multiple view geometry. Chapters 2-4 cover projective geometry and transformations.*

3. **Faugeras, O.** (1993). *Three-Dimensional Computer Vision.* MIT Press.
   - *Referenced by Birchfield; provides rigorous treatment of camera geometry.*

4. **OpenCV Documentation.** "Basic concepts of the homography explained with code." Available at: https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
   - *Practical implementation guide with Python examples.*

5. **Mundy, J.L. & Zisserman, A.** (1992). *Geometric Invariance in Computer Vision.* MIT Press.
   - *Classic reference on geometric invariants and their applications.*

### Additional Resources

6. **Szeliski, R.** (2010). *Computer Vision: Algorithms and Applications.* Springer.
   - *Chapter 2 covers image formation; Chapter 9 covers image stitching.*

7. **Prince, S.J.D.** (2012). *Computer Vision: Models, Learning, and Inference.* Cambridge University Press.
   - *Modern treatment with probabilistic perspective.*

---

## Project Files

| File | Description |
|------|-------------|
| `PROJECT_OVERVIEW.md` | This document — project organization and structure |
| `main.py` | Main entry point with interactive menu |
| `src/homography.py` | Core homography computation functions |
| `src/perspective_correction.py` | Interactive perspective correction tool |
| `src/image_stitching.py` | Image stitching implementation |
| `src/visualization.py` | Visualization utilities |
| `requirements.txt` | Python dependencies |

---

## Notes for the Author

### Before Writing the Paper
1. Read Birchfield's article carefully — it's well-written and accessible
2. Focus on understanding before explaining
3. Work through examples by hand

### Writing Tips
- Use your own words; quote sparingly with citation
- Include figures — they help enormously
- Define terms before using them
- Give concrete examples for abstract concepts

### For the Presentation
- Practice the demos multiple times
- Know your material well enough to answer questions
- It's okay to say "I don't know, but here's how I'd find out"

---

*Last updated: November 2025*
