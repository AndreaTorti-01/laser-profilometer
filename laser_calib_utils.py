"""Shared utilities for laser plane calibration scripts.

This module centralizes common geometry, calibration and image-processing
routines used by both the Plücker-line and RANSAC-based laser plane
calibration approaches.

Key functionalities:
  - Chessboard object point generation & detection
  - Camera intrinsic calibration wrapper
  - Board plane recovery from pose
  - Image laser stripe enhancement + thresholding (fixed or Otsu)
  - 2D image line fitting & conversion to 3D plane through camera center
  - Plücker line utilities (intersection of two planes, point on line)
  - Ray / plane intersection
  - Reference origin computation & plane offset alignment

All functions include concise docstrings for quick reference.
"""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from typing import Iterable, List, Sequence, Tuple, Optional

# ---------------------- Data containers ----------------------

@dataclass
class CameraCalibrationResult:
    cam_mtx: np.ndarray
    dist_coefs: np.ndarray
    rvecs: Sequence[np.ndarray]
    tvecs: Sequence[np.ndarray]

# ---------------------- Chessboard helpers ----------------------

def generate_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    """Create (N,3) array of checkerboard corner coordinates on Z=0 plane.

    cols, rows: number of inner corners along X and Y directions.
    square_size: edge length in meters.
    """
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return objp * square_size

def detect_checkerboard(gray: np.ndarray, pattern_size: Tuple[int,int], criteria) -> Tuple[bool, Optional[np.ndarray]]:
    """Locate checkerboard corners and refine to subpixel accuracy.

    Returns (success, corners) where corners is an (N,1,2) float32 array or None.
    """
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not ret:
        return False, None
    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    return True, corners2

def calibrate_camera(images: Iterable[str], pattern_size: Tuple[int,int], objp: np.ndarray, criteria) -> CameraCalibrationResult:
    """Perform intrinsic calibration over a list of image paths.

    Only images where the checkerboard is found contribute.
    Returns CameraCalibrationResult. Raises SystemExit if none found.
    """
    objpoints, imgpoints = [], []
    gray_shape = None
    for fn in images:
        img = cv2.imread(fn)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = detect_checkerboard(gray, pattern_size, criteria)
        if not ok:
            continue
        objpoints.append(objp)
        imgpoints.append(corners)
        gray_shape = gray.shape[::-1]
    if not objpoints:
        raise SystemExit('No checkerboards detected for calibration')
    ret, cam_mtx, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)
    return CameraCalibrationResult(cam_mtx=cam_mtx, dist_coefs=dist_coefs, rvecs=rvecs, tvecs=tvecs)

# ---------------------- Plane and pose utilities ----------------------

def board_plane_from_pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return plane [nx, ny, nz, D] for checkerboard (Z=0 in board frame).
    R: 3x3 rotation (object->camera), t: 3x1 translation.
    """
    n = R @ np.array([0.0, 0.0, 1.0])
    p0 = t.reshape(3)
    D = -float(n @ p0)
    return np.array([*n, D], dtype=float)

def board_normal_d_from_pose(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return (normal, D) for board plane."""
    n = R @ np.array([0.,0.,1.])
    p0 = t.reshape(3)
    D = -float(n @ p0)
    return n, D

def plane_from_image_line(a: float, b: float, c: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Convert normalized 2D image line a u + b v + c = 0 to 3D plane through camera center.

    Returns plane [nx, ny, nz, D] with D=0 (passes through origin). The normal is normalized.
    """
    A = a * fx
    B = b * fy
    C = a * cx + b * cy + c
    n = np.array([A, B, C], dtype=float)
    norm = np.linalg.norm(n)
    if norm == 0:
        raise ValueError('Degenerate line -> zero normal')
    n /= norm
    return np.array([n[0], n[1], n[2], 0.0], dtype=float)

# ---------------------- Plücker line utilities ----------------------

def plucker_from_two_planes(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (direction, moment) of line = intersection of planes P and Q.

    Planes P,Q: [nx, ny, nz, D] with equation n·X + D = 0.
    direction = nP x nQ, moment = Dp * nQ - Dq * nP.
    """
    n1, d1 = P[:3], P[3]
    n2, d2 = Q[:3], Q[3]
    d = np.cross(n1, n2)
    if np.linalg.norm(d) == 0:
        raise ValueError('Parallel planes -> no line')
    m = d1 * n2 - d2 * n1
    return d, m

def point_from_plucker(direction: np.ndarray, moment: np.ndarray) -> np.ndarray:
    """Closest point on line to origin: (d x m) / ||d||^2."""
    denom = float(direction @ direction)
    return np.cross(direction, moment) / denom

# ---------------------- Ray / plane intersection ----------------------

def intersect_ray_plane(ray: np.ndarray, n: np.ndarray, D: float) -> Optional[np.ndarray]:
    """Intersect ray (origin=0, direction=ray) with plane n·X + D = 0.
    Returns 3D point or None if parallel or behind origin.
    """
    denom = float(n @ ray)
    if abs(denom) < 1e-9:
        return None
    s = -D / denom
    if s <= 0:
        return None
    return s * ray

# ---------------------- Stripe detection & line fitting ----------------------

def enhance_blue_difference(bgr: np.ndarray) -> np.ndarray:
    """Return enhanced channel highlighting blue over red/green."""
    b,g,r = cv2.split(bgr)
    return cv2.subtract(b, cv2.max(r,g))

def laser_enhance_and_threshold(img: np.ndarray, blur: Tuple[int,int]=(5,5), threshold: Optional[int]=40,
                                use_otsu: bool=False, morph_mode: Optional[str]='close', morph_kernel: int=3) -> Tuple[np.ndarray, np.ndarray]:
    """Enhance blue channel difference then threshold to produce a mask.

    Parameters:
      blur: Gaussian kernel size.
      threshold: fixed threshold if use_otsu is False; ignored otherwise.
      use_otsu: enable Otsu adaptive threshold.
      morph_mode: 'close', 'open', or None.
      morph_kernel: kernel size in pixels for morphology.

    Returns (mask, enhanced_image).
    """
    enhanced = enhance_blue_difference(img)
    enh_blur = cv2.GaussianBlur(enhanced, blur, 0)
    if use_otsu:
        _, mask = cv2.threshold(enh_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(enh_blur, int(threshold), 255, cv2.THRESH_BINARY)
    if morph_mode and morph_kernel > 0:
        k = np.ones((morph_kernel, morph_kernel), np.uint8)
        if morph_mode == 'close':
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        elif morph_mode == 'open':
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask, enh_blur

def fit_image_line_from_mask(mask: np.ndarray, min_pixels: int=10) -> Tuple[float,float,float]:
    """Fit normalized line a u + b v + c = 0 using cv2.fitLine on mask pixels.
    Returns (a,b,c) with sqrt(a^2+b^2)=1.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < min_pixels:
        raise ValueError('Not enough pixels for line fit')
    pts = np.vstack([xs, ys]).T.astype(np.float32)
    vx, vy, x0, y0 = map(float, cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()) # robust to gaps
    a = vy; b = -vx; c = -(a*x0 + b*y0)
    norm = (a*a + b*b)**0.5
    a /= norm; b /= norm; c /= norm
    return a,b,c

# ---------------------- Reference origin alignment ----------------------

def compute_reference_origin(origins: Sequence[np.ndarray], mode: str='mean') -> np.ndarray:
    """Select a common 3D reference origin from a list of board origins.
    mode: 'mean' or 'first'. Returns (3,) array (zeros if empty).
    """
    if not origins:
        return np.zeros(3)
    if mode == 'mean':
        return np.mean(origins, axis=0)
    if mode == 'first':
        return origins[0]
    raise ValueError(f'Unknown reference origin mode {mode}')

def align_plane_offset(normal: np.ndarray, D_initial: float, ref_origin: np.ndarray) -> float:
    """Shift plane offset so that plane passes through ref_origin.
    Given plane n·X + D_initial = 0, new D satisfies n·ref + D = 0 → D = -n·ref.
    Ignores the original D magnitude (explicit alignment overrides it).
    """
    return -float(normal @ ref_origin)

# ---------------------- Orientation helpers ----------------------

def enforce_positive_z(normal: np.ndarray, D: float) -> Tuple[np.ndarray, float]:
    """Flip normal & D so that normal[2] >= 0 for consistency."""
    if normal[2] < 0:
        return -normal, -D
    return normal, D
