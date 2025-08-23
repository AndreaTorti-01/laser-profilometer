"""Laser plane calibration using brightness-weighted RANSAC on aggregated 3D laser stripe points.

High‑level idea:
Every image shows a slice of the laser line lying on the checkerboard plane. Instead of
representing lines explicitly, we collect many 3D points along those lines (one per bright
pixel) and then fit a single plane robustly with RANSAC, weighting each point by how bright
the laser looks (brighter → more trust).

Pipeline overview:
    1. Calibrate the camera intrinsics using all checkerboard detections (standard cv2.calibrateCamera).
    2. For each image:
             a. Recover checkerboard pose (R,t) → plane equation of board in camera frame.
             b. Detect laser stripe mask + intensity (blue channel enhanced).
             c. For each laser pixel: build its camera ray and intersect with checkerboard plane → 3D point.
             d. Store point with weight proportional to enhanced pixel intensity.
    3. Run a weighted RANSAC:
             - Randomly sample 3 weighted points to hypothesize a plane.
             - Count weighted inliers within distance threshold.
             - Keep the best hypothesis.
    4. Refine plane on inliers via weighted PCA (SVD of centered, weight-scaled points).
    5. Normalize & orient plane normal (positive Z) for consistency.

Outputs:
    calibration_ransac.npz: {cam_mtx, dist_coefs, plane_normal, plane_D}

Usage:
    python laser_plane_ransac.py --calib-folder log_reconstruction/laser_plane_calibration --pattern-cols 11 --pattern-rows 6 --square-size 0.024
"""

from __future__ import annotations
import cv2, glob, os, argparse, random
import numpy as np
from typing import Tuple, List
from laser_calib_utils import (
    generate_object_points, calibrate_camera, detect_checkerboard,
    board_normal_d_from_pose, intersect_ray_plane, laser_enhance_and_threshold,
    fit_image_line_from_mask, compute_reference_origin, align_plane_offset,
    enforce_positive_z
)

###############################
# Tunable Parameters (edit here)
###############################
# Laser detection
LASER_GAUSSIAN_BLUR = (5,5)
LASER_USE_OTSU = True            # If False, use fixed threshold
LASER_FIXED_THRESHOLD = 40       # Used only if LASER_USE_OTSU=False
LASER_MORPH_OPEN = 3             # Kernel size for morphological open (0 to disable)
MAX_POINTS_PER_IMAGE_DEFAULT = 1500  # Fallback if CLI not provided

# RANSAC
RANSAC_ITERS_DEFAULT = 2500
RANSAC_DIST_THRESH_DEFAULT = 0.002   # meters
RANSAC_POWER_WEIGHT = 1.2            # Exponent applied to normalized weights

# Orientation
FORCE_NORMAL_POS_Z = True
REFERENCE_ORIGIN_MODE = 'mean'   # 'mean' or 'first' to match plucker script alignment

###############################
# End tunable parameters
###############################

# ---------------- Geometry helpers -----------------

## Removed local detection + geometry helpers; using laser_calib_utils

# ---------------- RANSAC plane fitting -----------------

def plane_from_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray):
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return None, None
    n /= norm
    d = -n @ p1
    return n, d

def weighted_ransac(points: np.ndarray, weights: np.ndarray, iters=2000, dist_thresh=0.002, rng=None):
    if rng is None:
        rng = random.Random(0)
    N = len(points)
    best_inliers = None
    best_score = -1.0
    w_cum = np.cumsum(weights)
    total_w = w_cum[-1]

    def sample_point():
        r = rng.random() * total_w
        idx = np.searchsorted(w_cum, r)
        return idx

    for _ in range(iters):
        idxs = set()
        attempts = 0
        while len(idxs) < 3 and attempts < 10:
            idxs.add(sample_point())
            attempts += 1
        if len(idxs) < 3:
            continue
        i1,i2,i3 = list(idxs)
        n,d = plane_from_points(points[i1], points[i2], points[i3])
        if n is None:
            continue
        # Distance of all points
        dist = np.abs(points @ n + d)
        inliers = dist < dist_thresh
        score = weights[inliers].sum()
        if score > best_score:
            best_score = score
            best_inliers = inliers
            best_plane = (n.copy(), d)

    if best_inliers is None:
        raise RuntimeError('RANSAC failed.')

    # Weighted least squares refit on inliers
    P = points[best_inliers]
    w = weights[best_inliers]
    w_norm = w / (w.sum() + 1e-12)
    cent = (P * w_norm[:,None]).sum(axis=0)
    Q = (P - cent) * np.sqrt(w_norm[:,None])
    _,_,Vt = np.linalg.svd(Q, full_matrices=False)
    n_refined = Vt[-1]
    n_refined /= np.linalg.norm(n_refined)
    d_refined = -n_refined @ cent

    return n_refined, d_refined, best_inliers, best_score

# ---------------- Main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calib-folder', default='log_reconstruction/laser_plane_calibration')
    ap.add_argument('--pattern-cols', type=int, default=11)
    ap.add_argument('--pattern-rows', type=int, default=6)
    ap.add_argument('--square-size', type=float, default=0.024)
    ap.add_argument('--max-points-per_image', type=int, default=MAX_POINTS_PER_IMAGE_DEFAULT)
    ap.add_argument('--ransac-iters', type=int, default=RANSAC_ITERS_DEFAULT)
    ap.add_argument('--dist-thresh', type=float, default=RANSAC_DIST_THRESH_DEFAULT)
    ap.add_argument('--visualize', action='store_true')
    args = ap.parse_args()

    pattern_size = (args.pattern_cols, args.pattern_rows)
    # Prepare object points (Z=0 plane)
    objp = generate_object_points(pattern_size[0], pattern_size[1], args.square_size)

    images = sorted(glob.glob(os.path.join(args.calib_folder, '*.png')))
    if not images:
        raise SystemExit('No images found')

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # --- Stage 1: intrinsic calibration ---
    calib_res = calibrate_camera(images, pattern_size, objp, criteria)
    cam_mtx = calib_res.cam_mtx
    dist_coefs = calib_res.dist_coefs
    fx, fy = cam_mtx[0,0], cam_mtx[1,1]
    cx, cy = cam_mtx[0,2], cam_mtx[1,2]

    all_points = []
    all_weights = []
    board_origins = []  # store checkerboard origins (t) for reference alignment

    # --- Stage 2: Collect weighted 3D laser points ---
    for fn in images:
        img = cv2.imread(fn)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = detect_checkerboard(gray, pattern_size, criteria)
        if not ok:
            continue
        _, rvec, tvec = cv2.solvePnP(objp, corners, cam_mtx, dist_coefs)
        R,_ = cv2.Rodrigues(rvec)
        board_origins.append(tvec.reshape(3))
        n_board, d_board = board_normal_d_from_pose(R, tvec)

        mask, intensity = laser_enhance_and_threshold(
            img,
            blur=LASER_GAUSSIAN_BLUR,
            threshold=LASER_FIXED_THRESHOLD,
            use_otsu=LASER_USE_OTSU,
            morph_mode='open',
            morph_kernel=LASER_MORPH_OPEN
        )
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        # Randomly subsample if too many points
        if len(xs) > args.max_points_per_image:
            idx = np.random.choice(len(xs), args.max_points_per_image, replace=False)
            xs = xs[idx]; ys = ys[idx]
        # Compute rays and intersections
        for x,y in zip(xs, ys):
            # Normalized camera ray
            ray = np.array([(x - cx)/fx, (y - cy)/fy, 1.0])
            P = intersect_ray_plane(ray, n_board, d_board)
            if P is None:
                continue
            w = float(intensity[y, x])
            if w <= 0:
                continue
            all_points.append(P)
            all_weights.append(w)

    if not all_points:
        raise SystemExit('No laser points collected')

    P = np.vstack(all_points)
    W = np.array(all_weights, dtype=float)
    # Normalize weights to prevent overflow
    W /= W.max()
    # Enhance contrast between bright and dim using exponent
    W = W**RANSAC_POWER_WEIGHT

    # --- Stage 3: Weighted RANSAC plane fit ---
    n, d, inliers, score = weighted_ransac(P, W, iters=args.ransac_iters, dist_thresh=args.dist_thresh)

    # Ensure normal orientation (z positive arbitrary)
    n, d = enforce_positive_z(n, d)

    # Align D to chosen reference origin (same logic as Plücker script)
    ref_origin = compute_reference_origin(board_origins, REFERENCE_ORIGIN_MODE)
    d = align_plane_offset(n, d, ref_origin)
    # --- Stage 4: Persist results ---
    np.savez('calibration_ransac.npz', cam_mtx=cam_mtx, dist_coefs=dist_coefs, plane_normal=n, plane_D=d)
    print('Saved calibration to calibration_ransac.npz')
    print(f'RANSAC plane normal: {n}, D: {d}, inliers: {inliers.sum()} / {len(P)}')

    if args.visualize:
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(P[~inliers,0], P[~inliers,1], P[~inliers,2], s=1, c='lightgray')
            ax.scatter(P[inliers,0], P[inliers,1], P[inliers,2], s=2, c='red')
            # Draw plane patch
            pts_in = P[inliers]
            xlim = (pts_in[:,0].min(), pts_in[:,0].max())
            ylim = (pts_in[:,1].min(), pts_in[:,1].max())
            xx, yy = np.meshgrid(np.linspace(*xlim, 10), np.linspace(*ylim, 10))
            zz = (-n[0]*xx - n[1]*yy - d)/n[2]
            ax.plot_surface(xx, yy, zz, alpha=0.3, color='cyan')
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.set_title('RANSAC Weighted Plane Fit')
            plt.show()
        except Exception as e:
            print('Visualization skipped:', e)

if __name__ == '__main__':
    main()
