"""Laser plane calibration using Plücker lines (MATLAB-style pipeline).

High‑level idea:
We see the laser as a straight 3D line laying on the checkerboard plane for each frame.
Each frame therefore gives us one 3D line (the intersection of 2 planes):
    (a) The physical checkerboard plane recovered from pose estimation.
    (b) The plane that contains the camera center and the 2D image laser stripe.
Collect many such 3D lines (each has a direction) → the laser plane's normal is orthogonal
to all those directions. We recover it with a simple SVD. We then estimate the plane offset.

Detailed steps per image:
    1. Detect checkerboard -> pose (R, t) via solvePnP.
    2. Detect & threshold blue-ish laser pixels → fit a 2D image line (normal form a u + b v + c = 0).
    3. Form two planes:
             - Checkerboard plane: normal = R * (0,0,1); passes through board origin (t).
             - Laser image plane: all 3D points whose projections lie on the 2D image line (goes through camera center).
    4. Intersect the two planes → Plücker line (direction + moment).

Accumulation:
    5. Stack all line direction vectors; plane normal is singular vector with smallest singular value.
    6. Get a representative point from each line (closest point to origin) and compute median offset.

Outputs: calibration_plucker.npz containing intrinsics, distortion, plane normal and D.

Usage:
    python laser_plane_plucker.py --calib-folder log_reconstruction/laser_plane_calibration --pattern-cols 11 --pattern-rows 6 --square-size 0.024
"""

from __future__ import annotations
import cv2, glob, os, argparse
import numpy as np
from typing import List, Tuple
from laser_calib_utils import (
    generate_object_points, calibrate_camera, detect_checkerboard,
    board_plane_from_pose, plane_from_image_line, plucker_from_two_planes,
    point_from_plucker, laser_enhance_and_threshold, fit_image_line_from_mask,
    compute_reference_origin, align_plane_offset, enforce_positive_z
)

###############################
# Tunable Parameters (edit here)
###############################
# Laser stripe detection
LASER_GAUSSIAN_BLUR = (5, 5)        # Kernel for smoothing enhanced channel
LASER_THRESHOLD = 40                # Fixed threshold after enhancement (0-255)
LASER_MORPH_KERNEL_SIZE = 3         # Closing kernel size (pixels)
MIN_STRIPE_PIXELS = 10              # Minimum pixels required to attempt line fit

# Line direction normalization policy
FORCE_DIRECTION_POS_Z = True        # Flip line directions so d[2] >= 0 for stability

# Plane offset aggregation
OFFSET_AGGREGATION = 'median'       # 'median' or 'mean'

# Plane D alignment reference selection ('mean' of board origins or 'first')
REFERENCE_ORIGIN_MODE = 'mean'      # 'mean' | 'first'

# Visualization defaults (only used when --visualize)
VIS_LINE_COLOR = (0, 0, 255)        # BGR for sample line endpoints

###############################
# End tunable parameters
###############################

# ---------------- Utility geometry -----------------

## Removed local geometry & detection helpers; now imported from laser_calib_utils

# --------------- Main processing ---------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calib-folder', default='log_reconstruction/laser_plane_calibration')
    ap.add_argument('--pattern-cols', type=int, default=11)
    ap.add_argument('--pattern-rows', type=int, default=6)
    ap.add_argument('--square-size', type=float, default=0.024)
    ap.add_argument('--visualize', action='store_true', help='Show debug windows')
    args = ap.parse_args()

    pattern_size = (args.pattern_cols, args.pattern_rows)
    # Prepare object points (Z=0 plane)
    objp = generate_object_points(pattern_size[0], pattern_size[1], args.square_size)

    images = sorted(glob.glob(os.path.join(args.calib_folder, '*.png')))
    if not images:
        raise SystemExit('No images found in folder: ' + args.calib_folder)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # --- Stage 1: intrinsic calibration ---
    calib_res = calibrate_camera(images, pattern_size, objp, criteria)
    cam_mtx = calib_res.cam_mtx
    dist_coefs = calib_res.dist_coefs
    fx, fy = cam_mtx[0,0], cam_mtx[1,1]
    cx, cy = cam_mtx[0,2], cam_mtx[1,2]

    directions = []
    moments = []
    board_origins = []            # store board origin positions t (camera frame)

    # --- Stage 2: per-image extraction of Plücker line ---
    for fn in images:
        img = cv2.imread(fn)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        success, corners2 = detect_checkerboard(gray, pattern_size, criteria)
        if not success:
            continue
        # Pose
        _, rvec, tvec = cv2.solvePnP(objp, corners2, cam_mtx, dist_coefs)
        R,_ = cv2.Rodrigues(rvec)
        board_origins.append(tvec.reshape(3))
        P_board = board_plane_from_pose(R, tvec)

        mask, enhanced = laser_enhance_and_threshold(
            img,
            blur=LASER_GAUSSIAN_BLUR,
            threshold=LASER_THRESHOLD,
            use_otsu=False,
            morph_mode='close',
            morph_kernel=LASER_MORPH_KERNEL_SIZE
        )
        try:
            a,b,c = fit_image_line_from_mask(mask, MIN_STRIPE_PIXELS)
        except ValueError:
            continue
        P_laser_img = plane_from_image_line(a,b,c, fx, fy, cx, cy)

        # Intersect planes → line (d=direction, m=moment)
        d, m = plucker_from_two_planes(P_board, P_laser_img)
        # Normalize direction sign for stability
        if FORCE_DIRECTION_POS_Z and d[2] < 0:  # enforce consistent orientation
            d = -d
            m = -m
        directions.append(d/np.linalg.norm(d))
        moments.append(m)

        if args.visualize:
            vis = img.copy()
            # draw line
            h,w = vis.shape[:2]
            # line endpoints from a u + b v + c = 0 => v = (-a u - c)/b
            xs = np.array([0, w-1])
            ys = (-a*xs - c)/b
            for x,y in zip(xs, ys):
                if 0 <= y < h:
                    cv2.circle(vis, (int(x), int(y)), 4, (0,0,255), -1)
            cv2.imshow('stripe', vis)
            cv2.waitKey(1)

    if not directions:
        raise SystemExit('No laser stripes processed.')

    Dmat = np.vstack(directions)
    # --- Stage 3: global plane normal via SVD (find vector orthogonal to all directions) ---
    # Plane normal is singular vector with smallest singular value of directions^T
    _, _, Vt = np.linalg.svd(Dmat)
    plane_normal = Vt[-1] / np.linalg.norm(Vt[-1])
    plane_normal, _ = enforce_positive_z(plane_normal, 0.0)

    # Compute representative points from each line and average their projection
    pts = []
    for d, m in zip(directions, moments):
        p0 = point_from_plucker(d, m)
        pts.append(p0)
    pts = np.vstack(pts)
    # Compute preliminary offset from line points
    proj = pts @ plane_normal
    if OFFSET_AGGREGATION == 'median':
        D_pre = -np.median(proj)
    elif OFFSET_AGGREGATION == 'mean':
        D_pre = -np.mean(proj)
    else:
        raise ValueError(f"Unknown OFFSET_AGGREGATION {OFFSET_AGGREGATION}")

    ref_origin = compute_reference_origin(board_origins, REFERENCE_ORIGIN_MODE)
    D_offset = align_plane_offset(plane_normal, D_pre, ref_origin)

    # Note: This enforces both calibrations to output D s.t. n·ref_origin + D = 0, aligning reference.

    np.savez('calibration_plucker.npz', cam_mtx=cam_mtx, dist_coefs=dist_coefs, plane_normal=plane_normal, plane_D=D_offset)
    print('Saved calibration to calibration_plucker.npz')
    print(f'Plucker plane normal: {plane_normal}, D: {D_offset}')

    if 'cv2' in globals():
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
