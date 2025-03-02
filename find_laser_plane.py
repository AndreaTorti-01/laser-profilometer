# check all

import cv2
import numpy as np
import glob
import pickle
import os

def compute_laser_plane():
    # Load camera intrinsics
    with open('calibration/calibration.pkl', 'rb') as f:
        cameraMatrix, distCoeffs = pickle.load(f)

    folder = "laser_plane_extraction_images"
    images = glob.glob(os.path.join(folder, '*.png'))

    chessboardSize = (11, 6)
    all_laser_points_3d = []

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)
        if not ret:
            continue

        # Find plane equation
        objp = np.zeros((chessboardSize[0]*chessboardSize[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
        _, rvec, tvec = cv2.solvePnP(objp, corners, cameraMatrix, distCoeffs)
        R, _ = cv2.Rodrigues(rvec)
        normal_camera = R @ np.array([0, 0, 1])
        point_camera = R @ objp[0] + tvec.squeeze()
        D = -normal_camera.dot(point_camera)
        plane_eq = np.append(normal_camera, D)

        # Simple blue-laser detection (BGR threshold)
        lower_blue = np.array([100, 0, 0], dtype=np.uint8)
        upper_blue = np.array([255, 80, 80], dtype=np.uint8)
        mask = cv2.inRange(img, lower_blue, upper_blue)
        pts = np.argwhere(mask > 0)  # shape: (N, 2) [y, x]

        # For each pixel that is blue, do ray-plane intersection
        h, w = img.shape[:2]
        fx, fy = cameraMatrix[0, 0], cameraMatrix[1, 1]
        cx, cy = cameraMatrix[0, 2], cameraMatrix[1, 2]

        for y, x in pts:
            # Ray from camera center through pixel
            Xcam = np.array([(x - cx)/fx, (y - cy)/fy, 1.0])
            t = -(plane_eq[3] / (plane_eq[:3].dot(Xcam)))
            intersection_3d = t * Xcam
            all_laser_points_3d.append(intersection_3d)

    # Fit a plane to the 3D laser points
    all_laser_points_3d = np.array(all_laser_points_3d)
    centroid = np.mean(all_laser_points_3d, axis=0)
    A = all_laser_points_3d - centroid
    _, _, Vt = np.linalg.svd(A)
    normal_laser = Vt[-1, :]
    D_laser = -normal_laser.dot(centroid)
    laser_plane_eq = np.append(normal_laser, D_laser)
    print("Laser plane equation:", laser_plane_eq)

    # Overlay example visualization on the first valid image
    if len(images) > 0:
        sample_img = cv2.imread(images[0])
        # Project a few points from the fitted laser plane
        grid_x, grid_y = np.meshgrid(range(0, sample_img.shape[1], 50),
                                     range(0, sample_img.shape[0], 50))
        for gx, gy in zip(grid_x.flatten(), grid_y.flatten()):
            # Ray-plane intersection
            Xcam = np.array([(gx - cx)/fx, (gy - cy)/fy, 1.0])
            denom = laser_plane_eq[:3].dot(Xcam)
            if abs(denom) > 1e-6:
                t = -laser_plane_eq[3] / denom
                if t > 0:
                    inter_3d = t * Xcam
                    # Reproject to 2D for drawing (trivial here because it coincides)
                    cv2.circle(sample_img, (gx, gy), 3, (0,0,255), -1)

        cv2.imshow("Laser Plane Overlay", sample_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

compute_laser_plane()