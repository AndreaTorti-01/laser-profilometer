import cv2
import numpy as np
import glob
import os
import open3d as o3d

# --- 1. Load calibration ---
calib = np.load("calibration.npz")
cam_mtx = calib["cam_mtx"]
dist_coefs = calib["dist_coefs"]
plane_normal = calib["plane_normal"]
plane_D = calib["plane_D"]

fx, fy = cam_mtx[0,0], cam_mtx[1,1]
cx, cy = cam_mtx[0,2], cam_mtx[1,2]

# --- 2. Load images ---
folder = "log_reconstruction/final_3d_scan/"
image_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.png')])

# --- 3. Parameters ---
move_per_frame = 0.008  # meters per frame (adjust as needed)
points_3d = []

# --- Debug output folder ---
debug_folder = "debug_laser_detection"
os.makedirs(debug_folder, exist_ok=True)

# --- 4. Process each image ---
for idx, img_path in enumerate(image_files):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Blue laser threshold (tune as needed)
    lower_blue = np.array([40, 80, 80])
    upper_blue = np.array([255, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    debug_img = img.copy()
    green = np.zeros_like(debug_img)
    green[:, :, 1] = 255
    alpha = 0.4
    mask_bool = mask > 128
    debug_img[mask_bool] = cv2.addWeighted(debug_img[mask_bool], 1 - alpha, green[mask_bool], alpha, 0)
    for y in range(mask.shape[0]):
        row = mask[y]
        x_vals = np.where(row > 128)[0]
        if len(x_vals) == 0:
            continue
        x = int(np.mean(x_vals))
        # Undistort the pixel coordinate
        pts = np.array([[[x, y]]], dtype=np.float32)
        undistorted = cv2.undistortPoints(pts, cam_mtx, dist_coefs, P=cam_mtx)
        x_u, y_u = undistorted[0,0]
        # Draw detected point for debug
        cv2.circle(debug_img, (x, y), 1, (0, 0, 255), -1)
        # Back-project to 3D laser plane using undistorted coordinates
        ray = np.array([(x_u-cx)/fx, (y_u-cy)/fy, 1.0])
        t = -plane_D / (plane_normal.dot(ray))
        pt3d = t * ray
        pt3d_shifted = pt3d + idx * move_per_frame * plane_normal
        points_3d.append(pt3d_shifted)
    debug_name = os.path.splitext(os.path.basename(img_path))[0] + "_laser.png"
    cv2.imwrite(os.path.join(debug_folder, debug_name), debug_img)

# --- 5. Show point cloud using open3d ---
points_3d = np.array(points_3d)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d)
o3d.visualization.draw_geometries([pcd])
