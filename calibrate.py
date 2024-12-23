import numpy as np
import cv2
import glob # unix-like pathname pattern expansion
import pickle # serializing and de-serializing Python object structures
import os

# ===== Settings =====
imgsfolder = "images" # folder containing the images
chessboardSize = (6, 6) # number of INNER CORNERS of the checkerboard
squareSize = 0.024 # size of the squares in meters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # termination criteria for the subpixel iterative algorithm
images = glob.glob(imgsfolder + '/*.png') # get all the images in the folder
flags = None # flags for findChessboardCorners
# flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE 

# ===== Code =====

# create calibration folder if it doesn't exist
if not os.path.exists('calibration'):
    os.makedirs('calibration')
    print("calibration Directory created!")

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
objp = objp * squareSize

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # opencv uses BGR instead of RGB

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize, flags) # could add flags from the documentation

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria) # subpixel iterative algorithm
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img', 1920, 1080)
        cv2.imshow('img', img)
        cv2.waitKey(0)
    else:
        print('Chessboard not found in ' + fname)
    
cv2.destroyAllWindows()

# Calibrate the camera
frameSize = gray.shape[::-1]
ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# Save the camera calibration result for later use (rvecs and tvecs are not saved)
pickle.dump((cameraMatrix, dist), open( "calibration/calibration.pkl", "wb" )) # saves cameraMatrix and dist as tuple

# ===== Test the calibration =====

# first image of glob
img = cv2.imread(images[0])
h,  w = img.shape[:2]
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

# Undistort
dst = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
cv2.imwrite('calibration/caliResult1.png', dst)

# Undistort with Remapping (alternative equivalent approach)
mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# crop
x, y, wc, hc = roi
dst = dst[y:y+hc, x:x+wc]
cv2.imwrite('calibration/caliResult2.png', dst)

# Reprojection Error (the closer to 0 the better)
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist) # project 3D points to the image plane
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )

# Compute the focal length in mm for both specific sensor size and 35mm standard sensor size
def compute_focal_length(cameraMatrix, sensor_width_mm, sensor_height_mm, image_width_px, image_height_px):
    focal_length_px = (cameraMatrix[0, 0] + cameraMatrix[1, 1]) / 2
    sensor_diagonal_mm = (sensor_width_mm ** 2 + sensor_height_mm ** 2) ** 0.5
    image_diagonal_px = (image_width_px ** 2 + image_height_px ** 2) ** 0.5
    return (focal_length_px * sensor_diagonal_mm) / image_diagonal_px

sensor_sizes = [(23.2, 15.4), (36.0, 24.0)]  # specific sensor size and 35mm standard sensor size
for sensor_width_mm, sensor_height_mm in sensor_sizes:
    focal_length_mm = compute_focal_length(cameraMatrix, sensor_width_mm, sensor_height_mm, w, h)
    print("Focal length for sensor {:.1f}x{:.1f} mm: {:.2f} mm".format(sensor_width_mm, sensor_height_mm, focal_length_mm))
