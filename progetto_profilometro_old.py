import cv2
import numpy as np

image = cv2.imread('image.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# show loaded image
cv2.imshow('image', gray)
cv2.waitKey(0)

def find_camera_intrinsics(gray):
    # Convert the grayscale image to binary (black and white)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Define the criteria for terminating the corner sub-pixel refinement algorithm
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
    
    # Prepare object points for a 7x7 checkerboard with 2.5cm squares
    objp = np.zeros((7 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2) * 2.5  # Multiply by square size
    
    # Find the chessboard corners in the binary image
    ret, corners = cv2.findChessboardCorners(binary, (7, 7), None)
    if ret:
        # Refine corner locations to sub-pixel accuracy
        corners2 = cv2.cornerSubPix(binary, corners, (11, 11), (-1, -1), criteria)
        # Prepare data for camera calibration
        objpoints = [objp]
        imgpoints = [corners2]
        # Calibrate the camera to find intrinsic parameters
        _, mtx, _, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        return mtx
    else:
        # Return None if corners are not found
        return None

print("begin function")
# Get the camera intrinsic matrix
mtx = find_camera_intrinsics(gray)
print(mtx)


def find_chessboard_plane_equation(image, mtx):
    # Convert the input image to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Define criteria for corner sub-pixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points for a 7x7 checkerboard with 2.5cm squares
    objp = np.zeros((7 * 7, 3), np.float32)
    objp[:, :2] = np.indices((7, 7)).T.reshape(-1, 2) * 2.5  # Multiply by square size
    
    # Find the chessboard corners in the grayscale image
    ret, corners = cv2.findChessboardCorners(gray_img, (7, 7), None)
    if ret:
        # Refine corner locations to sub-pixel accuracy
        corners_subpix = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
        # Solve the PnP problem to get rotation and translation vectors
        _, rvecs, tvecs = cv2.solvePnP(objp, corners_subpix, mtx, None)
        # Convert rotation vectors to rotation matrices
        R, _ = cv2.Rodrigues(rvecs)
        # Define the normal vector of the chessboard plane in the world coordinate system
        normal_world = np.array([0, 0, 1])
        # Transform the normal vector to the camera coordinate system
        normal_camera = R @ normal_world
        # Choose a point on the plane (first object point transformed to camera coordinates)
        point_camera = R @ objp[0] + tvecs.flatten()
        # Calculate the plane equation coefficient 'd' using the point-normal form
        d = -normal_camera @ point_camera
        # Combine normal vector and 'd' to form the plane equation coefficients
        plane_eq = np.append(normal_camera, d)
        return plane_eq
    else:
        # Return None if corners are not found
        return None

# Find the plane equation of the chessboard
plane_eq = find_chessboard_plane_equation(image, mtx)
if plane_eq is not None:
    print("Plane equation coefficients (nx, ny, nz, d):", plane_eq)
else:
    print("Chessboard pattern not found.")