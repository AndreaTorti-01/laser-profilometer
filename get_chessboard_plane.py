import cv2
import numpy as np
import pickle

def get_chessboard_plane_equation(image, intrinsic_matrix, distortion_coefficients, pattern_size):
    # Prepare object points in the chessboard coordinate space
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[1], 0:pattern_size[0]].T.reshape(-1, 2)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners in the image
    ret, img_points = cv2.findChessboardCorners(gray, pattern_size, None)
    if not ret:
        return None

    # Refine the corner positions
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img_points = cv2.cornerSubPix(gray, img_points, (11, 11), (-1, -1), criteria)

    # Solve for rotation and translation vectors
    _, rvec, tvec = cv2.solvePnP(objp, img_points, intrinsic_matrix, distortion_coefficients)

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Normal vector in the world coordinate system
    normal_world = np.array([0, 0, 1])

    # Transform the normal vector to the camera coordinate system
    normal_camera = R @ normal_world

    # A point on the plane in the camera coordinate system
    point_camera = R @ objp[0] + tvec.squeeze()

    # Compute the plane equation coefficients
    D = -normal_camera.dot(point_camera)
    plane_equation = np.append(normal_camera, D)

    return plane_equation, rvec, tvec

# Load the intrinsic matrix and distortion coefficients
with open('calibration/calibration.pkl', 'rb') as f:
    intrinsic_matrix, distortion_coefficients = pickle.load(f)

# Load the image
image = cv2.imread("images/img0.png")

# Define the chessboard pattern size
pattern_size = (6, 6)

# Get the plane equation and pose
plane_equation, rvec, tvec = get_chessboard_plane_equation(
    image,
    intrinsic_matrix,
    distortion_coefficients,
    pattern_size
)

# Print the plane equation coefficients
print("Plane equation:", plane_equation)

# Visualize the found plane on top of the original image
if plane_equation is not None:
    # Draw the detected chessboard corners
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        image2 = image.copy()
        cv2.drawChessboardCorners(image2, pattern_size, corners, ret)

    # Create a grid of points on the plane
    x_vals = np.linspace(0, pattern_size[1]-1, pattern_size[1])
    y_vals = np.linspace(0, pattern_size[0]-1, pattern_size[0])
    xx, yy = np.meshgrid(x_vals, y_vals)
    zz = np.zeros_like(xx)
    plane_points = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T

    # Project the plane points onto the image
    plane_points_2d, _ = cv2.projectPoints(
        plane_points,
        rvec,
        tvec,
        intrinsic_matrix,
        distortion_coefficients
    )

    # Draw the projected plane points on the image
    for point in plane_points_2d:
        x, y = int(point[0][0]), int(point[0][1])
        cv2.circle(image, (x, y), 8, (0, 255, 0), -1)

    # Display image2
    cv2.namedWindow('Chessboard Corners', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Chessboard Corners', 1920, 1080)
    cv2.imshow('Chessboard Corners', image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Draw the plane normal vector for every point
    for point in plane_points_2d:
        center = (int(point[0][0]), int(point[0][1]))
        normal_end = center - 100 * plane_equation[:2]  # Invert the arrow to point up and make it longer
        cv2.arrowedLine(image, center, tuple(normal_end.astype(int)), (0, 0, 255), 2)

    # Display the image
    cv2.namedWindow('Plane Visualization', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Plane Visualization', 1920, 1080)
    cv2.imshow('Plane Visualization', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Chessboard corners not found.")