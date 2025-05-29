import cv2
import numpy as np
import glob, os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- Parameters ---
calib_folder = "log_reconstruction/laser_plane_calibration"
pattern_size = (11, 6)         # inner corners
square_size = 0.024            # meters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
images = sorted(glob.glob(os.path.join(calib_folder, "*.png")))

# --- 1) CALIBRATE ---
# prepare object points
objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
objp *= square_size

objpoints, imgpoints = [], []
for fn in images:
    img = cv2.imread(fn); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not ret: continue
    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    objpoints.append(objp); imgpoints.append(corners2)
ret, cam_mtx, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# --- 2) RECOVER EACH BOARD AND LASER LINE IN CAMERA COORDS ---
all_boards = []
all_line_points = []                      # <-- collect 3D endpoints of laser stripe
fx, fy = cam_mtx[0,0], cam_mtx[1,1]       # <-- intrinsics
cx, cy = cam_mtx[0,2], cam_mtx[1,2]

for fn in images:
    img = cv2.imread(fn)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not ret:
        continue
    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    _, rvec, tvec = cv2.solvePnP(objp, corners2, cam_mtx, dist_coefs)
    R, _ = cv2.Rodrigues(rvec)
    pts3 = (R @ objp.T + tvec).T
    all_boards.append(pts3)

    # --- detect blue laser stripe in this view ---
    mask = cv2.inRange(img, np.array([100,0,0]), np.array([255,80,80]))
    lines = cv2.HoughLinesP(mask,1,np.pi/180,50,minLineLength=50,maxLineGap=10)
    if lines is None:
        continue
    x1,y1,x2,y2 = max(lines[:,0,:], key=lambda l: np.hypot(l[2]-l[0],l[3]-l[1]))
    # board plane in camera coords
    board_normal = R @ np.array([0.,0.,1.])
    board_point0 = (R @ objp[0] + tvec.squeeze())
    D_b = -board_normal.dot(board_point0)
    # back-project endpoints
    for x,y in [(x1,y1),(x2,y2)]:
        ray = np.array([(x-cx)/fx, (y-cy)/fy, 1.0])
        t = -D_b / (board_normal.dot(ray))
        all_line_points.append(t * ray)

# --- 3) FIT LASER PLANE THROUGH ALL LASER LINE POINTS ---
A = np.array(all_line_points)
centroid = A.mean(axis=0)
_,_,Vt = np.linalg.svd(A - centroid)
normal = Vt[-1]
D = -normal.dot(centroid)

# --- 4) PICK ONE BOARD TO VISUALIZE ---
print("Available boards:")
for i, fn in enumerate(images):
    print(f"{i}: {fn}")
idx = int(input("Select board index: "))
board_pts = all_boards[idx]

# --- 5) 3D PLOT ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# build and sort quad faces for chessboards using row-major ordering
nx, ny = pattern_size
chess_faces = []
for pts in all_boards:
    grid = pts.reshape(ny, nx, 3)   # swap dims: rows, cols
    for j in range(ny-1):
        for i in range(nx-1):
            quad = [
                grid[j,   i],
                grid[j,   i+1],
                grid[j+1, i+1],
                grid[j+1, i]
            ]
            depth = np.mean([v[2] for v in quad])
            chess_faces.append((depth, quad))

# draw quads from farthest to nearest
for _, quad in sorted(chess_faces, key=lambda x: x[0], reverse=True):
    poly = Poly3DCollection([quad], facecolors='cyan', edgecolors='gray', alpha=0.3)
    ax.add_collection3d(poly)

# highlight chosen board with quad faces
hb = board_pts.reshape(ny, nx, 3)
hb_quads = []
for j in range(ny-1):
    for i in range(nx-1):
        hb_quads.append([
            hb[j,   i],
            hb[j,   i+1],
            hb[j+1, i+1],
            hb[j+1, i]
        ])
highlight = Poly3DCollection(hb_quads, facecolors='red', edgecolors='k', alpha=0.5)
ax.add_collection3d(highlight)

# create plane mesh
# pick extents from chosen board
xlim = (board_pts[:,0].min(), board_pts[:,0].max())
ylim = (board_pts[:,1].min(), board_pts[:,1].max())
xx, yy = np.meshgrid(
    np.linspace(*xlim, 10),
    np.linspace(*ylim, 10)
)
zz = (-normal[0]*xx - normal[1]*yy - D)/normal[2]

ax.plot_surface(xx, yy, zz, color='g', alpha=0.4)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title('Chessboards + fitted laser plane')
plt.show()
