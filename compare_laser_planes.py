"""Compare laser planes from Plücker and RANSAC calibrations using Open3D.

Loads two calibration .npz files (defaults: calibration_plucker.npz, calibration_ransac.npz)
containing fields: plane_normal (3,), plane_D (scalar).

Visualizes both planes as colored square patches plus their normals (arrows) and a legend.
Always shows a coordinate frame (axes). GUI panel support removed: only legacy visualization is used.

Usage:
    python compare_laser_planes.py \
            --plucker calibration_plucker.npz \
            --ransac calibration_ransac.npz \
            --half-size 0.25
"""
from __future__ import annotations
import argparse, sys
import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("This script requires open3d. Install with: pip install open3d", file=sys.stderr)
    sys.exit(1)


def load_plane(npz_path: str):
    data = np.load(npz_path)
    n = data['plane_normal'].astype(float)
    D = float(data['plane_D'])
    norm = np.linalg.norm(n)
    if norm == 0:
        raise ValueError(f"Plane normal zero in {npz_path}")
    n /= norm
    return n, D


def make_plane_mesh(n: np.ndarray, D: float, center: np.ndarray, half_size: float, color, resolution: int = 10):
    """Return an Open3D TriangleMesh for a square patch of the plane.
    Center the patch at the provided 3D center projected onto plane to align different planes.
    Also duplicate faces reversed to ensure backface visibility even with single-sided shading.
    """
    # Project desired center onto plane to align visually
    # plane: n·X + D = 0
    dist = n @ center + D
    p0 = center - dist * n
    # Build basis (u,v) spanning plane
    # Pick a vector not parallel to n
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)

    # Create grid points
    us = np.linspace(-half_size, half_size, resolution)
    vs = np.linspace(-half_size, half_size, resolution)
    vertices = []
    for a in us:
        for b in vs:
            vertices.append(p0 + a * u + b * v)
    vertices = np.array(vertices)

    # Triangles
    triangles = []
    for i in range(len(us) - 1):
        for j in range(len(vs) - 1):
            idx0 = i * len(vs) + j
            idx1 = (i + 1) * len(vs) + j
            idx2 = (i + 1) * len(vs) + (j + 1)
            idx3 = i * len(vs) + (j + 1)
            triangles.append([idx0, idx1, idx2])
            triangles.append([idx0, idx2, idx3])
    tri_arr = np.array(triangles)
    # Duplicate reversed for back face
    tri_rev = tri_arr[:, ::-1]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(np.vstack([tri_arr, tri_rev]))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh


def make_normal_arrow(n: np.ndarray, D: float, scale: float, color, center: np.ndarray):
    # Base point: project requested center onto plane for consistent overlay
    dist = n @ center + D
    p0 = center - dist * n
    arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.005*scale, cone_radius=0.01*scale,
                                                   cylinder_height=0.7*scale, cone_height=0.3*scale)
    # Default arrow points in +Z. Need rotation from z-axis to n.
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, n)
    c = np.dot(z, n)
    if np.linalg.norm(v) < 1e-12:
        R = np.eye(3) if c > 0 else -np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        s = np.linalg.norm(v)
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
    arrow.rotate(R, center=np.zeros(3))
    arrow.translate(p0)
    arrow.paint_uniform_color(color)
    return arrow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plucker', default='calibration_plucker.npz')
    ap.add_argument('--ransac', default='calibration_ransac.npz')
    ap.add_argument('--half-size', type=float, default=0.25, help='Half side length of plane patches (meters)')
    ap.add_argument('--resolution', type=int, default=12)
    ap.add_argument('--arrow-scale', type=float, default=0.15)
    args = ap.parse_args()

    n1, D1 = load_plane(args.plucker)
    n2, D2 = load_plane(args.ransac)

    # Choose common center = midpoint of closest points of each plane to origin
    c1 = -D1 * n1
    c2 = -D2 * n2
    common_center = 0.5 * (c1 + c2)

    plane1 = make_plane_mesh(n1, D1, common_center, args.half_size, color=[0.1, 0.6, 1.0], resolution=args.resolution)
    plane2 = make_plane_mesh(n2, D2, common_center, args.half_size, color=[1.0, 0.3, 0.1], resolution=args.resolution)

    arrow1 = make_normal_arrow(n1, D1, args.arrow_scale, [0.0, 0.0, 0.8], common_center)
    arrow2 = make_normal_arrow(n2, D2, args.arrow_scale, [0.8, 0.0, 0.0], common_center)

    geoms = [
        ("PluckerPlane", plane1),
        ("RANSACPlane", plane2),
        ("PluckerNormal", arrow1),
        ("RANSACNormal", arrow2),
        ("Axes", o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.arrow_scale*2.0))
    ]

    legend_items = [
        ((0.1,0.6,1.0,1.0), 'Plücker plane'),
        ((1.0,0.3,0.1,1.0), 'RANSAC plane'),
        ((0.0,0.0,0.8,1.0), 'Plücker normal'),
        ((0.8,0.0,0.0,1.0), 'RANSAC normal'),
    ]

    # Metrics
    angle = np.degrees(np.arccos(np.clip(n1 @ n2, -1.0, 1.0)))
    # Distance between planes along normals at common center
    # Signed distances of common center to each plane should be near zero; differences in D give relative offset.
    # Compute offset at origin: difference of D / |n| with n normalized => D difference (normals normalized)
    offset_diff = abs(D1 - D2)
    print(f"Angle between normals: {angle:.6f} deg")
    print(f"Offset |D1 - D2|: {offset_diff:.6f} m")
    print("Legend:")
    for c,lbl in legend_items:
        print(f"  {lbl} color RGBA={c}")
    legacy_geoms = [g for _, g in geoms]
    o3d.visualization.draw_geometries(legacy_geoms, window_name='Laser Plane Comparison')

if __name__ == '__main__':
    main()
