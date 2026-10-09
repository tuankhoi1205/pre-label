from itertools import combinations, product
import warnings
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation
from .schema import Cuboid, ensure_same_frame


def rotation(q_wxyz):
    q = np.asarray(q_wxyz, dtype=float)
    return Rotation.from_quat(q[[1, 2, 3, 0]])


def quaternion(rot):
    q = rot.as_quat()
    return q[[3, 0, 1, 2]].tolist()


def yaw_quaternion(yaw):
    return quaternion(Rotation.from_euler("Z", yaw))


def pose_matrix(translation, q_wxyz):
    t = np.eye(4)
    t[:3, :3] = rotation(q_wxyz).as_matrix()
    t[:3, 3] = translation
    return t


def transform_box(box, target_from_source, target_frame):
    t = np.asarray(target_from_source, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all() or not np.allclose(t[3], [0, 0, 0, 1]):
        raise ValueError("Invalid homogeneous transform")
    if not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(t[:3, :3]), 1):
        raise ValueError("Transform must be rigid, without reflection")
    center = t[:3, :3] @ box.center_xyz + t[:3, 3]
    r = Rotation.from_matrix(t[:3, :3]) * rotation(box.quaternion_wxyz)
    return box.updated(center_xyz=center.tolist(), quaternion_wxyz=quaternion(r), coordinate_frame=target_frame)


def corners(box):
    local = np.array(list(product([-0.5, 0.5], repeat=3))) * box.size_lwh
    return rotation(box.quaternion_wxyz).apply(local) + box.center_xyz


def cvat_points(box):
    if box.coordinate_frame != "lidar_sensor":
        raise ValueError("CVAT point clouds and cuboids must both be in lidar_sensor coordinates")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # Euler gimbal lock is an equivalent rotation.
        euler = rotation(box.quaternion_wxyz).as_euler("XYZ").tolist()
    return [*box.center_xyz, *euler, *box.size_lwh, *([0.0] * 7)]


def box_from_cvat(points, **metadata):
    if len(points) != 16 or not np.isfinite(points).all():
        raise ValueError("Expected CVAT 2.20 3D cuboid: exactly 16 finite values")
    return Cuboid(
        center_xyz=list(points[:3]), size_lwh=list(points[6:9]),
        quaternion_wxyz=quaternion(Rotation.from_euler("XYZ", points[3:6])), **metadata)


def _halfspaces(box):
    axes = rotation(box.quaternion_wxyz).as_matrix().T
    normals = np.concatenate([axes, -axes])
    offsets = normals @ np.array(box.center_xyz) + np.tile(np.array(box.size_lwh) / 2, 2)
    return normals, offsets


def iou3d(a, b):
    """Exact convex intersection of oriented cuboids, including roll and pitch."""
    ensure_same_frame(a, b)
    ca, cb = corners(a), corners(b)
    if np.any(np.minimum(ca.max(0), cb.max(0)) <= np.maximum(ca.min(0), cb.min(0))):
        return 0.0
    na, da = _halfspaces(a)
    nb, db = _halfspaces(b)
    normals, offsets = np.concatenate([na, nb]), np.concatenate([da, db])
    vertices = []
    for indices in combinations(range(12), 3):
        n = normals[list(indices)]
        if abs(np.linalg.det(n)) < 1e-10:
            continue
        p = np.linalg.solve(n, offsets[list(indices)])
        if np.all(normals @ p <= offsets + 1e-8):
            vertices.append(p)
    if len(vertices) < 4:
        return 0.0
    vertices = np.unique(np.round(vertices, 9), axis=0)
    try:
        intersection = ConvexHull(vertices).volume
    except QhullError:
        return 0.0
    va, vb = np.prod(a.size_lwh), np.prod(b.size_lwh)
    intersection = min(intersection, va, vb)
    return float(np.clip(intersection / (va + vb - intersection), 0, 1))


def geometry_delta(a, b):
    ensure_same_frame(a, b)
    return {
        "center_m": float(np.linalg.norm(np.array(a.center_xyz) - b.center_xyz)),
        "size_m": float(np.max(np.abs(np.array(a.size_lwh) - b.size_lwh))),
        "rotation_rad": float((rotation(a.quaternion_wxyz).inv() * rotation(b.quaternion_wxyz)).magnitude()),
    }
