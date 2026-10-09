import math
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from prelabel.geometry import (box_from_cvat, corners, cvat_points, geometry_delta, iou3d,
                               pose_matrix, quaternion, transform_box, yaw_quaternion)
from prelabel.schema import Cuboid


@pytest.mark.parametrize("yaw", [0, math.pi / 2, -0.72, math.pi, 2 * math.pi])
def test_cvat_roundtrip_asymmetric_box(box, yaw):
    b = box.updated(quaternion_wxyz=yaw_quaternion(yaw))
    points = cvat_points(b)
    assert len(points) == 16
    assert points[6:9] == [4.6, 1.7, 1.5]
    metadata = {k: v for k, v in b.to_dict().items() if k not in ("center_xyz", "size_lwh", "quaternion_wxyz")}
    restored = box_from_cvat(points, **metadata)
    np.testing.assert_allclose(corners(b), corners(restored), atol=1e-8)


def test_ninety_degree_is_not_dimension_swap(box):
    b = box.updated(quaternion_wxyz=yaw_quaternion(math.pi / 2))
    np.testing.assert_allclose(np.ptp(corners(b), axis=0), [1.7, 4.6, 1.5], atol=1e-8)
    assert cvat_points(b)[6:9] == box.size_lwh


@pytest.mark.parametrize("angles", [[0.2, -0.4, 0.7], [0.3, math.pi / 2, -0.8]])
def test_three_intrinsic_xyz_independent_matrix(box, angles):
    x, y, z = angles
    rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    b = box.updated(quaternion_wxyz=quaternion(Rotation.from_matrix(rx @ ry @ rz)))
    points = cvat_points(b)
    if abs(y) < math.pi / 2 - 0.01:
        np.testing.assert_allclose(points[3:6], angles, atol=1e-8)
    restored = box_from_cvat(points, sample_token=box.sample_token, frame_id=0, prediction_id="restored", class_name="car", confidence=None)
    np.testing.assert_allclose(corners(b), corners(restored), atol=1e-8)


def test_sensor_ego_global_roundtrip(box):
    ego_from_lidar = pose_matrix([1.5, -0.7, 1.8], quaternion(Rotation.from_euler("XYZ", [0.02, -0.03, 1.2])))
    global_from_ego = pose_matrix([90, -31, 2], quaternion(Rotation.from_euler("XYZ", [0.1, 0.03, -0.5])))
    ego = transform_box(box, ego_from_lidar, "ego")
    glob = transform_box(ego, global_from_ego, "global")
    result = transform_box(glob, np.linalg.inv(global_from_ego @ ego_from_lidar), "lidar_sensor")
    np.testing.assert_allclose(corners(box), corners(result), atol=1e-8)


def test_transform_rejects_scaling(box):
    t = np.eye(4)
    t[0, 0] = 2
    with pytest.raises(ValueError, match="rigid"):
        transform_box(box, t, "ego")


def test_iou_analytic_cases(box):
    b = box.updated(center_xyz=[0, 0, 0], size_lwh=[4, 2, 2])
    assert iou3d(b, b) == pytest.approx(1)
    shifted = b.updated(center_xyz=[1, 0, 0])
    assert iou3d(b, shifted) == pytest.approx(12 / 20)
    rotated = b.updated(quaternion_wxyz=yaw_quaternion(math.pi / 2))
    assert iou3d(b, rotated) == pytest.approx(8 / 24)
    assert iou3d(b, b.updated(center_xyz=[0, 0, 2])) == 0
    assert iou3d(b, b.updated(center_xyz=[100, 0, 0])) == 0


def test_iou_full_rotation_invariance(box):
    a = box.updated(quaternion_wxyz=quaternion(Rotation.from_euler("XYZ", [0.3, 0.4, 0.5])))
    b = a.updated(center_xyz=[4.4, 2, 1.8])
    t = pose_matrix([40, -8, 12], quaternion(Rotation.from_euler("XYZ", [-0.7, 0.6, 1.1])))
    expected = iou3d(a, b)
    transformed_a = transform_box(a, t, "lidar_sensor")
    transformed_b = transform_box(b, t, "lidar_sensor")
    assert iou3d(transformed_a, transformed_b) == pytest.approx(expected, abs=1e-7)
    assert iou3d(b, a) == pytest.approx(expected)


def test_geometry_refuses_coordinate_mismatch(box):
    with pytest.raises(ValueError, match="same sample"):
        iou3d(box, box.updated(coordinate_frame="global"))


def test_quaternion_sign_is_same_orientation(box):
    assert geometry_delta(box, box.updated(quaternion_wxyz=(-np.array(box.quaternion_wxyz)).tolist()))["rotation_rad"] == pytest.approx(0)


@pytest.mark.parametrize("changes", [{"size_lwh": [1, 0, 3]}, {"center_xyz": [1, float("nan"), 3]}, {"quaternion_wxyz": [2, 0, 0, 0]}, {"confidence": 1.2}])
def test_schema_rejects_invalid_boxes(box, changes):
    with pytest.raises(ValueError):
        box.updated(**changes)
