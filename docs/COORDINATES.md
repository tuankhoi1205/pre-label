# Quy ước hình học và dữ liệu

## Schema nội bộ

Mỗi `Cuboid` chứa:

| Field | Quy ước |
|---|---|
| `sample_token` | nuScenes keyframe sample token |
| `frame_id` | Chỉ số manifest, bắt đầu 0; không tự coi là CVAT frame |
| `prediction_id` | `run_signature_prefix:sample_token:native_prediction_index` |
| `class_name`, `confidence` | Class theo đúng thứ tự checkpoint; score trong [0,1], hoặc null cho manual/GT |
| `center_xyz` | **Tâm hình học** của cuboid, mét |
| `size_lwh` | Chiều dài local X, chiều rộng local Y, chiều cao local Z, mét |
| `quaternion_wxyz` | Quaternion đơn vị **scalar first** [w,x,y,z]; local box → coordinate frame |
| `coordinate_frame` | `lidar_sensor` của keyframe tương ứng; các hàm transform hỗ trợ `ego`/`global` |
| `model`, `checkpoint` | Tên model/version và SHA-256 checkpoint thực tế |
| `annotation_id`, `identity_source` | ID server và nguồn khôi phục identity sau review |

Một sample có một hệ LiDAR riêng; không so sánh hai box ở hai sample khác nhau dù cả hai cùng ghi `lidar_sensor`. Các hàm metric từ chối so sánh khác coordinate frame/sample. Đơn vị mét/radian. Không dùng Euler làm schema chính; giữ quaternion để tránh mất roll/pitch khi đổi frame.

## nuScenes raw → checkpoint

Raw `.pcd.bin` là little-endian float32, 5 cột `[x,y,z,intensity,ring]`. Devkit `LidarPointCloud.from_file` reshape thành 5 rồi bỏ ring cho một số phương thức khác; detector cần đọc đủ 5 theo config. Tool kiểm tra byte count là bội 20, nonempty, finite, ring nguyên 0–31. Byte count tự nó không chứng minh schema của tệp tùy ý; file phải lấy từ bảng `sample_data` của nuScenes.

Các giá trị XYZ được giữ nguyên trong **hệ cảm biến LIDAR_TOP**, không đổi thứ tự trục hay coi cảm biến trùng ego. Trong quy ước box LiDAR của MMDetection3D, yaw quanh +Z, yaw=0 ứng với local +X, góc dương quay +X sang +Y, dims `(x_size,y_size,z_size)`. Chiều vật lý so với xe ego phải dùng calibration, không suy ra từ tên trục.

Config đã pin dùng `LoadPointsFromFile(load_dim=5,use_dim=5)` rồi `LoadPointsFromMultiSweeps(sweeps_num=9,use_dim=[0,1,2,3,4],pad_empty_sweeps=True,remove_close=True)`:

1. Keyframe ring column được thay bằng **time lag=0**.
2. Tối đa 9 sweep trước đó từ chuỗi `sample_data.prev`, gần nhất trước, cùng scene.
3. Mỗi sweep loại điểm có cả `|x|<1` và `|y|<1` trong hệ sweep (quy tắc source, không phải bán kính tròn).
4. Sweep XYZ biến đổi về LiDAR keyframe; intensity giữ nguyên; cột cuối là `(timestamp_key_us - timestamp_sweep_us)/1e6`, giây.
5. Đầu scene không có sweep: không truyền key `lidar_sweeps`, để loader chuẩn pad 9 bản keyframe đã remove-close. Khi có ít hơn 9, giữ tất cả sweeps có thật, theo loader.
6. Các bước test augmentation/range filter/pack và NMS dùng config gốc. Không dùng `inference_detector(model, one_bin)` vì hàm demo đó không cung cấp sweep metadata thực.

Adapter chạy `Compose` của test pipeline, `pseudo_collate` một sample rồi `model.test_step` dưới `torch.inference_mode`. Load model và Compose một lần cho phần pending. Chỉ đặt `test_mode=True` ở loader sweeps để giữ chọn sweep xác định; không thay range/augmentation/NMS/số sweeps. Config pillar khác voxel ở feature encoder/range filter, nên vẫn dùng đúng pipeline của từng config, không gán lại pipeline thủ công chung.

## Pose và transport field của loader 1.4.0

Với column vectors, `T_A_B` đổi tọa độ từ B sang A:

```text
T_global_sensor = T_global_ego @ T_ego_sensor
T_keylidar_sweeplidar = inverse(T_global_keylidar) @ T_global_sweeplidar
p_key = R @ p_sweep + t
q_key_box = q_key_source * q_source_box
```

`T_ego_sensor` từ `calibrated_sensor`, `T_global_ego` từ `ego_pose` của **chính sample_data đó**. Không dùng ego pose keyframe cho sweep cũ. Tất cả quaternion metadata nuScenes scalar first.

Source `LoadPointsFromMultiSweeps` 1.4.0 tiêu thụ field tên `lidar2sensor` bằng biểu thức row-vector:

```text
p_loaded = p_raw @ field[:3,:3] - field[:3,3]
```

Vì vậy adapter encode `field[:3,:3]=R.T`, `field[:3,3]=-t`. Đây là **transport payload cho biểu thức source**, không phải ma trận `inverse(T_keylidar_sweeplidar)` chuẩn. Manifest lưu SE(3) đúng, chỉ `model_input` encode field này. Không tái sử dụng field đó như một pose tổng quát. Có test độc lập so sánh biểu thức loader với phép nhân homogeneous và một test native loader tùy chọn trong môi trường MMDetection3D.

## Prediction → nội bộ → CVAT

`LiDARInstance3DBoxes.tensor` chứa `[bottom_x,bottom_y,bottom_z,x_size,y_size,z_size,yaw,...]`, origin `(0.5,0.5,0)`. Dùng **`boxes.gravity_center`** và **`boxes.dims`**, không đọc 3 số đầu tensor thành tâm hình học. Với box upright, geometric Z = bottom Z + height/2; yaw giữ dấu/radian như model. Velocity được giữ trong raw tensor nhưng không dùng trong cuboid tĩnh.

Đối với GT nuScenes, `get_sample_data(...use_flat_vehicle_coordinates=False)` trả box đã đổi global → ego → sensor. `Box.center` là tâm hình học; devkit `wlh=[width,length,height]` nên đổi thành `[length,width,height]`. Quaternion giữ đủ rotation; không ép yaw-only. GT chỉ được gọi ở evaluation.

PCD CVAT chứa keyframe XYZ với identity VIEWPOINT. Cuboid xuất cùng `lidar_sensor`; `cvat_points` từ chối box ego/global. Source CVAT 2.20.0 đã kiểm chứng:

```text
points = [cx,cy,cz, rx,ry,rz, sx,sy,sz, 0,0,0,0,0,0,0]
             tâm      Euler       LWH       padding
```

Đúng **16 số**; trường `rotation` riêng của shape được đặt 0 (rotation 2D, không phải yaw 3D). Canvas dùng unit `THREE.BoxGeometry(1,1,1)` tâm ở origin, rồi `setPosition`, `setScale`, `setRotation`. Euler mặc định của THREE là **intrinsic XYZ**, tức ma trận `Rx(rx) @ Ry(ry) @ Rz(rz)` với column vectors; dùng SciPy `as_euler('XYZ')`/`from_euler('XYZ')`, chữ hoa. Không dùng extrinsic `xyz` khi có roll/pitch. Gimbal lock có thể đổi giá trị Euler nhưng rotation/corners tương đương.

## Kiểm chứng

Tests dùng dims `[4.6,1.7,1.5]` để phát hiện đảo L/W, yaw 0/90/arbitrary, rotation XYZ độc lập, quaternion đổi dấu, gimbal lock, sensor→ego→global→sensor, native sweep transport và CVAT points roundtrip. IoU có các nghiệm thể tích biết trước và invariant khi xoay/translate cả hai box. API fake/live kiểm tra label/frame/geometry sau PATCH.

**Chưa xác minh trực quan trên CVAT thật.** Bước bắt buộc trước batch là xem PCD/cuboid trong UI, chỉnh/xóa/thêm/Save/export, rồi đối chiếu manifest. Các link source có số dòng trong [SOURCES.md](SOURCES.md).
