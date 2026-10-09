# Nguồn chính thức và hợp đồng đã kiểm tra

Kiểm tra ngày **08/10/2026**. Các source sau đã tải và đọc từ repository upstream; bản tải tạm nằm trong `.research/` (gitignored). Không dùng thứ tự `points` suy đoán từ ví dụ 2D.

| Nội dung | Nguồn/version | Vị trí đã đọc |
|---|---|---|
| CVAT 16-value cuboid, position/rotation/scale | [canvas3dView.ts, v2.20.0](https://github.com/cvat-ai/cvat/blob/v2.20.0/cvat-canvas3d/src/typescript/canvas3dView.ts#L339) | serialization L339/L471/L846; decode L991–993 |
| Unit centered cuboid, Euler mặc định THREE | [cuboid.ts, v2.20.0](https://github.com/cvat-ai/cvat/blob/v2.20.0/cvat-canvas3d/src/typescript/cuboid.ts#L35) | BoxGeometry L35, rotation.set L130–132 |
| Dataset binding 3D fields | [bindings.py, v2.20.0](https://github.com/cvat-ai/cvat/blob/v2.20.0/cvat/apps/dataset_manager/bindings.py#L1945) | position=0:3, rotation=3:6, scale=6:9 |
| CVAT task/data/annotations/requests REST | [views.py, v2.20.0](https://github.com/cvat-ai/cvat/blob/v2.20.0/cvat/apps/engine/views.py) | data POST → rq_id; PATCH action=create; RequestViewSet |
| Task dimension không writable, labels/attributes/shape points | [serializers.py, v2.20.0](https://github.com/cvat-ai/cvat/blob/v2.20.0/cvat/apps/engine/serializers.py#L1138) | TaskWriteSerializer, DataSerializer, ShapeSerializer |
| CVAT source values và dimension | [models.py, v2.20.0](https://github.com/cvat-ai/cvat/blob/v2.20.0/cvat/apps/engine/models.py) | `SourceType.AUTO`, DimensionType |
| CenterPoint checkpoint/config/memory | [model zoo, v1.4.0](https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/configs/centerpoint/README.md) | voxel01 circle-NMS 5.2 GB, pillar02 circle-NMS 4.6 GB |
| Voxel model test pipeline | [voxel config, v1.4.0](https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/configs/centerpoint/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d.py) | load_dim/use_dim=5, 9 sweeps, test augmentation/range filter |
| Circle NMS overlay | [voxel circle config, v1.4.0](https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/configs/centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py) | nms_type=circle |
| Pillar alternative | [pillar config, v1.4.0](https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/configs/centerpoint/centerpoint_pillar02_second_secfpn_8xb4-cyclic-20e_nus-3d.py) | giữ pipeline riêng của profile, 5 features/9 prior sweeps |
| LiDAR boxes origin/dims/yaw | [lidar_box3d.py, v1.4.0](https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/mmdet3d/structures/bbox_3d/lidar_box3d.py#L15) | bottom origin 0.5/0.5/0, x/y/z size, yaw +X→+Y |
| Tâm hình học kế thừa | [base_box3d.py, v1.4.0](https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/mmdet3d/structures/bbox_3d/base_box3d.py#L142) | gravity_center Z = bottom_center Z + tensor height/2 |
| LoadPointsFromMultiSweeps | [loading.py, v1.4.0](https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/mmdet3d/datasets/transforms/loading.py#L298) | time lag, remove-close, pad and native transport expression |
| Inference API limitations | [inference.py, v1.4.0](https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/mmdet3d/apis/inference.py#L121) | generic one-file demo supplies timestamp=1, không supply actual LiDAR sweeps |
| Compatibility bounds | [mmdet3d/__init__.py, v1.4.0](https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/mmdet3d/__init__.py) | MMCV<2.2, MMEngine<1, MMDetection<3.4 |
| Dependency metadata | [MMDetection3D 1.4.0 metadata](https://pypi.org/pypi/mmdet3d/1.4.0/json), [devkit 1.1.11 metadata](https://pypi.org/pypi/nuscenes-devkit/1.1.11/json) | MIM extra mmdet<3.3; devkit matplotlib<3.6/shapely<2 |
| MMCV wheel có sẵn | [cu117/torch1.13 wheel index](https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html) | mmcv-2.1.0-cp310 manylinux |
| PyTorch historical wheel commands | [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/#v1131) | 1.13.1 / torchvision0.14.1 / CUDA11.7 |
| nuScenes binary và box dims | [devkit data_classes.py](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py) | float32 reshape(-1,5); Box wlh; corners local X=length |
| nuScenes transforms | [nuscenes.py](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py) | get_sample_data/get_boxes sensor/ego/global |
| CVAT UI cuboid review | [3D object annotation](https://docs.cvat.ai/docs/annotation/manual-annotation/modes/3d-object-annotation/) | Shape/Track, rotate/resize and Save workflow |

Đã tải wheel **nuscenes_devkit-1.1.11-py3-none-any.whl** từ PyPI và đọc trực tiếp `nuscenes/utils/data_classes.py` cùng `nuscenes/nuscenes.py` trong đúng bản pin. Xác nhận float32 reshape 5 cột, wlh, get_sample_data với `use_flat_vehicle_coordinates=False`, và global→ego→sensor bằng inverse quaternions/translation. Các link master ở bảng phục vụ đọc thuận tiện; hợp đồng runtime đã đối chiếu wheel 1.1.11. Core dùng API devkit thay vì viết lại `get_sample_data`; chạy với dataset thật vẫn là bước nghiệm thu còn thiếu.

Hai URL checkpoint trong `assets.MODEL_PROFILES` đã trả **HTTP 200 với HEAD**, nhưng chưa tải/verify full checkpoint hoặc load model trong phiên triển khai này. Full hash sẽ được tính/kiểm tra lúc `fetch-model` và `infer`. Không ghi số mAP/NDS/memory của model zoo thành kết quả thực nghiệm của dự án.

Các source có license upstream riêng; project dùng dependency/API và các phép chuyển đổi do mình triển khai, không vendoring code detector/CVAT. Archive MMDetection3D được tải theo yêu cầu qua `fetch-model` vào `models/` và giữ license upstream.
