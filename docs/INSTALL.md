# Môi trường cài đặt

Luồng Windows Docker local có nút CenterPoint và hướng cuboid: làm theo [README](../README.md#cài-đặt-trên-windows-với-docker-desktop). Phần dưới dành cho CLI Linux/WSL2 và server REST riêng.

## Detector Linux/WSL2

Chọn Ubuntu 22.04, Python **3.10** và NVIDIA GPU. Trên Windows, cài WSL2/Ubuntu bằng quy trình hệ điều hành của bạn, bật GPU passthrough và xác nhận `nvidia-smi` trong WSL. Project không tự cài WSL/Docker hay thay driver của máy.

| Thành phần | Phiên bản/profile |
|---|---|
| Python detector | 3.10 |
| PyTorch | 1.13.1+cu117 |
| torchvision | 0.14.1+cu117 |
| CUDA runtime của wheel | 11.7 |
| MMCV có CUDA ops | 2.1.0, wheel cu117/torch1.13 |
| MMEngine | 0.10.5 |
| MMDetection | 3.2.0 |
| MMDetection3D | 1.4.0 |
| nuScenes devkit | 1.1.11 |
| NumPy / SciPy | 1.26.4 / 1.12.0 |
| Numba / llvmlite | 0.59.1 / 0.42.0 |
| OpenCV | 4.10.0.84 |
| Matplotlib / Shapely | 3.5.3 / 1.8.5.post1 |
| CVAT REST target | 2.20.0 |

MMDetection3D source yêu cầu `MMCV >=2.0.0rc4,<2.2.0`, `MMEngine >=0.8.0,<1.0.0` và `MMDetection >=3.0.0rc5,<3.4.0`. Metadata extra MIM của package 1.4.0 có giới hạn MMDetection `<3.3.0`; chọn **3.2.0** để thỏa cả hai. Các pin trên nằm trong miền này, và OpenMMLab có wheel MMCV 2.1.0 Linux/Python 3.10/cu117/torch1.13. Bộ này là lựa chọn tương thích theo source/wheel; **chưa được chạy detector thật trên máy hiện tại**. Xem [nguồn](SOURCES.md).

Con số CUDA 12.7 trong `nvidia-smi` của máy chỉ là mức API driver hỗ trợ, không chứng minh đã có CUDA Toolkit hoặc torch CUDA. Dùng runtime cu117 trong wheel; không cần `nvcc` nếu toàn bộ ops dùng wheel. Nếu pip định build MMCV từ source, dừng và kiểm tra Python/torch/CUDA/wheel; `--only-binary=mmcv` trong script giúp phát hiện trường hợp này.

```bash
# Trong Ubuntu có python3.10 và python3.10-venv:
bash scripts/install_linux.sh
source .venv/bin/activate
python -m pip check
python -c 'import torch, mmcv, mmdet3d; from mmcv.ops import nms3d; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), mmcv.__version__, mmdet3d.__version__)'
```

Không cài `mmcv-lite` hoặc `mmcv-full` 1.x vào môi trường này. Không dùng Python 3.12 cho bộ detector PyTorch 1.13. Các dependency trực tiếp được pin; transitive packages được pip giải quyết, nên đây chưa phải lock đầy đủ theo platform. Sau khi môi trường inference thật đã chạy tốt, lưu `pip freeze` cùng log doctor để tái lập chính xác.

nuScenes-devkit 1.1.11 yêu cầu Matplotlib `<3.6.0` và Shapely `<2.0.0`. Pin Matplotlib3.5.3/Shapely1.8.5.post1 phù hợp Python3.10; extra `dataset` cũng cần Python3.10, dù core chạy được Python3.12. Không ép cài Matplotlib3.5 từ source trên Python3.12 để thực hiện pipeline này.

Nếu CUDA OOM: đảm bảo không có process GPU khác và batch luôn là 1; ưu tiên GPU đủ bộ nhớ. Profile `mini-pillar.yaml` dùng checkpoint CenterPoint pillar chính thức, không phải PointPillars thuần. Mức tham chiếu 4.6 GB vẫn lớn hơn 4 GB nên không bảo đảm giải quyết được OOM. Không có adapter OpenPCDet/PointPillars để fallback tự động; phải triển khai/kiểm chứng origin/dims/yaw/features trước khi thêm chúng.

## Dataset

Tải **nuScenes v1.0-mini** từ [nuScenes](https://www.nuscenes.org/nuscenes), theo điều kiện sử dụng của bộ dữ liệu, và giải nén:

```text
/data/nuscenes/
  v1.0-mini/{sample.json,sample_data.json,scene.json,...}
  samples/LIDAR_TOP/*.pcd.bin
  sweeps/LIDAR_TOP/*.pcd.bin
  maps/...
```

Đặt `NUSCENES_ROOT=/data/nuscenes`. Tệp nuScenes tên `.pcd.bin` là float32 raw 5 cột, không phải PCD header. `prepare` chuyển XYZ sang PCD binary riêng cho CVAT và kiểm tra keyframe/sweeps. Không cần sinh `nuscenes_infos*.pkl` hay database sampler cho inference; devkit và manifest cung cấp poses/sweeps. `create-task` chuẩn bị sáu ảnh camera keyframe của cùng sample và upload làm context; ảnh context chưa có overlay cuboid 3D.

## CVAT 2.20.0

Chưa có server URL của người dùng để kiểm tra phiên bản đang chạy. Adapter vì thế pin release **2.20.0** đã đọc source. Nếu server của bạn khác phiên bản, `doctor --check-cvat` dừng; kiểm chứng source/API/Euler và live test trước khi mở hỗ trợ phiên bản đó.

Server có thể ở máy khác; trên host Linux có Docker/Compose, dùng cấu hình upstream:

```bash
git clone --branch v2.20.0 --depth 1 https://github.com/cvat-ai/cvat.git cvat-server
cd cvat-server
CVAT_VERSION=v2.20.0 docker compose up -d
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Lệnh này là hướng dẫn, chưa chạy tại workspace. Nếu cần cấu hình domain/TLS/storage, làm theo [CVAT installation](https://docs.cvat.ai/docs/administration/basics/installation/) và file Compose của chính tag. Chạy CVAT trên host khác giúp tránh tranh VRAM với detector.

Trong `.env` của project:

```dotenv
CVAT_URL=http://localhost:8080
CVAT_USERNAME=your-user
CVAT_PASSWORD=your-password
# Hoặc CVAT_TOKEN=<token>, CVAT_AUTH_SCHEME=Token.
```

Token được ưu tiên khi cả hai cách cùng được điền. Có hỗ trợ header Bearer theo biến `CVAT_AUTH_SCHEME`, nhưng server 2.20 mặc định token API dùng `Token`. REST adapter không dùng SDK nên không có lệch phiên bản SDK. Nếu viết client SDK bổ sung cho target này, pin `cvat-sdk==2.20.0` và chạy lại live test.

Task dimension 3D được CVAT suy ra từ PCD sau xử lý data; API tạo task của release này không có trường ghi `dimension`. Tool upload tất cả PCD trong một multipart request, chờ `/api/requests/{rq_id}`, rồi kiểm tra `dimension`, metadata, labels và annotation readback. MVP phù hợp 20 frame; upload hàng nghìn PCD cần thêm chunk/TUS và giới hạn HTTP server.

## Core trên Windows

Python 3.10–3.12, không yêu cầu torch/CUDA cho geometry, REST, dry-run trên prediction đã có hoặc synthetic smoke:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest -q
python scripts/synthetic_smoke.py
```

Máy hiện tại chỉ dùng Python 3.12.14 được bundle bởi Codex để chạy core, cùng thư viện trong `.test-deps`. Đây không phải môi trường detector. Nếu chạy trực tiếp từ bundle này, cần `PYTHONPATH` trỏ `.test-deps` và `src`; nên tạo Python/venv thông thường để sử dụng dự án lâu dài.
