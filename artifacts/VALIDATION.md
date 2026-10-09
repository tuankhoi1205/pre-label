# Bằng chứng kiểm chứng — 08/10/2026

## Cập nhật 09/10/2026: task upload trực tiếp

Function local đã cập nhật để tự khớp binary XYZ/XYZI PCD với raw nuScenes, chuẩn bị calibration/sweeps và dùng chung một model GPU. Task #4 `test` gồm 10 PCD `20.pcd`–`29.pcd` và 60 ảnh context. Cả 10 input qua native CVAT provider/gateway đều trả ready/HTTP 200, khớp scene-1094 và có 9 prior sweeps. Đây là kiểm tra input bằng `validate_only`: không chạy detector hoặc tạo box; người dùng vẫn nhấn Annotate. Task #4 còn 0 box; Task #2 giữ 1.123 box và SHA256 annotation ban đầu.

Bằng chứng: [task4-auto-input-check.json](task4-auto-input-check.json). Bộ test đầy đủ đạt 103 passed/2 skipped; sau sửa kiểm tra integrity và ghi PCD atomic, nhóm liên quan đạt 48 passed/1 skipped: [full tests](auto-upload-test-results.xml), [final checks](auto-upload-final-test-results.xml). Các lượt skipped yêu cầu môi trường live/detector riêng. README và hướng dẫn CVAT đã ghi luồng upload trực tiếp và giới hạn format/dataset.

## Cập nhật: inference thật, camera và hướng cuboid

Người dùng đã nhấn Annotate: 20 frame thật hoàn tất trong khoảng 45,5 giây, Task #2 có 1.123 cuboid. 120 ảnh camera được gắn theo đúng sample token; provider CVAT trả 6 JPEG/frame cho đủ 20 frame và SHA256 annotation trước/sau không đổi. UI có checkbox Cuboid orientation và sáu camera context. Webpack build thành công, 41 test context/REST/UI/journal đạt. Bằng chứng: [real-inference-complete.json](real-inference-complete.json), [context-orientation-test-results.xml](context-orientation-test-results.xml), [ảnh giao diện](cvat-camera-orientation.png).

Các mục bên dưới ghi lịch sử kiểm chứng trước lượt inference thành công. Peak VRAM, kiểm định alignment/đầu–đuôi, quality sau review và thời gian annotator vẫn cần thực nghiệm.


**87 passed, 2 skipped, 0 failures/errors**. Chạy Python 3.12.14 trên Windows11, core dependencies được pin trong `.test-deps`; JUnit XML tại [test-results.xml](test-results.xml). Test REST CVAT thật chạy riêng: **1 passed**, dữ liệu PCD/cuboid tổng hợp; kết quả tại [live-cvat-test-results.xml](live-cvat-test-results.xml).

```powershell
$env:PYTHONPATH = "$PWD\.test-deps;$PWD\src"
$env:PYTHONIOENCODING = 'utf-8'
& 'C:\Users\khoik\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest -q --tb=short --basetemp '.test-tmp/check-ui-3' --junitxml artifacts/test-results.xml
```

`--basetemp` dùng thư mục con workspace vì TEMP mặc định không truy cập được trong môi trường kiểm thử này. Khi tái chạy ở máy bình thường dùng venv và `python -m pytest -q`; không dùng lại thư mục basetemp có dữ liệu cần giữ, vì pytest dọn thư mục đó.

Đã chạy CLI help, `doctor`, `report-time`, `compileall` và `scripts/synthetic_smoke.py`. Snapshot tại [environment.json](environment.json); thời gian gán nhãn thật [chưa được đo](reports/annotation_time_summary.json).

| Nhóm | Đã kiểm chứng |
|---|---|
| Geometry | Bất đối xứng L/W, yaw0/90/arbitrary, intrinsic XYZ/roll/pitch/gimbal lock, quaternion sign, SE(3) roundtrip, exact IoU analytic cases/rigid invariance |
| Dataset | BIN5 columns/ring/finite, binary XYZ PCD, calibration/ego sweep transform, manifest/resume không đọc GT |
| Detector contract | Time lag seconds, native transport formula, per-class thresholds/no second NMS, one model load/pending-only resume bằng adapter test |
| CVAT fixture | Task/data/PATCH/readback, reversed metadata, human edit/delete/add retention, rerun, lost-response recovery, uncertain deletion, restored/lost ID, duplicate identity |
| Evaluation/time | Maximum-cardinality assignment, zero matches, TP/FP/FN/IoU, edit tolerances/ratios, identity unavailable, breaks excluded, overlap rejection, actual CSV/JSON orchestration |
| CLI/provenance | YAML/env validation, actionable error code, immutable manifest/baseline, lock, safe archive extraction |
| Nuclio/UI contract | PCD base64 lookup, lazy model load, cached provenance, cuboid16/metadata, UI task không infer, native journal giữ sửa/xóa, commit recovery/uncertain deletion, patch version/idempotence |

Đã tải source CVAT v2.20.0 chính thức, áp dụng patch lên source thật, kiểm tra cú pháp Python/PowerShell; `docker compose config --quiet` thành công với upstream + serverless + overlay. Docker Desktop 4.94.0, WSL 3.0.1 và Docker Linux đã hoạt động sau reboot. Bằng chứng tại [local-cvat-setup.json](local-cvat-setup.json).

UI đã build webpack thành công. Task #2 có **20 frame nuScenes mini thật**, 10 lớp cuboid, annotation trống. PCD frame 0 qua API CVAT vẫn khớp SHA256 với manifest; point cloud hiển thị trong UI 3D. Bằng chứng: [real-task-ready.json](real-task-ready.json), [ảnh point cloud](real-task-pointcloud.png). Image detector đã build; wheel CUDA/checkpoint đã xác minh checksum. CUDA voxelization và nạp checkpoint CenterPoint thật thành công: [detector-runtime.json](detector-runtime.json). Native MMDetection3D loader chạy riêng **1 passed**: [native-loader-test-results.xml](native-loader-test-results.xml). Function Nuclio ready/healthy, thấy GPU, manifest, dataset và checkpoint mount. Model **CenterPoint 3D (nuScenes)** đã hiện trong Automatic annotation, đủ mapping 10 lớp/metadata, nút Annotate bật: [centerpoint-ready.png](centerpoint-ready.png). Chưa chạy Annotate. Export UI đã qua test giữ identity sau sửa/xóa/metadata bị xóa và từ chối journal sai/pending. Chi tiết tại [CVAT_BUTTON.md](../docs/CVAT_BUTTON.md).

**Chưa kiểm chứng:** forward inference và peak VRAM trên GPU 4GB, cuboid do detector tạo và alignment trong UI, inference 20 frame thật, thời gian/quality annotator. Người dùng yêu cầu tự nhấn Annotate nên chưa chạy inference thay người dùng. Tệp [synthetic smoke metrics](synthetic_smoke/metrics.json) và CSV đi kèm được tạo từ **một cuboid tổng hợp**, không phải nuScenes prediction/GT hoặc kết quả server.

**Cập nhật sau khi người dùng nhấn Annotate:** request bắt đầu 14:17 và đứng ở 0/20 frame. Worker bị treo trong `getaddrinfo('host.docker.internal')`, trạng thái D/wchan `rtnl_dumpit`; GPU 0%, chưa có shapes/journal. Docker/WSL không phục hồi tại chỗ, người dùng đã reboot. Script phục hồi đã chạy; request canceled được gỡ. Dashboard GET báo ready nhưng invoke vẫn dùng IP processor cũ sau restart. Bổ sung route Docker DNS riêng cho CenterPoint; gateway CVAT thật đã gọi được handler bằng payload trống trong 0,07 giây. Không chạy detector, task vẫn có 0 shapes và không còn request. [Bằng chứng đường gọi](cvat-route-recovery.json), [bằng chứng sự cố](annotation-stall.json), [ảnh menu](centerpoint-recovered.png). Test UI/journal/patch chạy lại: **21 passed**, [JUnit XML](routing-recovery-test-results.xml). Forward inference 20 frame vẫn cần người dùng nhấn Annotate.
