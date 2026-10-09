# Môi trường kiểm tra ngày 08/10/2026

Workspace `C:\Users\khoik\Documents\GitHub\pre-label` ban đầu trống, không có `.git` hoặc code/AGENTS.md áp dụng được tìm thấy. Source mới được xây trực tiếp trong thư mục này.

**Cập nhật sau reboot:** WSL 3.0.1 và Docker Linux 29.8.2 hoạt động; GPU RTX3050 4096 MiB nhận được trong container. CVAT API 2.20.0 tại localhost:8080 và Nuclio tại localhost:8070 hoạt động. NuScenes mini đã giải nén trong `data/nuscenes`, có 404 LiDAR keyframes và 3531 sweeps. Image detector đã build; CUDA voxelization và nạp checkpoint chính thức thành công. Function CenterPoint ready/healthy, thấy GPU và dữ liệu mount. Model đã hiện trong Automatic annotation của [task #2](http://localhost:8080/tasks/2), nút Annotate bật. Bảng bên dưới ghi tình trạng ban đầu; trạng thái triển khai mới nhất tại [local-cvat-setup.json](../artifacts/local-cvat-setup.json).

| Hạng mục | Kết quả đã quan sát |
|---|---|
| OS | Windows 11, NT 10.0.26200 |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | 4096 MiB |
| Driver | 566.07 |
| nvidia-smi CUDA label | 12.7; không phải CUDA Toolkit cài đặt |
| Python trên PATH / py launcher | Không tìm thấy |
| Python bundle dùng cho core tests | 3.12.14, Codex workspace runtime |
| nvcc / Docker CLI | Không tìm thấy |
| WSL | Có wsl.exe, thông báo WSL chưa cài |
| PyTorch/MMCV/MMEngine/MMDetection3D/devkit | Chưa cài trong runtime kiểm thử |
| nuScenes dataset | Chưa được cung cấp đường dẫn; NUSCENES_ROOT chưa cấu hình |
| Model config/checkpoint | Chưa có trong thư mục models |
| CVAT URL/auth/version thật | Chưa cấu hình, không gọi server |

Core dependencies cài riêng vào `.test-deps`: NumPy1.26.4, SciPy1.12.0, PyYAML6.0.2, requests2.32.3 và pytest8.3.5. Driver giữ nguyên; Docker Desktop và WSL đã được cài theo yêu cầu. Các thư viện detector nằm trong image Docker Python3.10. Snapshot core ban đầu tại `artifacts/environment.json`.

Kiểm thử offline đã kiểm chứng geometry, PCD binary, native loader transport formula, confidence filter, resume một model cho nhiều pending frame, CVAT REST giả lập, edit/identity và timer/evaluation. Bằng chứng pytest tại `artifacts/test-results.xml`; trạng thái hiện tại được ghi ở [ACCEPTANCE.md](ACCEPTANCE.md).

**Phần còn cần kiểm chứng:**

Sự cố WSL/worker treo ở 0% đã được phục hồi sau reboot. CenterPoint dùng Docker DNS thay vì host gateway/IP invocation lưu cũ; gateway CVAT thật đã đến handler trong phép thử payload trống. Task #2 giữ đủ 20 frame, annotation trống; xem [cvat-route-recovery.json](../artifacts/cvat-route-recovery.json). Chưa chạy forward inference.

1. Detector thật cần Linux/WSL2, Python3.10, torch/MMCV CUDA ops và đủ VRAM. 4 GB nhỏ hơn mức tham chiếu model zoo; chưa đo peak inference trên máy này.
2. Người dùng đã nhấn Annotate trên task #2; 20 frame thật hoàn tất, có 1.123 cuboid. Đã gắn 120 ảnh camera theo sample token và bổ sung checkbox hướng cuboid.
3. Chỉnh/duyệt chất lượng và thời gian cần thực nghiệm của người dùng; chưa có dữ liệu để báo chất lượng/thời gian thật.

Hướng dẫn gỡ từng chặn ở [INSTALL.md](INSTALL.md) và chuỗi smoke/batch ở [README](../README.md). Không coi các test tổng hợp là đã đạt 20 frame thật hay hiển thị đúng trong CVAT.
