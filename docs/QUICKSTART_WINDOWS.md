# Bắt đầu trên máy Windows hiện tại

**Cập nhật triển khai:** Docker/WSL và CVAT 2.20.0 đã hoạt động sau reboot, nuScenes mini và checkpoint đã có. [Task #2](http://localhost:8080/tasks/2) chứa 20 frame LiDAR thật, chưa có annotation. Dùng [hướng dẫn nút Automatic annotation](CVAT_BUTTON.md) cho luồng hiện tại. Phần dưới lưu tình trạng kiểm tra ban đầu và cách cài từ đầu; không cần cài/reboot lại.

Kiểm tra lại ngày 08/10/2026: Windows11, RTX3050 Laptop 4096MiB, RAM khoảng15.4GiB, ổ C còn khoảng273GiB. WSL chưa cài; không tìm thấy Python/Docker/nvcc trên PATH. Python3.12 bundle chỉ có core dependencies; chưa có devkit/PyTorch/MMCV/MMDetection3D. Chưa có `.env`, models hoặc dataset cấu hình; người dùng xác nhận chưa có CVAT và nuScenes mini. `http://localhost:8080/api/server/about` chưa truy cập được. Snapshot: `artifacts/recheck/environment.json`.

**Chưa đủ môi trường để chạy pipeline thật.** Có thể chuẩn bị CVAT và thử gán nhãn PCD trước; inference trên GPU4GB phải thử batch1 sau cài đặt, chưa đảm bảo đủ VRAM. Nếu OOM, dùng GPU khác có đủ bộ nhớ; không tự giảm số sweeps của checkpoint.

Luồng CLI bên dưới chạy tool Python ngoài CVAT và gửi cuboid qua API. Theo lựa chọn mới của bạn, đã bổ sung tích hợp Nuclio + patch CVAT 3D để dùng nút **Automatic annotation → Annotate**; xem [CVAT_BUTTON.md](CVAT_BUTTON.md). WSL3.0.1 đã cài/bật features và cần khởi động lại Windows. Docker Desktop4.94.0 có ở thư mục user; engine chưa hoạt động. Snapshot đầu trang là trước bước cài này.

## 1. WSL2 và Docker

Mở **PowerShell Run as administrator**:

```powershell
wsl --install -d Ubuntu-22.04
```

Khởi động lại khi Windows yêu cầu, mở Ubuntu và tạo tài khoản Linux. Kiểm tra từ PowerShell:

```powershell
wsl --list --verbose
```

Ubuntu cần chạy WSL version2. Quy trình chính thức: [Microsoft WSL installation](https://learn.microsoft.com/en-us/windows/wsl/install).

Cài và mở [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/), dùng WSL2/Linux containers; bật integration cho Ubuntu trong Settings → Resources → WSL Integration. Sau đó mở terminal Ubuntu:

```bash
docker version
docker compose version
nvidia-smi
sudo apt update
sudo apt install -y git python3.10 python3.10-venv
```

`docker version` phải có phần Server; `nvidia-smi` phải thấy GPU trong WSL trước inference. Nếu lệnh nào lỗi, xử lý bước đó trước khi chạy script detector.

## 2. Chạy CVAT riêng

Trong Ubuntu, dùng thư mục riêng cho server:

```bash
mkdir -p ~/projects
cd ~/projects
git clone --branch v2.20.0 --depth 1 https://github.com/cvat-ai/cvat.git cvat-server
cd cvat-server
export CVAT_VERSION=v2.20.0
docker compose up -d
docker compose ps
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Chờ services khởi động xong; nếu tạo tài khoản lúc server chưa ready bị lỗi, kiểm tra `docker compose logs --tail=50 cvat_server` rồi chạy lại lệnh createsuperuser. Mở [CVAT local](http://localhost:8080) trên Windows và đăng nhập.

Giữ version2.20.0 vì REST adapter hiện audit đúng release đó. [Compose upstream của tag](https://github.com/cvat-ai/cvat/blob/v2.20.0/docker-compose.yml) có image `cvat/server:${CVAT_VERSION:-v2.20.0}` và UI tương ứng. Đây là hướng dẫn cài, chưa thực hiện trên máy này.

**Có thể thử CVAT ngay khi server chạy**, chưa cần dataset hoặc detector: tạo task mới, label `car`, upload tệp `artifacts/synthetic_smoke/synthetic.pcd` trong project, Submit, mở Job và vẽ cuboid bằng Shape. Đây chỉ là point cloud tổng hợp. Chỉnh/Save để kiểm tra UI3D hoạt động.

## 3. Dataset và môi trường tool

Tải **nuScenes v1.0-mini** từ [nuScenes](https://www.nuscenes.org/nuscenes), giải nén vào ví dụ:

```text
C:\Users\khoik\Documents\GitHub\pre-label\data\nuscenes\
  v1.0-mini\sample.json, sample_data.json, ...
  samples\LIDAR_TOP\...
  sweeps\LIDAR_TOP\...
  maps\...
```

Trong Ubuntu:

```bash
cd /mnt/c/Users/khoik/Documents/GitHub/pre-label
bash scripts/install_linux.sh
source .venv/bin/activate
cp -n .env.example .env
```

Điền `.env` bằng editor của bạn, dùng **đường dẫn Linux khi chạy trong WSL**:

```dotenv
NUSCENES_ROOT=/mnt/c/Users/khoik/Documents/GitHub/pre-label/data/nuscenes
CVAT_URL=http://localhost:8080
CVAT_USERNAME=ten_tai_khoan_cvat
CVAT_PASSWORD=mat_khau_cvat
CVAT_TOKEN=
```

Nếu Docker/CVAT đặt ở máy khác, CVAT_URL là địa chỉ server đó. Không gửi mật khẩu vào chat. Sau đó:

```bash
prelabel doctor --check-cvat
prelabel fetch-model
```

Script pin Python3.10/torch1.13.1+cu117/MMCV2.1/MMEngine0.10.5/MMDetection3.2/MMDetection3D1.4; xem [INSTALL](INSTALL.md). `doctor` xác nhận môi trường và API version; một frame inference thật mới xác nhận được backend CUDA/VRAM.

## 4. Một frame → CVAT

Trong thư mục project, venv đã active:

```bash
prelabel --run-dir runs/smoke prepare --limit 1
prelabel --run-dir runs/smoke infer
prelabel --run-dir runs/smoke upload --dry-run
prelabel --run-dir runs/smoke upload
```

Tool tự tạo task3D, upload PCD, chờ server xử lý, lấy mapping frame, tạo label và cuboid; command cuối in `task_url`. Không cần tạo task thủ công cho luồng này.

Mở link → chọn Job → Start/Open. Cuboid nháp đã có. Xem top/side/front để kiểm tra hướng, tâm, L/W/H; chỉnh box, xóa box sai, vẽ box thiếu bằng **Draw new cuboid → label → Shape**, rồi **Save**. Quy trình thao tác: [CVAT3D annotation](https://docs.cvat.ai/docs/annotation/manual-annotation/modes/3d-object-annotation/).

Nếu `infer` báo CUDA out of memory thì chưa có predictions để upload. Profile pillar (`configs/mini-pillar.yaml`) là phương án nhẹ hơn được hỗ trợ, nhưng mức bộ nhớ tham chiếu4.6GB vẫn không đảm bảo chạy trên4GB; cần thử ở run mới hoặc chuyển detector sang GPU đủ VRAM.

## 5. Export và batch

Sau Save trong CVAT:

```bash
prelabel --run-dir runs/smoke export
prelabel --run-dir runs/smoke evaluate
```

Kết quả ở `runs/smoke/reviewed.json` và `runs/smoke/reports/`. Ground truth chỉ dùng evaluate. Khi một frame đã hiển thị đúng, dùng run mặc định mới cho20frame:

```bash
prelabel prepare
prelabel infer
prelabel upload
# Chỉnh/duyệt và Save trong CVAT.
prelabel export
prelabel evaluate
```

Giữ `runs/smoke` và run batch riêng để mapping/annotation ID không bị đổi. Chạy lại upload cùng run sẽ bảo toàn sửa/xóa của người dùng và tránh tạo prediction trùng.
