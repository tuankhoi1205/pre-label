# CVAT 3D Pre-label Tool

Pipeline Python cho **nuScenes LiDAR → CenterPoint pretrained → cuboid chuẩn hóa → CVAT 3D → người dùng chỉnh/duyệt → export → đánh giá**. MVP dùng REST API; không cần CVAT SDK. Backend duy nhất đã triển khai là CenterPoint qua MMDetection3D 1.4.0, với config và checkpoint chính thức.

Chạy trực tiếp từ **Tasks → Actions → Automatic annotation → CenterPoint 3D (nuScenes) → Annotate** trên CVAT Docker local. Task mới có LiDAR và sáu ảnh camera theo từng sample nuScenes. Giao diện bổ sung **Appearance → Cuboid orientation** để hiện trục X đỏ, Y xanh lá, Z xanh dương; +X là hướng đầu box theo yaw của detector.

Bạn cũng có thể tự tạo task trên giao diện, upload ZIP chứa binary float32 XYZ/XYZI PCD cùng ảnh context, rồi nhấn **Annotate**. Function tự nhận diện PCD nuScenes bằng nội dung và chuẩn bị calibration/sweeps trong nền; không phải chạy prepare cho mỗi task. Máy chạy model cần giữ raw nuScenes, metadata và sweeps trong `data/nuscenes`. PCD đã lọc/đổi tọa độ hoặc từ dataset khác chưa được adapter này hỗ trợ. Xem [luồng upload trực tiếp](docs/CVAT_BUTTON.md).

**Đã chạy thật:** ngày 08/10/2026, người dùng nhấn Annotate cho 20 frame nuScenes mini trên RTX 3050 Laptop 4 GB; request hoàn tất trong khoảng 45,5 giây và Task #2 có 1.123 cuboid. Đã bổ sung 120 ảnh camera, kiểm tra provider CVAT cho đủ 20 frame và xác nhận annotation không thay đổi. Peak VRAM, chất lượng sau review và thời gian thao tác annotator chưa được đo. Xem [bằng chứng kiểm chứng](artifacts/VALIDATION.md). Số liệu trong `artifacts/synthetic_smoke/` là dữ liệu tổng hợp.

Repo chỉ chứa source, cấu hình, tài liệu và bằng chứng nhỏ. **Dataset, checkpoint, source CVAT tải về, môi trường Python, prediction/run và `.env` không được commit.** Người clone repo tải chúng theo hướng dẫn bên dưới; `.env.example` chỉ chứa mẫu cấu hình.

## Kiến trúc và thư mục

```text
configs/                    YAML cho CenterPoint voxel và pillar
src/prelabel/
  dataset.py                devkit, chọn scene/split, PCD, manifest, GT chỉ khi evaluate
  detector.py               adapter, pipeline checkpoint, sweeps, raw/normalized, resume
  schema.py / geometry.py   tâm hình học, LWH, quaternion, SE(3), CVAT Euler, IoU 3D
  cvat.py                   REST, task/data, mapping, append, journal, readback, export
  evaluation.py             ghép một-một, quality/edit metrics, CSV/JSON/Markdown
  timing.py                 start/pause/resume/stop, thời gian chủ động
  assets.py / doctor.py      model chính thức và kiểm tra môi trường
  cli.py                    CLI
  serverless.py / ui_task.py Nuclio detector và tạo task trước inference UI
deployment/                 CVAT 3D patch, Docker/Nuclio, script PowerShell local
scripts/                    cài Linux và smoke hình học CPU
tests/                      hình học, pipeline, REST giả lập, live test tùy chọn
docs/                       cài đặt, tọa độ, thực nghiệm, nguồn, nghiệm thu
artifacts/                  bằng chứng kiểm thử và smoke tổng hợp
runs/<run>/                 manifest, prediction, CVAT state, export, reports (gitignored)
models/ / data/             checkpoint/config và dataset (gitignored)
```

Các mốc triển khai: (1) kiểm tra môi trường/hợp đồng dữ liệu, (2) smoke một frame và xác nhận hình học trong CVAT, (3) batch 20 frame, (4) chỉnh/duyệt/export và thực nghiệm. Source đã có đường chạy cho các mốc; các bước phụ thuộc dịch vụ/dataset thật vẫn cần thực hiện.

## Cài đặt trên Windows với Docker Desktop

### 1. Chuẩn bị phần mềm

- Windows có WSL2 và [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/), chạy **Linux containers**. Không cần cài Ubuntu riêng khi dùng luồng Docker này.
- NVIDIA GPU và driver hỗ trợ GPU trong WSL2. Detector chạy trong container Linux, không cài MMCV/CUDA ops trực tiếp vào Python Windows.
- [Python 3.10–3.12](https://www.python.org/downloads/windows/) trên Windows, chọn **Add Python to PATH**. Python host chỉ dùng cho script chuẩn bị/build; container dùng Python 3.10 pin sẵn. Không cần Codex để chạy project.
- Docker Compose v2 đi kèm Docker Desktop. Máy đã kiểm chứng dùng Docker Linux được cấp khoảng 8 GB RAM; build lần đầu cần tải nhiều image/wheel và có thể mất vài phút hoặc lâu hơn tùy mạng.

Clone bằng GitHub Desktop, mở PowerShell **trong thư mục gốc repo** chứa `README.md`, `deployment/`, `configs/` và `src/`. Các lệnh dưới đây đều chạy tại đó.

```powershell
python --version
docker info --format '{{.OSType}}'
docker compose version
```

Docker phải trả về `linux`. Nếu vừa cài/bật WSL và Windows yêu cầu restart, khởi động lại rồi mở Docker Desktop trước khi tiếp tục. Không cần restart ở mỗi lần chạy project.

### 2. Tải source CVAT đúng phiên bản

Source upstream được tải vào `.local/` và được patch khi Start; không cần đưa toàn bộ CVAT vào repo này.

```powershell
New-Item -ItemType Directory -Path .local/downloads, .local/upstream -Force | Out-Null
curl.exe -fL --retry 5 https://codeload.github.com/cvat-ai/cvat/zip/refs/tags/v2.20.0 -o .local/downloads/cvat-v2.20.0.zip
Expand-Archive -LiteralPath .local/downloads/cvat-v2.20.0.zip -DestinationPath .local/upstream
Test-Path .local/upstream/cvat-2.20.0/docker-compose.yml
```

Kết quả cuối phải là `True`. Chỉ giải nén ở lần đầu; nếu source đã có thì bỏ qua bước này. Patch kiểm tra đúng CVAT **2.20.0**, bổ sung model 3D/nút hướng box và tuyến gọi CenterPoint bằng Docker DNS.

### 3. Tải nuScenes mini riêng

Đăng nhập [nuScenes download](https://www.nuscenes.org/download), tải **v1.0-mini.tgz**. Giữ file dataset ở máy của bạn; repo không phân phối lại dữ liệu.

```powershell
$archive = 'C:\duong-dan-toi-file-da-tai\v1.0-mini.tgz'
$target = Join-Path $PWD 'data/nuscenes'
New-Item -ItemType Directory -Path $target -Force | Out-Null
tar.exe -xf $archive -C $target
Test-Path (Join-Path $target 'v1.0-mini/sample.json')
Test-Path (Join-Path $target 'samples/CAM_FRONT')
```

Thay `$archive` bằng đường dẫn thật đến file đã tải. Hai kiểm tra cuối phải là `True`. Cấu trúc:

```text
data/nuscenes/
  v1.0-mini/       sample.json, sample_data.json, calibrated_sensor.json, ...
  samples/         LIDAR_TOP/, CAM_FRONT/, CAM_BACK/, và các camera còn lại
  sweeps/          LIDAR_TOP/
  maps/
```

Model cần raw LiDAR/intensity, sweeps và poses trên disk dù CVAT đã nhận PCD. Ảnh camera được ghép theo sample token; thời gian camera có thể lệch vài mili giây so với LiDAR và được lưu trong manifest context.

### 4. Khởi động CVAT và tạo tài khoản

```powershell
Copy-Item -LiteralPath .env.example -Destination .env
.\deployment\local.ps1 Start
.\deployment\local.ps1 Status
.\deployment\local.ps1 CreateUser
```

Chỉ copy `.env` khi chưa có file này. `CreateUser` nhập tài khoản/mật khẩu trong terminal. Mở [CVAT local](http://localhost:8080) và đăng nhập. Mở `.env` bằng editor, điền `CVAT_USERNAME` và `CVAT_PASSWORD` của tài khoản vừa tạo, hoặc `CVAT_TOKEN`. Giữ `CVAT_URL=http://localhost:8080`, để `CVAT_ORG` trống nếu dùng task cá nhân.

Container tự dùng `NUSCENES_ROOT=/workspace/data/nuscenes` và `CVAT_URL=http://cvat-server:8080`. Nếu chạy CLI trực tiếp trên Windows, đặt `NUSCENES_ROOT` trong `.env` thành đường dẫn tuyệt đối đến dataset trên Windows.

`Start` build giao diện rồi bật CVAT/Nuclio. Các cổng dùng: **8080** cho CVAT, **8070** cho Nuclio. Giữ source CVAT tại `.local/upstream/cvat-2.20.0` vì compose mount các file đã patch.

### 5. Build runtime, tải model và tạo task

```powershell
.\deployment\local.ps1 BuildTool
.\deployment\local.ps1 Tool fetch-model
.\deployment\local.ps1 Tool doctor --check-cvat
.\deployment\local.ps1 Tool --run-dir runs/ui-centerpoint prepare --limit 20
.\deployment\local.ps1 Tool --run-dir runs/ui-centerpoint create-task
.\deployment\deploy-function.ps1 -RunDir runs/ui-centerpoint
```

`BuildTool` tải và xác minh wheel CUDA chính thức rồi build image detector. `fetch-model` tải source/config MMDetection3D **v1.4.0** cùng checkpoint CenterPoint từ OpenMMLab vào `models/`. `prepare` chọn 20 frame của `mini_val`, tạo PCD/manifest/sweeps. `create-task` upload PCD và sáu ảnh camera/frame, tự tạo 10 label cùng metadata, rồi in URL task; command này chưa chạy inference.

`deploy-function` build và đăng ký processor GPU với Nuclio. Khi hoàn tất, model phải xuất hiện trong Automatic annotation. Runtime đã pin: Python **3.10**, torch **1.13.1+cu117**, torchvision **0.14.1+cu117**, MMCV **2.1.0**, MMEngine **0.10.5**, MMDetection **3.2.0**, MMDetection3D **1.4.0**, nuScenes-devkit **1.1.11**. Không thay riêng một thư viện nếu chưa kiểm tra tương thích CUDA ops.

### 6. Pre-label và kiểm định trên giao diện

1. Mở URL task do `create-task` in ra; task ID trên mỗi máy có thể khác nhau.
2. Chọn **Actions → Automatic annotation → CenterPoint 3D (nuScenes)**, kiểm tra mapping label rồi nhấn **Annotate**.
3. Chờ request hoàn tất, mở Job ở **Standard 3D**.
4. Bật **Appearance → Cuboid orientation**. Mũi tên **đỏ +X** chỉ hướng đầu box, **xanh lá +Y** là chiều rộng, **xanh dương +Z** là chiều cao. Chọn box để xem các hình chiếu Top/Side/Front và chỉnh yaw bằng handle xoay.
5. Xem ảnh camera tương ứng bên cạnh LiDAR; dùng nút **+** của bố cục để thêm góc camera cần xem. Ảnh context là ảnh gốc để đối chiếu, chưa có phép chiếu cuboid 3D lên ảnh 2D.
6. Kiểm tra lớp, vật thể thừa/bỏ sót, tâm, kích thước, chiều cao và hướng; chỉnh/xóa/thêm cuboid rồi **Save**.

Mũi tên thể hiện **hướng model dự đoán**, cần đối chiếu ảnh để xác nhận đầu–đuôi. Không dùng confidence cao thay cho review. Ground truth nuScenes chỉ được đọc ở bước `evaluate`, không tạo pre-label từ GT.

### 7. Export, đánh giá và dừng

Sau khi đã Save trên CVAT:

```powershell
.\deployment\export-ui.ps1 -RunDir runs/ui-centerpoint
.\deployment\local.ps1 Tool --run-dir runs/ui-centerpoint evaluate
.\deployment\local.ps1 Stop
```

Export đọc journal và annotation từ CVAT, giữ liên kết prediction/ID ban đầu kể cả box đã bị sửa hoặc xóa. Xem `runs/ui-centerpoint/reports/summary.md`, `metrics.json`, CSV và `reviewed.json`. Báo cáo hình học của project không phải mAP/NDS chính thức nuScenes.

`Stop` giữ volumes, tasks và nhãn. Bật lại bằng `Start`. Backup cả Docker volumes và `runs/` nếu cần chuyển annotation sang máy khác; GitHub chỉ lưu source.

### Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `No models available` | Kiểm tra `deploy-function` đã hoàn tất và Nuclio [localhost:8070](http://localhost:8070) báo ready; tải lại trang CVAT rồi mở modal |
| Task có LiDAR nhưng không có camera | Task tạo trước khi bổ sung context cần gắn camera theo [hướng dẫn CVAT](docs/CVAT_BUTTON.md); task mới từ `create-task` tự có camera |
| Không thấy `Cuboid orientation` | Chạy `local.ps1 Start` để patch/build UI, rồi tải lại trang; chỉ hiển thị trong workspace 3D |
| Dataset/config/checkpoint thiếu | Kiểm tra cấu trúc giải nén, mount `data/nuscenes`, và hoàn tất `fetch-model` |
| `400` khi Annotate task tự upload | Bản cập nhật tự nhận diện binary XYZ/XYZI của nuScenes; deploy lại function nếu đang dùng bản cũ. Nếu lỗi nói PCD không khớp, kiểm tra raw dataset và giữ nguyên tọa độ/thứ tự điểm/intensity |
| CUDA out of memory | Đóng process dùng GPU khác; thử run mới một frame. Không tự giảm sweeps hay đổi checkpoint của run đã có. Peak VRAM cần kiểm tra trên từng máy |
| Docker/WSL unresponsive hoặc request treo sau reboot | Khởi động lại WSL/Windows theo thông báo Docker, mở Docker Desktop rồi chạy `deployment/recover-prelabel.ps1`; không xóa volumes/journal |
| Đổi model/config/threshold/frame selection | Dùng **run/task mới** và deploy function cho run đó; không ghi đè run cũ |

Recovery script hiện được viết cho local integration và có guard gỡ request canceled của Task #2 trên máy đã kiểm chứng; không dùng nó để hủy request khác. Xem [CVAT_BUTTON.md](docs/CVAT_BUTTON.md) để biết cơ chế journal/rerun và lỗi transport đã xử lý.

## Đưa source lên GitHub bằng GitHub Desktop

1. Mở **File → Add local repository…**, chọn thư mục gốc project `pre-label`.
2. Nếu chưa có `.git`, dùng liên kết **create a repository** trong thông báo. Đặt Name là `pre-label`; Local path là **thư mục cha** chứa `pre-label`, tránh tạo thêm `pre-label/pre-label`. Không tạo README hoặc `.gitignore` mới vì repo đã có hai file này.
3. Trong **Changes**, kiểm tra source/docs/configs/deployment được chọn; `.env`, `data/`, `models/`, `runs/`, `.local/`, `.venv/`, `.test-deps/` không xuất hiện. `.env.example` được commit vì chỉ là mẫu.
4. Nhập Summary, ví dụ `Initial commit: CVAT CenterPoint 3D pre-label`, rồi nhấn **Commit to main** (hoặc tên branch hiện tại).
5. Nhấn **Publish repository**, chọn tài khoản và quyền public/private, rồi **Publish repository**. Dataset/model vẫn chỉ nằm trên máy và không được gửi lên GitHub.
6. Những lần sau: chỉnh code → **Commit to main** → **Push origin**.

Hướng dẫn chính thức: [tạo/commit/publish với GitHub Desktop](https://docs.github.com/en/desktop/overview/creating-your-first-repository-using-github-desktop). Khi clone sang máy khác, thực hiện lại các bước tải CVAT, nuScenes và checkpoint trong README; annotation trên Docker volumes không nằm trong Git.

## Cài CLI trực tiếp trên Linux/WSL2

Detector dùng Ubuntu/WSL2 và Python 3.10. Hướng dẫn đầy đủ, bảng phiên bản pin và CVAT server: [docs/INSTALL.md](docs/INSTALL.md).

```bash
# Sau khi đã có Ubuntu/WSL2, Python 3.10 và NVIDIA CUDA hoạt động:
bash scripts/install_linux.sh
source .venv/bin/activate
cp .env.example .env
# Điền NUSCENES_ROOT, CVAT_URL và thông tin đăng nhập vào .env.
prelabel doctor --check-cvat
prelabel fetch-model
```

`fetch-model` tải source MMDetection3D tag `v1.4.0`, bao gồm các base config, và checkpoint từ OpenMMLab. Kiểm tra SHA-256 prefix trong tên checkpoint chính thức và lưu full hash trong provenance. Chỉ dùng checkpoint đáng tin cậy vì PyTorch 1.13 sử dụng pickle khi load.

Để kiểm thử core trên máy không có GPU:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux: source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest -q
python scripts/synthetic_smoke.py
```

Muốn `prepare`/`evaluate` mà chưa cài detector: dùng **Python 3.10**, rồi `python -m pip install -e ".[dataset,test]"`. Devkit pin Matplotlib cũ, không dùng Python 3.12 cho extra dataset. Core hỗ trợ Python 3.10–3.12; môi trường dataset/detector pin Python 3.10. Lệnh `python -m prelabel` tương đương `prelabel`.

## Một frame trước khi chạy batch

Tất cả đường dẫn YAML tương đối được tính từ `project_root`; đường dẫn dataset/model tuyệt đối và `${ENV_VAR}` cũng được hỗ trợ. `--config` và `--run-dir` đứng **trước** tên command. `scenes` dùng tên scene thuộc split đã chọn; `max_frames` là tổng số frame trên các scene theo thứ tự tên scene/thời gian.

```bash
prelabel --run-dir runs/smoke prepare --limit 1
prelabel --run-dir runs/smoke infer
prelabel --run-dir runs/smoke upload --dry-run
# Kiểm tra runs/smoke/dry_run.json; offline IDs âm là placeholder.
prelabel --run-dir runs/smoke upload
```

Command cuối in link `/tasks/<id>`. Mở task/job trong CVAT và chọn **Shape**, không dùng Track trong MVP. Xem point cloud và cuboid ở phối cảnh/top/side/front; kiểm tra chiều dài/rộng/cao, hướng yaw, tâm và chiều cao so với mặt đường. Tạo một cuboid kích thước bất đối xứng để đối chiếu khi cần. **Readback API chỉ xác nhận dữ liệu server; vẫn phải xem UI để nghiệm thu hiển thị.**

Người dùng sửa vị trí/kích thước/hướng/label, xóa một box, vẽ thêm một box và Save. Sau đó:

```bash
prelabel --run-dir runs/smoke export
prelabel --run-dir runs/smoke evaluate
prelabel --run-dir runs/smoke upload
# Rerun giữ các chỉnh sửa và không dựng lại box đã xóa.
```

Sau khi kiểm tra hiển thị đúng, dùng một run/task mới cho batch, với `configs/mini.yaml` mặc định 20 frame:

```bash
prelabel prepare
prelabel infer
prelabel upload --dry-run
prelabel upload
# Chỉnh, duyệt chất lượng và Save trong CVAT.
prelabel export
prelabel evaluate
```

Nếu thiếu dataset/checkpoint/CUDA/CVAT, command sẽ báo lỗi cụ thể; không tự tạo prediction giả. Core smoke không thể thay thế bước inference thật.

## GPU 4 GB và phương án thay thế

Default CenterPoint voxel 0.1 có mức bộ nhớ tham chiếu **5.2 GB** trong model zoo; pillar 0.2 là **4.6 GB**. Đây là số liệu tham chiếu, không phải peak inference trên máy này. Batch=1 có thể khác; không đảm bảo chạy được trên 4 GB. Khuyến nghị một GPU Linux có đủ bộ nhớ, thường từ 8 GB cho thử nghiệm này, nhưng vẫn phải đo thực tế.

Profile pillar vẫn dùng **CenterPoint**, dùng cùng adapter và 10 sweeps:

```bash
prelabel --config configs/mini-pillar.yaml fetch-model
prelabel --config configs/mini-pillar.yaml --run-dir runs/pillar-smoke prepare --limit 1
prelabel --config configs/mini-pillar.yaml --run-dir runs/pillar-smoke infer
```

Không giảm số sweeps hay đổi checkpoint ngầm khi OOM. PointPillars thuần/OpenPCDet chưa được triển khai; đó là adapter tiếp theo nếu cần, với hợp đồng origin/dims/yaw/features riêng. Đổi backend không tự giải quyết việc thiếu Linux/CUDA hay VRAM. Xem [docs/INSTALL.md](docs/INSTALL.md).

## Mapping, resume và bảo toàn chỉnh sửa

- `manifest.json` ánh xạ sample token, LiDAR sample_data token, BIN, PCD, frame nội bộ, poses/sweeps và hash. CVAT frame thực tế được lấy theo **tên file trong `/data/meta`**, không giả định thứ tự upload.
- `infer` load model một lần cho các frame còn thiếu, giữ pipeline/NMS của checkpoint, lọc confidence theo class, ghi raw output và normalized boxes. Đổi checkpoint/config/threshold/đầu vào yêu cầu run mới.
- `baseline.json` giữ prediction ban đầu; `cvat_state.json` giữ task, mapping, pending transaction và annotation ID. Hãy backup cả run khi chuyển máy hoặc chỉnh sửa lâu dài.
- `upload` chỉ PATCH `action=create` cho box chưa áp dụng. Không PUT toàn bộ annotation. IDs đã áp dụng vẫn được ghi nhận sau khi box bị người dùng xóa.
- Task mới tự tạo 10 label và text attributes `prediction_id`, `confidence`, `model`, `checkpoint`, `run_id`. `cvat.task_id` và `label_mapping` dùng task đã có dữ liệu; task đó cần cùng PCD và các attributes này. Thiếu label/attribute sẽ báo lỗi thay vì sửa schema của task đang dùng.
- `upload --dry-run` là offline, không gửi request. `--dry-run --check-server` kiểm tra task đã tồn tại và dùng ID thật; cần `task_id` hoặc saved state, và chỉ thực hiện read-only API sau login.
- Trường hợp phản hồi PATCH bị mất: rerun nhận dạng box đã commit bằng `prediction_id`. Nếu không biết box chưa commit hay đã bị người dùng xóa, tool lưu `upload_uncertain.json` và dừng việc ghi; xem [nghiệm thu/khôi phục](docs/ACCEPTANCE.md).
- Không chạy đồng thời nhiều uploader cho cùng run/task. Lock bảo vệ process trên một run directory; không phải distributed lock cho nhiều máy.

`export` đọc REST JSON trực tiếp để giữ annotation IDs. Sau export/import qua định dạng khác, chạy `export --ids-recreated`; chỉ `prediction_id` còn nguyên mới khôi phục được identity. Mất cả ID lẫn attribute thì báo cáo chỉnh sửa là **unavailable**, vẫn tính chất lượng hình học. Không suy đoán ID bằng IoU. Khi copy một cuboid, xóa metadata `prediction_id`/`run_id` khỏi bản copy nếu đó là box thêm mới; identity trùng sẽ bị từ chối.

MVP chỉ export cuboid dạng Shape. Tracks/tags và shape khác được giữ trong `cvat_export_raw.json` nhưng export chuẩn hóa sẽ báo unsupported, tránh bỏ mất nhãn âm thầm. `confidence/model/checkpoint` cũng có bản sao trong baseline để không phụ thuộc việc người dùng sửa attributes.

## Báo cáo và thời gian

Sau `evaluate`, xem:

```text
runs/<run>/reports/
  metrics.json              trước/sau, per-class, edit metrics, timing và policy
  quality_summary.csv       số frame/GT/ghép/thừa/bỏ sót/precision/recall/mean IoU
  matched_pairs.csv         từng cặp prediction–GT và rotated 3D IoU
  edits.csv                 giữ nguyên/sửa/xóa và độ thay đổi hình học
  summary.md                bảng tổng hợp
  annotation_sessions.csv   khi đã đo thời gian chủ động
```

IoU là giao thể tích của hai cuboid xoay 3D, hỗ trợ cả roll/pitch. Ghép tối đa số cặp hợp lệ theo frame/class, rồi tối đa tổng IoU; threshold mặc định 0.5. Mean IoU chỉ tính trên cặp ghép và luôn đi kèm TP/FP/FN, precision/recall và số mẫu. Đây là đánh giá hình học của dự án; **không phải mAP/NDS chính thức nuScenes**.

Ground truth chỉ được đọc trong `evaluate`. Dùng mini để xác minh pipeline, không coi nó là tập benchmark độc lập cho checkpoint huấn luyện nuScenes. Cách lọc GT, tolerance và thiết kế thí nghiệm: [docs/EVALUATION.md](docs/EVALUATION.md).

Ví dụ đo thao tác gán nhãn, dừng timer trong lúc nghỉ/chờ xử lý:

```bash
prelabel timer start --session manual-A --workflow manual --participant annotator1 --difficulty medium --quality-standard protocol-v1 --frame-ids 0,1
prelabel timer pause --session manual-A
prelabel timer resume --session manual-A
prelabel timer stop --session manual-A --cuboids 12

prelabel timer start --session assisted-B --workflow assisted --participant annotator1 --difficulty medium --quality-standard protocol-v1 --frame-ids 2,3
prelabel timer stop --session assisted-B --cuboids 11
prelabel report-time
```

Manual frames phải gán từ đầu ở task sạch, không xem pre-label/GT. Dùng tập frame khác có độ khó tương đương; không dùng lại frame đã nhớ. Timer dùng chung manifest của thí nghiệm để ghi frame IDs; việc phân công task sạch và review chất lượng do người tổ chức thực hiện. Không đo thì báo `not_measured`, không điền số tiết kiệm dự kiến.

## Kiểm thử

```bash
python -m pytest -q
# Trong môi trường detector đã pin: native LoadPointsFromMultiSweeps conformance.
python -m pytest -q -m detector
# Live REST test: tạo task với PCD/cuboid tổng hợp, không phải detector thật.
export PRELABEL_LIVE_CVAT=1
export CVAT_KEEP_TEST_TASK=1  # Giữ task để xem hình học; nếu bỏ biến này, test xóa task do nó tạo.
python -m pytest -s -q -m live
```

Hợp đồng tọa độ và các đoạn source đã kiểm chứng: [docs/COORDINATES.md](docs/COORDINATES.md), [docs/SOURCES.md](docs/SOURCES.md).
