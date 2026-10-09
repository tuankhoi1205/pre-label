# CenterPoint từ nút Automatic annotation của CVAT local

Luồng bạn chọn: **Tasks → Actions → Automatic annotation → CenterPoint 3D (nuScenes) → Annotate**. Sau khi request hoàn tất, mở Job để chỉnh cuboid và Save. Không phải import annotation thủ công; inference được gọi từ nút có sẵn.

Triển khai gồm CVAT 2.20.0 + UI đã patch + Nuclio 1.13.0 + function CenterPoint GPU. Source CVAT đã tải tại `.local/upstream/cvat-2.20.0`. Windows dùng dashboard API Nuclio để triển khai; không cần nuctl vì local platform của CLI này gọi `/bin/sh`.

**Trạng thái kiểm chứng:** người dùng đã chạy CenterPoint cho 20 frame thật trên RTX3050 Laptop 4GB; request hoàn tất trong khoảng 45,5 giây và [task #2](http://localhost:8080/tasks/2) có 1.123 cuboid. Task đã được gắn 120 ảnh camera và annotation được giữ nguyên. UI local có **Appearance → Cuboid orientation**. 41 test context/REST/UI/journal đạt; webpack build thành công. Peak VRAM và chất lượng sau review chưa được đo.

## Ảnh camera và hướng box khi review

Task mới tạo bằng `create-task` tự nạp sáu camera của đúng sample nuScenes. Bật **Appearance → Cuboid orientation**: X đỏ là hướng đầu box theo yaw, Y xanh lá là trục chiều rộng, Z xanh dương là trục chiều cao. Chọn box để xem Top/Side/Front. Ảnh camera là ảnh gốc làm context, chưa tự chiếu box 3D lên ảnh 2D.

Với task LiDAR cũ chưa có camera, cài core CLI trên host rồi chạy lệnh maintenance dưới đây từ thư mục repo. Script xác minh tên PCD/frame, hash ảnh và snapshot annotation; chỉ bổ sung RelatedFile/context của CVAT 2.20, không chạy detector hay ghi lại box. Thay task ID/run tương ứng của bạn:

```powershell
python -m pip install -e .
$dockerPath = (Get-Command docker).Source
python deployment/attach-context-images.py --docker $dockerPath --task-id 2 --run-dir runs/ui-centerpoint --dataset-root data/nuscenes
```

Lần gắn camera trên máy này đã kiểm tra 6 JPEG/frame cho đủ 20 frame; SHA256 annotation trước/sau giống nhau. Bằng chứng nằm trong `runs/ui-centerpoint/context/task-2-attachment.json` (run được gitignore).

## 1. Bật Docker Linux

Docker Desktop 4.94.0 được tìm thấy ở `%LOCALAPPDATA%\Programs\DockerDesktop`; CLI nằm trong `resources\bin\docker.exe`. Script local tự tìm vị trí này, không cần Python/Docker trên PATH.

WSL ban đầu báo `Wsl/CallMsi/Install/REGDB_E_CLASSNOTREG`. Đã cài MSI Microsoft thành công, phiên bản3.0.1.0 và kernel6.18.40.1; bật Windows Subsystem for Linux + VirtualMachinePlatform. Người dùng đã reboot, Docker Linux hiện hoạt động; không cần reboot lần nữa. Kết quả lưu trong `.local/downloads/wsl-setup-result.json` và `artifacts/local-cvat-setup.json`.

Nếu cần chạy lại bước cài sau này, mở PowerShell **Run as Administrator**, vào project và chạy:

```powershell
cd C:\Users\khoik\Documents\GitHub\pre-label
.\deployment\setup-wsl.ps1
```

Script xác minh chữ ký Microsoft của MSI, cài WSL, bật hai Windows features cần thiết, ghi log và **không tự khởi động lại**. Khởi động lại nếu Windows yêu cầu, sau đó mở Docker Desktop và dùng WSL2/Linux containers. Hoàn tất màn hình khởi tạo Docker nếu được hiển thị. Không cần Ubuntu riêng để dùng các script PowerShell này.

Kiểm tra:

```powershell
wsl --version
.\deployment\local.ps1 Status
```

`Status` cần Docker engine chạy; danh sách trống trước khi bật CVAT là bình thường. WSL/MSI chính thức: [Microsoft WSL releases](https://github.com/microsoft/WSL/releases). Yêu cầu Docker/WSL: [Docker Windows installation](https://docs.docker.com/desktop/setup/install/windows-install/).

## 2. Bật CVAT và Nuclio

```powershell
.\deployment\local.ps1 Start
.\deployment\local.ps1 Status
.\deployment\local.ps1 CreateUser
```

Start áp dụng patch kiểm tra đúng CVAT 2.20.0, build image UI riêng rồi khởi chạy compose upstream + serverless + overlay. Lần đầu tải image/build có thể mất thời gian. Máy này đã có tài khoản `admin`, mật khẩu ngẫu nhiên lưu trong `.env`; không cần chạy CreateUser lần nữa. CreateUser dùng cho tài khoản mới và nhập username/email/password trực tiếp trong terminal.

Overlay dùng Traefik 2.11.58 để hỗ trợ Docker 29, và chuẩn bị Alpine 3.17 chính thức dưới tag mirror đã ngừng hoạt động mà Nuclio 1.13 hardcode. Analytics mặc định tắt để giảm RAM. Nguồn: [Traefik release](https://github.com/traefik/traefik/releases/tag/v2.11.58), [Nuclio local platform](https://github.com/nuclio/nuclio/blob/1.13.0/pkg/platform/local/platform.go).

Mở [CVAT local](http://localhost:8080), đăng nhập. [Nuclio local](http://localhost:8070) dùng xem build/log của function. Việc server chạy chưa có nghĩa CenterPoint đã được đăng ký: cần hoàn tất bước dataset/model bên dưới.

## 3. Chuẩn bị model và PCD cho 20 frame

Tải **nuScenes v1.0-mini** từ [nuScenes download](https://www.nuscenes.org/download), giải nén vào `data/nuscenes/` trong project. Cấu trúc phải có `v1.0-mini/`, `samples/`, `sweeps/`, `maps/`. Trang chính thức yêu cầu tài khoản và đồng ý điều khoản dataset; thao tác này cần bạn thực hiện trong tài khoản của mình. [Hướng dẫn devkit](https://github.com/nutonomy/nuscenes-devkit#nuscenes-setup).

Tạo `.env` từ `.env.example`, điền CVAT_USERNAME/CVAT_PASSWORD hoặc CVAT_TOKEN bằng editor; không đưa mật khẩu vào chat. Container tự đặt NUSCENES_ROOT và CVAT_URL cho Docker network, nên không cần đổi đường dẫn Windows sang Linux trong `.env`.

```powershell
.\deployment\local.ps1 BuildTool
.\deployment\local.ps1 Tool fetch-model
.\deployment\local.ps1 Tool --run-dir runs/ui-centerpoint prepare --limit 20
.\deployment\local.ps1 Tool --run-dir runs/ui-centerpoint create-task
.\deployment\deploy-function.ps1
```

`create-task` chỉ tạo task/label có metadata và đưa PCD vào CVAT, **chưa chạy inference**. Nó in URL task. `deploy-function` dùng [dashboard API chính thức](https://docs.nuclio.io/en/1.13.x/reference/api/README.html) để build processor trên image Python/CUDA pin sẵn, mount project và đăng ký model. Dashboard dùng base image local với `NUCLIO_DASHBOARD_NO_PULL_BASE_IMAGES=true`. Khi dataset còn đang tải, có thể đăng ký trước bằng `.\deployment\deploy-function.ps1 -RegisterOnly`. Khi bấm Annotate, function mới kiểm tra manifest/input và load model GPU; các request tiếp theo dùng lại model. Nếu chưa prepare sẽ báo lỗi rõ. Dữ liệu raw và calibrated sweeps vẫn có sẵn trong mount.

Nếu bạn cần tự tải lại source CVAT:

```powershell
curl.exe -fL https://codeload.github.com/cvat-ai/cvat/zip/refs/tags/v2.20.0 -o .local/downloads/cvat-v2.20.0.zip
Expand-Archive -LiteralPath .local/downloads/cvat-v2.20.0.zip -DestinationPath .local/upstream
```

## 4. Nhấn nút pre-label

1. Mở [task #2 đã tạo](http://localhost:8080/tasks/2), hoặc task mà create-task in ra nếu triển khai lại. Nếu vẫn thấy **No models available**, đóng modal, tải lại trang bằng Ctrl+R rồi mở lại menu; danh sách model được tải khi trang khởi tạo.
2. Mở **Actions → Automatic annotation**.
3. Chọn **CenterPoint 3D (nuScenes)** ở Model.
4. Kiểm tra mapping label và metadata: `prediction_id`, `confidence`, `model`, `checkpoint`, `run_id`. Các tên khớp sẽ tự map.
5. Nhấn **Annotate**, chờ progress hoàn tất. Lần đầu gồm thời gian load checkpoint.
6. Mở Job; kiểm tra box trong top/side/front, chỉnh/xóa/thêm cuboid và **Save**.

Bạn có thể tạo task bằng giao diện, upload ZIP chứa PCD và ảnh context, rồi nhấn **Annotate** trực tiếp. Với PCD nuScenes chưa nằm trong manifest demo, function tự đối chiếu toàn bộ XYZ/intensity với keyframe trong dataset đã mount, lấy calibration và tối đa 9 sweeps trước đó, rồi lưu manifest/cache riêng trong `runs/ui-centerpoint/uploads/`. Không cần chạy lệnh prepare cho từng task. Tên file PCD có thể tùy ý; file phải là binary float32 XYZ hoặc XYZI, giữ nguyên tọa độ và thứ tự điểm. Cả mini_train và mini_val đều được nhận diện. Model GPU được dùng chung, không nạp thêm một model cho mỗi frame. Task có label Cuboid/Any sẽ được bổ sung năm text attributes provenance nếu thiếu, nên bạn chỉ cần map class trên modal.

Raw nuScenes, metadata và sweeps vẫn phải có trong `data/nuscenes` của máy chạy model. PCD từ dataset khác, PCD đã đổi hệ tọa độ/lọc điểm, ASCII hoặc binary_compressed chưa được adapter này hỗ trợ; lỗi model được đưa vào thông báo CVAT. Inference dùng raw LiDAR có intensity và sweeps, không dùng XYZ preview để thay thế input model. Confidence/class thresholds lấy từ YAML. Các manifest tự động lưu theo từng PCD; CVAT vẫn export annotation bình thường, còn CLI đánh giá theo run cần manifest/frame mapping tương ứng.

**Sự cố lượt đầu trên máy này:** request bắt đầu 14:17 ngày 08/10/2026, đứng ở 0/20 frame khi `getaddrinfo('host.docker.internal')` treo trong kernel `rtnl_dumpit`; function chưa nhận inference, GPU 0% và task chưa có nhãn. Đã hủy request và đổi `CVAT_NUCLIO_INVOKE_METHOD=dashboard` cho server/worker. GET Nuclio qua Docker network trả 200/ready. Khởi động lại Docker cũng mắc ở stopping, nên cần reboot Windows để giải phóng worker/lock. Sau reboot, mở Docker Desktop rồi chạy:

```powershell
.\deployment\recover-prelabel.ps1
```

**Đã phục hồi sau reboot:** script đã áp dụng cấu hình và gỡ request canceled. Còn phát hiện Nuclio lưu IP processor cũ (`172.18.0.17`) trong khi container nhận IP mới (`172.18.0.16`), khiến dashboard invoke báo no route to host. Patch bổ sung cho riêng CenterPoint gọi `http://nuclio-nuclio-pth-prelabel-centerpoint:8080` bằng Docker DNS; các model khác giữ đường invoke theo cấu hình. Gateway CVAT thật đã đến handler, trả lỗi payload trống đúng trong 0,07 giây, không chạy inference. 21 test UI/journal/patch đạt. Bằng chứng: [cvat-route-recovery.json](../artifacts/cvat-route-recovery.json), [ảnh menu sau phục hồi](../artifacts/centerpoint-recovered.png).

Script áp dụng patch/cấu hình mới, gỡ đúng record RQ đã canceled (CVAT 2.20 còn coi record này là conflict), kiểm tra đường gọi tới handler và giữ task/volumes. Script không chạy inference. Sau đó tải lại task #2 và tự nhấn Annotate. Không cần reboot thêm khi Docker đã hoạt động.

Nếu báo CUDA out of memory, xem log function ở Nuclio; GPU 4GB chưa được xác nhận đủ cho profile voxel/pillar. Không tự giảm sweeps để ép model chạy. Có thể dùng profile pillar ở **run/task mới**, fetch-model và deploy lại với `-Config configs/mini-pillar.yaml -RunDir runs/ui-pillar`; vẫn phải thử một frame hoặc dùng GPU đủ bộ nhớ.

## 5. Chạy lại và giữ nhãn đã sửa

Backend ghi journal sau từng frame trong Docker volume CVAT: `data/<data_id>/prelabel3d/centerpoint.json`. Frame đã tạo nhãn không được ghi lại khi bấm Annotate lần nữa, kể cả box đã bị bạn xóa. Manual box giữ nguyên. UI 3D không cho bật Clean previous annotations; backend cũng từ chối cleanup cho function này.

Nếu commit bị gián đoạn và không xác định box đã lưu rồi bị xóa hay chưa lưu, request dừng để kiểm tra journal, không tự tạo lại. Không xóa journal hoặc volume để thử retry. Đổi model/config/selection cần run/task mới. Không dùng `docker compose down -v` để dừng; dùng:

```powershell
.\deployment\local.ps1 Stop
```

Sau khi Save trên giao diện, export và evaluate luồng UI bằng:

```powershell
.\deployment\export-ui.ps1 -RunDir runs/ui-centerpoint
.\deployment\local.ps1 Tool --run-dir runs/ui-centerpoint evaluate
```

Script đọc bản snapshot journal bền vững từ container CVAT local, kiểm tra task/frame/metadata/hình học ban đầu với prediction, rồi tạo baseline và mapping ID cho exporter REST. Các ID gốc vẫn có sau khi box bị xóa hoặc attributes bị sửa. Journal pending hoặc thiếu frame sẽ bị từ chối; phải hoàn tất Annotate trước khi đánh giá cả run. Script không sửa annotation trên CVAT. `reviewed.json`, `cvat_export_raw.json`, `baseline.json`, `ui_journal_snapshot.json` và `reports/` lưu trong run. Nếu export/import đã tạo lại server IDs, dùng `-IdsRecreated`; mất cả ID và prediction_id sẽ báo không xác định edit ratio.

Đánh giá lấy ground truth chỉ ở bước evaluate. Nếu chưa có phiên chỉnh sửa/đo thời gian của annotator thì kết quả chưa chứng minh chất lượng sau review hay tiết kiệm thời gian.

## Những thay đổi trong CVAT

- `detector-runner.tsx`: cho task3D chọn detector có toàn bộ labels kiểu cuboid; task2D giữ các model2D. Dùng modal/button/progress sẵn có.
- `lambda_manager/views.py`: dispatch đúng function CenterPoint sang adapter 3D và chặn cleanup trước khi xóa dữ liệu.
- `prelabel3d.py`: tạo standalone cuboids qua serializer/DB API CVAT, append nhãn và lưu ID/journal bền vững trong volume; không thay toàn bộ annotations.
- `deployment/nuclio/main.py`: bridge detector protocol có sẵn. CVAT 2.20 gửi nguyên PCD base64 qua field tên `image`; handler không decode nó thành ảnh.

Source kiểm tra: [CVAT UI v2.20.0](https://github.com/cvat-ai/cvat/blob/v2.20.0/cvat-ui/src/components/model-runner-modal/detector-runner.tsx), [CVAT Lambda manager](https://github.com/cvat-ai/cvat/blob/v2.20.0/cvat/apps/lambda_manager/views.py), [Nuclio function configuration](https://docs.nuclio.io/en/1.13.x/reference/function-configuration/function-configuration-reference.html). Đây là tích hợp tùy biến cho release đã pin, cần build và kiểm thử thật trước khi coi là sẵn sàng sử dụng.
