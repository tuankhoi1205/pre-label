# Chính sách đánh giá và thực nghiệm

## Chất lượng hình học

`evaluate` đọc prediction trước review, snapshot ban đầu, export sau review và GT devkit. Tất cả box ở hệ `lidar_sensor` đúng sample. GT gồm 10 detection classes của checkpoint, ít nhất `min_lidar_points=1`, center trong ROI mặc định `[-51.2,-51.2,-5,51.2,51.2,3]`; policy được ghi vào JSON. Predictions/reviewed được chấm đầy đủ; box ngoài GT policy có thể là extra. Chỉnh policy trước thực nghiệm và giữ cố định khi so sánh.

IoU 3D = `volume(intersection) / volume(union)`. Hai cuboid là giao 12 halfspaces; giao ba mặt phẳng cho các candidate vertices, giữ vertex nằm trong mọi halfspace, lấy thể tích convex hull. Broad-phase AABB chỉ bỏ cặp không giao; metric cuối có xét rotation đủ 3 trục. Không thay bằng IoU BEV hoặc AABB IoU.

Với từng `(sample_token,class_name)`, lập ma trận rotated 3D IoU. Cặp hợp lệ khi IoU ≥ threshold (mặc định 0.5). Hungarian assignment tối đa **số cặp hợp lệ**, sau đó tối đa tổng IoU; mỗi prediction/GT ghép tối đa một lần. Không ghép xuyên frame/class và không coi GT là input detector.

```text
TP = số cặp ghép
FP = prediction không ghép (box thừa)
FN = GT không ghép (box bỏ sót)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
mean_matched_iou = tổng IoU cặp ghép / TP
```

Mẫu số bằng 0 hoặc không có cặp: dùng null, không tự điền 0/1 cho mean. Báo số frame, tổng prediction, tổng GT, số cặp, threshold, FP/FN và per-class. JSON giữ IDs box thừa/bỏ sót; CSV giữ từng cặp. So sánh before/after với cùng GT/policy.

mAP/NDS chính thức nuScenes chưa triển khai; báo `not_computed`. Chúng dùng quy tắc devkit khác phép ghép IoU này. Không ghi IoU thành mAP hoặc NDS. Mini phục vụ smoke và kiểm tra pipeline; checkpoint pretrained nuScenes có thể đã thấy các scene mini trong training/validation gốc, nên không dùng mini để khẳng định khả năng tổng quát.

## Mức độ chỉnh sửa

Snapshot `baseline.json` bất biến và `cvat_state.json.applied` ánh xạ prediction → annotation ID. Export ưu tiên ID server, sau đó attribute `prediction_id` khi ID bị mất. Label thay đổi không làm mất identity.

Một prediction là edited khi label khác hoặc:

- Norm L2 của thay đổi tâm > **0.02 m**.
- Max absolute thay đổi L/W/H > **0.02 m**.
- Geodesic rotation angle của `R_before^-1 R_after` > **0.01 rad**.

Tolerances đều có trong YAML và report. Quaternion q và -q cho cùng rotation; không so trực tiếp phần tử Euler. Không tính thay đổi confidence/model attribute là sửa hình học. Đổi frame/sample của cùng ID không được coi là geometry edit hợp lệ; tool báo lỗi mapping.

`unchanged`, `edited`, `deleted` chia cho tổng prediction ban đầu; ba tỷ lệ cộng 1 khi baseline nonempty. `added` báo riêng, không đưa vào mẫu số ban đầu. Baseline rỗng thì các tỷ lệ null.

Sau export/import tạo lại ID, dùng `export --ids-recreated`. Còn prediction attributes thì phục hồi và ghi `prediction_attribute_after_id_loss`. Mất cả ID/attribute dẫn đến `unresolved_after_id_loss`; toàn bộ tỷ lệ edit/delete/add là unavailable vì không phân biệt được box thêm mới với box được sửa. Chất lượng IoU vẫn tính được. Identity trùng khi copy giữ attributes bị từ chối; sửa metadata của bản copy trước export.

## Thời gian chủ động

Timer do annotator điều khiển: `start → pause → resume → stop`. Mỗi session có workflow manual/assisted, người gán, difficulty, quality protocol, manifest frame IDs và số cuboid cuối được chấp nhận. Chỉ các khoảng đang active được cộng. Không bật timer khi inference, upload, chờ server hay nghỉ; các thời gian hệ thống báo riêng từ pipeline.

```text
active_seconds = tổng(end_i - start_i) của các active intervals đã đóng
seconds_per_frame = active_seconds / số frame đã gán
seconds_per_cuboid = active_seconds / số cuboid cuối được chấp nhận
savings_percent = 100 * (manual_seconds_per_frame - assisted_seconds_per_frame)
                        / manual_seconds_per_frame
```

Manual phải là gán từ đầu trên task sạch; assisted là thời gian người xem/sửa/duyệt pre-label. Đảm bảo không tính cùng phiên làm việc hai lần. Giá trị âm của savings có nghĩa assisted chậm hơn. Không có session thì `not_measured`; timer còn active/paused không đưa vào kết quả.

`report-time` chỉ tính savings cho cùng người/difficulty/quality standard và tập frame không giao nhau. Không gộp một phần trăm chung khi có nhiều strata; giữ từng comparison trong JSON. Difficulty/quality do người tổ chức nhập, không được tool tự xác minh; kết quả không chứng minh tiêu chuẩn chất lượng chỉ vì strings trùng.

Thiết kế khuyến nghị:

1. Chọn hai tập frame tương đương theo số đối tượng, khoảng cách, occlusion, mật độ điểm, loại scene và class. Không xem GT khi annotating.
2. Phân công đối trọng giữa người tham gia: A manual/B assisted cho một nhóm, đảo điều kiện cho nhóm khác. Mỗi người không gán cùng frame ở cả hai điều kiện; dùng frame IDs chung của manifest để phát hiện overlap.
3. Đưa cùng quy trình chất lượng, cùng thao tác Save/review và training công cụ cho hai nhóm. Manual task không chứa prediction; tách task review khỏi task timer baseline.
4. Reviewer độc lập duyệt kết quả theo protocol; chỉ chốt `--cuboids` và stop sau duyệt. Nếu cần sửa thêm, giữ session paused rồi resume để không đếm lại frame/cuboid.
5. Công bố số người/frame/cuboid, cách ghép difficulty, thứ tự điều kiện, phân bố thời gian và khác biệt chất lượng. Nên phân tích thống kê theo người/scene khi mở rộng nghiên cứu.

Giới hạn MVP: frame count/cuboid count do người dùng khai báo; timer không theo dõi mouse/keyboard hay tự phát hiện idle. Timer từ chối tính cùng frame hai session cho cùng người/workflow; dùng pause/resume đến khi review hoàn tất. Số frame aggregate là số lượt gán khi nhiều người cùng tham gia, không nhất thiết là số sample duy nhất. Nếu cần nhiều người chia thao tác trên cùng frame, mở rộng session ledger trước khi dùng báo cáo aggregate.
