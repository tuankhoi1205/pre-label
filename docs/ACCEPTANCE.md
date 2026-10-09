# Nghiệm thu và phục hồi

## Trạng thái

Kiểm thử cuối trên core runtime: **87 passed, 2 skipped**, không có failure/error. Hai skipped là native MMDetection3D loader conformance và live CVAT REST integration; cả hai đã chạy riêng và đạt (mỗi bộ 1 passed). Chi tiết máy/command và artifacts ở [artifacts/VALIDATION.md](../artifacts/VALIDATION.md).

| Tiêu chí | Bằng chứng/trạng thái |
|---|---|
| Geometry asymmetry/yaw/full rotation/SE(3)/CVAT encoding | Đã qua tests CPU tổng hợp |
| PCD binary 5-float source validation/XYZ output | Đã qua tests tổng hợp |
| Sweeps/time lag/transport | Công thức và native MMDetection3D loader conformance đều đạt |
| Model load-once/resume/config invalidation | Đã qua test adapter thay thế, nạp checkpoint/CUDA ops và forward thật cho 20 frame |
| CVAT task/upload/frame metadata/readback | Đã qua fixture và test REST thật với PCD/cuboid tổng hợp |
| Rerun giữ sửa/xóa/thêm, không trùng | Đã qua REST fixture và commit-response-loss scenarios |
| Export mapping sample_token, ID recovery/ambiguity | Đã qua REST fixture |
| Rotated IoU/matching/edit/time/report orchestration | Đã qua dữ liệu tổng hợp |
| Inference thật trên ≥20 nuScenes mini frames | **Đã hoàn tất 20 frame**, Task #2 có 1.123 cuboid; request khoảng 45,5 giây |
| Cuboid hiển thị trong CVAT UI và camera context | Có box thật, công tắc hướng và 6 camera/frame; chất lượng alignment/đầu–đuôi cần annotator review |
| Người dùng chỉnh/xóa/thêm/Save/export thật | **Chưa thực hiện** |
| IoU trước/sau thật và thời gian annotator | **Chưa có dữ liệu** |

Người dùng đã tự chạy pre-label thật trên Task #2. 120 ảnh camera được thêm vào đúng 20 sample và hash annotation trước/sau không đổi. 41 test context/REST/UI/journal đạt. Còn cần chỉnh/duyệt/Save/export thật để nghiệm thu chất lượng và thời gian annotator. `artifacts/synthetic_smoke/metrics.json` vẫn ghi rõ dữ liệu tổng hợp, không đại diện cho chất lượng detector.

## Checklist chạy thật

1. Làm theo INSTALL; lưu `doctor --check-cvat`, `pip freeze`, checkpoint provenance.
2. Run smoke mới với `prepare --limit 1`, `infer`, inspect raw/normalized và `upload --dry-run`.
3. Upload; đối chiếu point cloud và cuboid trong UI, đặc biệt bottom→geometric center, L/W bất đối xứng, hướng yaw, số box/frame/label. Lưu ảnh/chú thích xác minh của annotator nếu cần bằng chứng nghiên cứu.
4. Chỉnh một box, xóa một box, vẽ thêm một box, Save. `export`, đối chiếu sample mapping; `upload` lại và đảm bảo edit/delete giữ nguyên, không trùng.
5. Tạo batch run/task mới; infer thật ≥20 frame. Không đổi selection/model trên run đã có task.
6. Thực nghiệm manual từ đầu và assisted trên các frame tương đương khác nhau; timer và review cùng tiêu chuẩn. Không lấy GT để gán nhãn nháp.
7. `export`, `evaluate`; kiểm tra CSV/JSON về sample count, matching threshold, policy/tolerance, box thừa/bỏ sót và thời gian chưa đo/null.

## Journal và lỗi giữa chừng

Run có `.lock` khi CLI đang thao tác. Nếu process bị kill và lock còn, kiểm tra process PID ghi trong tệp trước khi xóa **chính `.lock` của run đó**. Không xóa state/baseline để ép rerun trên task có chỉnh sửa.

Task tạo mới có tên chứa run ID. Nếu mất phản hồi khi create, rerun tìm exact task name để tránh tạo task thứ hai. Data upload ghi `data_attempted` và `rq_id`; rerun tiếp tục Requests polling hoặc truy vấn request create của task. Không tự gửi lại upload khi trạng thái không rõ.

Annotation PATCH ghi danh sách `pending` trước request; sau response lưu IDs trước readback. Nếu client mất response mà shape tồn tại, rerun phục hồi bằng attribute. Nếu pending shape không tồn tại, không phân biệt chắc chắn chưa commit với đã bị người dùng xóa. Khi đó:

1. Backup run và xem `upload_uncertain.json`, `cvat_state.json`, CVAT annotations/request logs.
2. Nếu xác nhận request chưa commit và chưa có thao tác người dùng: loại **chỉ các ID pending đó** khỏi `pending` trong journal, rồi rerun để tạo chúng.
3. Nếu xác nhận box đã commit rồi bị người dùng xóa: thêm entry `applied[prediction_id]` với annotation ID gốc từ log/backup; không upload box đó lại. Nếu không lấy được ID, giữ trạng thái chưa xác định và không tuyên bố edit metrics đầy đủ.
4. Nếu không có bằng chứng để phân biệt: không ép upload. Giữ nguyên task và phục hồi từ backup hoặc bắt đầu task/run mới có ghi rõ provenance.

Tool không tự ghi đè/chọn thay người dùng trong tình huống uncertain, vì khôi phục một box có thể phá quyết định xóa của annotator. Đây là chặn dựa trên trạng thái commit thực tế, không phải yêu cầu xác nhận chung trước mọi upload.

Readback mismatch giữ raw prediction/baseline/state để kiểm tra. Do IDs đã được ghi trước bước verify, chạy lại không tạo bản sao. Cần tìm nguyên nhân label/frame/geometry trước khi chấp nhận task. Nếu người dùng chỉnh đồng thời ngay sau upload, readback có thể phát hiện thay đổi; tổ chức upload hoàn tất rồi mới bắt đầu review.

Nếu di chuyển run sang máy Linux, cập nhật YAML/root, giữ manifest và các artifacts CVAT. LiDAR paths trong manifest tương đối dataroot, PCD tương đối run. Config/checkpoint paths không tham gia signature (hash/content và settings mới là identity), nên cùng assets có thể resume qua máy; thay `device` hoặc bất kỳ settings detector sẽ đòi run mới. Không chạy hai uploader từ hai bản copy run trên cùng task.
