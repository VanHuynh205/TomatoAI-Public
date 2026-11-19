---
id: SPEC-desktop-ca-chua
companions:
  - hien-trang-va-nguon.md
  - du-lieu-va-thuc-nghiem.md
  - kien-truc-va-cham-soc.md
  - ke-hoach-8-tuan.md
  - hop-dong-va-mo-rong.md
sources: []
---

# App desktop offline nhận diện tình trạng lá cà chua

Ngày lập và rà soát: 01/11/2026. Trạng thái: đủ phạm vi để lập kế hoạch coding; dữ liệu, stack và chất lượng vẫn cần vượt các gate thực thi. Chưa huấn luyện hoặc xây app.

Đọc [kế hoạch 8 tuần](ke-hoach-8-tuan.md) để bắt đầu. Các quyết định kỹ thuật dưới đây là đề xuất có điều kiện; dữ kiện đã kiểm nằm trong [hiện trạng](hien-trang-va-nguon.md).

## Why

Một người cùng AI cần hoàn thành đồ án trong khoảng hai tháng, làm khoảng 3 giờ mỗi tối. Sản phẩm nhận một ảnh lá cà chua, phân loại bằng mô hình đã fine-tune và hiển thị hướng dẫn chăm sóc có nguồn ngay trên máy Windows. EfficientNet-B0 là mô hình ưu tiên tích hợp; ResNet50 là mô hình đối chứng bắt buộc trong thực nghiệm. Kế hoạch không phụ thuộc việc tìm được chuyên gia địa phương hoặc hai thành viên còn lại.

## Capabilities

- CAP-1
  - intent: Người phát triển có thể chọn dữ liệu, truy nguồn và chia các tập độc lập để huấn luyện có thể kiểm tra lại.
  - success: Có manifest nguồn/quyền/nhãn/group/split/exclusion; không giao group hoặc ảnh trùng đã xác minh giữa các tập; ghi rõ phần chưa biết.
- CAP-2
  - intent: Người phát triển có thể huấn luyện và so sánh EfficientNet-B0 với ResNet50 bằng transfer learning.
  - success: Có baseline đóng băng và bản fine-tune cho mỗi backbone, cùng split; lưu cấu hình, checkpoint, dự đoán, macro-F1 và chỉ số từng lớp từ lần chạy thật.
- CAP-3
  - intent: Người dùng có thể nhận diện một ảnh lá trên desktop khi không có Internet.
  - success: Bản đóng gói chạy trên Windows mục tiêu bằng CPU, không cần cài Python/CUDA; ảnh hợp lệ trả đúng schema, mapping và preprocessing đã khóa.
- CAP-4
  - intent: Người dùng có thể đọc chăm sóc phù hợp với từng lớp được hỗ trợ và biết nguồn, điều kiện áp dụng.
  - success: Mỗi lớp trong model có thẻ tương ứng; thẻ có nội dung đối chiếu nguồn, giới hạn địa lý, ghi nhận người/ngày kiểm tra; không giả nhận chuyên gia duyệt. Bằng chứng thiếu thì hiển thị thiếu thông tin.
- CAP-5
  - intent: Người dùng nhận phản hồi rõ khi ảnh lỗi, kết quả bất định hoặc mô hình không sẵn sàng; ảnh được giữ trên máy.
  - success: Các ca file giả, ảnh quá lớn, model/mapping lỗi, gọi API không hợp lệ và nhận diện liên tiếp được kiểm thử; app không tự tải/gửi ảnh hoặc lưu ảnh sau phiên.
- CAP-6
  - intent: Người phát triển có thể nộp, trình diễn và giải thích lại đồ án từ artifact đã lưu.
  - success: Gói nộp có mã nguồn, hướng dẫn, cấu hình, split, checkpoint, model card, attribution, kết quả thật, báo cáo, slide, video dự phòng và bằng chứng chạy offline.
- CAP-7
  - intent: Người phát triển có thể thêm chức năng hoặc thay mô hình sau retrain mà giữ được ranh giới module và ý nghĩa dữ liệu.
  - success: Kiểm thử chứng minh thay bundle hợp lệ cùng kiến trúc/lớp không cần sửa UI/API; thêm lớp cần head/mapping/thẻ phù hợp; bundle không tương thích bị từ chối; khôi phục bản phát hành trước chạy được. Các module có test ranh giới và contract.

## Constraints

- Ngân sách: một người, 3 giờ/tối; 8 tuần có 144 giờ công việc chính và 24 giờ dự phòng. AI không thay người chịu trách nhiệm về kết quả.
- Máy: RTX 3050 4 GB VRAM, RAM 16 GB, Ryzen 5 5625U. Train từng mô hình; bản sử dụng phải có CPU inference.
- `ai/` chứa ML; `backend/` chỉ triển khai API; `frontend/` chỉ UI và mã gọi API; `desktop/` chứa host, composition và đóng gói; `infra/` chứa dữ liệu; tài liệu sinh ra trong `docs/`.
- Vault thực tế là `infra/library/`. Giữ raw, nhãn gốc, snapshot đã seal và metadata duyệt hiện có. Không tự đổi `rag_ready`, `allowed` hoặc expert flags.
- EfficientNet-B0 hiện có chỉ là pretrained ImageNet-1K; phải thay head và train bằng dữ liệu lá đã chọn. ResNet50 cũng bắt đầu từ pretrained weights có provenance.
- Bộ dữ liệu không có giấy phép hoặc quyền chưa rõ không được mặc định đưa vào training/gói phát hành. Kiểm tra đúng bản phân phối; tên dataset không chứng minh nguồn ảnh.
- Không lấy test để chọn mô hình, siêu tham số hay ngưỡng. Báo riêng kết quả nội nguồn và dữ liệu thực địa đủ điều kiện.
- Chuyên gia không là điều kiện bắt buộc của bản đồ án. Đối chiếu tài liệu và nhãn nghiên cứu không được gọi là xác nhận bệnh học.
- ID lớp mang ý nghĩa ổn định; thứ tự output chỉ lấy từ bundle mô hình. UI và chăm sóc không dùng số thứ tự nhãn hoặc tên dịch để nối dữ liệu.
- Phiên bản model, dataset/split, preprocessing, tri thức, schema và API được ghi riêng. Bundle sai hash, schema, số lớp hoặc thiếu thẻ phải bị từ chối trước dùng.
- Cập nhật model MVP bằng gói phát hành đã kiểm, giữ bản trước để khôi phục; không tự tải, thay nóng hoặc nạp checkpoint tùy ý. Không xây plugin framework cho nhu cầu mở rộng chưa có.

## Non-goals

- Huấn luyện từ đầu, ensemble hai backbone, grid search lớn hoặc tự gán lại nhãn chuyên môn bằng AI.
- Chẩn đoán nhiều lá/nhiều bệnh trong một cảnh, phân đoạn, chấm mức độ nặng hoặc xác định tác nhân bằng xét nghiệm.
- Chatbot/RAG/LLM chạy local, cloud API runtime, tài khoản, đồng bộ, IoT, tưới/bón tự động, app mobile hoặc UI 3D.
- Hệ plugin, service locator, event bus hoặc hệ tự cập nhật tổng quát. Việc mở rộng dựa trên module có trách nhiệm rõ và contract được kiểm thử.
- Kê thuốc, liều lượng hoặc lịch phun từ nguồn nước ngoài cho người dùng Việt Nam; cam kết độ chính xác ngoài vườn khi chưa có benchmark phù hợp.

## Success signal

Người dùng mở bản đóng gói khi tắt mạng, chọn ảnh một lá, nhận kết quả hoặc thông báo chưa đủ cơ sở, rồi đọc thẻ chăm sóc có nguồn. Người bảo vệ đồ án tái tạo được bảng so sánh hai mô hình từ artifact, giải thích được leakage và giới hạn dữ liệu.

Mục tiêu thực nghiệm đề xuất: macro-F1 nội bộ ít nhất 0,85 và recall từng lớp ít nhất 0,70; mục tiêu hiệu năng: warm CPU p95 không quá 3 giây/ảnh trên máy mục tiêu. Chốt trước thực nghiệm chính. Đây là tiêu chí để đánh giá, không phải kết quả đã đạt; nếu không đạt thì ghi nguyên kết quả và nguyên nhân, không giảm ngưỡng để nhận hoàn thành chất lượng.

## Assumptions

- Windows 10/11 x64 là nền tảng bản nộp; lịch dự kiến 02/11–27/12/2026, chưa phải hạn nộp chính thức.
- MVP ưu tiên 8 lớp PlantVillage hiện có grouping sau khi kiểm lại dữ liệu/quyền; số lớp chính thức khóa cuối tuần 1. Chỉ mở 10 lớp khi đủ bằng chứng trong ngân sách.
- React/TypeScript/Vite, FastAPI, pywebview/WebView2 và PyInstaller là stack đề xuất; phải vượt thử đóng gói nhỏ ở tuần 1.
- Thẻ chăm sóc là bản tổng hợp đối chiếu nguồn, do chủ dự án kiểm tra; chưa có thẩm định chuyên môn độc lập.

## Open Questions

- Hạn nộp chính thức và rubric có bắt buộc 10 lớp, bộ dữ liệu cụ thể hoặc framework desktop cụ thể không?
- Bộ hơn 10.000 ảnh nhà kính chính xác nằm ở đâu, URL/commit nào? Hiện chưa xác định trong checkout.
- Nghiên cứu nào dùng đúng phiên bản `tomato_vn` này, và có bằng chứng địa điểm thu ảnh hay không?
- Bản PlantVillage raw hiện có tương ứng bản phát hành và điều khoản nào? Metadata local khai CC BY-SA 3.0; cần khóa provenance trước sử dụng.

Các câu hỏi này có đầu việc xử lý trong lịch. Không chặn việc học pipeline, kiểm kê, chuẩn bị contract hoặc thử đóng gói bằng fixture tự tạo.

Quy tắc tương thích và mở rộng nằm trong [hợp đồng module](hop-dong-va-mo-rong.md). Kế hoạch coding nằm dưới `docs/superpowers/plans/`; việc lập kế hoạch không cấp trạng thái đạt cho các gate dữ liệu, model hoặc phát hành.
