# Nhận diện tình trạng lá cà chua

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

React, TypeScript, Vite và CSS global trong `frontend/`. Giao diện chạy trong pywebview/WebView2 trên Windows; FastAPI và suy luận CPU chạy cùng máy. Các lựa chọn này đến từ spec và kế hoạch đã được người dùng yêu cầu triển khai.

## Users

Người dùng chọn ảnh một lá cà chua để tham khảo kết quả phân loại và tài liệu chăm sóc. Chủ đồ án sử dụng cùng luồng để trình diễn và giải thích giới hạn của mô hình.

## Product Purpose

Nhận một ảnh JPEG/PNG, trả kết quả hoặc thông báo chưa đủ cơ sở, rồi hiển thị thẻ chăm sóc gắn đúng ID lớp. Nguồn, phạm vi địa lý, ngày đối chiếu và giới hạn phải đọc được ngay trong thẻ.

## Operating Context

Ứng dụng desktop Windows dùng CPU, hướng đến hoạt động offline. Ảnh không được gửi ra dịch vụ ngoài hoặc lưu sau phiên. Mở nguồn là thao tác chủ động của người dùng qua trình duyệt ngoài và cần kết nối mạng.

## Capabilities and Constraints

- Một luồng nhận diện, không tài khoản, lịch sử, cloud, chatbot/RAG hoặc 3D.
- File tối đa 10 MB và 20 triệu điểm ảnh; một lượt suy luận tại một thời điểm.
- Điểm mô hình không được trình bày như xác nhận bệnh học. Kết quả bất định không tự mở thẻ chăm sóc.
- Quyền nguồn và tình trạng thẩm định phải có bằng chứng. Holdout pv8-v1 đã dùng cho M7; giữ lịch sử này khi retrain. Việc phát hành cần qua gate riêng.
- Bố cục người dùng đã chọn ngày 05/11/2026: ảnh và nút bên trái, kết quả/chăm sóc bên phải; màn nhỏ xếp dọc.

## Evidence on Hand

Nguồn chuẩn là `docs/spec-desktop-ca-chua/` và các kế hoạch `docs/superpowers/plans/2026-11-01-*`. Đối chiếu checkpoint ngày 06/11/2026: M1–M7, A1–A7 và D2/D4 đã có bằng chứng triển khai. M6 đã xuất hai bundle; D4 có tám thẻ được Công Văn duyệt ngày 05/11/2026, composition desktop, browser E2E và smoke B0 CPU/HTTP. Bằng chứng chi tiết nằm trong `docs/progress/2026-11-05-d4/acceptance-audit.md`. M7 đã đánh giá cả hai bundle trên 1.714 ảnh PlantVillage và đo CPU trên Ryzen 5 5625U. G5 đạt trong phạm vi đó. D3 đã dựng và kiểm gói onedir CPU cục bộ. [Bàn giao D3](docs/verification/release.md) ghi artifact và giới hạn. Bước kế tiếp là D5 kiểm Windows sạch/ngắt mạng. D1/G0 còn kiểm native, D5/D6 còn kiểm gói offline và rollback trên Windows sạch. Hiệu quả nhận diện thực địa cần dữ liệu độc lập. [Nghiệm thu G5](docs/verification/m7-quality.md)

## Accessibility & Inclusion

Giao diện tiếng Việt, semantic HTML, thao tác bàn phím, nhãn input và thông báo trạng thái truy cập được. Giữ nội dung đọc được khi màn hẹp, tên tệp dài và zoom; tôn trọng reduced motion.
