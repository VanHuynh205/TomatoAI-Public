# Tomato Desktop

Tomato Desktop nhận diện dấu hiệu bệnh trên ảnh lá cà chua bằng mô hình G2 ConvNeXt-Tiny + BiGRU. Ứng dụng chạy trên CPU Windows, xử lý ảnh tại máy và hiển thị thẻ chăm sóc tiếng Việt có nguồn.

## Bắt đầu trên máy mới

- Mở [hướng dẫn Windows](docs/SETUP_WINDOWS.md#chạy-app-có-sẵn) để tải, xác minh và chạy gói có `TomatoDesktop.exe`.
- Mở [hướng dẫn chạy từ mã nguồn](docs/SETUP_WINDOWS.md#chạy-từ-mã-nguồn) để đặt bundle G2 và chạy `Start.ps1`.
- Mở [hướng dẫn phát triển](docs/SETUP_WINDOWS.md#phát-triển-và-kiểm-thử) để build frontend, chạy kiểm thử hoặc đóng gói Windows.

State dict G2 đã huấn luyện và gói Windows được phân phối dưới dạng ZIP qua GitHub Releases. Cây mã nguồn giữ nguyên đường dẫn module, tệp cấu hình và kiểm thử. [GitHub Releases](https://github.com/VanHuynh205/TomatoAI-Public/releases/tag/v0.1.0-g2-convnext-tiny-bigru)

## Tiếp tục phát triển

| Đường dẫn | Nội dung |
| --- | --- |
| `ai/` | Model, runtime suy luận, huấn luyện, hiệu chuẩn và công cụ đánh giá |
| `backend/` | API nội bộ, giới hạn đầu vào và catalog chăm sóc |
| `desktop/` | Launcher Windows, build PyInstaller, thông báo thành phần và kiểm thử |
| `frontend/` | React, TypeScript, OpenAPI types, kiểm thử và giao diện đã build |
| `infra/knowledge/` | Catalog và đầu vào chăm sóc đã duyệt |
| `tests/` | Kiểm thử AI và kiểm tra checkout Public |

[Kế hoạch CNN/BiGRU](docs/superpowers/plans/2026-11-18-cnn-bigru-colab-training.md) và [kết quả benchmark](docs/experiments/cnn-bigru-benchmark-2026-11-18.md) mô tả cách chọn G2. [Bàn giao DINOv2 cho hướng video](docs/training/IMAGE_TO_VIDEO_HANDOFF.md) áp dụng cho bản DINOv2 trước đó.

## Giấy phép

Mã dự án dùng Apache-2.0. Nội dung chăm sóc và thành phần bên thứ ba giữ thông báo và phạm vi giấy phép riêng. Xem hồ sơ model trước khi tái sử dụng checkpoint. [LICENSE](LICENSE) · [NOTICE](NOTICE) · [Model card](MODEL_CARD.md) · [Ghi công dữ liệu](DATASET_ATTRIBUTIONS.md)
