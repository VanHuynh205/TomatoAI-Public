# Model card G2 ConvNeXt-Tiny + BiGRU

G2 là model desktop mặc định. Kiến trúc gồm backbone ConvNeXt-Tiny và đầu phân loại BiGRU theo hai trục. Lượt train chọn checkpoint raw ở epoch 15. Suy luận CPU dùng ảnh 288 px, lật ngang ở bước test-time augmentation và hiệu chuẩn temperature 1,2840329428784063. Ngưỡng phục vụ là 0,7854081940644378. [Cấu hình inference](ai/cnn_gru_inference.py) · [Kết quả benchmark](docs/experiments/cnn-bigru-benchmark-2026-11-18.md)

Model có tám nhãn lá cùng lớp âm `not_tomato_leaf`. Điểm mô hình dùng để xếp hạng nhãn và quyết định từ chối. Sai khác giữa các tập thực địa cần được xem khi áp dụng ở vùng mới. Xem [catalog chăm sóc](infra/knowledge/CONTENT-LICENSE.md) và [phạm vi dự án](PRODUCT.md).

Giải nén `TomatoDesktop-G2-model.zip` tại gốc repo. `cnn-gru-release.json` khóa SHA-256 cho metadata và state dict. Bundle chạy G2 đã huấn luyện, gồm backbone và đầu phân loại trong cùng `model.safetensors`. [Đường dẫn cài đặt](docs/SETUP_WINDOWS.md#chạy-từ-mã-nguồn)

| Tệp | SHA-256 |
| --- | --- |
| `model.safetensors` | `85e0450a49a95f77e640a9e1ff63d8d9dc4befb5fe8bcd8674caa16d316a1a9e` |
| `leaf-classifier.json` | `3a9d075b70208952c3e484187980432cb2ac867c9a9f330734eec69fcf4e4630` |

Trọng số tiền huấn luyện ConvNeXt-Tiny của Meta có thông báo MIT. Hồ sơ train ghi nhận 241 ảnh PlantSeg v7 theo CC BY-NC 4.0. Bản G2 được chia sẻ để cộng tác phi thương mại. Hồ sơ hiện có chưa xác nhận quyền phân phối đầy đủ cho trọng số tinh chỉnh. [Thông báo thành phần](desktop/notices/NOTICE.md) · [Giấy phép ConvNeXt](desktop/notices/convnext/LICENSE) · [Kế hoạch và provenance](docs/superpowers/plans/2026-11-18-cnn-bigru-colab-training.md)

[DINOv2 v1 280](https://github.com/VanHuynh205/TomatoAI-Public/releases/tag/v0.1.0-dinov2-v1-280) là mốc đối chứng và bản desktop trước G2.
