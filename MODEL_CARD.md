# DINOv2 v1, suy luận 280px

## Danh tính và thay đổi

- Chủ dự án: [VanHuynh205](https://github.com/VanHuynh205).
- Backbone: [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base/tree/f9e44c814b77203eaa57a6bdbbd535f21ede1415), ViT-B/14 của Meta AI Research (FAIR), revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`, Apache License 2.0.
- Bài báo: Maxime Oquab và cộng sự, *DINOv2: Learning Robust Visual Features without Supervision*, 2023, [arXiv:2304.07193](https://arxiv.org/abs/2304.07193). Xem [ghi công đầy đủ](THIRD_PARTY_NOTICES.md).
- Dự án thêm đầu phân loại chín nhãn, tinh chỉnh bốn khối cuối và lớp chuẩn hóa cuối của backbone. Checkpoint epoch 6 được train ở 224px. Runtime v1 dùng **cùng checkpoint** ở 280px; không thay model bằng DINOv3, model v2 hoặc backbone chưa tinh chỉnh.
- Tiền xử lý: EXIF transpose, RGB, fit/pad bicubic, mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`; trung bình logits của ảnh gốc và ảnh lật ngang. Runtime CPU, `eval()` và `torch.inference_mode()`.
- Các nhãn: bacterial_spot, early_blight, late_blight, leaf_mold, septoria_leaf_spot, spider_mites, yellow_leaf_curl, healthy và not_tomato_leaf. Runtime không cung cấp thẻ chăm sóc cho not_tomato_leaf.

280px là cấu hình suy luận của đồ án, không phải tên một checkpoint pretrained riêng của Meta. Metadata train `size: 224` phải giữ nguyên. Validator kiểm `evaluation_size: 280`, epoch 6, revision, cấu hình ViT-B/14 và hashes của năm đầu vào.

## Dữ liệu và giới hạn quyền

Train: 6.969 ảnh; validation: 1.642 ảnh. [Danh sách nguồn, tác giả, phiên bản, điều kiện và khoảng trống](DATASET_ATTRIBUTIONS.md) phân biệt tập train/validation với đánh giá ngoài nguồn. Repo không phân phối ảnh huấn luyện.

Chủ dự án yêu cầu phát hành checkpoint tinh chỉnh qua GitHub Release để cộng tác làm tiếp phần video. **Giấy phép riêng của checkpoint chưa được chốt; quyền dữ liệu còn các điểm chưa xác minh.** Việc chủ dự án yêu cầu upload không phải xác nhận clearance từ các nguồn dữ liệu. Apache của DINOv2 gốc và của mã ứng dụng không tự giải quyết điều khoản PlantVillage hoặc quyền ảnh thu từ web của PlantDoc. Việc bổ sung ghi công không thay thế xin quyền hoặc rà soát pháp lý. Người tái sử dụng phải đánh giá quyền phù hợp với cách dùng của mình.

Lần cập nhật giấy phép/khả năng tái tạo này không train lại, không chọn checkpoint mới và không chỉnh weights. Không có độ chính xác mới hoặc cam kết độ chính xác trên mọi ảnh/địa phương. Test synthetic trong repo chỉ kiểm hợp đồng API/UI và cơ chế inference; chúng không đánh giá chất lượng phân loại thực tế.

## Bundle kỹ thuật cần có khi đã được cấp quyền

| Tệp | SHA-256 của bản lưu v1 hiện tại |
| --- | --- |
| classifier.safetensors | `eb2337fbaca6915e9ccae39868fc305dbd5a8a8d95e0347abbf67a5cc1b05efa` |
| config.json | `09a44ebed52dc88f73c7d27a1762a90f87e21c19dd014fe7b203e06de1db1732` |
| leaf-classifier.json | `9814d67e774b1742f22177aafc3aaa0bbd7579aa9a1826d745176f09f8fe10d3` |
| model.safetensors | `3274d01322f079a2b7d0df0048500b1094d67f8f308c47b989ad71e8c566a4ff` |
| preprocessor_config.json | `14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828` |

Tệp thứ sáu `dinov2-release.json` khai báo `model_version: dinov2-v1-280`, `evaluation_size: 280` và đúng năm hash trên. Hash giúp đối chiếu bản lưu; hash không phải chữ ký, bằng chứng quyền sử dụng hoặc nguồn tải weights. Không sửa manifest để làm weights khác vượt kiểm tra.

Đặt bundle được cấp quyền vào `ai/artifacts/dinov2-v1-280/`, theo [hướng dẫn chạy](README.md). Runtime tải model cục bộ, không tải pretrained thay thế. Không cần tải dataset hoặc chạy huấn luyện.

## Phạm vi sử dụng

Ứng dụng hỗ trợ tham khảo trên ảnh lá cà chua trong phạm vi nhãn đã học. Ánh sáng, nền, giống cây, nhiều bệnh cùng lúc và ảnh ngoài đối tượng có thể làm sai kết quả. Điểm softmax không phải xác suất chẩn đoán đã hiệu chuẩn. Thẻ chăm sóc chưa có thẩm định chuyên gia tại Việt Nam; giữ cảnh báo và provenance của từng thẻ.
