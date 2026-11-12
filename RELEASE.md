# Hồ sơ phát hành Tomato Desktop

## Mô hình của bản Windows

DINOv2 v1 dùng backbone [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base) tại revision f9e44c814b77203eaa57a6bdbbd535f21ede1415. Đồ án thêm đầu phân loại chín nhãn, rồi tinh chỉnh bốn khối cuối và lớp chuẩn hóa của backbone. Checkpoint epoch 6 được huấn luyện ở 224 px. Ứng dụng dùng cùng checkpoint với ảnh đầu vào 280 px. Phần gốc của Meta giữ [Apache 2.0](desktop/notices/dinov2/LICENSE) và [thông báo ghi công](THIRD_PARTY_NOTICES.md).

## Nguồn dùng để học và chọn checkpoint

Số liệu lấy từ manifest của lượt DINOv2 v1, gồm 6.969 ảnh train và 1.642 ảnh validation.

| Nguồn và bản phát hành | Train | Validation | Giấy phép hoặc thông báo của nguồn |
| --- | ---: | ---: | --- |
| [PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage) | 2.800 | 600 | CC BY-SA 3.0. [Bài báo phát hành dữ liệu](https://arxiv.org/pdf/1511.08060) nêu điều khoản cùng giấy phép cho thuật toán huấn luyện trên dữ liệu. |
| [PlantCity v2](https://data.mendeley.com/datasets/w8kh2xkspx/2) | 1.735 | 462 | CC BY 4.0. |
| [CCMT v1](https://data.mendeley.com/datasets/bwh3zbpkpv/1) | 963 | 299 | CC BY 4.0. |
| [Tomato Leaf Dataset v2 của Imtiaz và cộng sự](https://data.mendeley.com/datasets/bpfd9cns5g/2) | 273 | 30 | CC BY 4.0. |
| [PlantDoc Classification](https://github.com/pratikkayal/PlantDoc-Dataset) | 685 | 125 | Repo ghi CC BY 4.0 và mô tả ảnh thu từ web. Số liệu gồm lá cà chua và lá cây khác. |
| [SoilClass v1](https://data.mendeley.com/datasets/6mx34nshm4/1) | 368 | 95 | CC BY 4.0. |
| [Open Images V7](https://storage.googleapis.com/openimages/web/factsfigures_v7.html) | 145 | 31 | Google công bố nhãn theo CC BY 4.0 và liệt kê ảnh theo CC BY 2.0. Google yêu cầu người dùng tự xác minh quyền từng ảnh. |

Rachin v2 chỉ dùng để kiểm thử bản v1. Taiwan Tomato và TOM2024 dùng cho phép so sánh ngoài nguồn. Các tập này nằm ngoài train và validation của checkpoint v1.

## Điểm quyền dữ liệu còn mở

- PlantVillage. Cần làm rõ phạm vi điều khoản trong bài báo đối với backbone Apache 2.0 đã tinh chỉnh và đầu phân loại. Khi hỏi tác giả hoặc rà pháp lý, nêu rõ bản nguồn, cách dùng ảnh và các tệp trọng số dự định phân phối.
- PlantDoc. Manifest có ID hoặc đường dẫn cho 810 ảnh train và validation. Hồ sơ hiện thiếu ghi công tác giả ảnh gốc ở cấp từng ảnh. Cần làm rõ quyền đối với phần ảnh thu từ web và ảnh hưởng tới checkpoint.
- Open Images. Đủ 176 mục train và validation có metadata ghi tác giả, URL và giấy phép từ nguồn. Phạm vi đối chiếu hiện tại là metadata, chưa xác minh độc lập quyền từng ảnh.

Creative Commons giải thích rằng điều kiện của giấy phép phụ thuộc vào việc sử dụng có cần quyền tác giả hay không và việc phát hành có chia sẻ tác phẩm hoặc bản chuyển thể hay không. Giấy phép của checkpoint chưa được chốt. [Hướng dẫn của Creative Commons](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/)

## Phát hành Windows

Mã ứng dụng do đồ án viết dùng [Apache License 2.0](LICENSE), kèm [NOTICE](NOTICE); thông báo MIT của các revision cũ vẫn được giữ. Catalog care-v1 giữ [CC BY-SA 4.0 và ghi công nguồn](infra/knowledge/CONTENT-LICENSE.md). Chi tiết tác giả, version/DOI, thay đổi và giới hạn quyền dataset nằm trong [DATASET_ATTRIBUTIONS.md](DATASET_ATTRIBUTIONS.md); [MODEL_CARD.md](MODEL_CARD.md) ghi đúng danh tính và hashes của v1. Theo yêu cầu của chủ dự án, checkpoint được đưa vào bản cộng tác [v0.1.0-dinov2-v1-280](https://github.com/VanHuynh205/TomatoAI-Public/releases/tag/v0.1.0-dinov2-v1-280), dù các điểm quyền dữ liệu ở trên chưa được giải quyết. Không gán Apache-2.0 cho checkpoint.

Release có ZIP Windows đầy đủ gồm `TomatoDesktop.exe`, `_internal/`, `release-manifest.json`, hướng dẫn prerequisites và ZIP model riêng cho người dùng source. `SHA256SUMS.txt` ghi hash của hai ZIP. Giữ nguyên toàn bộ thư mục sau giải nén; không phân phối riêng EXE. EXE chưa ký Authenticode. [Báo cáo kiểm Windows](docs/verification/2026-11-13-windows-release.md) phân biệt kiểm integrity, chạy frozen/native và test fixture; không tuyên bố đã đo lại accuracy hoặc kiểm mọi máy Windows. [Hướng dẫn bàn giao](docs/training/IMAGE_TO_VIDEO_HANDOFF.md) chỉ rõ mã ảnh gốc và giới hạn warm-start trước khi phát triển video.

### Dựng gói cho người bảo trì

Dùng Windows x64, CPython 3.12.14 và uv. Lấy model ZIP từ Release, kiểm checksum rồi đặt đủ sáu tệp ở `ai/artifacts/dinov2-v1-280/`. Tám bản thảo care-v1 đã duyệt trong `infra/knowledge/drafts/` có nguyên bytes theo allowlist để kiểm lại catalog khi build. App không tự tải weights. Build không train hoặc sửa tham số model; giữ nguyên các thông báo quyền chưa hoàn tất khi phân phối tiếp.

Chạy PowerShell từ thư mục gốc. Dùng một thư mục dist mới nếu đã có gói trước đó.

~~~powershell
uv venv .venv-runtime --python 3.12.14
uv pip sync --python .venv-runtime/Scripts/python.exe --torch-backend cpu desktop/requirements-runtime-cpu.lock
uv pip install --python .venv-runtime/Scripts/python.exe --no-deps --no-build-isolation .
& ./.venv-runtime/Scripts/python.exe -m PyInstaller desktop/app.spec
& ./.venv-runtime/Scripts/python.exe -m desktop.release build-manifest --directory dist/TomatoDesktop
& ./.venv-runtime/Scripts/python.exe -m desktop.release verify --directory dist/TomatoDesktop
~~~
