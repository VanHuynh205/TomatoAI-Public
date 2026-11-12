# Kiểm chứng bản cộng tác Windows DINOv2 v1, 13/11/2026

Bản [v0.1.0-dinov2-v1-280](https://github.com/VanHuynh205/TomatoAI-Public/releases/tag/v0.1.0-dinov2-v1-280) đưa app Windows và weights hiện có lên GitHub theo yêu cầu chủ dự án. Không huấn luyện lại, đổi checkpoint hay thêm hỗ trợ video. Đây là bản prerelease để cộng tác.

## Nội dung và cách lấy

| Asset | Kích thước byte | SHA-256 |
| --- | ---: | --- |
| [ZIP Windows x64](https://github.com/VanHuynh205/TomatoAI-Public/releases/download/v0.1.0-dinov2-v1-280/TomatoDesktop-0.1.0-dinov2-v1-280-win-x64.zip) | 616452751 | `45a45119ca0fece7f302f602b39ca989b502bb136dd3e69046b5b1c61f6bc480` |
| [ZIP model riêng](https://github.com/VanHuynh205/TomatoAI-Public/releases/download/v0.1.0-dinov2-v1-280/dinov2-v1-280-model.zip) | 322521803 | `c22c1704e7bdd22357bf0c5902d56efdbe90a09b2e7f2edfdcf47a7c9a60189f` |

[SHA256SUMS.txt](https://github.com/VanHuynh205/TomatoAI-Public/releases/download/v0.1.0-dinov2-v1-280/SHA256SUMS.txt) đi kèm hai ZIP. Windows ZIP chứa toàn bộ thư mục onedir, Python CPU runtime, model, frontend, care export, notices và manifest. Model ZIP chứa sáu tệp dưới `dinov2-v1-280/`, cùng `model-notices/` riêng; giải nén vào `ai/artifacts/` khi chạy source. Không kèm ảnh dataset, video, PDF nghiên cứu hay workspace riêng.

Giải nén toàn bộ Windows ZIP, mở `TomatoDesktop/TomatoDesktop.exe`. Máy cần Windows x64, .NET Framework 4.6.2+ và [Microsoft WebView2 Evergreen](https://developer.microsoft.com/en-us/microsoft-edge/webview2/). Không cần cài Python, Node, CUDA hay GPU cho gói desktop. EXE **NotSigned**, chưa có chữ ký Authenticode. Kiểm nguồn tải và checksum trước khi quyết định chạy; không tắt bảo vệ Windows.

## Mốc source và môi trường build

Build từ checkout public trên nền commit `83a8253ce40624c5a065493033e2867acd439e73`, cùng các thay đổi được ghi trong tag Release: cập nhật thông báo phân phối, tài liệu tải/chạy và bàn giao train, thêm hai tệp kiểm thử trainer; không đổi logic inference/launcher/spec. Các thông báo project và model đóng vào gói được đối chiếu nguyên bytes với source trước khi ZIP. Tài liệu hướng dẫn và báo cáo ngoài bundle có thể được hoàn thiện sau khi build mà không đổi executable.

Windows 11 x64; CPython 3.12.14; PyInstaller 6.22.3; Torch 2.14.1+cpu; torchvision 0.29.1+cpu; Transformers 4.57.6; safetensors 0.7.0; pywebview 6.2.1. Dùng [lock CPU](../../desktop/requirements-runtime-cpu.lock) và [spec public](../../desktop/app.spec), không dùng executable cũ từ workspace riêng. Build và probe chạy với `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.

Model giữ nguyên `dinov2-v1-280`: Meta `facebook/dinov2-base`, revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`, train 224px, epoch 6, suy luận 280px với TTA lật ngang. Backbone SHA-256 `3274d01322f079a2b7d0df0048500b1094d67f8f308c47b989ad71e8c566a4ff`; head `eb2337fbaca6915e9ccae39868fc305dbd5a8a8d95e0347abbf67a5cc1b05efa`. Sáu tệp model đóng gói trùng hash bản gốc, không sửa metadata để giả train ở 280px.

## Kiểm đã chạy

| Kiểm | Kết quả và phạm vi |
| --- | --- |
| Build PyInstaller từ source public | Thành công; tạo onedir bằng spec public và lock CPU |
| `desktop.release build-manifest`, `verify` | Thành công với candidate; kiểm inventory/hash, model, frontend, care và giới hạn bundle |
| ZIP CRC và giải nén | Cả hai ZIP qua CRC; manifest bản Windows sau giải nén được `desktop.release verify` chấp nhận |
| Frozen CPU inference | Probe console riêng dùng lại đúng PYZ/entry table và `_internal` của build mới; nạp weights thật và suy luận ảnh RGB tổng hợp 320×240; trả `model_version=dinov2-v1-280`, `status=uncertain`, ba nhãn. Probe không được thêm vào ZIP phát hành |
| Ví dụ handoff | Chạy nguyên đoạn Python trong [bàn giao](../training/IMAGE_TO_VIDEO_HANDOFF.md): nạp backbone/head local, đặc trưng `(1,1536)`, probabilities `(1,9)`, không train |
| Native desktop | Mở EXE từ candidate bằng agent-desktop; quan sát cửa sổ thật “Tomato — nhận diện lá cà chua”, WebView2 hiển thị giao diện chọn ảnh. Đây không phải browser fixture |
| Python regression | `python -m pytest -q`: **297 passed**, một cảnh báo deprecation FastAPI/Starlette về httpx |
| Frontend | Typecheck đạt; Vitest **33 passed**; API types khớp OpenAPI |
| Browser E2E | **9 passed**, e2e 0.22.0, Chromium, fixture backend riêng: ảnh PNG/JPEG, uncertain, ảnh giả, API lỗi, timeout/retry, bỏ response cũ, source bridge, keyboard/responsive/zoom/reduced motion |
| Secret scan | Gitleaks 8.30.1: bảy commit lịch sử và snapshot source được chọn, không phát hiện leak. Scanner không phải bằng chứng tuyệt đối không có secret |
| Review | Review tĩnh độc lập các thay đổi thông báo/docs/tests; không còn finding source chặn release trong phạm vi đã đọc |

Runtime test dùng virtualenv CPU tách biệt, import từ checkout public, không từ cây source riêng. `Start.ps1 -CheckOnly` tại public checkout chưa có `.venv-runtime` đã dừng đúng với thông báo thiếu môi trường; không gọi đó là bootstrap pass. Cùng lệnh kiểm bundle/care/WebView2 bên trong script được chạy thành công bằng interpreter verification tách biệt. Người dùng source chạy `Start.ps1` lần đầu để tạo môi trường theo README.

## Giới hạn

- Frozen synthetic smoke chứng minh nạp checkpoint và đường xử lý CPU; không đo accuracy, không chứng minh nhãn bệnh của một ảnh thực và không thay test tập dữ liệu giữ ngoài train.
- Native startup được quan sát trên máy build đã có WebView2/.NET. Chưa có chứng nhận clean-machine, mạng bị ngắt vật lý, mọi phiên bản Windows, antivirus/SmartScreen hay GPU. Cờ offline ngăn tải model qua Hugging Face, không tương đương test máy bị chặn mạng hoàn toàn.
- Chín browser E2E dùng fixture, không phải đo model thật hoặc toàn bộ thao tác file dialog của EXE. Không gọi probe console là thao tác nhận diện native end-to-end.
- Repo có trainer ảnh gốc và API nạp weights. Chưa có video dataloader, mô hình thời gian hay trainer video. Safetensors không chứa optimizer/scaler để resume chính xác run cũ.
- Chủ dự án yêu cầu phát hành trong khi các điểm quyền dataset/checkpoint còn mở. Upload không xác nhận clearance. Apache-2.0 chỉ cho source thuộc dự án, không tự tái cấp phép checkpoint hoặc dataset; xem [MODEL_CARD](../../MODEL_CARD.md), [DATASET_ATTRIBUTIONS](../../DATASET_ATTRIBUTIONS.md) và [NOTICE](../../NOTICE). Nội dung chăm sóc giữ CC BY-SA 4.0 và cờ chưa thẩm định chuyên môn.

Báo cáo [source public ban đầu](2026-11-13-public-release.md) giữ nguyên như bằng chứng lịch sử của đợt source-only; không dùng trạng thái chưa phân phối model trong báo cáo cũ để mô tả Release lần này.
