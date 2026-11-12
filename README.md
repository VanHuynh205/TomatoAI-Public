# Tomato Desktop

Tomato Desktop là ứng dụng Windows của [VanHuynh205](https://github.com/VanHuynh205) để nhận diện dấu hiệu bệnh trên ảnh lá cà chua. Ứng dụng suy luận tại chỗ trên CPU và hiển thị thẻ chăm sóc tiếng Việt. Kết quả không thay thế chẩn đoán chuyên môn.

## Trạng thái bản public

Bản desktop Windows và weights đã train được phân phối qua [GitHub Release v0.1.0-dinov2-v1-280](https://github.com/VanHuynh205/TomatoAI-Public/releases/tag/v0.1.0-dinov2-v1-280). Weights lớn hơn giới hạn tệp Git nên không nằm trong lịch sử source. Repo giữ source train ảnh, runtime, giao diện đã build, tám thẻ chăm sóc đã duyệt và các bài kiểm thử. Không train lại trong đợt phát hành này; chưa triển khai nhận diện hay train video.

Chủ dự án yêu cầu đưa model lên để cộng tác. **Các khoảng trống quyền dataset/checkpoint vẫn chưa giải quyết; upload không phải xác nhận clearance và không gán Apache-2.0 cho weights.** Xem các thông báo bên dưới trước khi tái sử dụng hoặc phân phối tiếp.

Model đúng của dự án là DINOv2 v1, backbone `facebook/dinov2-base` (ViT-B/14), checkpoint epoch 6 train ở 224px và suy luận ở **280px**. DINOv2 pretrained của Meta không thay thế được đầu phân loại lá đã train. Xem [model card](MODEL_CARD.md), [ghi công dataset](DATASET_ATTRIBUTIONS.md) và [hồ sơ phát hành](RELEASE.md).

## Tải và chạy desktop, không cần Python

1. Tải [TomatoDesktop-0.1.0-dinov2-v1-280-win-x64.zip](https://github.com/VanHuynh205/TomatoAI-Public/releases/download/v0.1.0-dinov2-v1-280/TomatoDesktop-0.1.0-dinov2-v1-280-win-x64.zip) và đối chiếu SHA-256 với [SHA256SUMS.txt](https://github.com/VanHuynh205/TomatoAI-Public/releases/download/v0.1.0-dinov2-v1-280/SHA256SUMS.txt).
2. Giải nén **toàn bộ ZIP** ra một thư mục trên Windows x64. Giữ nguyên `TomatoDesktop.exe`, `_internal/` và `release-manifest.json` cạnh nhau. Không chạy EXE ngay trong ZIP hoặc chỉ chép riêng EXE.
3. Máy cần [WebView2 Evergreen Runtime](https://developer.microsoft.com/en-us/microsoft-edge/webview2/) và .NET Framework 4.6.2 trở lên. Mở `TomatoDesktop.exe`, chờ nạp model rồi chọn ảnh để nhận diện. Gói đã có model và Python runtime CPU; không cần GPU, CUDA, Node.js hoặc tải thêm weights.

EXE chưa có chữ ký Authenticode. Nếu Windows cảnh báo, kiểm lại nguồn tải và checksum trước khi quyết định chạy; không tắt SmartScreen hoặc phần mềm bảo mật. Khi đã có prerequisites, suy luận chạy tại máy và không cần Internet; mở liên kết nguồn chăm sóc có thể cần Internet. Đây là bản cộng tác thử nghiệm, chưa phải kết quả kiểm trên mọi máy Windows.

## Chạy từ source trên Windows x64

Yêu cầu: Windows x64, [uv](https://docs.astral.sh/uv/getting-started/installation/), [Microsoft Edge WebView2 Evergreen Runtime](https://developer.microsoft.com/en-us/microsoft-edge/webview2/) và .NET Framework 4.6.2 trở lên. Bootstrap dùng CPython 3.12.14 x64 và các dependency CPU đã khóa; lần cài đầu cần Internet. Không cần GPU, CUDA, Node.js hoặc dataset huấn luyện để chạy giao diện đã build. Desktop host hiện không hỗ trợ macOS/Linux; chưa kiểm chứng Windows ARM64.

Tải [dinov2-v1-280-model.zip](https://github.com/VanHuynh205/TomatoAI-Public/releases/download/v0.1.0-dinov2-v1-280/dinov2-v1-280-model.zip), đối chiếu checksum trong Release rồi giải nén vào `ai/artifacts/`. Sau giải nén phải có đúng sáu tệp model ở vị trí sau (thư mục `model-notices/` đi kèm giữ các thông báo riêng):

```text
ai/artifacts/dinov2-v1-280/
  classifier.safetensors
  config.json
  dinov2-release.json
  leaf-classifier.json
  model.safetensors
  preprocessor_config.json
```

Từ PowerShell tại thư mục repo:

```powershell
./Start.ps1
# Kiểm lại môi trường đã cài, hashes model, catalog và WebView2, không mở app:
./Start.ps1 -CheckOnly
```

Script dừng trước khi cài dependency nếu thiếu bundle. Script không thay execution policy của máy, không tải weights và không train. Nếu chính sách PowerShell của tổ chức chặn script, dùng các lệnh thủ công dưới đây; không tắt chính sách bảo mật chỉ để chạy ứng dụng.

```powershell
uv venv .venv-runtime --python 3.12.14
uv pip sync --python .venv-runtime/Scripts/python.exe --torch-backend cpu desktop/requirements-runtime-cpu.lock
& ./.venv-runtime/Scripts/python.exe -m desktop
```

Không đồng bộ lock runtime vào môi trường train hoặc môi trường đang dùng của bạn. Sau khi cài đủ dependency, WebView2 và bundle hợp lệ, suy luận không cần Internet. Liên kết tài liệu nguồn mở trình duyệt và có thể cần Internet.

## Build giao diện và chạy kiểm thử

Dùng Node.js 26 trở lên để build lại frontend. Bản clone có sẵn `frontend/dist` cho runtime. Cài công cụ phát triển vào môi trường riêng:

```powershell
uv venv .venv-dev --python 3.12.14
uv pip sync --python .venv-dev/Scripts/python.exe --torch-backend cpu desktop/requirements-runtime-cpu.lock
uv pip install --python .venv-dev/Scripts/python.exe --no-deps -e .
uv pip install --python .venv-dev/Scripts/python.exe pytest ruff
& ./.venv-dev/Scripts/python.exe -m pytest
npm --prefix frontend ci
npm --prefix frontend run typecheck
npm --prefix frontend test
npm --prefix frontend run build
npm --prefix frontend run api:check
```

Các test fixture không cần weights thật và không đo độ chính xác model. Cấu hình E2E khởi động API fixture riêng tại `http://127.0.0.1:8765`; không trỏ vào ứng dụng production. Từ PowerShell ở thư mục repo:

```powershell
$env:TOMATO_E2E_SESSION_TOKEN = [guid]::NewGuid().ToString('N')
$env:TOMATO_E2E_PYTHON = (Resolve-Path .venv-dev/Scripts/python.exe).Path
Push-Location frontend
npx e2e run
Pop-Location
```

Không commit token. Các bài E2E hiện dùng locator/assertion tường minh, không có bước `agent.act`, nên không cần model API key. Runner cần trình duyệt Chromium phù hợp. Báo cáo nằm trong `frontend/.e2e/`.

Kết quả kiểm source ban đầu nằm trong [báo cáo source ngày 13/11/2026](docs/verification/2026-11-13-public-release.md). Kiểm bản Windows lần này được ghi riêng trong [báo cáo release Windows](docs/verification/2026-11-13-windows-release.md); test fixture không thay thế kiểm executable thật.

## Làm tiếp train ảnh và hướng video

Đọc [bàn giao DINOv2 ảnh sang video](docs/training/IMAGE_TO_VIDEO_HANDOFF.md) trước khi sửa trainer. Tài liệu chỉ đúng source ảnh gốc, cách nạp backbone/head đã train và lấy đặc trưng từng frame, schema manifest và quy tắc tách clip để tránh leakage. Có ví dụ warm-start từ weights phát hành. Bundle suy luận không chứa optimizer/scaler để resume chính xác lượt train cũ; repo chưa có trainer video hoặc mô hình thời gian. Không kèm raw dataset, ảnh hay video.

## Giấy phép và ghi công

Mã ứng dụng do dự án sở hữu dùng [Apache License 2.0](LICENSE), bản quyền ©2026 VanHuynh205. Giữ [NOTICE](NOTICE) khi phân phối theo điều khoản Apache. Các revision public trước lần đổi giấy phép này đã dùng MIT; [thông báo MIT cũ](desktop/notices/project-MIT-legacy.txt) vẫn được giữ và quyền đã cấp cho các bản cũ không bị thu hồi.

DINOv2 của Meta giữ Apache 2.0. [Thông báo bên thứ ba](THIRD_PARTY_NOTICES.md) ghi tác giả, revision và thay đổi của dự án. Apache của mã nguồn không cấp lại giấy phép cho dataset, checkpoint tinh chỉnh hoặc dependency.

Nội dung chăm sóc giữ [CC BY-SA 4.0 và ghi công theo từng nguồn](infra/knowledge/CONTENT-LICENSE.md). Repo không phân phối ảnh dataset hoặc PDF nguồn. [Ghi công dataset](DATASET_ATTRIBUTIONS.md) phân biệt nguồn train/validation với các tập chỉ dùng đánh giá, cùng những phần quyền chưa xác minh.
