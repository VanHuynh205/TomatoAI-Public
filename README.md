# Tomato Desktop

Tomato Desktop là ứng dụng Windows của [VanHuynh205](https://github.com/VanHuynh205) để nhận diện dấu hiệu bệnh trên ảnh lá cà chua. Ứng dụng suy luận tại chỗ trên CPU và hiển thị thẻ chăm sóc tiếng Việt. Kết quả không thay thế chẩn đoán chuyên môn.

## Trạng thái bản public

Repo có mã nguồn, giao diện đã build, tám thẻ chăm sóc đã duyệt cùng đầu vào để kiểm lại catalog và các bài kiểm thử. **Repo chưa phân phối checkpoint DINOv2 v1 đã tinh chỉnh**, vì chủ dự án chưa có xác nhận quyền phát hành model từ các nguồn dữ liệu còn vướng. Vì vậy, bản clone sạch chưa thể nhận diện ảnh ngay. Không có lệnh tải weights tự động hoặc train lại.

Model đúng của dự án là DINOv2 v1, backbone `facebook/dinov2-base` (ViT-B/14), checkpoint epoch 6 train ở 224px và suy luận ở **280px**. DINOv2 pretrained của Meta không thay thế được đầu phân loại lá đã train. Xem [model card](MODEL_CARD.md), [ghi công dataset](DATASET_ATTRIBUTIONS.md) và [hồ sơ phát hành](RELEASE.md).

## Chạy trên Windows x64 với bundle được cấp quyền

Yêu cầu: Windows x64, [uv](https://docs.astral.sh/uv/getting-started/installation/), [Microsoft Edge WebView2 Evergreen Runtime](https://developer.microsoft.com/en-us/microsoft-edge/webview2/) và .NET Framework 4.6.2 trở lên. Bootstrap dùng CPython 3.12.14 x64 và các dependency CPU đã khóa; lần cài đầu cần Internet. Không cần GPU, CUDA, Node.js hoặc dataset huấn luyện để chạy giao diện đã build. Desktop host hiện không hỗ trợ macOS/Linux; chưa kiểm chứng Windows ARM64.

Chỉ khi bạn có quyền sử dụng bundle DINOv2 v1 tương ứng, đặt đủ sáu tệp vào:

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

## Giấy phép và ghi công

Mã ứng dụng do dự án sở hữu dùng [Apache License 2.0](LICENSE), bản quyền ©2026 VanHuynh205. Giữ [NOTICE](NOTICE) khi phân phối theo điều khoản Apache. Các revision public trước lần đổi giấy phép này đã dùng MIT; [thông báo MIT cũ](desktop/notices/project-MIT-legacy.txt) vẫn được giữ và quyền đã cấp cho các bản cũ không bị thu hồi.

DINOv2 của Meta giữ Apache 2.0. [Thông báo bên thứ ba](THIRD_PARTY_NOTICES.md) ghi tác giả, revision và thay đổi của dự án. Apache của mã nguồn không cấp lại giấy phép cho dataset, checkpoint tinh chỉnh hoặc dependency.

Nội dung chăm sóc giữ [CC BY-SA 4.0 và ghi công theo từng nguồn](infra/knowledge/CONTENT-LICENSE.md). Repo không phân phối ảnh dataset hoặc PDF nguồn. [Ghi công dataset](DATASET_ATTRIBUTIONS.md) phân biệt nguồn train/validation với các tập chỉ dùng đánh giá, cùng những phần quyền chưa xác minh.
