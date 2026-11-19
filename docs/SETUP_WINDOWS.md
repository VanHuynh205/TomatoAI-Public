# Cài Tomato Desktop trên Windows x64

Ứng dụng dùng G2 ConvNeXt-Tiny + BiGRU trên CPU. Cài Microsoft Edge WebView2 Evergreen Runtime và .NET Framework 4.6.2 trở lên trước khi mở ứng dụng. [WebView2 Evergreen](https://developer.microsoft.com/en-us/microsoft-edge/webview2/)

## Chạy app có sẵn

1. Tải `TomatoDesktop-Windows-x64-g2.zip` và `SHA256SUMS.txt` từ bản G2. [GitHub Releases](https://github.com/VanHuynh205/TomatoAI-Public/releases/tag/v0.1.0-g2-convnext-tiny-bigru)
2. Chạy `Get-FileHash .\TomatoDesktop-Windows-x64-g2.zip -Algorithm SHA256`. Chuỗi băm phải trùng dòng mang cùng tên ZIP trong `SHA256SUMS.txt`.
3. Giải nén toàn bộ ZIP. Mở `desktop/dist/field-guide-20261118/TomatoDesktop/TomatoDesktop.exe` trong thư mục giải nén. Giữ `_internal/` và `release-manifest.json` cùng thư mục với EXE.
4. Chọn ảnh PNG hoặc JPEG rồi bấm **Nhận diện**. Ứng dụng hiển thị phiên bản `cnn-bigru-20261118-g2-convnext_tiny_axial_bigru` và tri thức `care-v1`.

Windows có thể hiện cảnh báo nếu EXE chưa ký Authenticode. Đối chiếu nguồn tải và checksum trước khi quyết định chạy. Máy đã có prerequisites có thể suy luận khi mất Internet. Các liên kết nguồn chăm sóc mở qua trình duyệt.

## Chạy từ mã nguồn

Cài uv để lấy CPython 3.12.14 x64 và dependency CPU theo lockfile. [Hướng dẫn cài uv](https://docs.astral.sh/uv/getting-started/installation/)

```powershell
git clone https://github.com/VanHuynh205/TomatoAI-Public.git
Set-Location TomatoAI-Public
```

Tải `TomatoDesktop-G2-model.zip` cùng `SHA256SUMS.txt` từ bản G2. Kiểm checksum bằng `Get-FileHash`, sau đó giải nén ZIP vào **gốc repo** để đặt đủ ba tệp trong thư mục sau.

```text
ai/artifacts/cnn-bigru-20261118-g2-convnext_tiny_axial_bigru/
  cnn-gru-release.json
  leaf-classifier.json
  model.safetensors
```

```powershell
./Start.ps1
# Kiểm môi trường và bundle đã cài, rồi thoát.
./Start.ps1 -CheckOnly
```

Bootstrap dùng `.venv-runtime` và giao diện đã build trong `frontend/dist/`. Nếu máy giới hạn chạy PowerShell script, các lệnh sau cài cùng lock và mở app:

```powershell
uv venv .venv-runtime --python 3.12.14
uv pip sync --python .venv-runtime/Scripts/python.exe --torch-backend cpu desktop/requirements-runtime-cpu.lock
& ./.venv-runtime/Scripts/python.exe -m desktop
```

Nếu báo thiếu bundle, kiểm tra đường dẫn trên và số tệp. Nếu lỗi xác thực checksum, lấy lại ZIP đúng phiên bản. Nếu báo thiếu WebView2, cài Evergreen Runtime rồi chạy lại.

## Phát triển và kiểm thử

Dùng Node.js 26 trở lên cho frontend. Duy trì môi trường phát triển riêng với runtime.

```powershell
uv venv .venv-dev --python 3.12.14
uv pip sync --python .venv-dev/Scripts/python.exe --torch-backend cpu desktop/requirements-runtime-cpu.lock
uv pip install --python .venv-dev/Scripts/python.exe pytest ruff mypy
npm --prefix frontend ci
npm --prefix frontend run typecheck
npm --prefix frontend run api:check
npm --prefix frontend test
npm --prefix frontend run build
& ./.venv-dev/Scripts/python.exe -m pytest tests/test_public_release.py backend/tests desktop/tests
```

E2E Chromium dùng fixture cục bộ và khởi động máy chủ ở `127.0.0.1:8765`.

```powershell
$env:TOMATO_E2E_SESSION_TOKEN = [guid]::NewGuid().ToString('N')
$env:TOMATO_E2E_PYTHON = (Resolve-Path .venv-dev/Scripts/python.exe).Path
Push-Location frontend
npx --no-install e2e run
Pop-Location
```

Các thử nghiệm dữ liệu và huấn luyện lịch sử cần đúng dataset, manifest hoặc checkpoint theo thí nghiệm. Đọc [kế hoạch dự án](../LO_TRINH_8_TUAN.md) và [kế hoạch CNN/BiGRU](superpowers/plans/2026-11-18-cnn-bigru-colab-training.md) trước khi lấy lại đầu vào huấn luyện.

Dùng bundle G2 và frontend đã build khi đóng gói Windows onedir.

```powershell
& ./.venv-runtime/Scripts/python.exe -m PyInstaller desktop/app.spec
& ./.venv-runtime/Scripts/python.exe -m desktop.release build-manifest --directory dist/TomatoDesktop
& ./.venv-runtime/Scripts/python.exe -m desktop.release verify --directory dist/TomatoDesktop
```
