# Kiểm chứng bản source public ngày 13/11/2026

Code commit được kiểm: `8d807d4d91ba8573fc3dac4e85f6cc1b6c81990e`, so với baseline `5c6365938dc7b96b2378a37310b85796dd230e88`. Không train lại, không thay model, không đưa weights hoặc ảnh dataset lên Git.

## Kết quả chạy thật

Windows x64, CPython 3.12.14, Node 26.2.0, npm 11.13.0, uv 0.13.5. Clone độc lập từ Git tree của code commit bằng `git clone --no-hardlinks`; không sao chép file untracked của workspace sang clone kiểm chứng.

| Kiểm tra | Kết quả | Giới hạn |
| --- | --- | --- |
| Tạo môi trường CPU mới và sync [lockfile](../../desktop/requirements-runtime-cpu.lock) | Cài đủ 66 package chính xác, gồm torch 2.14.1+cpu | Máy đang có Python/uv; không phải Windows VM trắng |
| Care exporter và Windows prerequisites trong môi trường CPU mới | Đạt | Không mở native WebView2 window |
| `python -m pytest -q` từ clone mới | **284 passed**, 1 warning, 40,93 giây | Dùng môi trường CPU mới + pytest 9.2.0; không có editable install private; không phải 284 lượt suy luận model thật |
| Ruff trên `tests`, `backend/tests`, `desktop/tests` | Đạt | Lint phạm vi test đã phục hồi |
| `npm ci --ignore-scripts --no-audit --no-fund` từ clone mới | 168 package | Không chạy install scripts hoặc audit advisory npm |
| Frontend typecheck và Vitest | **33 passed**, typecheck đạt | DOM/unit tests |
| `npm run api:check` | Types khớp OpenAPI | Đã cố định LF khi checkout Windows |
| `npm run build` rồi so Git diff | Build đạt; dist và schema không đổi | Build frontend, không phải executable |
| E2E Chromium từ clone mới với môi trường CPU mới | **9 passed**, 43,03 giây cả startup | API fixture, không dùng model thật |
| [Bootstrap](../../Start.ps1) thiếu bundle | Exit 1, báo đủ các file thiếu, không tải weights/train | Đây là trạng thái chủ đích của clone public hiện tại |
| CPU smoke với source public + bundle v1 cục bộ | Đạt; cả 5 SHA-256 đầu vào không đổi trước/sau | Ảnh RGB tổng hợp, không chứng minh accuracy, không sao chép weights vào repo |
| Gitleaks 8.30.1 | Không phát hiện secret trong staged source và lịch sử đến code commit | Không bảo đảm không còn secret chưa phát hiện hoặc ở repo khác |

Lệnh cài runtime đã chạy:

```powershell
uv venv .venv-runtime --python 3.12.14
uv pip sync --python .venv-runtime/Scripts/python.exe --torch-backend cpu desktop/requirements-runtime-cpu.lock
```

Lệnh E2E đã chạy từ frontend, với token sinh mới chỉ trong environment, không lưu hoặc in giá trị:

```powershell
$env:TOMATO_E2E_SESSION_TOKEN = [guid]::NewGuid().ToString('N')
$env:TOMATO_E2E_PYTHON = (Resolve-Path ../.venv-runtime/Scripts/python.exe).Path
npx e2e run
```

Run ID E2E của clone mới: `01a1d137-aebe-785d-aab5-c0302f64ada4`. Báo cáo JSON do runner tạo nằm tại `frontend/.e2e/report.json` trong clone kiểm chứng cục bộ, được ignore để không công bố log/runtime environment. [Suite đã commit](../../frontend/tests/desktop.e2e.ts) bao phủ PNG/JPEG, uncertain không care, fake PNG 422, lỗi API, timeout/retry, bỏ response cũ, source bridge, keyboard/viewport nhỏ, 640px, zoom 200% và reduced motion.

Warning duy nhất của pytest là deprecation `fastapi.testclient`/`httpx`: upstream khuyên dùng `httpx2`. Không đổi lock chỉ để xóa warning trong đợt giấy phép này.

## Các lỗi phát hiện và sửa

- Checkout Windows đổi API declaration và đầu vào HTML sang CRLF: API check lệch bytes, HTML build khác bytes. [Attributes](../../.gitattributes) cố định hai đầu vào LF. Clone mới đã qua API check/build và giữ nguyên dist.
- `Write-Error` của Windows PowerShell format/wrap thông báo thiếu bundle, làm regression không đọc được chuỗi nguyên vẹn. Bootstrap ghi trực tiếp stderr, giữ exit 1; regression đã đi từ fail sang pass.
- Hook pre-push phát hiện CLI OpenAPI còn thiếu. Môi trường dev cũ có editable finder lấy module từ checkout private, nên các lượt pytest trước đó chưa đủ chứng minh tính độc lập. Đã tái hiện bằng regression kiểm origin module (fail đúng đường dẫn private), khôi phục [CLI](../../backend/openapi.py), rồi chạy lại toàn bộ bằng môi trường CPU mới không có editable private: 284 pass. CLI trong clone mới xuất đúng JSON contract, không import torch. Hook được giữ nguyên; dùng đúng Python 3.12 và pytest launcher trong env kiểm chứng thay cho công cụ global/lỗi trampoline cũ.
- Tám draft care dùng CRLF được hash-phê duyệt từ trước. Giữ nguyên bytes và nhận dạng `cr-at-eol` cho whitespace check; không chuẩn hóa lại draft, sửa hash hoặc nới care allowlist. Exporter trong clone mới đã tái tạo đúng catalog.

## Review và bất biến

Review độc lập, chỉ đọc, theo differential-review với baseline nêu trên không tìm thấy blocker cụ thể trong bootstrap, phạm vi Apache, notices, test và danh tính v1. Đây là review có phạm vi, không phải audit toàn bộ dependency hoặc xác nhận pháp lý.

Đối chiếu Git xác nhận mã inference và các module backend/runtime đã có trong baseline, [CPU lock](../../desktop/requirements-runtime-cpu.lock), [catalog](../../infra/knowledge/care-v1.json), [allowlist](../../infra/knowledge/allowlist.json) và frontend dist không đổi. Tám draft được phục hồi nguyên bytes, hash đều khớp allowlist. Model vẫn train 224px/chọn epoch 6, inference 280px và trung bình logits ảnh gốc/lật ngang; không đổi metadata 224 thành 280, không thay bằng backbone chưa fine-tune.

## Phần chưa đạt hoặc chưa chạy

**Clone public chưa thể suy luận thật ngay** vì thiếu checkpoint được cấp quyền phân phối. [Attribution dataset](../../DATASET_ATTRIBUTIONS.md) ghi rõ ba nhóm quyền cần giải quyết: PlantVillage, PlantDoc và Open Images. [Model card](../../MODEL_CARD.md) mô tả bundle sáu file/hashes; thông báo thiếu file không phải lỗi cài dependency.

Chưa phát hành hoặc kiểm executable PyInstaller, clean-machine/offline ZIP hay native pywebview end-to-end; chưa đánh giá lại accuracy, độ phù hợp lâm sàng/nông học hoặc kiểm ảnh OOD thực. Các test fixture và smoke tổng hợp không thay thế những kiểm chứng đó. Không công bố binary chứa checkpoint để né gate quyền.
