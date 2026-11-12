# Cài Tomato Desktop trên máy Windows mới

Bạn có thể chạy app có sẵn hoặc cài mã nguồn để sửa chương trình. Nếu bạn chỉ muốn chọn ảnh lá và xem kết quả, làm theo phần A. Nếu bạn cần phát triển phần video từ mô hình ảnh, làm phần B rồi đọc phần C.

Bản phát hành này dùng đúng weights DINOv2 v1 đã train: checkpoint epoch 6, train ở 224px, suy luận ở 280px. Bạn không cần train lại để mở app hoặc chạy ví dụ nạp model.

## Chọn cách cài

| Nhu cầu của bạn | Phần cần làm | Bạn cần cài thêm gì? |
| --- | --- | --- |
| Mở app và nhận diện ảnh | [A. Chạy app có sẵn](#a-chạy-app-có-sẵn) | WebView2 và .NET Framework nếu máy còn thiếu |
| Đọc, sửa Python hoặc dùng weights cho hướng video | [B. Chạy từ mã nguồn](#b-chạy-từ-mã-nguồn) | Các thành phần của A, Git và uv |
| Sửa giao diện React | [C. Môi trường phát triển](#c-môi-trường-phát-triển-và-bàn-giao-video) | Node.js 26 trở lên, ngoài môi trường Python |

`Weights` là các tệp chứa tham số model đã học. `Runtime` là thành phần cần để chạy chương trình. Gói app đã kèm weights và Python runtime CPU. Bạn tải model riêng khi dùng mã nguồn.

## Máy cần đáp ứng điều kiện nào?

- Dùng Windows x64 trên máy Intel/AMD. Trong **Settings > System > About > System type**, kiểm tra dòng tương đương `64-bit operating system, x64-based processor`. Dự án chưa kiểm chứng Windows ARM64; chưa hỗ trợ desktop macOS/Linux.
- Máy cần Microsoft .NET Framework 4.6.2 trở lên và WebView2 Evergreen Runtime. Phần A1 chỉ cách cài. Bản build đã chạy trên Windows 11 x64; dự án chưa có kết quả thử trên từng phiên bản Windows khác.
- Với bản app có sẵn, dành khoảng 2GB trống cho ZIP và thư mục giải nén, chưa tính bộ cài Microsoft. ZIP khoảng 616MB, thư mục app khoảng 1,1GB. Đây là dự trù lưu trữ theo gói hiện tại, không phải phép đo RAM tối thiểu.
- Bạn cần Internet để tải app, bộ cài hoặc dependency lần đầu. Sau khi cài đủ, app suy luận tại máy. Khi bấm liên kết tài liệu chăm sóc, bạn cần mạng để mở trang nguồn.

Bạn không cần GPU, CUDA, tài khoản Hugging Face hoặc API key để nhận diện ảnh bằng bản này. Cài từ source cần thêm chỗ cho môi trường Python và cache tải dependency; không dùng mốc 2GB ở trên cho toàn bộ môi trường phát triển.

## A. Chạy app có sẵn

### A1. Cài thành phần Microsoft

Bạn có thể bỏ qua thành phần mà máy đã cài. Nếu bộ cài báo đã có phiên bản phù hợp hoặc mới hơn, đóng bộ cài rồi tiếp tục. Máy do trường hoặc công ty quản lý có thể cần quản trị viên hỗ trợ; không đổi chính sách bảo mật của máy.

#### WebView2 Evergreen Runtime

1. Mở [trang WebView2 của Microsoft](https://developer.microsoft.com/en-us/microsoft-edge/webview2/).
2. Tìm mục **Download the WebView2 Runtime**.
3. Trong **Evergreen Standalone Installer**, chọn **x64**. Bộ cài này có đủ runtime để cài khi không có mạng.
4. Mở tệp vừa tải, kiểm tra nhà phát hành Microsoft rồi làm theo hộp thoại cài đặt. Chấp nhận yêu cầu quyền quản trị nếu bạn có quyền cài trên máy.
5. Mở lại Tomato Desktop sau khi cài xong.

Chọn Evergreen Standalone Installer, không chọn Fixed Version. Việc máy có trình duyệt Microsoft Edge chưa đủ để kết luận app tìm thấy WebView2 Runtime đã đăng ký trong Windows. Tomato kiểm WebView2 từ phiên bản 86 trở lên; khi cài mới, dùng bản Evergreen mà Microsoft đang cung cấp.

#### .NET Framework

1. Mở [trang tải .NET Framework 4.8 của Microsoft](https://dotnet.microsoft.com/en-us/download/dotnet-framework/net48).
2. Chọn bộ cài **Runtime** dành cho người chạy ứng dụng, rồi làm theo hướng dẫn của Microsoft cho phiên bản Windows của bạn. Bạn không cần Developer Pack để mở Tomato.
3. Khởi động lại Windows nếu bộ cài yêu cầu, rồi mở lại app.

App yêu cầu .NET Framework 4.6.2 trở lên; 4.8 đáp ứng yêu cầu đó trên hệ điều hành tương thích. Một bản .NET Framework mới hơn mà Windows đã có cũng đáp ứng ngưỡng kiểm của app. .NET 8/9/10 là dòng sản phẩm khác; đừng chọn chúng thay cho .NET Framework trong bước này.

### A2. Tải đúng gói app

Mở [Release v0.1.0-dinov2-v1-280](https://github.com/VanHuynh205/TomatoAI-Public/releases/tag/v0.1.0-dinov2-v1-280). Trong mục **Assets**, tải hai tệp:

| Tệp | Link tải | Công dụng |
| --- | --- | --- |
| ZIP Windows, khoảng 616MB | [TomatoDesktop-0.1.0-dinov2-v1-280-win-x64.zip](https://github.com/VanHuynh205/TomatoAI-Public/releases/download/v0.1.0-dinov2-v1-280/TomatoDesktop-0.1.0-dinov2-v1-280-win-x64.zip) | App, Python runtime CPU và weights đã train |
| Danh sách checksum | [SHA256SUMS.txt](https://github.com/VanHuynh205/TomatoAI-Public/releases/download/v0.1.0-dinov2-v1-280/SHA256SUMS.txt) | Hash để đối chiếu ZIP tải về |

Bạn không cần tải model ZIP riêng cho cách A. Hai mục GitHub tạo sẵn mang tên **Source code (zip)** và **Source code (tar.gz)** chứa mã nguồn, không phải app Windows đóng gói.

Đợi trình duyệt tải xong. Nếu trình duyệt thêm `(1)` vào tên tệp vì bạn tải hai lần, dùng đúng tên đang có trên máy khi chạy lệnh kiểm bên dưới.

### A3. Kiểm tệp tải về bằng SHA-256

Checksum là một chuỗi tính từ nội dung tệp. So khớp checksum giúp bạn phát hiện ZIP tải thiếu hoặc khác với gói phát hành; nó không thay chữ ký số hay kiểm tra bảo mật của Windows.

1. Mở thư mục chứa ZIP, thường là **Downloads**.
2. Bấm thanh địa chỉ của File Explorer, gõ `powershell` rồi nhấn Enter. Cửa sổ PowerShell sẽ mở tại thư mục đó. Bạn không cần chạy quyền Administrator.
3. Chép khối lệnh dưới đây vào PowerShell, nhấn Enter. Nếu bạn đổi tên ZIP, sửa tên trong lệnh cho khớp.

```powershell
$zip = '.\TomatoDesktop-0.1.0-dinov2-v1-280-win-x64.zip'
$expected = '45a45119ca0fece7f302f602b39ca989b502bb136dd3e69046b5b1c61f6bc480'
$actual = (Get-FileHash -LiteralPath $zip -Algorithm SHA256 -ErrorAction Stop).Hash
if ($actual -ne $expected) { throw 'Checksum khong khop. Dung lai va tai lai ZIP tu Release.' }
'Checksum khop.'
```

Bạn cần thấy `Checksum khop.`. Hash này cũng nằm trong [SHA256SUMS.txt](https://github.com/VanHuynh205/TomatoAI-Public/releases/download/v0.1.0-dinov2-v1-280/SHA256SUMS.txt). Nếu lệnh báo không tìm thấy tệp, kiểm tra thư mục PowerShell đang đứng bằng `Get-Location` và tên ZIP trong File Explorer. Nếu hash khác, dừng tại đây và tải lại từ Release.

### A4. Giải nén toàn bộ thư mục

1. Trong File Explorer, nhấp chuột phải ZIP rồi chọn **Extract All... / Giải nén tất cả...**.
2. Chọn một thư mục bạn có quyền ghi, chẳng hạn thư mục `TomatoAI` do bạn tạo trong Documents. Không chọn Program Files cho lần thử này.
3. Bấm **Extract**. Chờ giải nén xong rồi mở thư mục `TomatoDesktop` bên trong.

Bạn cần thấy cấu trúc sau. Windows có thể ẩn đuôi `.exe`; bật **File name extensions / Phần mở rộng tên tệp** trong menu View nếu bạn cần phân biệt tên.

```text
Thu-muc-ban-chon/
  README-WINDOWS.txt
  TomatoDesktop/
    TomatoDesktop.exe
    release-manifest.json
    _internal/
      ...
```

Giữ EXE, manifest và `_internal` cạnh nhau. App kiểm danh sách tệp và hash khi khởi động; việc sửa, làm thiếu hoặc thêm tệp vào thư mục gói có thể khiến kiểm tra dừng app. Lưu ảnh cần nhận diện ở một thư mục khác. Nếu muốn tạo shortcut, đặt shortcut trên Desktop, bên ngoài thư mục `TomatoDesktop`.

Đừng mở EXE khi bạn còn đang xem nội dung ZIP. Cũng đừng chép riêng EXE ra Desktop: phần Python runtime và weights nằm trong thư mục đi kèm.

### A5. Mở app và thử một ảnh

1. Mở `TomatoDesktop.exe` từ thư mục đã giải nén. App cần thời gian nạp model vào RAM; đừng mở nhiều bản trong lúc chờ.
2. Khi app mở cửa sổ **Tomato — nhận diện lá cà chua**, bấm **Chọn ảnh lá**.
3. Chọn một ảnh **JPEG hoặc PNG tĩnh** trên máy. Dùng ảnh không quá **10MB (10.000.000 byte)** và **20 triệu điểm ảnh**. Ví dụ ảnh 4000 × 3000 có 12 triệu điểm ảnh, còn ảnh 6000 × 4000 vượt giới hạn dù dung lượng tệp dưới 10MB. App chưa nhận HEIC, WebP hoặc video. Đổi đuôi tệp sang `.jpg` không chuyển được định dạng ảnh.
4. Sau khi giao diện hiển thị ảnh xem trước, bấm **Nhận diện**.
5. Đọc nhãn và thẻ chăm sóc nếu app có kết quả phù hợp. Nếu app hiện **Chưa đủ cơ sở xác định**, thử ảnh rõ hơn của một lá; app sẽ không chọn thẻ chăm sóc cho kết quả này. **Điểm mô hình** dùng để xếp hạng nhãn, không phải xác nhận bệnh học. Bạn có thể mở **Xem điểm các nhãn** để kiểm tên model `dinov2-v1-280`.

Bạn đã hoàn thành cài đặt khi app mở được giao diện, nhận ảnh hợp lệ và trả kết quả hoặc trạng thái chưa đủ chắc chắn sau khi bạn bấm Nhận diện. Việc một ảnh cho kết quả chưa chắc chắn không đồng nghĩa bạn cài sai model.

EXE chưa có chữ ký Authenticode, nên Windows có thể hiện cảnh báo nhà phát hành chưa xác định. Kiểm lại repo, Release và checksum trước khi quyết định chạy. Nếu chính sách máy chặn app, gửi thông tin cho quản trị viên hoặc chủ dự án; không tắt Defender/SmartScreen, thêm ngoại lệ antivirus hoặc bỏ qua cảnh báo để tiếp tục.

### A6. Mở lại, cập nhật và gỡ bản portable

Những lần sau, mở EXE trong cùng thư mục đã giải nén. Bạn không cần cài lại Python hay tải lại weights. Bạn cũng không cần giữ PowerShell mở khi chạy EXE.

Khi lấy bản phát hành mới, giải nén vào thư mục mới và kiểm checksum của bản đó. Tránh chép đè từng tệp vào thư mục bản cũ vì manifest gắn với đúng nội dung một gói. ZIP hiện tại là bản portable, không có trình gỡ cài đặt riêng; sau khi đóng app, bạn có thể xóa thư mục app và ZIP nếu không cần giữ. Các runtime Microsoft dùng chung với ứng dụng khác nên không cần gỡ chúng cùng Tomato.

## B. Chạy từ mã nguồn

Bạn cần cách này khi muốn đọc hoặc sửa Python, nạp weights cho thử nghiệm của mình, hay làm tiếp phần video. Bạn không cần thực hiện phần B chỉ để mở app có sẵn.

Các lệnh dưới đây dùng **PowerShell**, không phải Command Prompt hay Git Bash. Chạy từng khối theo thứ tự và dừng khi gặp lỗi. Dấu `#` mở đầu dòng ghi chú; bạn không gõ dấu nhắc `PS>`.

### B1. Cài Git và uv

Cài WebView2/.NET theo [A1](#a1-cài-thành-phần-microsoft) trước khi mở desktop từ source.

- Lấy **Git for Windows x64 Setup** ở [trang Git chính thức](https://git-scm.com/install/windows), rồi chạy bộ cài. Bạn cần chọn tùy chọn cho phép dùng Git từ command line trong quá trình cài.
- `uv` quản lý Python và cài dependency. Nếu máy có WinGet, mở PowerShell và chạy lệnh mà [tài liệu uv](https://docs.astral.sh/uv/getting-started/installation/#winget) hướng dẫn:

```powershell
winget install --id=astral-sh.uv -e
```

Nếu máy không có `winget`, dùng cách cài Windows phù hợp trên [trang cài uv](https://docs.astral.sh/uv/getting-started/installation/). Không thay execution policy để chạy một installer mà chính sách máy không cho phép.

Đóng PowerShell cũ sau khi cài Git/uv, mở cửa sổ mới để nhận thay đổi PATH. Kiểm tra:

```powershell
git --version
uv --version
```

Bạn cần thấy phiên bản của cả hai. Nếu Windows báo `not recognized`, kiểm tra lại cài đặt và mở terminal mới trước khi làm tiếp. Bootstrap dùng `uv` để tạo CPython **3.12.14 x64** riêng; bạn không cần gỡ bản Python đang dùng cho công việc khác.

### B2. Lấy repo và đứng đúng thư mục

Ví dụ sau tạo thư mục làm việc trong thư mục người dùng của bạn. Chỉ chạy `git clone` khi chưa có thư mục repo cùng tên tại vị trí đó.

```powershell
New-Item -ItemType Directory -Path "$HOME\TomatoAI" -Force | Out-Null
Set-Location "$HOME\TomatoAI"
git clone https://github.com/VanHuynh205/TomatoAI-Public.git
Set-Location .\TomatoAI-Public
Get-Location
Get-Item .\Start.ps1, .\pyproject.toml
```

Bạn cần thấy thư mục kết thúc bằng `TomatoAI-Public` và hai tệp [Start.ps1](../Start.ps1), [pyproject.toml](../pyproject.toml). Các lệnh từ B3 trở đi đều chạy tại đây, không chạy trong thư mục `frontend` hoặc `ai`.

Nếu bạn không dùng Git, mở [repo](https://github.com/VanHuynh205/TomatoAI-Public), chọn **Code > Download ZIP**, giải nén toàn bộ rồi mở PowerShell tại thư mục chứa [Start.ps1](../Start.ps1). Cách này cho phép đọc/chạy source, nhưng không có lịch sử Git và không dùng được `git pull`. GitHub thường đặt tên thư mục ZIP là `TomatoAI-Public-main`; tên khác không sao, vị trí của các tệp bên trong mới cần đúng.

### B3. Tải model riêng và kiểm checksum

Tải [dinov2-v1-280-model.zip](https://github.com/VanHuynh205/TomatoAI-Public/releases/download/v0.1.0-dinov2-v1-280/dinov2-v1-280-model.zip), khoảng 323MB. Repo Git và ZIP source không kèm weights. Giữ model v1 này để dùng với runtime hiện tại; việc tải DINOv2 pretrained của Meta không bổ sung được đầu phân loại lá đã train.

Khối lệnh dưới giả sử trình duyệt lưu model ZIP ở Downloads của tài khoản đang dùng. Nếu bạn lưu chỗ khác, sửa giá trị `$modelZip` thành đường dẫn thực tế. Dùng dấu nháy khi đường dẫn có khoảng trắng.

```powershell
$modelZip = "$HOME\Downloads\dinov2-v1-280-model.zip"
$expected = 'c22c1704e7bdd22357bf0c5902d56efdbe90a09b2e7f2edfdcf47a7c9a60189f'
$actual = (Get-FileHash -LiteralPath $modelZip -Algorithm SHA256 -ErrorAction Stop).Hash
if ($actual -ne $expected) { throw 'Checksum model khong khop. Dung lai va tai lai ZIP.' }
'Checksum model khop.'
```

Chỉ tiếp tục khi lệnh in `Checksum model khop.`. Thư mục đích ở bước sau dành cho model trong source, khác với thư mục app portable ở phần A.

### B4. Đặt model đúng chỗ

Từ thư mục repo, chạy:

```powershell
New-Item -ItemType Directory -Path .\ai\artifacts -Force | Out-Null
Expand-Archive -LiteralPath $modelZip -DestinationPath .\ai\artifacts
Get-ChildItem .\ai\artifacts\dinov2-v1-280 -File | Select-Object Name,Length
```

Không dùng tùy chọn ghi đè nếu bạn đã có model tại đích. Kiểm tra bản đang có trước khi tải/giải nén thêm. Sau giải nén, cần có sáu tệp trực tiếp trong thư mục `dinov2-v1-280`, cùng thư mục thông báo riêng:

```text
TomatoAI-Public/
  Start.ps1
  pyproject.toml
  ai/
    artifacts/
      dinov2-v1-280/
        classifier.safetensors
        config.json
        dinov2-release.json
        leaf-classifier.json
        model.safetensors
        preprocessor_config.json
      model-notices/
        ...
  desktop/
  frontend/
    dist/
  infra/
    knowledge/
```

Sai vị trí thường gặp là có thêm một lớp thư mục, chẳng hạn `ai/artifacts/dinov2-v1-280/dinov2-v1-280/model.safetensors`. Đường dẫn cần đúng là `ai/artifacts/dinov2-v1-280/model.safetensors`. Giữ nguyên nội dung metadata và [các thông báo model](../MODEL_CARD.md); không sửa hash để bỏ qua lỗi kiểm.

### B5. Tạo môi trường và mở app

Ở PowerShell tại repo, chạy:

```powershell
./Start.ps1
```

Lần đầu, script kiểm đủ tên tệp model, kiểm có `uv`, tạo `.venv-runtime` với Python 3.12.14, rồi cài dependency theo [lock CPU](../desktop/requirements-runtime-cpu.lock). Tiếp đó, script kiểm Python, hash model, catalog chăm sóc và WebView2 trước khi mở app. Bạn cần mạng trong lúc `uv` tải Python/dependency; thời gian cài tùy đường truyền và máy.

Khi kiểm đầu vào đạt, terminal in tên `dinov2-v1-280` và dòng `Runtime inputs verified.`. Sau đó app mở cửa sổ để bạn thử theo [A5](#a5-mở-app-và-thử-một-ảnh). Giữ PowerShell mở trong phiên chạy từ source.

Sau lần cài đầu, đóng app rồi kiểm lại môi trường mà không mở cửa sổ:

```powershell
./Start.ps1 -CheckOnly
```

Bạn cần thấy `Runtime inputs verified.`. Tham số `-CheckOnly` không tạo môi trường và không cài dependency. Nếu chưa có `.venv-runtime`, chạy cách cài thường trước; đừng dùng CheckOnly làm lệnh cài đầu tiên.

`Start.ps1` không tải weights, đổi execution policy hoặc train model. Script đồng bộ lại dependency khi bạn chạy không có `-CheckOnly`; giữ `.venv-runtime` riêng cho runtime, không dùng nó làm môi trường train GPU.

### B6. Chạy thủ công khi PowerShell chặn script

Nếu PowerShell báo `running scripts is disabled` hoặc `cannot be loaded`, bạn có thể dùng các lệnh thủ công thay cho script, trong phạm vi chính sách của máy. Giữ nguyên execution policy. Tại repo, sau khi đã cài uv và đặt model, chạy các lệnh dưới đây. Nếu `.venv-runtime` đã tồn tại, bỏ qua lệnh `uv venv`; đừng tạo lại hoặc xóa môi trường chỉ vì script bị chặn.

```powershell
uv venv .venv-runtime --python 3.12.14
uv pip sync --python .venv-runtime/Scripts/python.exe --torch-backend cpu desktop/requirements-runtime-cpu.lock
```

Kiểm phiên bản Python trong môi trường đang dùng:

```powershell
& ./.venv-runtime/Scripts/python.exe -c "import struct,sys; print(sys.version); print(struct.calcsize('P')*8)"
```

Kết quả cần cho thấy Python `3.12.14` và kiến trúc `64`. Chạy kiểm đầu vào tương đương phần preflight của script:

```powershell
& ./.venv-runtime/Scripts/python.exe -c "from pathlib import Path; from ai.dinov2_inference import validate_dinov2_export; from desktop.release import validate_care_export; from desktop.prerequisites import require_webview2; print(validate_dinov2_export(Path('ai/artifacts/dinov2-v1-280'))); validate_care_export(Path('infra/knowledge'),Path('infra/knowledge')); require_webview2(); print('Runtime inputs verified.')"
```

Khi kiểm đạt, mở app:

```powershell
& ./.venv-runtime/Scripts/python.exe -m desktop
```

Bạn không cần kích hoạt virtualenv hoặc đổi PATH vì các lệnh gọi thẳng Python của dự án. Nếu chính sách tổ chức chặn cả các chương trình này, nhờ quản trị viên hỗ trợ.

### B7. Mở lại sau khi đã cài

Mở PowerShell tại repo rồi gọi Python trong môi trường runtime:

```powershell
& ./.venv-runtime/Scripts/python.exe -m desktop
```

Bạn dùng lệnh này để mở app mà không yêu cầu `uv` sync lại dependency. Nếu bạn cập nhật source và cần cài đúng lock hiện tại, dùng [Start.ps1](../Start.ps1). Đừng chép weights vào lịch sử Git; repo đã ignore thư mục artifact để giữ các tệp lớn ngoài source.

## C. Môi trường phát triển và bàn giao video

### C1. Tạo môi trường Python để sửa và kiểm thử

Môi trường riêng giúp bạn tránh làm thay đổi dependency của app đang chạy. Các lệnh sau cài bộ CPU đã khóa và đăng ký repo ở chế độ editable, tức Python lấy mã bạn sửa trong repo. Chúng không train model.

```powershell
uv venv .venv-dev --python 3.12.14
uv pip sync --python .venv-dev/Scripts/python.exe --torch-backend cpu desktop/requirements-runtime-cpu.lock
uv pip install --python .venv-dev/Scripts/python.exe --no-deps -e .
uv pip install --python .venv-dev/Scripts/python.exe pytest ruff
& ./.venv-dev/Scripts/python.exe -m pytest
```

Khi chuyển sang môi trường dev, dùng Python của `.venv-dev` trong các lệnh thử nghiệm. Không gọi nhầm Python hệ thống hoặc môi trường của dự án khác.

### C2. Chỉ cài Node.js khi sửa giao diện

Source đã có [frontend build](../frontend/dist/index.html), nên phần B không cần Node.js. Nếu bạn sửa React/TypeScript, cài [Node.js](https://nodejs.org/en/download) bản **26 trở lên** phù hợp Windows x64. Kiểm yêu cầu này thay vì chọn mặc định một bản LTS cũ hơn 26.

Mở terminal mới sau cài đặt, kiểm tra rồi build từ thư mục gốc repo:

```powershell
node --version
npm --version
npm --prefix frontend ci
npm --prefix frontend run typecheck
npm --prefix frontend test
npm --prefix frontend run build
npm --prefix frontend run api:check
```

`node --version` cần từ `v26` trở lên. Nếu PowerShell chặn `npm.ps1`, dùng `npm.cmd` thay cho `npm` trong các lệnh trên; không hạ execution policy. Sau khi build, đóng rồi mở lại desktop để nạp giao diện mới. Đừng chạy Vite rồi coi đó là bản desktop, vì frontend cần backend và bridge của host.

Cấu hình kiểm browser nằm trong [e2e.config.ts](../frontend/e2e.config.ts); [README](../README.md#build-giao-diện-và-chạy-kiểm-thử) ghi lệnh chạy E2E. Các test đó dùng API fixture và không đo chất lượng weights thật.

### C3. Dùng model ảnh làm nền cho video

Sau khi bạn chạy được source, đọc [bàn giao DINOv2 ảnh sang video](training/IMAGE_TO_VIDEO_HANDOFF.md). Tài liệu có ví dụ nạp backbone/head, lấy vector đặc trưng 1536 chiều từ frame, schema manifest và quy tắc tách clip tránh rò rỉ dữ liệu giữa train/validation/test.

Bạn có thể dùng weights phát hành làm điểm khởi tạo cho thử nghiệm tiếp theo. Bundle safetensors chỉ có tham số mô hình, không chứa optimizer/scaler để resume đúng trạng thái lượt train cũ. Repo chưa có video dataloader, mô hình thời gian hay trainer video.

Môi trường CPU ở C1 phục vụ nạp model, suy luận và kiểm thử. Trainer ảnh gốc cần GPU và bản Torch hỗ trợ CUDA như [hướng dẫn huấn luyện](training/IMAGE_TO_VIDEO_HANDOFF.md#môi-trường). Không tự đổi Torch trong `.venv-runtime` để thử train. Repo cũng không kèm dataset gốc; bạn cần chuẩn bị dữ liệu có quyền sử dụng theo manifest trong tài liệu bàn giao.

## D. Tra lỗi theo thông báo

| Triệu chứng hoặc thông báo | Bạn kiểm tra và xử lý thế nào? |
| --- | --- |
| Thiếu DLL, không nạp được thành phần khởi động | Kiểm bạn đã giải nén toàn bộ ZIP và giữ `_internal` cạnh EXE. Nếu antivirus cách ly tệp, giữ thông tin cảnh báo và liên hệ quản trị viên/chủ dự án; không khôi phục tệp hoặc tạo ngoại lệ trước khi xác minh. Tải lại gói chính thức và so checksum nếu cần. |
| App báo kiểm gói/manifest không đạt | Dùng thư mục giải nén mới từ ZIP đúng checksum. Giữ ảnh, log và shortcut ngoài thư mục app; không sửa manifest hoặc weights. |
| `Không tìm thấy Microsoft Edge WebView2 Runtime phù hợp` | Cài WebView2 Evergreen Standalone **x64** theo A1 rồi mở lại. Cài trình duyệt Edge riêng chưa thay thế bước này. |
| App báo cần `.NET Framework 4.6.2 trở lên` | Dùng bộ cài .NET Framework phù hợp Windows theo A1. Khởi động lại nếu Microsoft yêu cầu. |
| Windows/antivirus chặn EXE | Kiểm nguồn Release và checksum, lưu cảnh báo để đối chiếu. Gói hiện chưa ký số. Không tắt bảo vệ hoặc tự thêm ngoại lệ. |
| `git`, `uv`, `node` hoặc `npm` không được nhận diện | Cài đúng công cụ cho nhánh bạn đang làm và mở terminal mới. Người chạy app phần A không cần các lệnh này. |
| Không tìm thấy [Start.ps1](../Start.ps1) hoặc lock dependency | Mở PowerShell tại thư mục gốc repo, nơi có [pyproject.toml](../pyproject.toml), rồi chạy lại. |
| `Missing authorized DINOv2 v1 bundle files` | Xem cây thư mục ở B4. Clone Git chưa có weights; giải nén model ZIP sao cho sáu tệp nằm trực tiếp trong `ai/artifacts/dinov2-v1-280/`. |
| Hash model không khớp | So hash ZIP trước, rồi giải nén lại vào đích mới để đối chiếu. Giữ nguyên metadata; không sửa hash hoặc thay weights pretrained. |
| `Runtime environment missing` khi dùng CheckOnly | Chạy [Start.ps1](../Start.ps1) không có `-CheckOnly` trước, hoặc cài thủ công theo B6. |
| `running scripts is disabled`, `npm.ps1 cannot be loaded` | Với Python, dùng B6. Với npm, gọi `npm.cmd`. Giữ chính sách bảo mật của máy. |
| Cài dependency lỗi mạng, proxy hoặc chứng chỉ | Kiểm mạng và cấu hình proxy của tổ chức. Giữ thông báo lỗi đã che thông tin riêng, nhờ quản trị viên xử lý chứng chỉ; không tắt TLS verification hay thêm nguồn package không tin cậy. |
| Nút Nhận diện chưa bấm được hoặc ảnh bị từ chối | Chọn ảnh JPEG/PNG thật, đúng giới hạn ở A5, đợi xử lý ảnh. Bạn có thể xuất ảnh từ trình xem ảnh thành JPEG/PNG; chỉ đổi đuôi tệp không đủ. |
| App báo chưa đủ chắc chắn hoặc không phải lá cà chua | Thử ảnh rõ, đủ sáng, thấy lá thay vì ảnh toàn cảnh. Đây là kết quả suy luận, không phải lỗi cài đặt theo mặc định. Không tự tăng điểm hoặc bỏ lớp từ chối để làm kết quả trông chắc hơn. |
| App báo timeout khi nhận diện | Giữ thông báo, thử lại theo nút trên giao diện. Nếu lỗi lặp lại, gửi cấu hình máy và thông tin ảnh cho người bảo trì; dự án chưa có phép đo tốc độ bảo đảm cho từng CPU. |

### Thông tin cần gửi khi nhờ hỗ trợ

Gửi [issue trên repo](https://github.com/VanHuynh205/TomatoAI-Public/issues) với các thông tin sau:

- Bạn chạy EXE hay chạy source; bản Release hoặc commit đang dùng.
- Phiên bản Windows, dòng System type, và lỗi xảy ra lúc mở app, chọn ảnh hay bấm Nhận diện.
- Nguyên văn thông báo hoặc ảnh chụp hộp thoại; hash ZIP nếu bạn đã kiểm.
- Với source, lệnh vừa chạy và phần cuối output PowerShell. Che token, tên tài khoản, đường dẫn riêng hoặc thông tin proxy trước khi gửi.

Bản EXE hiện tại chưa lưu tệp log riêng để bạn gửi kèm. Khi nhờ hỗ trợ, gửi ảnh chụp hộp thoại và thông tin máy theo danh sách trên. Đừng tạo tệp log trong thư mục portable đã có manifest. Không đăng ảnh lá hoặc dữ liệu riêng tư lên issue nếu bạn chưa đồng ý chia sẻ.

## E. Phạm vi kiểm chứng và quyền sử dụng

[Báo cáo kiểm bản Windows](verification/2026-11-13-windows-release.md) ghi riêng native startup, frozen CPU inference và test fixture. Người bảo trì đã kiểm hash/manifest, mở desktop trên máy build và chạy inference synthetic với weights thật; các kiểm đó chưa thay thế một lần cài đầy đủ trên máy Windows sạch. Browser E2E dùng fixture không chứng minh chất lượng model trên ảnh thực hoặc video.

Mã ứng dụng thuộc dự án dùng [Apache-2.0](../LICENSE). Nội dung chăm sóc giữ [CC BY-SA 4.0](../infra/knowledge/CONTENT-LICENSE.md). Chủ dự án yêu cầu upload weights để cộng tác trong khi các điểm quyền dữ liệu/model còn mở; việc tải lên không xác nhận đã hoàn tất quyền phát hành và không gán Apache-2.0 cho checkpoint. Đọc [model card](../MODEL_CARD.md), [ghi công dataset](../DATASET_ATTRIBUTIONS.md) và [NOTICE](../NOTICE) trước khi tái phân phối.
Lưu cả `model-notices/` khi lấy model riêng. Repo không phân phối ảnh dataset hay video gốc.