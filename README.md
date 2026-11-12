# Tomato Desktop

Tomato Desktop là ứng dụng Windows của [VanHuynh205](https://github.com/VanHuynh205) để nhận diện dấu hiệu bệnh trên ảnh lá cà chua. Ứng dụng chạy suy luận tại chỗ và hiển thị thẻ chăm sóc tiếng Việt theo kết quả.

## Chạy trên Windows

Cài Python 3.12.14 x64, uv, Microsoft Edge WebView2 Evergreen Runtime và .NET Framework 4.6.2 trở lên. Bản công khai có mã nguồn, giao diện đã build và catalog chăm sóc. Để chạy nhận diện, bạn cần một bản DINOv2 đã tinh chỉnh tương thích mà bạn có quyền sử dụng. Kho này chưa phát hành checkpoint của dự án. Trọng số DINOv2 gốc của Meta chỉ trích xuất đặc trưng và không thay thế được bộ phân loại lá cà chua.

Đặt sáu tệp của bundle tương thích trong thư mục mô hình mặc định:

~~~text
ai/artifacts/dinov2-v1-280/
  classifier.safetensors
  config.json
  dinov2-release.json
  leaf-classifier.json
  model.safetensors
  preprocessor_config.json
~~~

Từ thư mục gốc của bản clone, chạy PowerShell:

~~~powershell
uv venv .venv-runtime --python 3.12.14
uv pip sync --python .venv-runtime/Scripts/python.exe --torch-backend cpu desktop/requirements-runtime-cpu.lock
& ./.venv-runtime/Scripts/python.exe -m desktop
~~~

Trong ứng dụng, chọn một ảnh lá cà chua để xem kết quả và thẻ chăm sóc nếu có. Để build lại giao diện, dùng Node.js 26 trở lên, chạy `npm --prefix frontend ci` rồi `npm --prefix frontend run build`.

## Giấy phép và ghi công

Mã và tài liệu do dự án viết dùng [MIT License](LICENSE). Khi chia sẻ toàn bộ hoặc một phần đáng kể của mã, giữ dòng bản quyền VanHuynh205 và thông báo giấy phép.

Backbone DINOv2 gốc do Meta AI Research, Maxime Oquab và cộng sự phát triển, theo Apache License 2.0. [Ghi công, model card và giấy phép gốc](THIRD_PARTY_NOTICES.md).

Nội dung chăm sóc giữ [CC BY-SA 4.0 và ghi công theo từng nguồn](infra/knowledge/CONTENT-LICENSE.md). Các thành phần bên thứ ba giữ giấy phép riêng.

Xem [hồ sơ phát hành](RELEASE.md) để tra nguồn huấn luyện, phạm vi ghi công và điều kiện phát hành bản Windows.
