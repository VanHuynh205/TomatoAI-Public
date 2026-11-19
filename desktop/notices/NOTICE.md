# Tomato Desktop: thông báo thành phần

Bản Windows CPU mặc định dùng G2 ConvNeXt-Tiny + BiGRU (`cnn-bigru-20261118-g2-convnext_tiny_axial_bigru`) và nội dung chăm sóc `care-v1`. Giấy phép của từng thành phần giữ phạm vi riêng. Thông báo này không cấp lại quyền cho toàn bộ ứng dụng.

`python/` chứa nguyên văn LICENSE, COPYING, NOTICE, AUTHORS và METADATA từ các wheel đúng phiên bản trong `requirements-runtime-cpu.lock`. Spec lấy trực tiếp từ môi trường build, giữ đường dẫn tương đối để các thông báo cùng tên không ghi đè nhau. License của PyTorch gồm `third_party/`; license tổng hợp của NumPy gồm OpenBLAS/LAPACK/GCC runtime; license tổng hợp của Pillow gồm các thư viện codec và FreeType. PyInstaller và hooks có giấy phép cùng ngoại lệ riêng trong bản gốc. CPython có thông báo tại `cpython/LICENSE.txt`.

`frontend/` giữ giấy phép của React, React DOM, Scheduler, GSAP và Lenis. Tệp `components.json` ghi phiên bản, nguồn và SHA-256 của từng bản giấy phép. Giao diện dùng Cambria và Segoe UI từ Windows. Matplotlib mang theo phông riêng trong gói Python.

`dinov2/LICENSE` chứa giấy phép Apache 2.0 cho tùy chọn DINOv2-base của các bản trước. [Model card tại revision nguồn](https://huggingface.co/facebook/dinov2-base/blob/f9e44c814b77203eaa57a6bdbbd535f21ede1415/README.md) · [Giấy phép gốc](https://github.com/facebookresearch/dinov2/blob/main/LICENSE).

`torchvision-weights/LICENSE` chứa giấy phép BSD 3-Clause cho trọng số tiền huấn luyện `torchvision.models` (`ResNet50 IMAGENET1K_V2`, `EfficientNetV2-S IMAGENET1K_V1`). `convnext/LICENSE` chứa giấy phép MIT của Meta Platforms cho trọng số tiền huấn luyện `ConvNeXt` (`convnext_tiny_22k_1k_384`). Khi phát hành công khai các trọng số được huấn luyện trên `dinov3-colab-v2`, hồ sơ provenance phải ghi rõ ràng buộc phi thương mại `CC BY-NC 4.0` của `241` ảnh `plantseg_v7` trong tập huấn luyện.

`care/CONTENT-LICENSE.md` ghi điều kiện CC BY-SA 4.0, tác giả nguồn, thay đổi của bản tiếng Việt và giới hạn thẩm định. Từng nguồn, phạm vi địa lý, URL, mục/trang và thông báo quyền vẫn nằm trong `knowledge/care-v1.json`, ngay cạnh bộ notices. Không kèm PDF, ảnh nguồn hoặc vault.

Quyền phân phối model/dữ liệu huấn luyện và các binary Microsoft phải được đối chiếu trước khi phát hành. Bản build không tự tải hoặc cài WebView2; runtime cài offline là thành phần riêng. Hồ sơ phát hành phải giữ các kết luận kiểm quyền và điều kiện redistribution tương ứng.

`proxy-tools/LICENSE.txt` giữ thông báo BSD từ mã nguồn upstream. Wheel `proxy-tools==0.1.0` khai MIT trong METADATA. Gói giữ cả hai thông báo để ghi lại chênh lệch giữa hai nguồn. [Giấy phép tại commit upstream](https://github.com/jtushman/proxy_tools/blob/ccd35a569b95bec271a59890780c7821756c548a/LICENSE.txt)

`openssl-3.5.8/` và `libffi-3.4.8/` giữ giấy phép của các DLL từ bản CPython 3.12.14 qua Astral. SHA-256 của ba DLL trong gói khớp các tệp trong môi trường uv. [Phiên bản nguồn tại bản build 20260901](https://github.com/astral-sh/python-build-standalone/blob/20260901/pythonbuild/downloads.py)

`torchvision-codecs/` giữ thông báo theo phiên bản DLL trong TorchVision 0.29.1. Metadata DLL xác định libjpeg-turbo 3.1.3 và zlib 1.3.2. API của hai DLL còn lại trả libpng 1.6.56 và libwebp 1.6.0.

Microsoft WebView2 SDK được ghi riêng trong `microsoft-webview2-sdk/`. Các DLL Visual C++ trong bản Windows chịu điều kiện phân phối Microsoft, ghi tại giấy phép CPython và danh mục Visual Studio. [Microsoft Visual Studio 2026 Distributable Code](https://learn.microsoft.com/en-us/visualstudio/releases/2026/redistribution)
