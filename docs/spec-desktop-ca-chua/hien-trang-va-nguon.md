# Hiện trạng đã kiểm tra và nguồn đối chiếu

Snapshot: 01/11/2026, workspace `C:/Users/admin/Downloads/DoAnCN_CaChua`. Đây là kiểm kê phục vụ lập kế hoạch, không phải audit toàn bộ dự án hoặc kiểm định nhãn.

## Workspace

| Thành phần | Quan sát | Hệ quả |
| --- | --- | --- |
| `backend/`, `frontend/`, `desktop/`, `docs/` | Trống trước lượt lập kế hoạch | Chưa có app/API/UI để tái sử dụng hoặc chạy E2E |
| `ai/` | Một checkpoint EfficientNet-B0, 21.444.401 byte | Có pretrained weights; chưa thấy pipeline hoặc checkpoint bệnh lá |
| `.venv-torch/` | Import được torch và torchvision | Dùng làm điểm khởi đầu, chưa cần cài lại |
| `_bmad/config.toml` | Tên cũ `Dataset_DOAn`, output `_bmad-output`, chưa có active initiative | Tài liệu mới vào `docs/` theo yêu cầu; không sửa cấu hình |
| `LO_TRINH_8_TUAN.md` | Bản 30/10/2026, giả định nhóm 3 người, 240 giờ và PySide6 Widgets | Giữ như lịch sử; kế hoạch hiện hành dùng ngân sách một người và ranh giới frontend/backend mới |
| Tài liệu agent | Thiếu `docs/agents/domain.md`, `docs/agents/issue-tracker.md`; thiếu skill project được dẫn | Không giả nhận đã đọc các file thiếu; chưa tạo issue tracker hoặc thay AGENTS |
| Graphify | Không thấy `graphify-out/graph.json` | Đọc file có phạm vi, không dựng graph cho cây chưa có app |

## Checkpoint và môi trường

Đã nạp checkpoint bằng `torch.load(..., map_location="cpu", weights_only=True)`, dựng `torchvision.models.efficientnet_b0(weights=None)` và `load_state_dict(..., strict=True)`: tất cả key khớp.

- SHA-256: `7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934`.
- Tensor `classifier.1.weight`: `(1000, 1280)`; đầu ra còn 1.000 lớp ImageNet.
- URL trong torchvision trùng tên file: [pretrained weights chính thức](https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth).
- Local: `torch 2.14.1+cu126`, `torchvision 0.29.1+cu126`; `torch.cuda.is_available()` trả `True`; tên thiết bị `NVIDIA GeForce RTX 3050 Laptop GPU`.
- Transform của weights: resize 256, center crop 224, bicubic, RGB về [0,1], mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.

Chưa chạy forward benchmark, epoch train, thử OOM hoặc đóng gói. Import CUDA thành công chưa chứng minh mọi workload chạy ổn. Cách dùng và preprocessing đối chiếu [torchvision EfficientNet-B0](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html).

## tomato_vn

Nguồn local: `infra/tomato_vn/data.yaml`, `README.dataset.txt`, `README.roboflow.txt`, toàn bộ file TXT dưới `train/labels`, `valid/labels`, `test/labels`.

Trang [Roboflow dataset v1](https://universe.roboflow.com/wildlife-l1e7w/tomato-leaf-disease-vietnam-iquff/dataset/1) và metadata local cùng ghi CC BY 4.0, Object Detection, 9.892 ảnh. Trang nguồn ghi ba output mỗi ảnh train và tăng cường hue. Tên “Vietnam” không xác nhận toàn bộ ảnh chụp ở Việt Nam; filename có mẫu PSU và nhiều kiểu nguồn khác nhau.

| Split gốc | Ảnh | File nhãn | Ảnh chứa nhiều class ID |
| --- | ---: | ---: | ---: |
| train | 8.661 | 8.661 | 4.125 |
| valid | 802 | 802 | 352 |
| test | 429 | 429 | 222 |

Số dưới đây là bounding box, không phải số ảnh độc lập hoặc số cây:

| ID gốc | Nhãn | Train | Valid | Test |
| --- | --- | ---: | ---: | ---: |
| 0 | Bacterial Spot | 12 | 0 | 0 |
| 1 | Early Blight | 3.103 | 252 | 132 |
| 2 | Healthy | 3.322 | 274 | 106 |
| 3 | Late Blight | 4.121 | 336 | 176 |
| 4 | Leaf Miner | 2.968 | 243 | 174 |
| 5 | Leaf Mold | 3.998 | 330 | 204 |
| 6 | Mosaic Virus | 3.870 | 299 | 192 |
| 7 | Septoria | 3.773 | 298 | 163 |
| 8 | Spider Mites | 2.563 | 249 | 122 |
| 9 | Yellow Leaf Curl Virus | 4.827 | 295 | 216 |

Kiểm tra sơ bộ không thấy dòng khác 5 token, file nhãn rỗng hoặc tọa độ giống hệt nhau mang nhiều class ID. Chưa kiểm miền giá trị tọa độ, decode ảnh, chất lượng box hoặc độ đúng của nhãn.

So prefix tên ảnh trước `.rf.` thấy 68 prefix chung train–valid, 15 train–test, 10 valid–test. Đây là ứng viên cùng ảnh nguồn; không cộng ba số thành số ảnh trùng duy nhất, không coi là kết quả hash/pHash. Cần kiểm tra trước khi tái sử dụng split. Nhiều class trong một ảnh không tự chứng minh nhãn sai, nhưng không thể gán toàn ảnh bằng nhãn dòng đầu.

Không thể dùng split hiện tại để báo recall Bacterial Spot trên valid/test vì support bằng 0. Không lấy macro-F1 của chín lớp để gọi thành benchmark đủ mười lớp.

## PlantVillage

Nguồn: `infra/tomato_plantvillage/TomatoAI/data/dataset-registry.json` và manifest run hiện hành `processed/plantvillage/pv-d9e4c21eec8e9172/manifest.csv` dưới cùng thư mục dữ liệu. Run `pv-018f15e5f3a9b2a1` là lịch sử, không gộp hai manifest.

| Nhãn nguồn | Tổng record | ready_for_split |
| --- | ---: | ---: |
| Bacterial spot | 2.127 | 2.127 |
| Early blight | 1.000 | 999 |
| Late blight | 1.909 | 920 |
| Leaf mold | 952 | 952 |
| Septoria leaf spot | 1.771 | 1.000 |
| Two-spotted spider mite | 1.676 | 1.676 |
| Target spot | 1.404 | 0 |
| Tomato yellow leaf curl virus | 5.357 | 2.738 |
| Tomato mosaic virus | 373 | 0 |
| Healthy | 1.591 | 999 |
| Tổng | 18.160 | 11.411 |

Còn 6.735 `needs_review` và 14 `excluded_exact_duplicate`. Mọi record đều `ready_for_training=False`. Những số này được đếm lại từ manifest; chưa hash/decode lại raw hoặc xác nhận nhãn.

Local `raw/plantvillage/README_HF.md` khai `license: cc-by-sa-3.0`. [README_HF upstream](https://github.com/spMohanty/PlantVillage-Dataset/blob/master/README_HF.md) cũng ghi như vậy; [loader upstream](https://github.com/spMohanty/PlantVillage-Dataset/blob/master/plant_village.py) có ghi chú cần kiểm chứng license. Vì vậy cần gắn bản raw với đúng release/commit và bằng chứng quyền; không mặc định CC0, không áp license của một mirror lên mọi bản PlantVillage.

## Kho chăm sóc và nguồn chưa xác định

Đã đọc `infra/library/00-Quan-tri/QUY-TAC-KHO.md`, `MA-TRAN-BAO-PHU.md` và phần đầu sổ nguồn. Đếm file Markdown trực tiếp: 130 trong `01-Hop-thu`, 129 trong `02-Nguon`, 3 trong `03-Benh`, 1 trong `05-Cham-soc`. Sổ/ma trận có các checkpoint lịch sử; con số 52 ở phần đầu không phải kiểm kê cuối cùng.

Số file không chứng minh số nguồn duy nhất, toàn văn đầy đủ, quyền hay mức duyệt. Chưa kiểm toàn bộ từng hồ sơ. Không thu/clip nguồn mới vào vault, không đổi metadata. Bộ nhà kính hơn 10.000 ảnh được người dùng mô tả chưa được định vị; không gán nhầm thành dữ liệu WUR về môi trường canh tác.

## Cách dùng BMad và công cụ trong lượt này

- Đọc skill `bmad`, chạy `knowledge.py --content` trên roots hiện hữu và đọc help module. Module help hiện dùng luồng Product Owner mới; `bmad-spec` là bản local cũ vẫn được giữ theo router tài khoản.
- Dùng `bmad-spec` để rút gọn yêu cầu và tạo companions, `.memlog.md`; không nhận đã chạy PRD, architecture workflow hay ticket workflow riêng.
- `resolve_config.py` và `resolve_customization.py` chạy được bằng Python sẵn có. Một lần gọi `uv` qua shell bị lỗi cache access; không cài hoặc sửa môi trường để vượt lỗi. Customization không có activation hooks.
- Xuất vào `docs/` theo chỉ dẫn người dùng thay cho output mặc định; chưa tạo initiative hoặc tracker. Checklist lịch không phải ticket store của BMad.
- Context-mode MCP xử lý kiểm kê; web search MCP đối chiếu tài liệu chính thức; Stop-slop/Humanizer dùng ở bước viết. Không dùng subagent, không cài plugin mới.

## Nguồn kỹ thuật cho các quyết định

| Nguồn | Quyết định sử dụng |
| --- | --- |
| [PyTorch transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) | Đóng băng backbone rồi fine-tune; không train từ đầu |
| [PyTorch AMP](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) | Thử mixed precision và accumulation khi cần, đo trên GPU thật |
| [pywebview architecture](https://pywebview.flowrl.com/guide/architecture) | UI web có thể hiển thị trong desktop qua local server |
| [pywebview freezing](https://pywebview.flowrl.com/guide/freezing) | Build React thành asset local rồi đóng gói với PyInstaller |
| [pywebview security](https://pywebview.flowrl.com/guide/security) | Local REST API cần bảo vệ CSRF/token phiên |
| [Microsoft WebView2 distribution](https://learn.microsoft.com/en-us/microsoft-edge/webview2/concepts/distribution) | Kiểm tra runtime và chuẩn bị bộ cài offline khi máy đích thiếu |
| [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en) | Ghi tác giả, nguồn, link license và thay đổi với dữ liệu áp dụng license này |
| [GitHub về repository không có license](https://docs.github.com/en/enterprise-cloud%40latest/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository) | Public repository không tự cấp đủ quyền tái sử dụng; bộ thiếu license không nằm trên đường găng |

Nguồn web kiểm tra ngày 01/11/2026; bản thư viện local và smoke test lúc triển khai mới quyết định API thực chạy. Các mục tiêu chất lượng/thời gian trong kế hoạch là đề xuất, không phải số liệu rút từ các trang này.
