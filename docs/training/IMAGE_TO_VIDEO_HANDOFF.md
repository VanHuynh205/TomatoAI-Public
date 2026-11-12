# Bàn giao mô hình ảnh DINOv2 để phát triển hướng video

Dùng weights `dinov2-v1-280` làm điểm khởi tạo từ mô hình ảnh đã huấn luyện. Repo hiện có mã huấn luyện ảnh gốc và API nạp backbone cùng đầu phân loại. Repo chưa có mô hình chuỗi thời gian, video dataloader hay vòng huấn luyện video; đợt bàn giao này không chạy huấn luyện mới.

## Gói cần lấy và phiên bản cần giữ

Mở [bản phát hành v0.1.0-dinov2-v1-280](https://github.com/VanHuynh205/TomatoAI-Public/releases/tag/v0.1.0-dinov2-v1-280), tải [gói weights dinov2-v1-280](https://github.com/VanHuynh205/TomatoAI-Public/releases/download/v0.1.0-dinov2-v1-280/dinov2-v1-280-model.zip). Giải nén để một thư mục chứa trực tiếp sáu tệp sau, rồi dùng thư mục đó làm `model_dir`:

```text
model.safetensors
classifier.safetensors
config.json
preprocessor_config.json
leaf-classifier.json
dinov2-release.json
```

Giữ nguyên metadata và weights. Đừng sửa `size` trong metadata thành 280 để làm cho nó trông giống kích thước chạy ứng dụng.

| Thuộc tính | Giá trị của bản v1 |
| --- | --- |
| Backbone | `facebook/dinov2-base`, ViT-B/14 |
| Revision | `f9e44c814b77203eaa57a6bdbbd535f21ede1415` |
| Huấn luyện | 224 × 224, 6 epoch; checkpoint được chọn ở epoch 6 |
| Suy luận desktop | 280 × 280, trung bình logits ảnh gốc và ảnh lật ngang trước softmax |
| Đầu phân loại | CLS token ghép với trung bình patch token; LayerNorm, Dropout 0,15, Linear 9 lớp |
| Tiền xử lý | EXIF transpose, RGB, fit-pad bicubic; mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| SHA-256 backbone đã huấn luyện | `3274d01322f079a2b7d0df0048500b1094d67f8f308c47b989ad71e8c566a4ff` |
| SHA-256 đầu phân loại | `eb2337fbaca6915e9ccae39868fc305dbd5a8a8d95e0347abbf67a5cc1b05efa` |

Đối chiếu `leaf-classifier.json`: `size=224`, `epochs=6`, `selected_epoch=6`, `test_time_horizontal_flip=true`. Đối chiếu `dinov2-release.json`: `model_version="dinov2-v1-280"`, `evaluation_size=280` và hash của năm tệp model. Metadata lượt chạy gốc ghi Tesla T4 và Torch `2.14.1+cu130`. Tên bản phát hành có 280 vì kích thước suy luận; nó không chứng minh mô hình từng học ở 280 px.

## Mã nguồn gốc dùng lại

Ba module dưới đây đã có trong repo public, trùng nội dung mã gốc sau khi chuẩn hóa xuống dòng. Không cần một bản trainer thứ hai.

| Mã | Vai trò |
| --- | --- |
| [dinov2_field.py](../../ai/dinov2_field.py) | Trainer v1, `Classifier`, sampler theo lớp/miền/nhóm, kiểm split và hash, TTA, export |
| [dinov2_improve.py](../../ai/dinov2_improve.py) | `load_export`, warm-start từ safetensors, thử nghiệm fine-tune ảnh tiếp theo và CLI dự đoán manifest |
| [dinov3_field.py](../../ai/dinov3_field.py) | Module dùng chung chứa `CLASSES`, `LeafImages`, `leaf_transform`, `load_manifest`, `metrics`, `save_atomic` |
| [dinov2_inference.py](../../ai/dinov2_inference.py) | Kiểm gói phát hành v1 và suy luận 280 px cho ứng dụng |

Tên module dùng chung có DINOv3 do lịch sử dự án. Trainer DINOv2 chỉ import các helper cần thiết; không chạy trainer DINOv3 và không đổi backbone.

Các script dựng dataset gốc phụ thuộc cây dữ liệu và biên nhận riêng tư, nên không nằm trong gói bàn giao. Bạn cần tự chuẩn bị dữ liệu có quyền sử dụng theo hợp đồng manifest bên dưới. Gói này không kèm ảnh, video hay tập dữ liệu gốc.

## Môi trường

Dùng Python 3.12. Từ thư mục gốc repo, trong virtualenv riêng:

```powershell
python -m pip install -e ".[ml,dino,test]"
python -m pytest tests/ai/test_dinov2_training.py tests/ai/test_dinov2_improve.py tests/ai/test_dinov2_inference.py -q
```

[pyproject.toml](../../pyproject.toml) đã khai báo Torch 2.14.1, torchvision 0.29.1, Transformers 4.57.6, safetensors 0.7.0 và Pillow 13.x. Không cần thêm thư viện cho các API bên dưới. Để chạy trainer ảnh gốc, bạn cần bản Torch hỗ trợ CUDA và GPU; CPU đủ cho kiểm thử synthetic và nạp model. Chọn wheel CUDA phù hợp driver theo hướng dẫn PyTorch. Dependency extras không khóa toàn bộ môi trường GPU của lượt Colab gốc.

## Nạp weights đã học để tiếp tục nghiên cứu

Chạy từ thư mục gốc repo. Đoạn này chỉ kiểm và nạp model, không train, không tải backbone mới:

```python
from pathlib import Path
import torch
from PIL import Image, ImageOps
from ai.dinov2_inference import validate_dinov2_export
from ai.dinov2_improve import load_export
from ai.dinov3_field import CLASSES, leaf_transform

model_dir = Path("ai/artifacts/dinov2-v1-280")  # cùng vị trí README/Start.ps1
assert validate_dinov2_export(model_dir) == "dinov2-v1-280"
device = torch.device("cpu")
model, metadata = load_export(model_dir, device)
model.eval()
backbone = model.backbone
head = model.head
assert tuple(metadata["class_ids"]) == CLASSES
transform = leaf_transform(
    280, metadata["preprocessing"]["mean"],
    metadata["preprocessing"]["std"], training=False,
)

# Ảnh tổng hợp chỉ để kiểm shape; không phải dữ liệu huấn luyện.
image = ImageOps.exif_transpose(Image.new("RGB", (320, 240))).convert("RGB")
pixels = transform(image).unsqueeze(0).to(device)
with torch.inference_mode():
    tokens = backbone(pixel_values=pixels).last_hidden_state
    features = torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=1)
    logits = head(features)              # [batch, 9]
    flipped_logits = model(pixels.flip(-1))
    probabilities = ((logits + flipped_logits) / 2).softmax(dim=1)
assert features.shape == (1, 1536)
assert probabilities.shape == (1, 9)
```

`load_export` gọi `AutoModel.from_pretrained` với `local_files_only=True`, `use_safetensors=True`, kiểm thông tin nạp backbone, rồi nạp `classifier.safetensors` vào head. Bạn có thể lấy `features` hoặc gọi `model(pixels)` cho từng frame trong phần nghiên cứu riêng. Đoạn ví dụ chưa định nghĩa cách gộp frame, mục tiêu loss hay nhãn clip.

### Warm-start và resume khác nhau

Gói sáu tệp chứa tham số backbone/head và metadata suy luận. Nó không chứa optimizer, GradScaler hay trạng thái epoch để tiếp tục chính xác vòng train cũ. ZIP kết quả Colab gốc đã kiểm danh sách entry cũng chỉ có năm tệp model, lịch sử và hai tệp kết quả test; không có checkpoint `.pt`.

Trainer v1 tạo `best.pt` gồm model + epoch được chọn, và `last.pt` gồm model, optimizer, scaler, cấu hình run, lịch sử và epoch hiện tại. Nó tự nạp `last.pt` nếu dùng lại thư mục output; cấu hình run phải trùng và `best.pt` phải còn. Vì cấu hình có số epoch, Torch version và tên GPU, đổi các giá trị đó có thể bị từ chối. Trainer không lưu toàn bộ RNG state, nên đừng gọi đây là bảo đảm tái lập bit-for-bit.

[dinov2_field.py](../../ai/dinov2_field.py) không có cờ `--initial`: gọi trainer v1 trong output mới sẽ khởi tạo từ backbone Meta đã pin revision, không tự nạp gói v1 đã học. Muốn dùng weights hiện có, dùng API `load_export` ở trên, hoặc CLI `dinov2_improve train --initial` với đúng hợp đồng development của thử nghiệm đó. CLI improve yêu cầu dữ liệu train đủ chín lớp, validation ảnh thực địa từ `rachin`, `plantcity`, `ghana_ccmt` và mẫu âm. Nó khởi tạo optimizer mới nếu output chưa có `last.pt`; đó là warm-start một thí nghiệm ảnh mới, không phải resume epoch 7 của v1. Không chạy nó lên dữ liệu video rồi tuyên bố đã có trainer video.

## Hợp đồng dữ liệu ảnh

`load_splits` của v1 nhận một thư mục gốc dữ liệu có `train-val.csv` và `test.csv`. Dùng các cột:

```csv
path,split,class_id,source,domain,group,sha256
images/train-healthy.jpg,train,healthy,my_garden,field,capture-train-001,<sha256-64-hex-of-image>
images/val-healthy.jpg,val,healthy,my_garden,field,capture-val-001,<sha256-64-hex-of-image>
images/test-negative.jpg,test,not_tomato_leaf,my_negatives,negative,capture-test-001,<sha256-64-hex-of-image>
```

Ví dụ này chỉ minh họa schema, chưa đủ điều kiện chạy train. Ghi hai dòng đầu vào manifest development và dòng cuối vào manifest test; thay placeholder bằng hash byte ảnh thật. Các đường dẫn phải nằm trong thư mục dữ liệu, trỏ tới tệp có thật và không lặp trong cùng manifest.

Thứ tự chín nhãn cố định:

```text
bacterial_spot, early_blight, late_blight, leaf_mold,
septoria_leaf_spot, spider_mites, yellow_leaf_curl, healthy,
not_tomato_leaf
```

Train phải đủ chín nhãn. Mỗi split phải có ảnh cà chua miền `field` và mẫu `not_tomato_leaf`. Nhãn cà chua dùng miền `field`, `web` hoặc `controlled`; sampler dành riêng cách cân bằng theo nguồn cho mẫu âm. `source` cần mô tả bộ nguồn thật, `group` cần cùng định danh cho các ảnh cùng đợt chụp/cùng lá có thể gây leakage. Giữ cùng group trong một split. Trainer v1 kiểm path, group và hash không giao nhau giữa split, rồi kiểm hash byte từng ảnh.

Với video do bạn chuẩn bị sau này, mọi frame của cùng clip phải cùng split; các clip cùng cây/đợt chụp cũng cần nhóm theo bằng chứng bạn có. Không coi tách ngẫu nhiên từng frame là test độc lập. Mã ảnh hiện tại không tự dựng group từ video và không phát hiện mọi ảnh gần trùng.

Lệnh gốc bên dưới phục vụ đọc/tái lập pipeline ảnh khi đã có đủ dữ liệu hợp lệ và quyền sử dụng. Đợt bàn giao không chạy lệnh này:

```powershell
python -m ai.dinov2_field --data C:\datasets\authorized-leaves --output C:\runs\image-v1 --size 224 --epochs 6 --batch 32 --samples 3072 --workers 2 --seed 42
```

Trainer đọc manifest test để kiểm split/hash trước vòng học, chọn checkpoint bằng validation, rồi chạy dự đoán test sau khi chọn. Không sửa test để tối ưu sampler hoặc chọn epoch. Nếu dùng dữ liệu khác, bạn không tái tạo được chính xác kết quả v1 chỉ bằng lệnh và weights.

## Giới hạn cần giữ khi làm hướng video

Mô hình đã học ảnh một lá chính. Nó chưa được kiểm định cho cảnh nhiều lá, chuyển động, mất nét, các bệnh ngoài taxonomy hoặc độ chính xác theo clip. Giữ lớp âm và thứ tự head khi tái sử dụng weights; nếu thay head hoặc thêm module thời gian, ghi phiên bản mới và đánh giá riêng trên các clip giữ ngoài train.

Quyền sử dụng dataset huấn luyện gốc vẫn còn điểm chưa giải quyết. Việc phát hành weights không cấp quyền đối với ảnh nguồn, không xác nhận mọi điều kiện giấy phép đã hoàn tất và không cho phép suy ra bạn được phân phối lại dataset. Người tiếp tục dự án cần kiểm quyền của dữ liệu mới và nguồn cũ trước khi dùng/phân phối. Không dùng kết quả của mô hình làm chẩn đoán bệnh chắc chắn ngoài vườn.

## Kiểm chứng của phần bàn giao

[test_dinov2_improve.py](../../tests/ai/test_dinov2_improve.py) là bản sao nguyên nội dung kiểm thử gốc cho sampler, split, selection và checkpoint round-trip. [test_dinov2_training.py](../../tests/ai/test_dinov2_training.py) kiểm module import đúng repo public, CLS/patch pooling, chín logit, TTA, giá trị không hữu hạn, sampler v1 và split/hash bằng ảnh tổng hợp. [test_dinov2_inference.py](../../tests/ai/test_dinov2_inference.py) kiểm integrity và suy luận runtime 280 px.

Ngày 13/11/2026, chạy ba tệp kiểm thử trên từ thư mục repo public bằng môi trường CPU: **15 passed**. Kiểm nguồn import xác nhận `ai`, `backend` và `desktop` nằm trong checkout public hiện tại. Ruff check và format check đạt cho hai tệp kiểm thử huấn luyện mới.

Các kiểm thử này không train model, không đánh giá độ chính xác trên dataset gốc và không chứng minh chất lượng video. Không có thay đổi giao diện hoặc luồng ứng dụng trong phần bàn giao nguồn huấn luyện; phần này không chạy browser E2E.
