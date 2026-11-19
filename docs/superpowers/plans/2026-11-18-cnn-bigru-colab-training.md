# Kế hoạch Huấn luyện ResNet50 / ConvNeXt + 2D Axial-BiGRU trên Colab Free & Tích hợp TomatoDesktop

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended, kết hợp `/swarm` và `/boost`) hoặc `superpowers:executing-plans` để thực thi từng Task. Các bước dùng cú pháp checkbox (`- [ ]`).

**Goal:** Huấn luyện mô hình phân loại 9 lớp bệnh lá cà chua (`ResNet50 + Cascaded 2D Axial-BiGRU + Gated Spatial Attention` và đối chứng `ConvNeXt-Tiny + Cascaded 2D Axial-BiGRU + Gated Spatial Attention` kèm nhánh ablation không GRU) trên Colab Free T4, vượt điểm thực địa lịch sử và nối an toàn vào ứng dụng `TomatoDesktop` trên máy Windows mà không dính bất kỳ thư viện hay định danh `dino`, `yolo`, `ultralytics`, `transformers`, `timm` nào ở runtime.

**Architecture:** Trích xuất bản đồ đặc trưng không gian $9 \times 9$ tại đầu vào `288x288` (`fit_pad bicubic`, `Pillow==13.0.0` đồng nhất hai đầu) từ backbone `torchvision.models` (`weights=None` khi suy luận), đi qua nhánh `Cascaded Row->Col BiGRU` + `Gated Spatial Attention` (*Ilse et al., ICML 2018*) bọc trong `FP32` và ghép với nhánh tắt `gap_norm(GAP(F4))` trước lớp phân loại 9 logits. Huấn luyện trên phân hoạch tất định 3 tập không rò rỉ (`train = 6.762`, `val_select = 1.422`, `val_calib = 1.527`) từ `dinov3-colab-v2` với bộ lấy mẫu rót đầy nước `cnn_bigru_weights_strict` ($N_{\text{eff}} = 4.781,5$) và hàm mất mát khớp tiên nghiệm `prior_matched_ce_loss` (`label_smoothing = 0.02`).

**Tech Stack:** CPython `3.12.14` (local) / Colab T4 GPU, `torch==2.14.1`, `torchvision==0.29.1`, `safetensors==0.7.0`, `Pillow==13.0.0`, `numpy==2.5.4`, `agent-browser` CDP, `pytest`, `ruff`, `mypy`, `@playwright/test` (`npx e2e run`).

**Spec:** [LO_TRINH_8_TUAN.md](file:///c:/Users/admin/code/DoAnCN_CaChua/LO_TRINH_8_TUAN.md), [dinov3-generalization-diagnosis-2026-11-11.md](file:///c:/Users/admin/code/DoAnCN_CaChua/docs/experiments/dinov3-generalization-diagnosis-2026-11-11.md), [2026-11-11-dinov2-field-improvement.md](file:///c:/Users/admin/code/DoAnCN_CaChua/docs/superpowers/plans/2026-11-11-dinov2-field-improvement.md), [2026-11-13-desktop-model-release.md](file:///c:/Users/admin/code/DoAnCN_CaChua/docs/superpowers/plans/2026-11-13-desktop-model-release.md).

## Global Constraints

- Giới hạn kích thước tệp trọng số: `_WEIGHTS_LIMIT = 256_000_000` byte (`244,14 MiB`, tại `ai/artifacts.py:26`); mỗi tệp JSON metadata `<= 1_000_000` byte.
- Hợp đồng lớp đầu ra: mạng xuất `9` logits theo đúng thứ tự `CLASSES` (`ai/dinov3_field.py:23-27`, chỉ số `0..7` là 8 nhãn cà chua, chỉ số `8` là `"not_tomato_leaf"`); `CnnGruPredictor.class_ids` trả `CLASSES[:-1]` (8 nhãn) để khớp `CareCatalog.validate_model_classes` (`backend/care.py:84-88`) và `care-v1.json`.
- Không rò rỉ dữ liệu: giữ `0` phần tử giao trên cả 5 trường (`group`, `sha256`, `original_sha256`, `normalized_sha256`, `path`) giữa `train`, `val_select`, `val_calib` và cả 3 bộ holdout (`legacy` `2.647`, `external` `460`, `SLIF` `21`). Cấm tuyệt đối chấm `Track B` (nếu có gộp `Rachin` `1.349` ảnh) trên `legacy/test.csv`.
- Không sửa `ai/preprocessing.py:80` (`crop_size == 224` cho `schema_version == 1`, được ghim bởi `tests/ai/test_models.py:433`); dùng duy nhất hàm `cnn_gru_eval_transform(size=288, ...)` trong `ai/cnn_gru_model.py` và cài cô lập `Pillow==13.0.0` vào `runtime-deps` trên Colab để tránh lệch `max |Δtensor| = 3,9216`.
- Bundle xuất ra tại `ai/artifacts/<bundle-id>/` chỉ chứa đúng 3 tệp: `CNN_GRU_FILES = ("leaf-classifier.json", "model.safetensors", "cnn-gru-release.json")`. File bản quyền đặt tại `desktop/notices/torchvision-weights/LICENSE` (`BSD-3-Clause`) và `desktop/notices/convnext/LICENSE` (`MIT`); ghi rõ giới hạn `CC BY-NC 4.0` của `241` ảnh `plantseg_v7` trong tài liệu provenance khi phát hành public.
- Không truyền dữ liệu nhị phân qua tham số dòng lệnh Windows hoặc chạy `files.upload()` thiếu `target_dir='/content'`. Khi kết nối Chrome CDP trên cổng `9222`, giữ cố định phiên với `--pin-tab` và không gọi `agent-browser close` trước khi tải xong kết quả.
- Quản lý `.scratch/` theo `AGENTS.md`: lưu `scratch-before.json`, `scratch-rules.json`, `scratch-inventory.csv`, `scratch-assessment.json` tại `docs/progress/2026-11-18-cnn-bigru/` (ngoài `.scratch/`).

## Review Focus

- `MAX_PIXELS` decompression-bomb và lệch nội suy `ImageOps.pad` vs `build_transform`: ghim tại `tests/ai/test_cnn_gru_model.py::test_eval_transform_rejects_oversized_and_is_deterministic`.
- Tràn `65504` hoặc dưới chuẩn `float16` (`var == 0` trên viền xám `fit_pad`, `exp >= 11.1` ở attention) dưới `torch.autocast("cuda", dtype=torch.float16)`: ghim tại `tests/ai/test_cnn_gru_model.py::test_head_forces_fp32_under_fp16_autocast`.
- Nổ `ValueError("Requested coverage is unattainable...")` khi truyền dòng âm `target == 8` vào `select_coverage_threshold`: ghim tại `tests/ai/test_cnn_gru_train.py::test_calibration_splits_indomain_and_negative_masks`.
- Đảo nhãn `raw_argmax` khi nhiệt độ `T` lớn gây hòa số học `float64`: ghim tại `tests/ai/test_cnn_gru_inference.py::test_high_temperature_tie_preserves_raw_logit_order` với khóa sắp xếp `(-scores[i], -raw_logits[i], i)`.
- Điều hướng nhầm vào `DinoV3Predictor` tại `desktop/server.py:38` khi thư mục chứa cả `cnn-gru-release.json` và `leaf-classifier.json`: ghim tại `desktop/tests/test_server.py::test_create_desktop_app_routes_cnn_gru_before_dinov3`.

---

## Tổng hợp Kết quả Kiểm toán, Thiết kế & Phản biện Chéo

### 1. Chẩn đoán lịch sử huấn luyện & 6 lỗi hệ thống đã chặn trước khi chạy

Đối chiếu bảng điểm trong [metrics-complete.json](file:///c:/Users/admin/code/DoAnCN_CaChua/infra/derived/benchmark-completion-20261112/metrics-complete.json), [dinov3-generalization-diagnosis-2026-11-11.md](file:///c:/Users/admin/code/DoAnCN_CaChua/docs/experiments/dinov3-generalization-diagnosis-2026-11-11.md) và [RUN-REPORT.md](file:///c:/Users/admin/code/DoAnCN_CaChua/infra/derived/dinov3-fast-track-v1/colab-v3-20261112/RUN-REPORT.md):

| Mô hình lịch sử | Tập dữ liệu huấn luyện | Số lớp | PV8 Test (1.714) mF1 | Legacy Field (1.730) mF1 | Taiwan/TOM2024 Toàn bộ (460) mF1 | Taiwan/TOM2024 Field (139) mF1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `EfficientNet-B0 E06 adapt` | PV8 + 273 ảnh PlantDoc | 8 | 0,9765 (E02) | 0,1423 | 0,2045 | 0,2563 |
| `YOLO26s-cls P07 refit` | PV8 + 273 ảnh PlantDoc | 8 | N/A | 0,1329 | 0,2756 | 0,2943 |
| `ResNet50 E08 adapt` | PV8 + 273 ảnh PlantDoc | 8 | **0,9867** (E04) | **0,1876** | **0,2607** | **0,3207** |
| `DINOv2 v1 (280 px)` | Đa nguồn thực địa v1 (6.969 ảnh) | 9 | N/A | 0,2147 | **0,7980** | **0,7637** |
| `DINOv3 v1` (ViT-S/16, 224 px) | Đa nguồn + Bapari/Hossain (7.781 ảnh) | 9 | N/A | 0,3700 | 0,6966 | 0,7604 |
| `DINOv3 v2` (ViT-S/16, 224 px) | Bản v1 + 288 ảnh PlantSeg/Chen (8.069 ảnh) | 9 | N/A | **0,3934** | 0,6857 | 0,7425 |

1. **Nguyên nhân `ResNet50 E08` thấp điểm thực địa:** `E08` vẫn thắng cả `EfficientNet-B0 E06` lẫn `YOLO26s-cls P07` trên cùng tập `pv8-plantdoc-v1`, nhưng bị kìm bởi 4 thiếu sót: (a) chưa từng học ảnh vườn đa nguồn (`PlantCity`, `Ghana CCMT`, `Imtiaz`, `PlantSeg`), (b) thiếu lớp thứ 9 `not_tomato_leaf`, (c) dùng trọng số cũ `IMAGENET1K_V1` (`76,13%`) đóng băng `layer1..layer3`, (d) tắt toàn bộ biến đổi hình học và màu sắc. Lưu ý `ai/train.py:358-360` đã giữ `_BatchNorm` ở `.eval()` suốt quá trình fine-tune, là thiết kế đúng cần kế thừa.
2. **Điểm mù tập Validation cũ:** Trong `dinov3-colab-v2/train-val.csv`, toàn bộ 4 nguồn bổ sung (`bapari_2025_v1` 436, `hossain_v1` 381, `plantseg_v7` 241, `chen_tomato_disease_2025` 42) đều vào 100% `train` (`0` mẫu ở `val`), còn tiêu chí lọc `domain == "field"` bỏ qua 3/9 lớp (`yellow_leaf_curl`, `spider_mites`, `not_tomato_leaf`). Hệ quả là `DINOv3 v2` tăng `val` từ `74,79%` lên `85,11%` nhưng tụt `TOM2024 Healthy` từ `73/77` xuống `61/77`.
3. **Ba lỗ hổng của `continuation_weights` (`ai/dinov3_field.py:103-141`):** Không đặt trần lặp cho miền `other` (`yellow_leaf_curl | other` lặp tới `4,77` lần/epoch, `healthy | other` lặp `8,48` lần/epoch ở lịch 18 epoch), ép 100% khối lượng lớp `spider_mites` lên nền phòng lab (`1,86` lần/epoch), và chia cứng theo số nguồn trước khi cắt trần làm thị phần `field` của `bacterial_spot` và `leaf_mold` tụt từ `75%` xuống `38%`.
4. **Lệch `fit_pad bicubic` (`max |Δtensor| = 3,9216`) và lệch phiên bản Pillow:** `ai/preprocessing.py:200-218` dùng `round` + `// 2`, còn `ai/dinov3_field.py:230-232` gọi `ImageOps.pad` dùng `math.floor` + `round(... * 0.5)`. Colab mặc định dùng `Pillow 11.3.0`, khác với `Pillow 13.0.0` trên Windows.
5. **Xung đột `log_prior_offset` với `label_smoothing = 0.02`:** Truyền thẳng `logits + delta` vào `CrossEntropyLoss(label_smoothing=0.02)` làm logit nền của `spider_mites` lệch `+0,8963 nat` (xác suất nền sai tăng `2,45` lần).
6. **Lỗi mẫu số `select_coverage_threshold` (`ai/calibration.py:125-140`) & bẫy tích hợp:** Đưa các dòng `target == 8` vào `select_coverage_threshold` làm nổ `ValueError("Requested coverage is unattainable...")`. Ở runtime, đặt nhánh kiểm tra model mới sau dòng `elif (model_dir / "leaf-classifier.json").is_file():` tại `desktop/server.py:38` sẽ điều hướng nhầm vào `DinoV3Predictor`, đồng thời `desktop/tests/test_lifecycle.py:207`, `browser_server.py:61` và `real_smoke.py:67-80` đang giả định cứng bundle cũ.

### 2. Dữ liệu, Phân hoạch 3 Tập Không Rò Rỉ & Water-Filling Sampler

- **Mã băm SHA-256 gốc đã xác minh:**
  - `infra/derived/dinov3-colab-v2/train-val.csv` (`9.711` ảnh, `3.801` nhóm): `24e3e7c2c40b03ac1bcc510073975d94e1c234733b43f092ebeec1c2e3d0b944` (ghép 100% từ `dinov3-colab-input.zip` `9.423` ảnh + `dinov3-additions.zip` `288` ảnh; `241` ảnh `plantseg_v7` mang `CC BY-NC 4.0`).
  - `legacy/test.csv` (`2.647` ảnh trong `dinov3-colab-holdout.zip`): `721ab09d690a3ed12aaf06c1c8c3d55e06cef6e480211e003d750b1968bbbe47`.
  - `external/test.csv` (`460` ảnh trong `dinov3-colab-holdout.zip`): `09e46e7668f421a613ddc83b99c5d813da290b11c69ac406d59dd1ad786bb8a0`.
  - `slif/test.csv` (`21` ảnh trong `dinov3-slif-pilot.zip`): `4b6c0ce43f0ab93b1c90385ca3004c50786b810e144977ae01d12bbe96049c27`.
- **Phân hoạch `Track A (v2-3way-9c, seed="tomato-cnn-bigru-v1")`:** Khóa 100% `hossain_v1` (`381` ảnh) trong `train`; tách 100% `bapari_2025_v1` (`436` ảnh) khỏi `train` chia sang `val_select` (`149` ảnh, `6g`) và `val_calib` (`287` ảnh, `7g`) làm chốt chặn OOD-source cho tam giác `healthy / bacterial_spot / septoria_leaf_spot`; phần còn lại chia theo tầng `(source, domain, class_id)` có `>= 6` nhóm theo tỷ lệ `74% / 13% / 13%` bằng băm `SHA-256`. Tổng: `train = 6.762` (`2.793g`), `val_select = 1.422` (`499g`), `val_calib = 1.527` (`509g`), `0` giao cắt ở cả 15 cặp tập trên cả 5 khóa định danh và `0` giao cắt `split_block`.
- **Water-filling sampler `cnn_bigru_weights_strict(rows, samples=4800)`:** Đặt trần cứng `caps = {"field": 1.40, "other": 1.50, "controlled": 0.85, "negative": 1.05}`, nâng $N_{\text{eff}}$ lên `4.781,5`, khôi phục thị phần thực địa `bacterial_spot`/`leaf_mold` lên `~75%`, và tính vector bù tiên nghiệm $\Delta_y = \ln(K \cdot \pi_y)$ = `[0.000000, +0.160417, 0.000000, +0.008588, +0.206235, -0.692273, -0.462333, +0.206235, +0.206235]`.
- **Hàm mất mát khớp tiên nghiệm (`Prior-Matched Label Smoothing`):**
  $$\mathcal{L}(z, y) = (1 - \varepsilon)\,\text{CE}(z + \Delta, y) - \varepsilon \sum_{k=1}^{K} \pi_k \ln \text{softmax}(z + \Delta)_k, \quad \pi = \text{softmax}(\Delta), \quad \varepsilon = 0,02$$
  Đảm bảo độ lệch xác suất nền giữa các lớp sai khi suy luận bằng `0,000000 nat`.

### 3. Kiến trúc Mạng & Danh mục Ứng viên (đã đo trên `AMD Ryzen 5 5625U` và `torchvision==0.29.1`)

- **Kiến trúc chuẩn `SpatialBiGruClassifier(backbone_name, num_classes=9, proj_dim=512, hidden_dim=256)`:**
  - Đầu vào `288x288` (`fit_pad bicubic`) cho bản đồ đặc trưng `F4` kích thước `(B, C4, 9, 9)`.
  - `CascadedAxialBiGRUGatedAttnHead` (bọc trọn trong `torch.amp.autocast(device_type, enabled=False)` chạy `FP32`):
    1. `gap = gap_norm(F4.mean(dim=(-2, -1)))` (`LayerNorm(C4, eps=1e-6)`; riêng `ConvNeXt` chép trực tiếp `tv.classifier[0].state_dict()` sang `gap_norm`).
    2. `X = GELU(in_norm(Conv1x1(C4 -> 512)(F4)))` với `in_norm = LayerNorm(512, eps=1e-6)`.
    3. `H_row = X + row_norm(RowBiGRU(512 -> 2x256)(X))` quét 9 bước theo hàng.
    4. `S = H_row + col_norm(ColBiGRU(512 -> 2x256)(H_row))` quét nối tiếp 9 bước theo cột (bao phủ đủ `81/81` ô trên phiến lá).
    5. Gated Spatial Attention (*Ilse et al., ICML 2018*): `a = softmax(w(tanh(V S) * sigmoid(G S)))`, `v_gru = gamma * gru_norm(sum_i a_i S_i)` (`gamma` khởi tạo `0.1`).
    6. `encode_image(x) = cat([gap, v_gru], dim=1)` có kích thước `(B, C4 + 512)` (hỗ trợ tái sử dụng cho chuỗi khung hình video), nối vào `Dropout(0.2) -> Linear(C4 + 512, 9)`.
- **Ánh xạ `ConvNeXt` về `torchvision` thuần (`convert_convnext_to_torchvision`):** Chuyển đổi trực tiếp checkpoint gốc Meta (`convnext_tiny_22k_1k_384.pth`, MIT License) sang `torchvision.models.convnext_tiny(weights=None)` (`182/182` keys `strict=True`, `max_logit_diff = 8.94e-8`), nhớ `.reshape(-1, 1, 1)` cho `gamma -> layer_scale` và ánh xạ `dwconv, norm, pwconv1, pwconv2` vào `block.{0, 2, 3, 5}`. Loại bỏ `ConvNeXt-V2` (`CC-BY-NC 4.0`) và `tf_efficientnetv2_s` (lệch padding `"SAME"`). Tắt `MixStyle` trên `ResNet50` do xung đột với `53` lớp `BN.eval()`.

| Mã nhánh | Mô hình & Trọng số gốc (Giấy phép) | Tham số | Tệp `.safetensors` | CPU `288px` (1-view / 2-flip TTA) | T4 dự kiến (18 ep) | Vai trò |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| **`M1` (Chính #1)** | `resnet50` (`IMAGENET1K_V2`, BSD-3) + `Axial-BiGRU-GSA` | 27,2M | `109,1 MB` | `81,6 ms` / `150,8 ms` | ~14 phút | Đúng tên mục tiêu `ResNet50 + GRU`, giữ toàn bộ `_BatchNorm` ở `.eval()`. |
| **`M2` (Chính #2)** | `convnext_tiny` (`22k_1k_384`, MIT) + `Axial-BiGRU-GSA` | 30,9M | `123,4 MB` | `88,8 ms` / `169,5 ms` | ~16 phút | CNN thuần `torchvision` dùng `LayerNorm` + kernel `7x7` bền vững ngoài vườn. |
| **`M0` (Ablation)** | `resnet50` (`IMAGENET1K_V2`, BSD-3) + `Linear GAP` | 23,5M | `94,3 MB` | `68,0 ms` / `136,0 ms` | ~10 phút | Chứng minh mức cải thiện thực chất của nhánh `+ BiGRU + GSA` trong báo cáo. |
| `M3` (Dự phòng) | `efficientnet_v2_s` (`IMAGENET1K_V1`, BSD-3) + `Axial-BiGRU-GSA` | 23,3M | `94,6 MB` | `79,3 ms` (1-view) | ~14 phút | Dùng thêm nếu dư quota T4 hoặc cần nhánh thuộc họ `EfficientNet` từ đề cương. |

### 4. Lịch Huấn luyện $C^0$, Hiệu chuẩn & Chiến lược Chạy Kép

- **Lịch `18 epochs x 150 steps = 2.700 steps` (`samples = 4.800`, `batch = 32` trên T4 hoặc `batch = 16, accum = 2` trên RTX 3050 4GB cục bộ):**
  - GĐ 1 (Epoch 1-3, `u = 1..450`): Đóng băng backbone (và `gap_norm` của `ConvNeXt`), `lr_head` tăng tuyến tính `8e-5 -> 8e-4` trong 150 bước rồi hạ Cosine về `3e-4` tại `u = 450`.
  - GĐ 2 (Epoch 4-15, `u = 451..2250`): Mở khóa `stage2`, `stage3`, `stage4` (`ResNet50`: `stage4=8e-5, stage3=2.5e-5, stage2=1e-5`; `ConvNeXt-Tiny`: `stage4=5e-5, stage3=2e-5, stage2=8e-6`) warmup tuyến tính trong Epoch 4 (`u = 451..600`), sau đó dùng một đường Cosine liên thông duy nhất đến `u = 2700` về sàn `0,05 * lr`; tích lũy Warmup-EMA $\beta_t = \min(0.992, \frac{1+t}{10+t})$ từ `u = 451`.
  - GĐ 3 (Epoch 16-18, `u = 2251..2700`): Tắt `RandomErasing`, giảm 50% `ColorJitter` để hội tụ cực tiểu phẳng sạch mà không có bước nhảy LR.
- **Chọn `best_epoch` trên `val_select` (2-view TTA FP32):** Chấm cả `raw` và `ema` bằng tuple `selection_key = (guardrail_passed, round(s_select, 8), round(recall_nonlab_healthy, 8), round(precision_nonlab_bs, 8))`, trong đó `guardrail_passed = (recall_nonlab_healthy >= 0.85) and (precision_nonlab_bs >= 0.70) and (recall_negative >= 0.90)`.
- **Hiệu chuẩn trên `val_calib` (`float64`):** Khóa `temperature_fit_scope = "in_domain_9logits"` trong `registration.json`, gọi `fit_temperature` và `select_coverage_threshold(..., negative_class_index=8)` chỉ trên các dòng `targets != 8` (`1.371` ảnh), sau đó kiểm tỷ lệ lọt lưới của mẫu âm thật (`targets == 8`, `156` ảnh) bằng `acceptance_mask`.

---

## Kế hoạch Triển khai Chi tiết Từng Bước (Bite-Sized Tasks)

### Task 1: Kiểm kê `.scratch/`, sinh manifest 3 tập Track A & lập `registration.json`

**Files:**
- Create: `docs/progress/2026-11-18-cnn-bigru/scratch-before.json`
- Create: `docs/progress/2026-11-18-cnn-bigru/scratch-rules.json`
- Create: `docs/progress/2026-11-18-cnn-bigru/registration.json`
- Create: `.scratch/cnn-bigru-20261118/manifests/train-val-calib.csv`

**Interfaces:**
- Produces:
  - `docs/progress/2026-11-18-cnn-bigru/registration.json` (khóa `seed="tomato-cnn-bigru-v1"`, `samples_per_epoch=4800`, `epochs=18`, `input_size=288`, `temperature_fit_scope="in_domain_9logits"`, `target_coverage=0.85`, 3 ngưỡng guardrail `0.85 / 0.70 / 0.90`, và SHA-256 của manifest 3 tập).
  - `.scratch/cnn-bigru-20261118/manifests/train-val-calib.csv` (9.711 dòng kèm cột `split` thuộc `{"train", "val_select", "val_calib"}`).

- [ ] **Step 1: Quét `lstat()` `.scratch/` tạo `scratch-before.json` và khai báo `scratch-rules.json`**

Chạy script chỉ đọc ghi 2 tệp inventory tại `docs/progress/2026-11-18-cnn-bigru/` (ngoài `.scratch/`), gán nhãn `KEEP_REUSE` cho `manifests/` và bundle dữ liệu tái lập, `KEEP_EVIDENCE` cho log/receipt/parity/evaluation, `DISPOSABLE` cho `parts/`, `__pycache__/`, `.pytest_cache/`, checkpoint tạm.

- [ ] **Step 2: Viết lệnh sinh `train-val-calib.csv` và kiểm chứng 0 rò rỉ 15 cặp tập**

Áp dụng hàm `build_three_way_split(v2, seed="tomato-cnn-bigru-v1")` xác nhận `train = 6.762`, `val_select = 1.422`, `val_calib = 1.527`, `0` giao cắt `group`, `sha256`, `original_sha256`, `normalized_sha256`, `path` trên cả 15 cặp tập với `legacy`, `external`, `slif`, và `0` giao cắt `split_block` giữa `train` và `(val_select | val_calib)`.

- [ ] **Step 3: Lưu `registration.json` và kiểm chứng lại bằng Python**

Run: `.\.venv-dev\Scripts\python.exe -B -c "import json; r=json.loads(open('docs/progress/2026-11-18-cnn-bigru/registration.json', encoding='utf-8').read()); assert r['counts']=={'train':6762,'val_select':1422,'val_calib':1527} and r['leakage_pairs']==0; print('REG_OK')"`
Expected: `REG_OK`

---

### Task 2: Xây dựng và kiểm thử `ai/cnn_gru_model.py` (TDD trên CPU)

**Files:**
- Create: `ai/cnn_gru_model.py`
- Test: `tests/ai/test_cnn_gru_model.py`

**Interfaces:**
- Produces in `ai/cnn_gru_model.py`:
  - `SUPPORTED_BACKBONES = ("resnet50", "convnext_tiny", "convnext_small", "efficientnet_v2_s")`
  - `cnn_gru_eval_transform(size: int, mean: Sequence[float], std: Sequence[float]) -> Callable[[Image.Image], torch.Tensor]`
  - `convert_convnext_to_torchvision(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]`
  - `class CascadedAxialBiGRUGatedAttnHead(nn.Module)`
  - `class SpatialBiGruClassifier(nn.Module)` với phương thức `encode_image(pixels: torch.Tensor) -> torch.Tensor` và `forward(pixels: torch.Tensor) -> torch.Tensor`
  - `build_cnn_gru_model(architecture: str, class_count: int = 9, *, use_gru_head: bool = True, hidden_dim: int = 256, proj_dim: int = 512, pretrained_weights: str | None = None, convnext_checkpoint: Path | None = None) -> SpatialBiGruClassifier`

- [ ] **Step 1: Viết 5 unit test thất bại trong `tests/ai/test_cnn_gru_model.py`**

Gồm:
- `test_factory_never_downloads_when_offline`: monkeypatch `torch.hub.download_url_to_file` và kiểm tra dựng `resnet50`, `convnext_tiny`, `efficientnet_v2_s`, cùng chế độ ablation `use_gru_head=False` khi không gọi mạng.
- `test_model_shapes_encode_image_and_weight_budget`: kiểm tra `encode_image((2, 3, 288, 288))` có shape `(2, C4 + 512)` (hoặc `(2, C4)` khi `use_gru_head=False`), `forward` có shape `(2, 9)` và tổng byte tensor `float32` `<= 256_000_000`.
- `test_eval_transform_rejects_oversized_and_is_deterministic`: kiểm tra từ chối ảnh vượt `MAX_PIXELS` trước khi giải mã, và chạy 2 lần trên ảnh bất đối xứng `(321, 197)` cho tensor trùng khớp tuyệt đối (`fit_pad bicubic`, `ImageOps.pad`).
- `test_head_forces_fp32_under_fp16_autocast`: bơm feature map chứa kênh hằng (`var == 0`) và biên độ lớn qua head dưới `torch.autocast("cpu", dtype=torch.bfloat16)` / giả lập half precision, xác nhận output và gradient hữu hạn 100%, không sinh `NaN`/`Inf`.
- `test_convert_convnext_to_torchvision_maps_all_keys_and_copies_gap_norm`: kiểm tra chuyển đổi đủ `182/182` khóa định dạng Meta/`timm` sang `torchvision.models.convnext_tiny(weights=None)` và chép đúng `classifier[0]` sang `head.gap_norm` (`eps=1e-6`).

- [ ] **Step 2: Chạy test xác nhận FAIL**

Run: `.\.venv-dev\Scripts\pytest.exe tests/ai/test_cnn_gru_model.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'ai.cnn_gru_model'`

- [ ] **Step 3: Cài đặt `ai/cnn_gru_model.py`**

Tuân thủ chính xác kiến trúc `CascadedAxialBiGRUGatedAttnHead` (bọc `autocast(enabled=False)` chạy `FP32`), giữ mọi `_BatchNorm` sẵn sàng cho `.eval()`, hỗ trợ `encode_image`, hàm duy nhất `cnn_gru_eval_transform`, và `convert_convnext_to_torchvision`.

- [ ] **Step 4: Chạy test, ruff và mypy cho Task 2**

Run: `.\.venv-dev\Scripts\pytest.exe tests/ai/test_cnn_gru_model.py -v && .\.venv-dev\Scripts\ruff.exe check ai/cnn_gru_model.py tests/ai/test_cnn_gru_model.py && .\.venv-dev\Scripts\mypy.exe ai/cnn_gru_model.py tests/ai/test_cnn_gru_model.py`
Expected: `5 passed`, `ruff` và `mypy` exited with code `0`.

---

### Task 3: Xây dựng và kiểm thử `ai/cnn_gru_train.py` (TDD trên CPU)

**Files:**
- Create: `ai/cnn_gru_train.py`
- Test: `tests/ai/test_cnn_gru_train.py`

**Interfaces:**
- Consumes: `ai/cnn_gru_model.py`, `ai/calibration.py` (`fit_temperature`, `select_coverage_threshold`, `acceptance_mask`), `ai/dinov3_field.py` (`CLASSES`, `save_atomic`, `validate_checkpoint_artifacts`).
- Produces in `ai/cnn_gru_train.py`:
  - `build_three_way_split(rows: list[dict[str, str]], seed: str = "tomato-cnn-bigru-v1", ratios: tuple[float, float, float] = (0.74, 0.13, 0.13)) -> dict[str, str]`
  - `cnn_bigru_weights_strict(rows: list[dict[str, str]], samples: int = 4800, *, caps: dict[str, float] | None = None) -> tuple[list[float], list[float]]`
  - `prior_matched_ce_loss(logits: torch.Tensor, target: torch.Tensor, log_prior_offset: torch.Tensor, label_smoothing: float = 0.02) -> torch.Tensor`
  - `continuous_c0_lr(global_step: int, total_steps: int, head_peak_lr: float, head_main_lr: float, backbone_lrs: dict[str, float]) -> dict[str, float]`
  - `selection_key(metrics: dict[str, float]) -> tuple[bool, float, float, float]`
  - `calibrate_and_export(...)` (tách `targets != 8` cho `select_coverage_threshold`, kiểm `targets == 8` qua `acceptance_mask`, xuất đúng 3 tệp `CNN_GRU_FILES` `<= 256_000_000` B và khóa `frozen-model.json`).

- [ ] **Step 1: Viết 6 unit test thất bại trong `tests/ai/test_cnn_gru_train.py`**

Gồm:
- `test_three_way_split_blocks_leakage_and_separates_bapari_hossain`: kiểm tra với fixture đa nguồn có `bapari_2025_v1`, `hossain_v1` và `split_block`, xác nhận `0` giao nhóm/hash giữa 3 tập.
- `test_water_filling_weights_and_prior_matched_loss_zero_bias`: kiểm tra tổng kỳ vọng bằng `samples`, không lớp/miền nào vượt `caps`, và nghiệm tối ưu hóa của `prior_matched_ce_loss` giữ 8 lớp sai trên `logits` trần đồng nhất trong sai số `1e-5`.
- `test_continuous_c0_lr_has_no_jump_at_stage_boundaries`: kiểm tra bước `450 -> 451` và `2250 -> 2251` liên tục, không nhảy vọt LR.
- `test_selection_key_never_returns_none_even_when_guardrail_fails`: xác nhận mọi epoch `guardrail_passed == False` vẫn có điểm xếp hạng tất định, nhưng luôn xếp sau mọi epoch `guardrail_passed == True`.
- `test_calibration_splits_indomain_and_negative_masks`: kiểm tra tập giả lập `val_calib` chứa 30% dòng âm `target == 8` vẫn chọn được `tau` qua `select_coverage_threshold` mà không nổ `ValueError`, đồng thời đo đúng tỷ lệ lọt lưới mẫu âm bằng `acceptance_mask`.
- `test_atomic_resume_and_export_bundle_contains_exactly_three_files`: chạy 2 epoch giả lập (ep 1 + resume ep 2 từ `last.pt` chứa `ema_model` và `ema_updates`), xác nhận xuất ra đúng 3 tệp `CNN_GRU_FILES` (không có `LICENSE.md`), `model.safetensors <= 256_000_000` B, và giữ nguyên `_BatchNorm` ở `.eval()`.

- [ ] **Step 2: Chạy test xác nhận FAIL**

Run: `.\.venv-dev\Scripts\pytest.exe tests/ai/test_cnn_gru_train.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'ai.cnn_gru_train'`

- [ ] **Step 3: Cài đặt `ai/cnn_gru_train.py`**

Cài đặt đầy đủ pipeline trainer hỗ trợ cả Colab CUDA AMP (`torch.backends.cuda.matmul.allow_tf32 = False` lúc eval, theo dõi `scaler.get_scale()` `updated`) lẫn CPU/local RTX 3050 (`grad_accum`).

- [ ] **Step 4: Chạy test, ruff và mypy cho Task 3**

Run: `.\.venv-dev\Scripts\pytest.exe tests/ai/test_cnn_gru_train.py -v && .\.venv-dev\Scripts\ruff.exe check ai/cnn_gru_train.py tests/ai/test_cnn_gru_train.py && .\.venv-dev\Scripts\mypy.exe ai/cnn_gru_train.py tests/ai/test_cnn_gru_train.py`
Expected: `6 passed`, `ruff` và `mypy` exited with code `0`.

---

### Task 4: Xây dựng `ai/cnn_gru_inference.py`, mở rộng `ai/dinov3_inference.py` & cập nhật `pyproject.toml`

**Files:**
- Create: `ai/cnn_gru_inference.py`
- Modify: `ai/dinov3_inference.py:108-129`
- Modify: `pyproject.toml:12-20`
- Test: `tests/ai/test_cnn_gru_inference.py`

**Interfaces:**
- Consumes: `ai/artifacts.py` (`_json`, `_local`, `_name`, `_read`, `_threshold`), `ai/calibration.py` (`scaled_probabilities`), `ai/cnn_gru_model.py` (`build_cnn_gru_model`, `cnn_gru_eval_transform`), `ai/dinov3_field.py` (`CLASSES`), `ai/contracts.py` (`Prediction`, `ScoredClass`).
- Produces in `ai/cnn_gru_inference.py`:
  - `CNN_GRU_MANIFEST = "cnn-gru-release.json"`
  - `MAX_WEIGHT_BYTES = 256_000_000`
  - `EXPORT_LIMITS = {"leaf-classifier.json": 1_000_000, "model.safetensors": MAX_WEIGHT_BYTES}`
  - `CNN_GRU_FILES = (*EXPORT_LIMITS, CNN_GRU_MANIFEST)`
  - `validate_cnn_gru_export(model_dir: Path, *, require_trained: bool = False) -> dict[str, Any]`
  - `class CnnGruPredictor` (`class_ids = CLASSES[:-1]`, `model_version: str`, `predict(image: Image.Image) -> Prediction`)
- Produces in `ai/dinov3_inference.py:108`:
  - `prediction_from_logits(logits: torch.Tensor, model_version: str, *, temperature: float = 1.0, threshold: float = 0.5) -> Prediction`

- [ ] **Step 1: Viết 5 unit test thất bại trong `tests/ai/test_cnn_gru_inference.py`**

Gồm:
- `test_validate_export_rejects_bad_hash_oversize_symlink_and_fixture`: kiểm tra từ chối sai SHA-256, `artifact_kind == "fixture"` khi `require_trained=True`, kích thước vượt `256_000_000` byte, và file thừa.
- `test_predictor_exposes_eight_care_classes_and_loads_safetensors_offline`: xác nhận `predictor.class_ids == CLASSES[:-1]` và không gọi mạng khi load.
- `test_high_temperature_tie_preserves_raw_logit_order`: kiểm tra với `logits = [[0.0, 1e-15, -5.0, ...]]` và `temperature = 20.0`, `prediction_from_logits` chọn đúng `early_blight` (`index=1`, khớp raw top-1) nhờ khóa `(-scores[i], -raw_logits[i], i)`.
- `test_predictor_rejects_not_tomato_leaf_even_at_high_score`: trả `status="uncertain", class_id=None` khi `raw_argmax == 8`.
- `test_predictor_tta_and_threshold_boundary`: kiểm tra lấy trung bình logits gốc + lật ngang trước khi chia `temperature`, nhận khi `top1 >= threshold` và từ chối khi `top1 < threshold`.

- [ ] **Step 2: Chạy test xác nhận FAIL**

Run: `.\.venv-dev\Scripts\pytest.exe tests/ai/test_cnn_gru_inference.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'ai.cnn_gru_inference'`

- [ ] **Step 3: Cài đặt `ai/cnn_gru_inference.py`, cập nhật `ai/dinov3_inference.py:108-129` và thêm `cnn_gru = ["numpy==2.5.4", "safetensors==0.7.0"]` (cùng `"safetensors==0.7.0"` vào `ml`) trong `pyproject.toml`**

- [ ] **Step 4: Chạy toàn bộ `tests/ai`, `ruff` và `mypy`**

Run: `.\.venv-dev\Scripts\pytest.exe tests/ai -v && .\.venv-dev\Scripts\ruff.exe check ai tests/ai && .\.venv-dev\Scripts\mypy.exe ai tests/ai`
Expected: toàn bộ `tests/ai` PASS (bao gồm `test_calibration.py`, `test_models.py`, `test_inference.py` và 3 file test mới), `ruff` và `mypy` exited with code `0`.

---

### Task 5: Đóng gói gói chạy Colab an toàn & chạy Smoke đối chứng cục bộ

**Files:**
- Create: `docs/progress/2026-11-18-cnn-bigru/colab_package.py`
- Create: `docs/progress/2026-11-18-cnn-bigru/colab_cells.py` (chứa `preflight_cell`, `upload_cell`, `train_cell`, `holdout_eval_cell`)
- Create: `.scratch/cnn-bigru-20261118/colab-bundle/cnn-gru-code.zip` (gói mỏng code + manifest SHA-256)
- Create: `.scratch/cnn-bigru-20261118/colab-bundle/parts/` (các mảnh `<= 50 MiB` kèm `manifest-parts.json` nếu dùng đường upload CDP)

**Interfaces:**
- Consumes: `ai/cnn_gru_model.py`, `ai/cnn_gru_train.py`, `ai/cnn_gru_inference.py`, `ai/calibration.py`, `ai/contracts.py`, `ai/artifacts.py`, `ai/dinov3_field.py`, `.scratch/cnn-bigru-20261118/manifests/train-val-calib.csv`, `infra/derived/dinov3-colab-v1/dinov3-colab-input.zip`, `dinov3-additions.zip`, `dinov3-colab-holdout.zip`, `dinov3-slif-pilot.zip`.
- Produces: script đóng gói kiểm chống Zip-slip/symlink, ghi bảng băm `bundle-hashes.json`, và chạy smoke 2 bước trên CPU/RTX 3050 cục bộ.

- [ ] **Step 1: Viết `colab_package.py` và `colab_cells.py`**

Đảm bảo trong `colab_cells.py`:
- `if not __debug__: raise RuntimeError("Assertions required")` ở đầu mỗi cell.
- `preflight_cell` sinh `run_nonce` thời gian thực, kiểm tra `torch.cuda.get_device_name(0)`, dung lượng `/content`, cài cô lập `Pillow==13.0.0` vào `runtime-deps` và kiểm `assert importlib.metadata.version('Pillow') == '13.0.0'`.
- `upload_cell` kiểm tra xem `/content/drive/MyDrive/TomatoAI/<run_id>/` có sẵn các ZIP hay không; nếu dùng `files.upload`, bắt buộc gọi `_ = files.upload(target_dir='/content')` cho từng mảnh `<= 50 MiB` và kiểm khớp SHA-256 trước khi ghép.
- `train_cell` chạy `M1`, `M2`, `M0`, lưu nguyên tử 2 pha `best-epoch-{e}-{selected_variant}.pt` trước `last.pt`, đồng bộ sang Drive cuối mỗi epoch, xuất 3 tệp `CNN_GRU_FILES` và khóa `frozen-model.json`.
- `holdout_eval_cell` kiểm khớp SHA-256 của `frozen-model.json`, giải nén 3 bộ holdout chống Zip-slip (`is_relative_to` + `not S_ISLNK`), tắt TF32, chấm 1 lần duy nhất, và xuất kèm 5 ảnh PNG không nén + logits CPU FP32 để kiểm parity `< 1e-5` khi về Windows.

- [ ] **Step 2: Đóng gói và chạy smoke 2 bước cục bộ**

Run: `.\.venv-dev\Scripts\python.exe -B docs/progress/2026-11-18-cnn-bigru/colab_package.py --smoke-steps 2`
Expected: sinh gói code + manifest sạch, xác minh smoke 2 bước `resnet50` và `convnext_tiny` giảm loss, xuất 3 tệp `CNN_GRU_FILES` hợp lệ dưới `256_000_000` B.

---

### Task 6: Thực thi Huấn luyện Pha 1 & Đánh giá Holdout Pha 2 (Colab T4 qua `agent-browser` / Dự phòng RTX 3050)

**Files:**
- Output: `.scratch/cnn-bigru-20261118/results/M1-resnet50-bigru/` (`model/`, `history.json`, `calibration.json`, `holdout-metrics.json`, `parity-png-fp32.json`)
- Output: `.scratch/cnn-bigru-20261118/results/M2-convnext-tiny-bigru/`
- Output: `.scratch/cnn-bigru-20261118/results/M0-resnet50-ablation/`
- Output: `docs/experiments/cnn-bigru-benchmark-2026-11-18.md` (bảng so sánh đối đầu trên `legacy`, `external TOM2024/Taiwan`, `SLIF`)

- [ ] **Step 1: Kết nối Chrome cổng `9222` bằng `agent-browser` với phiên cố định**

Run: `agent-browser --session colab-gru --cdp 9222 --pin-tab tab`
Kiểm tra: nếu người dùng đã bấm `Allow` trên Chrome, xác nhận tab `Untitled2.ipynb` và chạy tiếp Step 2 trên Colab T4 (tuyệt đối không gọi `agent-browser close` cho tới khi tải xong kết quả). Nếu Chrome vẫn khóa ở hộp thoại `Allow` hoặc T4 hết quota, chuyển sang Step 2B chạy tương đương toán học trên GPU cục bộ RTX 3050 4GB (`.venv-gpu` từ `ai/requirements-ml-cu126.lock` + `safetensors==0.7.0`, `batch=16, grad_accum=2`).

- [ ] **Step 2: Chạy Pha 1 (`M1`, `M2`, `M0`, mỗi nhánh `18 epochs x 4.800 mẫu`) và khóa `frozen-model.json`**

Xác nhận mỗi nhánh hoàn tất, lưu lịch sử `val_select`, fit `(T, tau)` trên `val_calib`, kiểm `model.safetensors <= 256_000_000` B.

- [ ] **Step 3: Chạy Pha 2 chấm Holdout (`legacy 2.647`, `external 460`, `SLIF 21`) & xuất báo cáo đối chứng**

Tải ZIP kết quả về `.scratch/cnn-bigru-20261118/results/`, kiểm SHA-256 khớp tuyệt đối, ghi bảng so sánh mF1 / accuracy / recall Healthy / negative rejection của `M0`, `M1`, `M2` so với `ResNet50 E08`, `YOLO26s-cls P07`, `DINOv2 v1`, `DINOv3 v1/v2` vào `docs/experiments/cnn-bigru-benchmark-2026-11-18.md`. Chỉ thăng cấp mô hình có `guardrail_passed == True` và vượt điểm thực địa lịch sử sang Task 7.

---

### Task 7: Nối Model Đạt Chuẩn vào `TomatoDesktop`, Cập nhật Test Suite & Chạy E2E Hai Lớp

**Files:**
- Create: `ai/artifacts/<winner-bundle-id>/` (`leaf-classifier.json`, `model.safetensors`, `cnn-gru-release.json`)
- Create: `desktop/notices/torchvision-weights/LICENSE` (`BSD-3-Clause`)
- Create: `desktop/notices/convnext/LICENSE` (`MIT`, nếu chọn `M2`)
- Create: `frontend/e2e.cnn-gru.config.ts`
- Create: `frontend/integration/cnn-gru.e2e.ts`
- Modify: `desktop/server.py:10-43` (đặt nhánh `CNN_GRU_MANIFEST` trước `leaf-classifier.json`; chỉ đổi `DEFAULT_MODEL_BUNDLE` khi model thắng đạt cổng holdout)
- Modify: `desktop/release.py:22,61,192-195,277-306`
- Modify: `desktop/app.spec:21-22,39,92-93`
- Modify: `desktop/notices/NOTICE.md:1-17`
- Modify: `desktop/tests/browser_server.py:15,58-65`
- Modify: `desktop/tests/real_smoke.py:21,67-80`
- Modify: `desktop/tests/test_lifecycle.py:206-208`
- Modify: `desktop/tests/test_server.py:93-121`
- Modify: `desktop/tests/test_package.py:25-116,275-556`

**Interfaces:**
- Consumes: `ai/cnn_gru_inference.py` (`CNN_GRU_FILES`, `CNN_GRU_MANIFEST`, `CnnGruPredictor`, `validate_cnn_gru_export`), `desktop/server.py` (`DEFAULT_MODEL_BUNDLE`).
- Produces: Tích hợp hoàn chỉnh `CnnGruPredictor` vào FastAPI runtime, release manifest validator, test suite và E2E thật.

- [ ] **Step 1: Chép bundle đạt chuẩn (đúng 3 tệp `CNN_GRU_FILES`) vào `ai/artifacts/<winner-bundle-id>/` và kiểm parity PNG FP32 CPU vs Colab**

Run: `.\.venv-dev\Scripts\python.exe -B -c "... verify parity max_abs_diff < 1e-5 ..."`
Expected: `PARITY_OK` với `max_abs_diff < 1e-5` và `class_id` khớp 100%.

- [ ] **Step 2: Cập nhật `desktop/server.py`, `desktop/release.py`, `desktop/app.spec`, `desktop/notices/` và các tệp test `desktop/tests/`**

Đảm bảo `create_desktop_app` kiểm `CNN_GRU_MANIFEST` trước `leaf-classifier.json`, `desktop/release.py` kiểm `{"safetensors", "numpy"} <= dependencies.keys()` và chặn trộn 2 định dạng model, `test_lifecycle.py:207` tham chiếu `DEFAULT_MODEL_BUNDLE`, `browser_server.py` và `real_smoke.py` nhận diện `CNN_GRU_MANIFEST`.

- [ ] **Step 3: Tạo `frontend/e2e.cnn-gru.config.ts` và `frontend/integration/cnn-gru.e2e.ts` (đọc động `manifest.threshold`)**

- [ ] **Step 4: Chạy toàn bộ cổng kiểm chứng Python, Frontend và E2E**

Run:
1. `.\.venv-dev\Scripts\ruff.exe check ai backend desktop tests && .\.venv-dev\Scripts\ruff.exe format --check ai backend desktop tests && .\.venv-dev\Scripts\mypy.exe`
2. `.\.venv-dev\Scripts\pytest.exe`
3. `.\.venv-dev\Scripts\python.exe -m desktop.tests.real_smoke --bundle ai/artifacts/<winner-bundle-id>`
4. `npm --prefix frontend run typecheck && npm --prefix frontend test && npm --prefix frontend run api:check`
5. `cd frontend && npx --no-install e2e run` (9/9 fixture tests)
6. `cd frontend && $env:TOMATO_E2E_MODEL_DIR="..\ai\artifacts\<winner-bundle-id>"; npx --no-install e2e run -c e2e.cnn-gru.config.ts`
Expected: tất cả PASS 100%, không bỏ qua hay nới lỏng assertion nào.

---

### Task 8: Kiểm kê cuối đợt & Dọn `.scratch/` có biên nhận

**Files:**
- Create: `docs/progress/2026-11-18-cnn-bigru/scratch-inventory.csv`
- Create: `docs/progress/2026-11-18-cnn-bigru/scratch-assessment.json`
- Create: `docs/progress/2026-11-18-cnn-bigru/cleanup-receipt.json`

- [ ] **Step 1: Kiểm kê toàn bộ `.scratch/` sau khi mọi tiến trình đã dừng**

Ghi `path`, `change_since_snapshot`, `label`, `reason`, `size_bytes`, `mtime_ns`, `sha256` vào `scratch-inventory.csv` và `scratch-assessment.json`; xác nhận `review_required == []` và `missing_since_snapshot == []`.

- [ ] **Step 2: Xóa từng tệp thường nhãn `DISPOSABLE` khớp inventory và xác minh lại `KEEP_*`**

Chỉ gọi `unlink()` khi `stat.S_ISREG` và toàn bộ thư mục cha lên tới `.scratch/` không có symlink, junction hoặc reparse point (`st_file_attributes & 0x400 == 0`), sau đó băm lại SHA-256 các tệp `KEEP_REUSE` và `KEEP_EVIDENCE` vào `cleanup-receipt.json`.
