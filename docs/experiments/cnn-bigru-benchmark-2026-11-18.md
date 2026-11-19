# So sánh G0–G3 với DINOv2 của TomatoDesktop

Ngày 18/11/2026, cả bốn model đã hoàn tất 22 epoch trên Colab T4. G2 đứng đầu nhóm mới về macro-F1 thực địa trên Legacy và Taiwan/TOM2024. So với DINOv2 mặc định tại thời điểm đo, G2 tăng 14,62 điểm phần trăm trên Legacy và giảm 15,66 điểm trên Taiwan/TOM2024. Trên toàn bộ 460 ảnh Taiwan/TOM2024, DINOv2 cũng giữ điểm cao nhất trong các model đã đối chiếu. Sau phép đo, chủ dự án chọn G2 làm mặc định cho desktop. DINOv2 vẫn là mốc đối chứng trong báo cáo này. [Cấu hình desktop](../../desktop/server.py) · [Kết quả theo model và tập](../progress/2026-11-18-cnn-bigru/benchmark-results.json)

## Kết quả trên cùng ảnh

Các cột F1 và accuracy dùng nhãn top-1 trước khi áp ngưỡng từ chối, đơn vị %. F1 là trung bình các F1 của những lớp có mẫu thật trong lát cắt. Legacy có 2.647 ảnh, trong đó 1.730 ảnh lá cà chua thực địa. Taiwan/TOM2024 có 460 ảnh, trong đó 139 ảnh lá thực địa. SLIF gồm 21 ảnh cùng nhãn Bacterial Spot, cột cuối đếm số ảnh đoán đúng. [Hàm chấm](../../ai/dinov2_field.py) · [Hồ sơ SLIF](../../infra/derived/dinov3-colab-v2/data-audit.md)

| Model | F1 Legacy, toàn bộ | F1 Legacy, thực địa | Acc. Legacy, thực địa | F1 Taiwan/TOM2024, toàn bộ | F1 Taiwan/TOM2024, thực địa | Acc. Taiwan/TOM2024, thực địa | SLIF, đúng/21 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 v1 280 | 62,55 | 21,47 | 33,53 | 79,80 | 76,37 | 76,98 | 13/21 |
| G0 · ResNet50 GAP | 64,48 | 33,43 | 39,83 | 54,07 | 33,20 | 35,25 | 14/21 |
| G1 · ResNet50 + BiGRU | 64,59 | 34,10 | 41,16 | 53,09 | 38,65 | 37,41 | 18/21 |
| G2 · ConvNeXt-Tiny + BiGRU | 65,80 | 36,09 | 42,37 | 64,16 | 60,70 | 55,40 | 8/21 |
| G3 · EfficientNet-V2-S + BiGRU | 64,70 | 35,83 | 42,60 | 58,54 | 51,82 | 41,73 | 10/21 |
| ResNet50 E08 | 34,25 | 18,76 | 22,31 | 26,07 | 32,07 | 30,22 | chưa đo |
| YOLO26s P07 | 36,75 | 13,29 | 17,57 | 27,56 | 29,43 | 25,18 | chưa đo |
| DINOv3 v1 | 56,72 | 37,00 | 41,45 | 69,66 | 76,04 | 78,42 | 14/21 |
| DINOv3 v2 | 60,22 | 39,34 | 41,39 | 68,57 | 74,25 | 71,22 | 16/21 |

G1 và G0 cùng dùng ResNet50. Nhánh BiGRU của G1 tăng macro-F1 thực địa từ 33,43% lên 34,10% trên Legacy và từ 33,20% lên 38,65% trên Taiwan/TOM2024. Ở SLIF, G1 đoán đúng 18/21 ảnh. Khi áp ngưỡng phục vụ người dùng, G1 trả kết quả cho 12 ảnh và đúng 11 ảnh. Các ảnh SLIF tập trung vào bốn cụm cảnh tạm xác định, nên 21 ảnh chưa đại diện cho 21 ca bệnh độc lập.

## Khác biệt so với model gốc

Chênh lệch dưới đây là macro-F1 thực địa của ứng viên trừ DINOv2, tính bằng điểm phần trăm. Khoảng 95% dùng 2.000 lần bootstrap ghép cặp theo nhóm ảnh, phân tầng theo nguồn và tập nhãn của nhóm, seed 42. Khoảng này phản ánh biến thiên khi lấy lại nhóm trong hai benchmark lịch sử, chưa điều chỉnh cho nhiều phép so sánh. [Mã bootstrap](../../ai/compare_dinov2_field.py)

| Nhánh | Δ Legacy | Khoảng 95% | Δ Taiwan/TOM2024 | Khoảng 95% |
|---|---:|---:|---:|---:|
| G0 | +11,96 | +9,81 đến +14,26 | −43,17 | −52,68 đến −29,66 |
| G1 | +12,62 | +10,21 đến +15,06 | −37,71 | −49,62 đến −14,00 |
| G2 | +14,62 | +12,41 đến +16,84 | −15,66 | −24,57 đến −4,39 |
| G3 | +14,36 | +12,02 đến +16,55 | −24,55 | −33,54 đến −9,49 |

## Lá khỏe và hành vi từ chối

Recall Healthy dùng top-1 trên lá khỏe thực địa, với 437 ảnh Legacy và 77 ảnh Taiwan/TOM2024. Coverage là tỷ lệ ảnh lá cà chua được trả nhãn trên toàn tập tương ứng. Cột từ chối âm dùng đúng quyết định sau ngưỡng của từng model. DINOv2 dùng ngưỡng 0,5. G0–G3 dùng ngưỡng đã hiệu chuẩn trước khi mở holdout.

| Model | Healthy Legacy (%) | Healthy Taiwan (%) | Coverage Legacy (%) | Coverage Taiwan (%) | Âm bị từ chối Legacy | Âm bị từ chối Taiwan |
|---|---:|---:|---:|---:|---:|---:|
| DINOv2 v1 280 | 61,56 | 88,31 | 85,58 | 73,85 | 287/296 | 70/70 |
| G0 | 63,84 | 25,97 | 70,06 | 36,92 | 280/296 | 70/70 |
| G1 | 68,19 | 25,97 | 70,06 | 40,51 | 283/296 | 69/70 |
| G2 | 43,48 | 50,65 | 64,06 | 46,67 | 287/296 | 69/70 |
| G3 | 62,01 | 33,77 | 55,89 | 36,67 | 286/296 | 70/70 |
| ResNet50 E08 | 0,00 | 7,79 | 95,28 | 91,54 | 19/296 | 5/70 |
| YOLO26s P07 | 0,46 | 10,39 | 92,43 | 87,95 | 34/296 | 13/70 |
| DINOv3 v1 | 57,44 | 94,81 | 51,25 | 64,62 | 281/296 | 70/70 |
| DINOv3 v2 | 54,46 | 79,22 | 57,51 | 64,36 | 282/296 | 70/70 |

G2 nhận đúng 39/77 lá khỏe ngoài thực địa Taiwan/TOM2024, DINOv2 nhận đúng 68/77. Trong 390 ảnh lá cà chua của toàn tập này, G2 trả nhãn cho 182 ảnh và đúng 138. DINOv2 trả nhãn cho 288 ảnh và đúng 234. Hai model có tỷ lệ nhận nhầm mẫu âm thấp, nhưng khác nhau nhiều về số lá thật được trả kết quả.

## Huấn luyện, chọn checkpoint và hiệu chuẩn

Kế hoạch ban đầu dự kiến 18 epoch và coi EfficientNet-V2-S là nhánh dự phòng. Run `cnn-bigru-20261118` thực tế hoàn tất cả bốn nhánh, mỗi nhánh 22 epoch. Cả bốn vượt điều kiện chọn trên `val_select` về recall lá khỏe ngoài lab, precision Bacterial Spot ngoài lab và recall mẫu âm. G3 đạt `S_select` cao nhất. [Kế hoạch trong repo](../superpowers/plans/2026-11-18-cnn-bigru-colab-training.md) · [Registration đang lưu](../progress/2026-11-18-cnn-bigru/registration.json)

| Nhánh | Epoch chọn | Trọng số chọn | Cập nhật optimizer | S_select | Nhiệt độ | Ngưỡng | Từ chối âm val_calib (%) |
|---|---:|---|---:|---:|---:|---:|---:|
| G0 | 14 | ema | 3299 | 0,8063 | 1,1110 | 0,6859 | 97,44 |
| G1 | 16 | ema | 3298 | 0,7947 | 1,2176 | 0,8113 | 98,72 |
| G2 | 15 | raw | 3299 | 0,8057 | 1,2840 | 0,7854 | 98,72 |
| G3 | 16 | ema | 3300 | 0,8120 | 1,2654 | 0,8750 | 96,79 |

Trainer chỉ tăng `global_step` khi optimizer thực sự cập nhật dưới AMP. Lịch dự kiến 3.300 bước. Cả bốn lịch sử đều có đủ epoch 1–22. Bản khóa của G0–G2 ghi SHA-256 registration `0724d22374babda4f6b66dffbd4d49583e0b7a27c67674e3584afe43695db27d`. G3 khớp bản còn lưu `15d68dec248fa231550ef984b431ac5bd2f658017a0821136c7a5973bdb0a5e2`. Chưa xác định được toàn bộ khác biệt cấu hình giữa hai bản. Cả bốn cùng ghi SHA-256 manifest phát triển `e0fe14587c36076d2cb7cc242e7aa989a0bc6fa84d3788db8c4ff1373267ee19`.

## Độ trễ và đối chiếu Windows

Đo trên Ryzen 5 5625U, PyTorch CPU 4 luồng, batch 1. Mỗi lượt gọi `predict` gồm tiền xử lý và TTA lật ngang, đầu vào là ảnh PIL đã giải mã. Sau một lượt làm nóng, lấy trung vị 15 lượt trên 5 ảnh PNG cho từng model. DINOv2 dùng 280 px, G0–G3 dùng 288 px. MB tính theo 1.000.000 byte. Số liệu là phép đo trên máy này.

| Model | Trọng số (MB) | Trung vị (ms/ảnh) | Sai lệch logit tối đa Colab GPU–CPU | Sai lệch tối đa Windows–Colab CPU | Nhãn và trạng thái khớp |
|---|---:|---:|---:|---:|---:|
| DINOv2 v1 280 | 346,41 | 818,03 | — | — | — |
| G0 | 94,37 | 171,33 | 7.54e-6 | 5.48e-6 | 5/5 |
| G1 | 109,11 | 185,67 | 2.50e-6 | 3.19e-6 | 5/5 |
| G2 | 123,45 | 275,01 | 3.28e-6 | 4.29e-6 | 5/5 |
| G3 | 94,64 | 174,21 | 1.43e-6 | 3.44e-6 | 5/5 |

## Bằng chứng và phạm vi kết luận

- Track A có 6.762 ảnh train, 1.422 ảnh chọn checkpoint và 1.527 ảnh hiệu chuẩn. Đã đối chiếu SHA-256 của các manifest và toàn bộ 3.128 ảnh holdout, kiểm tra 15 cặp tập theo `group`, `path` và ba trường SHA-256 có giá trị. Không tìm thấy giao cắt theo từng trường. Các benchmark này đã được dùng để so sánh trong quá khứ. Hiệu quả ở một vườn mới vẫn cần được đo trên một tập độc lập.
- Bốn model tạo đủ 12.512 dòng dự đoán. Khi ghép lại theo SHA-256 và đối chiếu nhãn, nguồn, miền, đường dẫn và nhóm ảnh, các metric tính lại trùng với JSON Colab. DINOv2 dùng bản dự đoán đã khóa hash cho Legacy và Taiwan/TOM2024. SLIF được chạy trực tiếp trong lượt này bằng bundle desktop đã xác minh. [Dự đoán SLIF của DINOv2](../progress/2026-11-18-cnn-bigru/dinov2-v1-280-slif-predictions.csv) · [Điểm và hash từng CSV](../progress/2026-11-18-cnn-bigru/benchmark-results.json)
- Gói train chứa trọng số, lịch sử, hiệu chuẩn và bản khóa cho cả bốn nhánh. SHA-256 `9c59ecbff409fb7eb60d4dd7a3c13ca8c402ab9410677f6927fe50b5a4fdc80f`. [Tải trọng số và lịch sử](../../.scratch/cnn-bigru-20261118/results/cnn-bigru-train-results.zip)
- Gói test chứa dự đoán từng ảnh, metric và các PNG đối chiếu. SHA-256 `09fed10b421eb8f04f27048c6e536fd72309bcc3fec81b822b76cbdd7102bafc`. [Tải kết quả holdout](../../.scratch/cnn-bigru-20261118/results/cnn-bigru-test-results.zip) · [Inventory](../progress/2026-11-18-cnn-bigru/scratch-inventory.csv)
