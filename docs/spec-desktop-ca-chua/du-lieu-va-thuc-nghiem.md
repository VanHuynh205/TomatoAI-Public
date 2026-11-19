# Dữ liệu, transfer learning và đánh giá

Mục tiêu: một pipeline dùng chung cho EfficientNet-B0 và ResNet50, một tập nhãn đã khóa và một protocol đánh giá. Đây là kế hoạch; chưa có dữ liệu mới được cấp trạng thái training-ready.

## 1. Chọn dữ liệu trong ngân sách hai tuần

Khuyến nghị dùng subset PlantVillage có grouping làm benchmark chính sau khi đối chiếu quyền, đường dẫn và chất lượng. Hiện có 11.411 record `ready_for_split` thuộc tám lớp; phải kiểm lại số ảnh/nhóm sau loại trùng, không lấy con số này làm số train cuối cùng.

Tám lớp dự kiến, với ID ổn định độc lập thứ tự thư mục:

| class_id | Nhãn nguồn | Tên hiển thị dự thảo |
| --- | --- | --- |
| bacterial_spot | Bacterial spot | Đốm vi khuẩn |
| early_blight | Early blight | Đốm vòng / cháy lá sớm |
| late_blight | Late blight | Mốc sương |
| leaf_mold | Leaf mold | Mốc lá |
| septoria_leaf_spot | Septoria leaf spot | Đốm lá Septoria |
| spider_mites | Two-spotted spider mite | Tổn thương do nhện hai chấm |
| yellow_leaf_curl | Tomato yellow leaf curl virus | Triệu chứng xoăn vàng lá theo nhãn TYLCV |
| healthy | Healthy | Lá khỏe theo nhãn dữ liệu |

Tên tiếng Việt cần đối chiếu tài liệu khi biên soạn thẻ. Tập lớp gồm bệnh, dịch hại và trạng thái khỏe. “Healthy” không khẳng định toàn cây không mắc bệnh.

Target Spot và Tomato mosaic virus chỉ vào thí nghiệm chính nếu xử lý được grouping/nguồn trong ngân sách tuần 1–2. Nếu rubric bắt buộc mười lớp, ghi đây là rủi ro tiến độ và giải quyết trước khi train chính. Không gộp bệnh để đủ số lớp.

`tomato_vn` là nhánh bổ sung có điều kiện. Mười nhãn của nó có Leaf Miner thay Target Spot; không đồng nhất hai schema theo index. “Mosaic Virus” hoặc “Spider Mites” chỉ map đến nhãn cụ thể hơn khi tài liệu nguồn đủ căn cứ. Chưa xác minh được thì giữ unresolved, không ép map.

Bộ nhà kính không có license nằm ngoài thí nghiệm chính và gói nộp. Có thể ghi URL, paper và trạng thái quyền khi xác định được; không mặc định sử dụng được chỉ vì nghiên cứu đã đăng GitHub. Không chờ liên hệ tác giả để bắt đầu phần việc độc lập.

## 2. Artifact dữ liệu

Giữ nguyên dữ liệu hiện có. Tạo dưới `infra/derived/<dataset-version>/`:

- `manifest.csv`: `sample_id`, `source_dataset`, `source_version`, `source_path`, `source_sha256`, `label_original`, `class_id`, `group_id`, `group_evidence`, `split`, `derived_path`, `bbox` nếu có.
- `exclusions.csv`: sample và lý do loại/chờ; không xóa raw.
- `label_map.json`: thứ tự class index, class_id, tên nguồn và tên hiển thị; version đi cùng model.
- `split.json`: seed, tỷ lệ mục tiêu, quy tắc nhóm, nguồn/hash manifest, số ảnh và số nhóm mỗi lớp/tập.
- `provenance.json`: tác giả/đơn vị, URL, version/commit, ngày lấy đã biết, license evidence, biến đổi và phạm vi sử dụng.

Kết quả audit viết trong `docs/data/`; CSV manifest là dữ liệu máy đọc, không phải tài liệu báo cáo. Chỉ dùng file thuộc allowlist dẫn xuất đã đối chiếu; không đổi `ready_for_training` trong snapshot cũ để làm test xanh.

## 3. Kiểm tra và chia tập

1. Kiểm đường dẫn cũ sau di chuyển thư mục; resolve theo root được cấu hình, không sửa raw. Kiểm ảnh tồn tại, decode, RGB/channels, kích thước, ảnh rỗng/hỏng.
2. Đối chiếu SHA-256 nguồn với manifest khi có; gom byte/pixel duplicate. Dùng pHash hoặc contact sheet để tìm ứng viên gần trùng, rồi xác nhận; pHash không tự chứng minh cùng lá.
3. Dùng `leaf_group_id` và bằng chứng hiện có; gộp cùng lá/cây/buổi chụp khi metadata cho phép. Ảnh crop, phiên bản màu, segmented, hue và augmentation cùng nguồn phải giữ cùng nhóm.
4. Nếu dùng nhiều nguồn, kiểm trùng xuyên dataset trước split. Tên khác nhau không chứng minh độc lập. Subset `tomato_vn` có ảnh kiểu PlantVillage không tự là external test.
5. Chia khoảng 70/15/15 theo nhóm, cố gắng giữ phân bố lớp. Không giao nhóm quan trọng hơn tỷ lệ chính xác. Seed khởi đầu đề xuất 42; lưu kết quả split, không chỉ seed.
6. Review contact sheet có phân tầng theo lớp/nguồn: đề xuất 20–30 ảnh mỗi lớp và mọi ca parser nghi vấn. Với một người, đây là rà định dạng và nhãn hiển nhiên bất nhất, không xác nhận bệnh học. Ghi mức bao phủ và ảnh chưa xem.
7. Kiểm mọi lớp có đủ nhóm trong train/validation/test. Nếu test một lớp dưới 30 nhóm độc lập, báo support thấp và độ bất định; số 30 là mốc lập kế hoạch, không bảo đảm đại diện thực địa.
8. Chốt manifest/split/hash. Chỉ train có augmentation ngẫu nhiên. Validation/test dùng biến đổi cố định, không chứa các bản tăng cường của nhóm train.

Ca không đủ bằng chứng vào quarantine hoặc tập khảo sát, không làm chuyên gia giả để duyệt hết. Không tự sinh nhãn bằng mô hình rồi coi là ground truth.

## 4. Nếu dùng tomato_vn cho classifier

YOLO TXT chứa `class x_center y_center width height` chuẩn hóa. Mỗi box có thể tạo một crop dẫn xuất, nhưng trước đó phải kiểm cả class và tọa độ hữu hạn, đúng miền, diện tích dương, ảnh tương ứng và nội dung box. Cần xác định annotation bao vùng lá hay tổn thương; không giả định mọi box là một lá hoàn chỉnh.

Crop giữ liên kết tới ảnh cha và source group. Mọi crop từ cùng ảnh/nhóm ở một split. Không lấy dòng nhãn đầu gán cho toàn ảnh chứa nhiều class. Các crop chồng nhau mang nhãn mâu thuẫn cần review; không chọn nhãn dựa trên suy đoán của AI.

Ưu tiên bản ảnh gốc chưa augmentation từ nguồn được phép nếu có. Nếu chỉ có bản export đã tăng cường, truy lại nhóm cha; không đưa bản hue đã biến đổi vào test rồi gọi đó là ảnh gốc. Trường hợp không truy được, để ngoài benchmark chính. Giao prefix `.rf.` hiện tại là danh sách cần kiểm, không phải quy tắc grouping đủ dùng.

Hạn mức nhánh này: tối đa 6 giờ khảo sát/chuyển đổi ở tuần 2; chưa đạt thì giữ benchmark PlantVillage và ghi giới hạn. Không tiêu hai tuần sửa toàn bộ dataset để lấy thêm số lượng ảnh.

## 5. Pipeline huấn luyện chung

Code dự kiến trong `ai/`: loader/manifest validation, model factory cho đúng hai backbone, train, evaluate, inference và tests. Chỉ tách module khi có trách nhiệm rõ; không viết hai training loop độc lập.

EfficientNet-B0: giữ checkpoint gốc bất biến; dựng đúng architecture, nạp state dict 1.000 lớp bằng `strict=True`, sau đó thay `classifier[1]` bằng lớp có K output. Nếu thay head trước, phải xử lý mismatch có kiểm tra; không dùng `strict=False` để che thiếu backbone.

ResNet50: dùng pretrained `ResNet50_Weights.IMAGENET1K_V1` làm lựa chọn khởi đầu, lưu weights/hash trước các buổi offline; thay `fc` thành K output. Không dùng weights ngẫu nhiên rồi gọi đó là transfer learning. Mỗi model có metadata preprocessing riêng.

| Thành phần | Cấu hình khởi đầu đề xuất | Điều kiện điều chỉnh |
| --- | --- | --- |
| Input | RGB, 224×224; normalization ImageNet | Giữ transform evaluate/inference khớp metadata |
| Batch vật lý | B0: 8; ResNet50: 4 | Nếu OOM giảm còn 4/2; đo thực tế |
| Batch hiệu dụng | 16 bằng accumulation khi cần | Giữ quy tắc loss scaling/optimizer step đúng; không coi tương đương BN batch lớn |
| Precision | CUDA autocast FP16 + GradScaler | Kiểm loss/gradient hữu hạn; fallback FP32 nếu bất ổn |
| DataLoader | `num_workers=0` lúc smoke; thử 2 sau đó | Windows spawn cần entry guard; tránh load toàn bộ ảnh vào RAM |
| Optimizer | AdamW, weight decay khởi đầu 1e-4 | Cùng lựa chọn ở thí nghiệm đối chứng |
| Train head | LR 1e-3, tối đa 5 epoch ban đầu | Backbone đóng băng; ghi validation sau mỗi epoch |
| Fine-tune | Head 1e-4, backbone 1e-5; tối đa 20 epoch | B0 mở phần block cuối; ResNet50 mở layer4 trước |
| Checkpoint tốt nhất | Validation macro-F1 cao nhất; tie dùng validation loss thấp hơn | Không dùng test để lựa chọn |
| Dừng sớm | Patience 5 epoch | Quy tắc cùng nhau cho hai mô hình, ghi số epoch thực chạy |
| Augmentation | Flip ngang, xoay nhẹ, crop nhẹ giữ vùng lá | Tránh hue mạnh và crop mất triệu chứng; chọn bằng validation |

Các con số là điểm bắt đầu, không đảm bảo vừa VRAM hoặc đạt chỉ số. Trong giai đoạn frozen, `requires_grad=False` chưa đủ để đóng băng BatchNorm: giữ backbone ở eval và chỉ head ở train. Khi fine-tune batch nhỏ, ưu tiên giữ BN running statistics cố định; ghi rõ trong cấu hình. Optimizer phải chứa đúng tham số vừa mở.

Smoke test trước job dài: overfit một tập nhỏ 16–32 mẫu train để phát hiện sai nhãn/loss/head; kết quả đó chỉ kiểm pipeline. Chạy 50–100 step để đo peak VRAM, thời gian/step và số step/epoch. Ước lượng `giờ chạy = epoch × (train_steps × thời_gian_step + validation_time) / 3600`; ghi đo thực, không hứa một đêm sẽ xong.

Lưu `last` để resume và `best` để evaluate. Resume cần epoch, optimizer, scaler, RNG states, seed, data/split hash và cấu hình; best inference chỉ cần state dict và metadata. Lưu nguyên tử, kiểm reload, giữ checkpoint gốc. Một GPU chạy một job; notebook thử nghiệm không được giữ thêm model trên GPU khi train chính.

## 6. Bốn thí nghiệm bắt buộc

| Run | Backbone | Giai đoạn | Mục đích |
| --- | --- | --- | --- |
| E01 | EfficientNet-B0 | Frozen backbone, train head | Baseline transfer learning |
| E02 | EfficientNet-B0 | Fine-tune từ best E01 | Đo tác động mở backbone |
| E03 | ResNet50 | Frozen backbone, train head | Đối chứng cùng split |
| E04 | ResNet50 | Fine-tune từ best E03 | So sánh chất lượng và chi phí |

Thêm baseline đoán lớp đông nhất để hiểu imbalance. Hai backbone dùng cùng nhóm mẫu, protocol metric, ngân sách epoch/tuning và quy tắc chọn checkpoint. Khác biệt transform/interpolation theo weights phải ghi rõ; không đánh đồng số step với chi phí tính toán.

Khởi đầu một seed. Nếu còn giờ máy, chạy thêm hai seed cho cấu hình cuối của mỗi backbone, rồi báo mean/std. Không bắt buộc nhiều seed để có MVP; một seed phải được ghi là hạn chế. Không dùng ba seed như một ensemble.

Nếu lớp lớn lấn át, phân tích recall trước. Chỉ thử thêm một biện pháp cân bằng, như weighted loss từ train, trong thí nghiệm có mã riêng. Không cùng lúc đổi loss, augmentation và split. Không tính trọng số từ test.

Artifact mỗi run ở `ai/runs/<run-id>/`: config, environment/version, training log, checkpoint, class map, preprocessing và data hashes. Báo cáo/bảng/biểu đồ ở `docs/experiments/<run-id>/`. Ghi provenance code bằng Git commit nếu có repository; nếu chưa có Git thì lưu manifest/hash source, không bịa commit.

## 7. Khóa và đánh giá

Cuối tuần 4–6, chọn cấu hình bằng validation. B0 là ưu tiên vì kích thước và yêu cầu dự án; vẫn chọn dựa trên chất lượng/latency đo được. Nếu ResNet50 tốt hơn đáng kể và CPU đáp ứng, ghi quyết định đổi model mặc định. App chỉ nạp một model mặc định; báo cáo vẫn có cả hai.

Trước khi mở test, khóa checkpoint/hash, class map, preprocessing, selection rule, quy tắc abstention và script metric. Tuần 7 chạy hai model cuối trên test cùng protocol:

- Accuracy, macro-F1, precision/recall/F1 và support từng lớp; confusion matrix theo thứ tự class cố định.
- Lưu dự đoán từng sample để tạo lại bảng; lớp support bằng 0 phải báo không đánh giá được, không giấu khỏi bảng.
- Báo chỉ số toàn bộ test trước abstention; thêm tỷ lệ được trả lời (coverage) và lỗi trong phần được trả lời. Không tính “từ chối” là dự đoán đúng.
- Phân tích một số lỗi đại diện, không chỉ ảnh đẹp. Bootstrap theo group để có khoảng bất định nếu ngân sách cho phép.
- Benchmark CPU: tách cold start/model load và warm inference; sau warm-up dùng ít nhất 100 lần đo, ghi p50/p95, cấu hình máy và có/không gồm decode/preprocess/API.
- Mục tiêu đề xuất macro-F1 ≥0,85, recall từng lớp ≥0,70 và warm CPU p95 ≤3 giây. Chốt ở tuần 1; chưa có bằng chứng đạt.

Threshold trên softmax chỉ hỗ trợ nói “chưa đủ cơ sở”, không chứng minh ảnh là lá cà chua hay phát hiện được mọi bệnh ngoài danh sách. Nếu hiệu chuẩn temperature/threshold, dùng validation; khi ít mẫu, báo chưa hiệu chuẩn thay vì gọi phần trăm là xác suất chẩn đoán.

Ảnh thực địa có nguồn/quyền/nhãn đáng tin và đã loại trùng mới dùng cho đánh giá ngoài miền. Ảnh tự chụp chưa có nhãn xác nhận có thể kiểm trải nghiệm và phân tích định tính, không tạo accuracy bằng nhãn AI. Nếu không có test thực địa đủ điều kiện, kết luận phải giới hạn ở benchmark đã đánh giá.

Nếu sửa mô hình sau khi xem test, ghi test đã tham gia phát triển; không tiếp tục gọi là đánh giá cuối độc lập. Sửa lỗi đóng gói mà không đổi model/preprocessing có thể chạy lại bộ kiểm kỹ thuật và ghi rõ phạm vi.
