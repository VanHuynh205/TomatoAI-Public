# Lộ trình 8 tuần: nhận diện bệnh ở cây cà chua

Ngày lập: 30/10/2026. Đối tượng: nhóm 3 sinh viên, mới học machine learning.

## 1. Đầu ra và giả định

Mục tiêu đề xuất: phân loại trạng thái lá cà chua từ một ảnh, so sánh EfficientNet-B0 với ResNet50 bằng transfer learning, và làm demo dùng mô hình đã chọn. EfficientNet-B0 là biến thể được đề xuất; nhóm mới chỉ xác nhận họ EfficientNet và ResNet50.

Sản phẩm cuối gồm dữ liệu có nguồn và split tái lập; code train/evaluate/inference; checkpoint và cấu hình; bảng so sánh; demo; báo cáo; slide và kịch bản bảo vệ. Bài toán phân loại ảnh một lá, không bao gồm đếm lá, khoanh vùng nhiều bệnh, xác định giống cà chua hoặc xác nhận bệnh bằng xét nghiệm.

Đã xác nhận: app Windows offline và báo cáo; GPU RTX 3050, VRAM 4 GB; chưa dùng Colab; thời gian khoảng hai tháng từ hiện tại, chưa có hạn nộp chính thức.

Ngân sách tạm tính: 10 giờ/người/tuần, tổng 240 giờ công. Giữ khoảng 15% dự phòng. Đây là giả định lập kế hoạch, chưa phải cam kết của nhóm. Lịch đề xuất: 30/10–24/12/2026, dự phòng 25–30/12 nếu hạn nộp cho phép. Không coi ngày dự phòng là hạn nộp đã được giảng viên xác nhận.

Stack đề xuất: Python, PyTorch/torchvision cho ML; PySide6 Widgets cho app Windows; PyInstaller đóng gói dạng thư mục trước để dễ kiểm tra. Nếu repository ứng dụng cũ có stack phù hợp và code dùng được, ưu tiên giữ stack đó. App tối thiểu có chọn/xem ảnh, nút nhận diện, tên lớp, điểm mô hình và thông báo giới hạn. Suy luận chạy nền để giao diện không bị treo; hỗ trợ CPU để demo trên máy không có NVIDIA. Đóng gói checkpoint và mapping; không tự tải weights hoặc gọi dịch vụ lúc sử dụng.

Với GPU 4 GB: train lần lượt từng mô hình, thử đầu vào 224×224 và batch size 8; giảm xuống 4 nếu thiếu VRAM. Đây là cấu hình khởi đầu cần đo, không bảo đảm vừa mọi môi trường. Dùng mixed precision nếu môi trường hỗ trợ và kiểm tra loss hữu hạn. Bắt đầu đóng băng backbone rồi fine-tune một phần; chạy thử vài batch trước job dài. Việc B/C phụ trách hai mô hình không có nghĩa chạy hai job đồng thời trên cùng GPU. Lưu checkpoint để phục hồi. Chỉ cân nhắc Colab khi máy local không đáp ứng thời gian, chưa cần dành một tuần học/cấu hình cloud.

## 2. Điểm xuất phát đã kiểm tra

- Manifest PlantVillage hiện hành có 18.160 record ảnh, 10 lớp. Có 14 record loại vì trùng byte; 18.146 bản ảnh dẫn xuất được báo cáo giữ lại.
- Đếm trực tiếp manifest: 11.411 record `ready_for_split`, 6.735 `needs_review`, 14 `excluded_exact_duplicate`, 0 record `ready_for_training=True`.
- Subset có grouping chỉ bao phủ 8 lớp. Target Spot và Tomato mosaic virus chưa có ảnh `ready_for_split`. Không gọi subset này là benchmark 10 lớp.
- Pilot PlantDoc có 16 record. Hồ sơ intake ghi chưa xác nhận nhãn canonical, chưa kiểm tra trùng với dữ liệu cũ và chưa sẵn sàng đánh giá ngoài miền.
- PlantExpert có 17 card kiến thức cần review. WUR, Dryad, Mendeley là dữ liệu canh tác nghiên cứu; không cần cho bài toán nhận diện ảnh hiện tại.
- Thư mục đang mở chủ yếu chứa dữ liệu và báo cáo. Không tìm thấy pipeline train, notebook thí nghiệm hoặc checkpoint mô hình trong phạm vi tìm kiếm. Một số báo cáo trỏ tới scripts ở repository cũ; cần kiểm tra và lấy lại phần dùng được trước khi viết mới.

Nguồn local: `data/TomatoAI/data/dataset-registry.json`; `data/TomatoAI/data/processed/plantvillage/pv-d9e4c21eec8e9172/manifest.csv`; `data/TomatoAI/data/reports/plantvillage/README.md`; `data/external-web-intake/2026-10-19-plantdoc-pilot-v1/checks.json`.

Đây là đối chiếu manifest và hồ sơ có sẵn, không phải kiểm định lại từng nhãn, decode lại toàn bộ ảnh hoặc hash lại tất cả file.

## 3. Chia trách nhiệm

| Thành viên | Chịu trách nhiệm chính | Phối hợp bắt buộc |
| --- | --- | --- |
| A | Dữ liệu, grouping, split, đánh giá và phân tích lỗi | B kiểm tra split; C kiểm tra tên lớp trong demo |
| B | Pipeline dùng chung, EfficientNet-B0, lưu và nạp checkpoint | C dùng cùng pipeline để chạy ResNet50 |
| C | ResNet50 ở giai đoạn thí nghiệm; demo, tích hợp và đóng gói sau đó | B hỗ trợ inference; A cung cấp bộ ca kiểm thử |

Cả ba học kiến thức cốt lõi, đọc kết quả và viết báo cáo phần mình. Không giao toàn bộ hiểu biết mô hình cho một người. Mỗi tuần họp 30–45 phút: xem artifact, lỗi còn lại và chốt tuần sau. Mỗi phần quan trọng phải có một người khác chạy lại.

## 4. Lịch thực hiện

| Tuần | Việc cụ thể | Đầu ra và điều kiện hoàn thành |
| --- | --- | --- |
| 1 — 30/10–05/11: Chốt bài toán, học nền tảng | Cả nhóm học Python/NumPy tối thiểu, tensor, Dataset/DataLoader, CNN, loss, overfitting, train/validation/test, precision/recall/F1. A đọc manifest và review queue. B/C chạy tutorial transfer learning trên dữ liệu mẫu. Trao đổi rubric với giảng viên. | Một trang phạm vi; sơ đồ dữ liệu; danh sách lớp; mỗi người giải thích được validation khác test. Cuối tuần quyết định hướng 10 lớp hay xin duyệt subset 8 lớp nếu chưa xử lý được grouping. |
| 2 — 06–12/11: Khóa dữ liệu và môi trường | A xử lý grouping/ảnh trùng/nhãn đáng ngờ trong phạm vi đã chọn. B/C chuẩn bị môi trường chung và loader. Chia train/validation/test khoảng 70/15/15 theo nhóm lá, cố gắng giữ tỷ lệ lớp. Chỉ augmentation ngẫu nhiên trên train. | Manifest split có version, seed, tên lớp, số ảnh và số nhóm mỗi tập. Kiểm tra không giao group/duplicate giữa các tập; mỗi lớp có đại diện đủ dùng. Loader chạy được trên hai máy hoặc môi trường dùng chung. Chưa đủ căn cứ thì giữ ảnh ngoài benchmark. |
| 3 — 13–19/11: Hai baseline | B và C dùng cùng pipeline, split và quy tắc chọn checkpoint. B chạy EfficientNet-B0, C chạy ResNet50: backbone pretrained đóng băng, train lớp phân loại. A kiểm tra dự đoán, phân bố lớp và log. | Hai baseline chạy đủ vòng train–validation–save–reload–predict; có loss curve, validation macro-F1 và cấu hình. Có baseline đoán lớp đông nhất làm mốc tối thiểu. Không dùng test để chọn cấu hình. |
| 4 — 20–26/11: Fine-tune có kiểm soát | B/C mở một phần backbone và fine-tune. A phân tích lớp nhầm, kiểm tra augmentation và cân bằng lớp nếu cần. Chỉ thay một yếu tố trong mỗi phép so sánh. | Bốn thí nghiệm chính: hai backbone đóng băng, hai backbone fine-tune. Cùng dữ liệu, ngân sách thử nghiệm tương đương; mọi khác biệt được ghi lại. Có checkpoint tốt nhất theo validation cho mỗi mô hình. |
| 5 — 27/11–03/12: Chốt lựa chọn, dựng demo | A/B xem validation macro-F1, recall từng lớp, thời gian suy luận và bộ nhớ. Nếu ngân sách cho phép, chạy lại hai cấu hình tốt nhất với ba seed. C dựng luồng chọn ảnh–suy luận–hiển thị bằng checkpoint thật. | Khóa cấu hình, checkpoint, mapping lớp, preprocessing và quy tắc hiển thị trước khi mở test. Demo đầu tiên chạy được. Ngưỡng confidence, nếu có, chọn trên validation và không được coi là thuật toán nhận diện ảnh ngoài miền. |
| 6 — 04–10/12: Đánh giá cuối | A chạy quy trình test đã khóa cho hai mô hình. B đo latency trên máy dự kiến demo. C kiểm tra ảnh lỗi, đường dẫn tiếng Việt, ảnh kích thước lớn và nạp model offline. Đánh giá ảnh ngoài nguồn chỉ khi có nhãn và kiểm tra độc lập phù hợp. | Bảng accuracy, macro-F1, precision/recall/F1 và support từng lớp; confusion matrix; danh sách lỗi có phân tích. Báo cáo thời gian khởi động và suy luận. Tách rõ kết quả PlantVillage và ngoài nguồn. Nếu sửa model sau khi xem test, ghi test đã tham gia quá trình phát triển. |
| 7 — 11–17/12: Hoàn thiện sản phẩm | C hoàn thiện giao diện/đóng gói Windows offline. B bảo đảm preprocessing và mapping giống lúc đánh giá; inference không tự tải weights. A chuẩn bị ca kiểm thử và hoàn thiện phần thực nghiệm. | Một người khác chạy demo từ hướng dẫn cài đặt. Tắt mạng và thử trên máy/môi trường khác, gồm chế độ CPU. Báo cáo bản gần hoàn chỉnh, ảnh giao diện và bảng kết quả thật. |
| 8 — 18–24/12: Khóa bản nộp, tập bảo vệ | Cả nhóm rà báo cáo, trích dẫn, giấy phép và khả năng chạy lại. Làm slide, video dự phòng, backup. Tập trình bày và hỏi đáp; sửa lỗi chặn nộp. | Gói nộp gồm code, hướng dẫn, manifest split, config, checkpoint, metrics, demo, báo cáo, slide. Mỗi người giải thích được đóng góp, leakage, mô hình, chỉ số và giới hạn. Không thêm tính năng mới trong tuần cuối. |

Viết báo cáo từ tuần 1: bài toán và phạm vi ở tuần 1; dữ liệu ở tuần 2; phương pháp ở tuần 3–4; kết quả ở tuần 5–6; sản phẩm và kết luận ở tuần 7.

## 5. Quy tắc thực nghiệm

1. Giữ raw bất biến. Mỗi lần chọn/lọc/split tạo manifest dẫn xuất có lý do và truy vết.
2. Ảnh của cùng lá và biến thể trùng phải nằm cùng một tập. Group ID tự sinh từ filename không chứng minh ảnh độc lập. Tỷ lệ 70/15/15 là gần đúng; ưu tiên không giao nhóm.
3. Nếu 10 lớp chưa có bằng chứng grouping đủ dùng, xin giảng viên duyệt benchmark 8 lớp. Khi đó model, UI, bảng kết quả và báo cáo đều phải ghi 8 lớp. Không tự gộp hai bệnh vào lớp khác.
4. Tách thống kê theo lớp: dữ liệu gồm healthy và spider mites, nên không gọi mọi nhãn là một bệnh truyền nhiễm. Giữ mapping nhãn nguồn, tên khoa học và tên hiển thị rõ ràng.
5. Dùng validation để chọn checkpoint, learning rate, augmentation, threshold và mô hình. Test chỉ phục vụ đánh giá cuối theo quy trình đã khóa.
6. Hai mô hình dùng cùng split, cách tính metric và ngân sách điều chỉnh tương đương. Chuẩn hóa đầu vào phải tương thích pretrained weights; mọi khác biệt cần được ghi.
7. Ưu tiên macro-F1; báo cả accuracy, recall và số mẫu từng lớp. Không đặt cam kết accuracy 99% khi chưa có thí nghiệm. Nếu giảng viên yêu cầu ngưỡng, chốt ngưỡng ở tuần 1.
8. Ghi seed, môi trường, phiên bản dữ liệu, config và checkpoint. Lưu dự đoán để tạo lại bảng kết quả mà không phải train lại.
9. Ảnh ngoài nguồn dùng để chỉnh mô hình là tập phát triển ngoài miền, không còn là test độc lập. Pilot 16 ảnh hiện tại chỉ dùng khảo sát sau review; không đủ để khẳng định độ chính xác ngoài vườn.
10. Confidence của classifier không bảo đảm ảnh là lá cà chua hoặc bệnh nằm trong danh sách hỗ trợ. Demo phải trình bày giới hạn này. Nội dung chăm sóc chỉ thêm nếu có nguồn và review; không cần để hoàn thành phần ML cốt lõi.

## 6. Cắt giảm khi chậm

- Cuối tuần 1 chưa chốt 10 lớp: chuẩn bị phương án 8 lớp cho giảng viên quyết định; không kéo dài rà dữ liệu vô hạn.
- Cuối tuần 3 chưa có baseline: dừng phát triển giao diện nâng cao, tập trung pipeline tối thiểu và kiểm tra loader/nhãn/loss.
- Cuối tuần 5 chưa có demo: tích hợp một checkpoint ổn định; giữ so sánh hai mô hình trong báo cáo.
- Thiếu thời gian/GPU: bỏ grid search, ensemble, cross-validation nhiều fold và Grad-CAM tùy chọn. Giữ hai mô hình, split đáng tin và đánh giá trung thực.
- Chưa có bộ ngoài miền đạt yêu cầu: báo giới hạn và chỉ đưa khảo sát định tính nếu phù hợp; không tạo nhãn giả để có bảng metric.
- Chưa duyệt tri thức: giữ demo nhận diện; không mở rộng sang chatbot, thuốc, tưới hoặc tối ưu năng suất.

## 7. Nguồn học chính

- [PyTorch: Transfer Learning for Computer Vision](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html): học và tái sử dụng cách đóng băng backbone, fine-tune, train và đánh giá.
- [scikit-learn: StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html): tham khảo nguyên tắc giữ nhóm không giao nhau và cố gắng bảo toàn tỷ lệ lớp. Đây không phải yêu cầu chạy k-fold cho toàn bộ đồ án.
- [PyTorch: Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/notes/amp_examples.html): tham khảo autocast và GradScaler khi train bằng mixed precision.
- [Qt for Python](https://doc.qt.io/qtforpython-6/): tài liệu PySide6 cho app desktop.
- [PyInstaller: Operating Mode](https://pyinstaller.org/en/stable/operating-mode.html): cách đóng gói Python và dependency thành bản chạy Windows; ưu tiên dạng thư mục khi kiểm tra ban đầu.

## 8. Thông tin đang chờ

Giờ làm mỗi người/tuần, ngày nộp chính thức và rubric giảng viên. Nhóm đã xác nhận app Windows offline và báo cáo, RTX 3050 4 GB, chưa dùng Colab. Lịch hiện tại giả định 10 giờ/người/tuần; cập nhật khi có ngân sách thời gian thực. Chưa có training, xác nhận nhãn hay đo chất lượng mô hình trong lượt lập kế hoạch.
