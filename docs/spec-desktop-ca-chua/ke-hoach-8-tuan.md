# Kế hoạch 8 tuần cho một người cùng AI

Ngày lập: 01/11/2026. Bạn đã xác nhận khoảng 3 giờ mỗi tối. Lịch đề xuất bắt đầu thứ Hai 02/11 và kết thúc Chủ nhật 27/12/2026; chưa có hạn nộp chính thức.

Mục tiêu: hai mô hình EfficientNet-B0 và ResNet50 có thực nghiệm so sánh; một app Windows offline dùng model đã chọn; hướng dẫn chăm sóc có nguồn cho từng lớp hỗ trợ; bộ báo cáo và bản nộp tái hiện được. Không phụ thuộc hai thành viên còn lại hoặc việc tìm được chuyên gia.

## Ngân sách và phạm vi

8 tuần × 7 tối × 3 giờ = **168 giờ**. Lập lịch 6 buổi việc chính/tuần: **144 giờ**; buổi thứ bảy giữ **24 giờ dự phòng** cho lỗi, việc cá nhân hoặc nghỉ. Không coi dự phòng là chỗ thêm tính năng. Giờ GPU tự chạy ghi riêng; AI có thể hỗ trợ code và tài liệu, nhưng bạn vẫn cần chạy, kiểm tra và hiểu kết quả.

Một buổi 3 giờ: khoảng 15 phút xem checkpoint công việc, 120 phút làm phần chính, 35 phút kiểm chứng và 10 phút ghi kết quả/việc tiếp. Kiến thức mới học ngay trong task liên quan. Chưa quen Python/React thì lấy giờ dự phòng cho học và giảm tính năng phụ, không bỏ kiểm dữ liệu.

Khuyến nghị kỹ thuật:

- EfficientNet-B0 là model đầu tiên và ưu tiên cho app; ResNet50 dùng cùng pipeline để đối chứng. Chạy lần lượt trên GPU 4 GB.
- Benchmark khởi đầu đề xuất tám lớp PlantVillage có grouping; chốt sau audit quyền/dữ liệu. `tomato_vn` chỉ bổ sung khi xử lý được format YOLO, nhãn và trùng nguồn.
- React/TypeScript cho UI, FastAPI cho API, pywebview cho cửa sổ và PyInstaller cho gói Windows; thử đóng gói sớm.
- Chăm sóc bằng thẻ có nguồn, không cần LLM/RAG hoặc GPU runtime. Đối chiếu nguồn do bạn thực hiện cùng AI; không gọi đó là chuyên gia kiểm định.

Phạm vi và tiêu chí đầy đủ ở [spec](spec-desktop-ca-chua.md), [protocol dữ liệu/ML](du-lieu-va-thuc-nghiem.md), [kiến trúc/chăm sóc](kien-truc-va-cham-soc.md). [Hiện trạng](hien-trang-va-nguon.md) phân biệt số đã kiểm với giả định. Lộ trình cũ tại root giữ như lịch sử; không dùng giả định 240 giờ của nhóm ba người cho lịch này.

Sau rà soát lần hai, [kế hoạch coding Superpowers](../superpowers/plans/2026-11-01-00-tomato-desktop-master.md) cụ thể hóa 48 buổi này bằng file, interface, test và dependency. Đây là cách triển khai cùng phạm vi, không cộng thêm một dự án hoặc ngân sách tám tuần khác.

## Mốc cuối tuần

| Tuần | Ngày dự kiến | Kết quả cần nhìn thấy | Điều kiện đi tiếp |
| --- | --- | --- | --- |
| 1 | 02–08/11 | Scope, nguồn, lớp dự kiến, rủi ro; cửa sổ desktop đóng gói thử | Có hướng dữ liệu và stack khả thi; không tự coi dữ liệu đã sẵn sàng |
| 2 | 09–15/11 | Manifest/split đã khóa, loader và smoke train | Không giao group; kiểm nguồn/quyền; mỗi lớp có support; đo được VRAM/thời gian |
| 3 | 16–22/11 | E01/E02 B0; API predict nhận checkpoint thật | Save/reload/inference khớp; có validation metrics thật |
| 4 | 23–29/11 | E03/E04 ResNet50; bảng validation và chọn model | Hai model cùng protocol; test còn niêm phong |
| 5 | 30/11–06/12 | Đủ thẻ chăm sóc cho lớp đã chọn, API tra thẻ | Map đầy đủ; đối chiếu nội dung/quyền, không giả nhận chuyên môn |
| 6 | 07–13/12 | App hoàn chỉnh luồng ảnh–kết quả–chăm sóc | UI/API test chạy được; lỗi và bất định có phản hồi |
| 7 | 14–20/12 | Test cuối, binary CPU offline, bằng chứng kiểm thử | Báo chỉ số thật, không runtime download, không cần Python/CUDA |
| 8 | 21–27/12 | Báo cáo, slide, demo/video và gói nộp | Có hướng dẫn chạy, attribution, artifact và backup; không thêm tính năng |

## 48 buổi việc chính

Mỗi dòng là một buổi tối khoảng 3 giờ, không phải cam kết mọi lần train xong trong 3 giờ. Các mã chưa được thực hiện. Task phụ thuộc task trước trong cùng luồng; chỉ mở nhánh độc lập khi chờ GPU. Ghi ngày, artifact, check thực chạy và lỗi còn lại khi đánh dấu xong.

### Tuần 1: quyết định và kiểm khả thi — 18 giờ

| Mã | Việc làm | Bằng chứng xong | Phụ thuộc |
| --- | --- | --- | --- |
| T01 | Đọc spec, đối chiếu rubric/hạn nộp nếu có; xác nhận Windows, ngân sách và tám lớp dự kiến; chọn tiêu chí chất lượng | Một biên bản quyết định; câu hỏi chưa có đáp án được ghi, không bịa | Kế hoạch này |
| T02 | Lập bảng nguồn/quyền cho đúng bản PlantVillage và tomato_vn; xác định bộ nhà kính nếu có URL | `docs/data/source-register.md`, bằng chứng license/version, bộ chưa rõ để riêng | T01 |
| T03 | Kiểm đường dẫn manifest, group metadata và mẫu trùng; tìm script cũ còn dùng được trước viết mới | Mẫu file resolve được; danh sách cần sửa đường dẫn ở bản dẫn xuất; phương án chống leakage | T02 |
| T04 | So schema hai dataset; rà class 0, ảnh nhiều nhãn và prefix giao tập; chốt benchmark chính và nhánh hoãn | Danh sách K lớp dự kiến, mapping nguồn, lý do chưa dùng toàn bộ tomato_vn | T03 |
| T05 | Thử host desktop, frontend build và health API tối thiểu; đóng gói onedir nhỏ | Binary mở cửa sổ/health khi ngắt mạng; ghi điều kiện WebView2; fixture chưa phải model bệnh | T01 |
| T06 | Chốt ranh giới module/contract; tạo skeleton ML và fixture tự tạo; đọc tensor, head, loss, train/val/test | Smoke loader/model shape; test mapping đầu tiên; checkpoint công việc tuần | T04, T05 |

Điểm dừng: nếu stack host không đóng gói được, dùng buổi dự phòng để xử lý và quyết định host trước xây UI. Nếu quyền bản dữ liệu còn mơ hồ, tiếp tục phần fixture/host nhưng chưa train chính trên bộ đó.

### Tuần 2: dữ liệu dùng được và pipeline — 18 giờ

| Mã | Việc làm | Bằng chứng xong | Phụ thuộc |
| --- | --- | --- | --- |
| T07 | Viết/tái dùng audit decode, hash, nhãn, group; tạo manifest dẫn xuất, giữ raw | Báo ảnh hỏng/trùng/thiếu; exclusions có lý do; test fixture lỗi | T03, T06 |
| T08 | Chia train/validation/test theo nhóm; kiểm phân bố và giao tập | Split/version/seed/hash; assertion không giao group và duplicate đã xác minh | T07 |
| T09 | Khảo sát một subset nhỏ tomato_vn để quyết định có bổ sung; nếu dùng crop thì kiểm box/ảnh cha và đối chiếu trùng | Crop pilot có provenance và test tọa độ, hoặc quyết định hoãn có lý do; không kéo dài nhánh | T04, T08 |
| T10 | Rà contact sheet theo lớp, khóa mapping/split/preprocessing; xác nhận điều kiện dữ liệu | `docs/data/dataset-card-v1.md`; số mẫu/group từng tập; chữ ký đối chiếu của chủ dự án, không là xác nhận chuyên gia | T07–T09 |
| T11 | Train smoke trên ít mẫu, kiểm loss, metric, best/last và reload | Chạy overfit nhỏ để kiểm pipeline; test save/reload; không gọi là accuracy tổng quát | T06, T10 |
| T12 | Đo 50–100 step trên cả backbone, thử AMP/batch, kiểm resume và runtime CPU | Peak VRAM, thời gian/epoch ước lượng, cấu hình vừa máy và kế hoạch job GPU | T11 |

Khóa dữ liệu ở đây. Nếu bổ sung sau đó, tạo version/split mới và ghi thí nghiệm nào không còn so sánh trực tiếp được. Tổng ngân sách khảo sát tomato_vn ở T04/T09 không quá 6 giờ.

### Tuần 3: EfficientNet-B0 và API đầu tiên — 18 giờ

| Mã | Việc làm | Bằng chứng xong | Phụ thuộc |
| --- | --- | --- | --- |
| T13 | Chạy E01 frozen backbone; học đọc loss curve và macro-F1 | Log/config/checkpoint baseline B0, train/val tách đúng | T12 |
| T14 | Phân tích validation E01, lỗi loader/head/BN nếu có; sửa có regression khi cần | Bảng từng lớp và quyết định fine-tune; không mở test | T13 |
| T15 | Mở block cuối, cấu hình optimizer và BN; bắt đầu E02 | Config E02, gradient/BN kiểm được, job chạy ổn | T14 |
| T16 | Hoàn tất hoặc resume E02, kiểm overfit và lưu best | Validation E01/E02, checkpoint tốt nhất theo rule | T15 |
| T17 | Đóng gói model artifact B0: metadata, transform, mapping; inference CPU | Một ảnh hợp lệ trả đúng shape/schema; reload dự đoán khớp trong tolerance đã ghi | T16 |
| T18 | API health/predict gọi inference dùng chung; không viết UI trong backend | API trả model_version/top-k/status từ model thật; smoke request và file lỗi | T17, T05 |

Chờ GPU thì viết phần phương pháp báo cáo hoặc rà nguồn thẻ. Không chạy ResNet50 song song với B0 trên cùng GPU.

### Tuần 4: ResNet50 và chọn cấu hình — 18 giờ

| Mã | Việc làm | Bằng chứng xong | Phụ thuộc |
| --- | --- | --- | --- |
| T19 | Lấy/kiểm pretrained ResNet50, chạy E03 bằng pipeline chung | URL/hash, head K lớp, config/log/checkpoint E03 | T12, T16 |
| T20 | Fine-tune E04 từ best E03, giữ protocol tương đương | Config/checkpoint E04, metric validation và epoch thực chạy | T19 |
| T21 | So bốn run theo validation, recall từng lớp, overfit và tài nguyên | Bảng so sánh từ log thật; giải thích chênh lệch, không tuyên bố model thắng trước số liệu | T16, T20 |
| T22 | Đo CPU hai model, kiểm transform/label parity; cân nhắc chạy thêm seed nếu còn giờ máy | Benchmark sơ bộ và quyết định model mặc định có lý do | T21 |
| T23 | Hoàn thiện validation upload/token/host/busy và test API | File giả/quá lớn/token sai bị từ chối; một model instance, ảnh không tự lưu | T18 |
| T24 | Chọn checkpoint, quy tắc bất định từ validation; viết bản protocol đánh giá cuối | Danh sách artifact cần khóa, tiêu chí lựa chọn và giới hạn OOD; test chưa dùng | T22, T23 |

Nếu chỉ đủ thời gian một seed, ghi hạn chế. Không bỏ ResNet50 hoặc tăng thử nghiệm đến mức mất thời gian tích hợp.

### Tuần 5: chăm sóc có nguồn — 18 giờ

| Mã | Việc làm | Bằng chứng xong | Phụ thuộc |
| --- | --- | --- | --- |
| T25 | Map mỗi class tới hồ sơ/clip; kiểm quyền, vùng và phần quản lý; xác định nguồn thay thế khi thiếu | Ma trận lớp–nguồn–claim–quyền, đủ nguồn ứng viên cho K lớp | T10 |
| T26 | Soạn và đối chiếu thẻ Bacterial spot, Early blight, Late blight | Ba bản thẻ có dẫn mục/trang và điều kiện; loại nội dung thuốc không phù hợp | T25 |
| T27 | Soạn và đối chiếu Leaf mold, Septoria, Spider mites | Ba bản thẻ; không nhầm nhện hai chấm với nhện rám | T25 |
| T28 | Soạn Yellow leaf curl và Healthy; bổ sung lớp khác chỉ nếu K đã khóa lớn hơn tám | Đủ K bản thẻ; không suy toàn cây khỏe hoặc xác nhận virus từ ảnh | T25 |
| T29 | Bạn đọc lại từng khuyến nghị, đối chiếu nguồn/quyền; ghi reviewer/date và phạm vi hiển thị | Thẻ source-checked; expert_verified false; không sửa gate vault cũ | T26–T28 |
| T30 | Xuất bundle allowlist cho app mới, API care và test mapping | K lớp map đúng thẻ; draft ngoài allowlist, ID sai trả lỗi; nội dung đọc offline được | T29, T23 |

Thiếu nguồn cho một hành động cụ thể thì bỏ hành động đó và ghi thiếu căn cứ. Giữ thẻ có thông tin nhận biết/quan sát và phần chăm sóc đủ bằng chứng; không bỏ toàn bộ chức năng chăm sóc để bù tiến độ.

### Tuần 6: app hoàn chỉnh luồng chính — 18 giờ

| Mã | Việc làm | Bằng chứng xong | Phụ thuộc |
| --- | --- | --- | --- |
| T31 | Dùng Impeccable đúng nhánh 2D, xác định bố cục tối thiểu; UI chọn ảnh/preview | Luồng chọn/đổi ảnh hoạt động, hướng dẫn chụp một lá rõ | T05, T24 |
| T32 | Nối predict, loading, kết quả, uncertain và lỗi | UI hiển thị schema thật; không gọi score là chắc chắn chẩn đoán | T31, T23 |
| T33 | Thẻ chăm sóc/nguồn/giới hạn theo class, nội dung tiếng Việt | Mỗi kết quả map đúng thẻ; uncertain không tự xác nhận bệnh | T32, T30 |
| T34 | Ghép desktop host, API và static build; readiness/shutdown/path | Chạy từ ngoài cwd dự án; tắt app không còn server; không asset CDN | T33 |
| T35 | Thiết lập hoặc dùng e2e runner theo skill, viết flow ảnh–kết quả–thẻ và lỗi | Tests có assertion chính xác, báo cáo/trace; blocker credential ghi riêng nếu có | T34 |
| T36 | Một vòng QA gộp desktop/cửa sổ nhỏ, keyboard/loading/error; sửa và xác nhận phạm vi đã sửa | Checklist UI hoàn tất; build frontend/type checks/test liên quan có kết quả | T35 |

Không thêm lịch sử, tài khoản, dashboard hay hiệu ứng 3D. Nếu cần cập nhật thiết kế, giữ phạm vi luồng chính và giới hạn QA của Impeccable.

### Tuần 7: đánh giá cuối và bản offline — 18 giờ

| Mã | Việc làm | Bằng chứng xong | Phụ thuộc |
| --- | --- | --- | --- |
| T37 | Khóa hash hai checkpoint, mapping, transform, threshold và script metric; rà leakage lần cuối | Protocol có ngày khóa, manifest test chưa bị dùng tuning | T24, T36 |
| T38 | Chạy đánh giá test cho hai model, lưu predictions và confusion matrix | Accuracy/macro-F1/per-class support thật; báo cả phần abstain và coverage | T37 |
| T39 | Phân tích lỗi và đo latency CPU cuối; làm model card | Bảng p50/p95/cold-start, hạn chế dataset/ngoài miền, mục tiêu đạt/chưa đạt rõ | T38 |
| T40 | Build CPU onedir; đóng asset/model/cards/license; chuẩn bị WebView2 offline nếu cần | Gói binary có manifest version/hash, kích thước đo được | T34, T39 |
| T41 | Chạy binary tắt mạng trên máy khác hoặc môi trường Windows sạch; không Python/CUDA | Video/checklist native; kiểm ảnh hợp lệ/lỗi/tiếng Việt; xác nhận không cần tải mạng | T40 |
| T42 | Sửa lỗi tích hợp/đóng gói từ T41, chạy lại tests bị ảnh hưởng và xác nhận | Báo cáo verification; chưa có máy khác thì ghi chưa xác minh, không tự tick | T41 |

Sau T38 không chỉnh model vì nhìn thấy lỗi test rồi gọi kết quả mới là test độc lập. Nếu buộc sửa model/preprocessing, ghi test contamination và lập đánh giá mới có phạm vi trung thực.

### Tuần 8: hồ sơ nộp và bảo vệ — 18 giờ

| Mã | Việc làm | Bằng chứng xong | Phụ thuộc |
| --- | --- | --- | --- |
| T43 | Hoàn thiện bài toán, dữ liệu, phương pháp, kiến trúc từ ghi chép các tuần | Báo cáo có nguồn, lý do transfer learning và giới hạn nhãn | T10, T24, T34 |
| T44 | Viết thực nghiệm, thảo luận và kết luận từ metrics thật | Bảng tái tạo từ predictions; không phóng đại hiệu quả ngoài vườn/Việt Nam | T38, T39 |
| T45 | Làm slide, sơ đồ luồng và kịch bản demo ngắn | Slide đúng số liệu; phần hạn chế và chăm sóc có nguồn | T43, T44 |
| T46 | Tập bảo vệ, chạy demo tắt mạng, quay video dự phòng | Trả lời được leakage, macro-F1, hai backbone, CPU, nguồn thẻ và hạn chế | T42, T45 |
| T47 | Rà hướng dẫn, license/attribution, dependency, artifact và nội dung riêng tư | Gói code/model/report không chứa raw có quyền chưa rõ, secret hoặc ảnh người dùng | T42–T46 |
| T48 | Khóa bản nộp, tạo hash, sao lưu hai nơi và thử giải nén/chạy | Gói nộp cuối, checklist, backup kiểm mở được; danh sách tồn tại nếu có | T47 |

Báo cáo được ghi dần từ tuần 1 trong 10 phút cuối buổi; tuần 8 dành để hoàn thiện và bảo vệ, không viết lại mọi thứ từ trí nhớ.

## Khi chậm hoặc dữ liệu không đạt

| Tình huống | Quyết định |
| --- | --- |
| Cuối tuần 1 chưa rõ bộ nhà kính | Để ngoài MVP, tiếp tục bộ có provenance; không chờ giấy phép chưa biết khi còn việc độc lập |
| Hai lớp PlantVillage thiếu grouping chưa xử lý xong | Giữ tám lớp nếu rubric cho phép; ghi danh sách unsupported; không train nhãn thiếu bằng chứng cho đủ số |
| tomato_vn không đạt trong T04/T09 | Hoãn bổ sung; không coi nó là external test Việt Nam; giữ báo cáo phát hiện để xử lý sau |
| Cuối tuần 3 chưa có B0 chạy thật | Dùng giờ dự phòng, giảm thử nghiệm phụ; tập trung loader/label/loss. Giữ cả hai backbone ở mức tối thiểu bắt buộc |
| GPU chậm/OOM | Giảm batch, dùng accumulation/AMP, fine-tune ít block; đo lại. Không mặc định phải mua GPU hoặc dùng cloud |
| Cuối tuần 6 chưa có app end-to-end | Dừng tăng seed, Grad-CAM và mọi tính năng phụ; dùng model đã chọn và bộ thẻ tối thiểu có nguồn |
| Chưa có expert | Bạn đối chiếu nguồn theo checklist, ghi chưa thẩm định chuyên môn; không tạo nhãn hoặc xác nhận bệnh giả |
| Không có test thực địa độc lập | Báo benchmark nội nguồn và khảo sát định tính; không tuyên bố chất lượng sử dụng ngoài vườn |
| Không đạt mục tiêu ML/latency | Giữ kết quả thật; nêu chưa đạt tiêu chí và phạm vi demo. Không đổi ngưỡng để nhận chất lượng đạt |

Không cắt: kiểm leakage, mapping lớp, hai mô hình đối chứng, chăm sóc có nguồn, chạy offline, xử lý lỗi cơ bản và báo cáo trung thực. Có thể cắt: thêm seed, Grad-CAM, mở lớp mới, lịch sử ảnh, tối ưu ONNX/quantization và UI trang trí.

## Bạn và AI làm gì mỗi buổi

Bạn chốt phạm vi, chạy trên máy thật, xem contact sheet/nguồn và ký nhận kết quả đã hiểu. AI khảo sát code, viết phần nhỏ, chạy kiểm tra được phép, phân tích lỗi và ghi tài liệu. AI không thay bạn xác nhận provenance chưa có, nhãn chuyên môn hoặc kết quả chưa chạy.

Khi bắt đầu buổi, đưa một mã task và checkpoint gần nhất. Ví dụ:

> Thực hiện T07 theo `docs/spec-desktop-ca-chua/ke-hoach-8-tuan.md`. Đọc spec và companions cần thiết, giữ raw bất biến. Tìm script có sẵn trước viết mới. Làm phần vừa một buổi, kiểm chứng và ghi artifact, lỗi còn lại, bước tiếp theo vào docs. Không tự train chính khi gate dữ liệu chưa đạt.

Khi kết thúc buổi, ghi vào nhật ký dưới `docs/progress/`: task, đầu vào/version, file đổi, lệnh/check thực chạy, kết quả, quyết định và task kế tiếp. Không ghi “xong” nếu chỉ có code chưa chạy. Nếu người khác hỗ trợ, giao đọc nguồn/đối chiếu báo cáo hoặc chạy binary theo checklist; đường găng vẫn do bạn kiểm soát.

## Skills và MCP theo giai đoạn

| Giai đoạn | Dùng khi cần | Vai trò |
| --- | --- | --- |
| Định hướng/phạm vi | `bmad`, `bmad-spec` local | Bộ tài liệu hiện tại; cập nhật quyết định vào memlog khi yêu cầu thay đổi |
| Triển khai từng task | `superpowers:subagent-driven-development` hoặc `superpowers:executing-plans`, chọn một cách thực thi | Theo coding plan đã tổng hợp; không chạy thêm `bmad-build` cho cùng task, không tự commit/push |
| Tìm nguyên nhân lỗi | `systematic-debugging` khi lỗi chưa rõ | Chuyển sang giai đoạn điều tra có evidence, không thêm workflow TDD thứ hai cho cùng việc |
| UI | Impeccable đúng nhánh 2D | Bố cục, nhãn và trạng thái; đọc project context trước khi sửa |
| Kiểm tra flow | `e2e`, browser/computer-use theo bề mặt | Browser cho UI/API; native smoke riêng cho gói desktop |
| Dữ liệu/log dài | Context-mode MCP | Kiểm kê, tổng hợp log và kết quả có phạm vi |
| Tra nguồn | Web search MCP; Obsidian Web Clipper khi có yêu cầu thu | Tài liệu chính thức/paper, giữ provenance và quyền |
| Nội dung viết | Stop-slop + Humanizer | Giữ báo cáo rõ, không bịa kết quả hoặc thẩm định |

Không cần cài thêm plugin, dùng cloud, tạo agent chạy nền hoặc thuê dịch vụ để thực hiện bước đầu. Các skill dự kiến chưa được coi là đã chạy trong lượt lập kế hoạch; xem nhật ký công cụ ở [hiện trạng](hien-trang-va-nguon.md).

## Bắt đầu với ba buổi đầu

1. T01: xác nhận phạm vi một lá, tám lớp dự kiến, mục tiêu và điều kiện nộp. Những gì chưa biết ghi vào quyết định mở.
2. T02: khóa nguồn/phiên bản/quyền của đúng dữ liệu sẽ dùng; không tải thêm hàng loạt.
3. T03: kiểm manifest và đường dẫn/group trên dữ liệu thật. Chỉ sau gate tuần 2 mới chạy thí nghiệm chính.

Chưa train, xây app, xuất thẻ chăm sóc hoặc chạy E2E trong lượt này. Kế hoạch hoàn thành; chất lượng dữ liệu, thời gian train và khả năng đóng gói còn phải được chứng minh bằng các task tương ứng.
