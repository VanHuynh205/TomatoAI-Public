# Hợp đồng module và quy tắc mở rộng

Cập nhật 01/11/2026 theo yêu cầu giữ cấu trúc khi thêm tính năng và train thêm. Đây là phần bổ sung của BMad spec, áp dụng cho kế hoạch coding.

## Ranh giới thay đổi

| Thay đổi | Phần được phép đổi | Bằng chứng không làm hỏng phần khác |
| --- | --- | --- |
| Train lại cùng backbone và tập lớp | Dữ liệu/config/model bundle và báo cáo | Contract inference, mapping và API/UI tests giữ nguyên |
| Thêm lớp bệnh | Dataset/map, head K lớp, model mới, thẻ chăm sóc | Model classes được care catalog phủ; UI hiển thị theo dữ liệu, không sửa nhánh if theo tên bệnh |
| Thêm backbone mới | Model factory/loader/config trong `ai/` | Artifact schema hợp lệ, inference contract và CPU smoke pass; không tự nạp code từ bundle |
| Thêm tính năng UI | Feature tương ứng trong `frontend/src/features/`, route/composition cần thiết | UI feature cũ và API contract vẫn pass |
| Thêm API/nghiệp vụ | Module tương ứng trong `backend/`, schema/OpenAPI và API client sinh lại | Giữ API v1 tương thích hoặc thêm v2 có chủ đích |
| Đổi host desktop | `desktop/`, công đoạn build/launch | API, frontend build và model bundle không phụ thuộc host cụ thể |
| Cập nhật nguồn chăm sóc | Bản tổng hợp/allowlist/knowledge version | Không đổi class_id; test nội dung/quyền/nguồn và UI thẻ giữ được |

Chưa tạo repository/service/plugin cho tính năng giả định. Chưa thêm DB chỉ để dành chỗ cho lịch sử. Khi thêm lưu lịch sử sau này, record phải giữ class_id, model_version và knowledge_version; không diễn giải lại kết quả cũ bằng mapping mới.

## Chiều phụ thuộc

`ai/` chứa tính toán và contract ML, không import `backend` hoặc `desktop`. `backend/` gọi interface inference và tra thẻ, không import desktop hoặc frontend. `desktop/` ghép các thành phần và phục vụ frontend build. Frontend chỉ truy cập API qua một client có kiểu; component không đọc trực tiếp model/card file hoặc gọi Python filesystem.

Trách nhiệm trong file nhỏ, đặt cạnh feature liên quan. `App.tsx`, launcher và route handler chỉ phối hợp. Không gom train, inference, chăm sóc và state UI vào một file dùng chung. Test ranh giới kiểm các import bị cấm và thư mục source, không ép một giới hạn dòng máy móc.

## Artifact và ID

Model bundle v1 gồm `model.pth` chứa tensor state dict và `metadata.json`. Metadata phải có version, architecture nằm trong allowlist, ordered class_ids duy nhất, cấu hình preprocessing đầy đủ, hash weights, phiên bản/hashes dữ liệu/split và provenance. Runtime kiểm hash/schema trước load, dùng `weights_only=True`, không import code khai trong metadata.

Classifier output thứ i ứng với `class_ids[i]`. Thẻ chăm sóc nối bằng class_id. Tên tiếng Việt chỉ là nội dung hiển thị. Thay tên hiển thị không đổi ID. Khác biệt thứ tự giữa bộ dữ liệu và model phải được map có kiểm tra, không sort lại lúc runtime.

Model với K output phải có đúng K class_ids. Thêm class vào JSON không làm model học lớp mới: phải train head mới, đánh giá và phát hành bundle mới. Care catalog có thể có thẻ dư; phải phủ mọi class_id của model đang chạy. Thiếu thẻ hoặc sai schema làm gói không đủ điều kiện phát hành.

Preprocessing có version và các giá trị resize/crop/interpolation/normalization được lưu; validate transform và inference dùng cùng implementation. Runtime không tự đổi transform theo defaults của phiên bản thư viện mới.

Phiên bản model, knowledge và API độc lập. Kiểm tương thích dựa trên schema và tập class IDs; không buộc các version string phải giống nhau.

## API và trạng thái

Backend là nguồn schema API/OpenAPI; frontend sinh kiểu response từ schema đó. Các task coding phải định nghĩa cùng tên field và status trong một contract, có kiểm tra drift. Mock chỉ mô phỏng đúng contract, không tự đặt shape riêng.

Phân biệt dự đoán `ok`/`uncertain` với lỗi HTTP. `uncertain` trả HTTP 200 và `class_id: null`; có thể giữ top-k để giải thích mức phù hợp, không mở chăm sóc đặc hiệu như bệnh đã xác nhận. Input sai, bận, model lỗi dùng HTTP code và error schema ổn định. UI xử lý theo code, không parse chuỗi thông báo.

Request wrapper duy nhất sở hữu token, timeout và giải mã lỗi. Khi chọn ảnh mới, response cũ không được ghi đè ảnh/kết quả hiện tại. Hủy request ở browser không có nghĩa CPU đã ngừng inference: backend giữ giới hạn một tác vụ đến khi compute thật sự kết thúc.

## Phát hành và thay model

Không hot reload trong MVP. Pipeline tạo bundle ở thư mục candidate mới, kiểm đầy đủ rồi đóng gói release mới. Giữ bundle/binary trước để rollback. Không ghi đè checkpoint đang dùng hoặc sửa bundle đã niêm phong.

Thử cập nhật bằng fixture có giá trị dự đoán biết trước để kiểm contract; đánh giá model thật là gate riêng. Fixture/dev mode phải được đánh dấu rõ và bị từ chối trong build release. Không để ảnh fixture hoặc model ngẫu nhiên trở thành bằng chứng sản phẩm nhận diện bệnh.

Đường dẫn asset dựa trên thư mục ứng dụng/resource, không theo current working directory. Gói chạy ở vị trí chỉ đọc; file tạm/log nếu cần nằm trong thư mục người dùng và không chứa ảnh/token. Không cần quyền administrator để thao tác nhận diện thông thường.

Host chọn port bằng socket đã bind `127.0.0.1:0` và chuyển chính socket đó cho server; không đóng rồi bind lại. Cửa sổ mở sau readiness; đóng app phải dừng server. Token phiên do host truyền qua cơ chế bridge giới hạn, không đưa vào URL hoặc log. API kiểm token trên mọi endpoint, Host hợp lệ và Origin khi header này có; GET same-origin có thể không gửi Origin. Không cho trang ngoài điều hướng vào cửa sổ có bridge đặc quyền.

## Điều kiện xác nhận khả năng mở rộng

1. Hai bundle cùng contract nhưng weights khác có thể thay qua bản release; UI/API tests không đổi.
2. Bundle thêm lớp với thẻ tương ứng hiển thị được qua cùng UI; thiếu thẻ bị từ chối trước dùng.
3. Class order đổi trong metadata nhưng tensor không đổi bị phát hiện trong export/parity test; checksum file đơn lẻ không chứng minh nhãn đúng.
4. Schema tương lai chưa hỗ trợ, hash sai và K lệch bị từ chối với lý do rõ.
5. Sau thử candidate không đạt, bản release trước vẫn mở và suy luận được.

Các phép kiểm trên thuộc kế hoạch coding. Chưa có implementation hoặc kết quả test cho chúng.
