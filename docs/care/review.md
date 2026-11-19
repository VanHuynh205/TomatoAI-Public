# Đối chiếu thẻ chăm sóc D4

Trạng thái ngày 05/11/2026: Công Văn chấp thuận toàn bộ tám thẻ thay thế từ factsheet Plantwise/CABI để tham khảo offline, với điều kiện đủ ghi công. Đã đối chiếu thông báo giấy phép trên cả tám PDF và các mục ghi công trong từng thẻ. Đã xuất catalog curated `care-v1` bằng exporter hiện hành. Đây là phê duyệt của chủ dự án, `expert_verified` vẫn false.

## Nguồn của bản trước

- Người dùng xác nhận nguồn cũ được lưu bằng Obsidian Web Clipper từ các trang tìm qua Google; không tìm thấy giấy phép. Đây là thông tin cách thu thập, không phải đồng ý xuất bản hoặc duyệt tám thẻ.
- [UMN Copyright notice](https://apps.extension.umn.edu/copyright.html) yêu cầu yêu cầu và chấp thuận bằng văn bản cho việc tái sử dụng nội dung. Hai nguồn về chăm sóc nền và nhện hai chấm chưa có chấp thuận này.
- [UC IPM Legal Notices](https://ipm.ucanr.edu/GENERAL/copyright.html) có ngoại lệ sao chép văn bản cho giáo dục, giữ ghi công và thông báo copyright. Trang hiện được đánh dấu archived; chưa chứng minh ngoại lệ cho bản biên soạn tiếng Việt trong phần mềm.
- Các factsheet NC State đã đọc không cung cấp giấy phép tái phân phối rõ trong phần hiển thị. Không áp điều khoản của thư viện NC State cho trang Extension khác phạm vi.

Bản nguồn cũ và packet/hashes được giữ nguyên trong `docs/progress/2026-11-05-d4/before-care-replacement/`. Không lấy giấy phép CABI gán cho UMN, NC State, UC IPM hoặc vault cũ.

## Bản được chấp thuận

[Bản đồ claim/nguồn](licensed-source-map.md) ghi tám tài liệu, tác giả, mục/trang, phạm vi và các phần bị loại. [Packet](draft-review-packet.md) được giữ nguyên với tám thẻ và SHA-256 tại thời điểm gửi duyệt. [Thông báo nội dung](../../infra/knowledge/CONTENT-LICENSE.md) giữ ghi công, thay đổi, ShareAlike và giới hạn bảo đảm. Phần “dễ nhầm” được ghi rõ là agent tổng hợp đối chiếu biểu hiện, không giả kết quả chẩn đoán đã thẩm định.

Từng PDF có thông báo CC BY-SA 4.0. [Điều khoản 2 và 3](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en) là căn cứ cho dịch/biên soạn và chia sẻ có điều kiện. Kiểm ghi công xác nhận mỗi nguồn giữ tên tài liệu/tác giả, ©CAB International, URL gốc, URL giấy phép, thông báo dịch/rút gọn, áp dụng cùng giấy phép và không có bảo đảm/chứng thực. UI hiển thị các trường này trong mục “Thông tin đối chiếu nguồn”; JSON giữ chúng khi dùng offline. Không đóng gói ảnh, logo hoặc nguyên PDF nguồn.

Trước phê duyệt, exporter đã từ chối cả tám draft; bằng chứng âm giữ tại `care-revision-red.log`, `care-revision-green.log` và `care-draft-gates.json`. Snapshot `before-care-approval/` giữ nguyên nội dung đã gửi duyệt. Chỉ cập nhật reviewer/ngày/trạng thái duyệt và metadata quyền theo xác nhận; nội dung triệu chứng, so sánh, quan sát, hành động, giới hạn và provenance giữ nguyên. Không thay đổi vault, metadata nguồn gốc, `rag_ready`, TEST hoặc logic publication gate.

## Phạm vi sử dụng đã chấp thuận

Tham khảo offline cho đồ án, văn bản do agent biên soạn dưới CC BY-SA 4.0, không đóng gói ảnh/bảng nguồn. Giữ phạm vi Kenya, Trung Quốc hoặc Campuchia; không chuyển lịch mùa vụ, liều thuốc hoặc mức hiệu quả nước ngoài thành hướng dẫn Việt Nam. Không coi nguồn ghi All Countries là đã kiểm tại Việt Nam. `expert_verified` vẫn false. Đây không phải thay giấy phép code/model/dataset.

## Dấu vết phê duyệt và xuất bản cục bộ

Người dùng trả lời câu hỏi `call_2225734ffd1a4a7d965a7e53c6dab9e9`: “nếu đã đủ điều kiện ghi công, tôi chấp thuận tất cả dùng để tham khảo offline, ngày duyệt là hôm nay, cứ lấy tên Công Văn làm người duyệt”. Ngày của xác nhận là 2026-11-05. Tám ID: bacterial_spot, early_blight, late_blight, leaf_mold, septoria_leaf_spot, spider_mites, yellow_leaf_curl, healthy. Không mở rộng xác nhận này sang nguồn cũ hoặc xác nhận chuyên gia.

- [Approval JSON](../progress/2026-11-05-d4/care-approval.json): nguyên văn xác nhận, hash packet và hash từng thẻ trước/sau metadata.
- [Allowlist](../../infra/knowledge/allowlist.json): đúng tám ID/file/hash, allowed_use=offline_reference_vn.
- [Catalog](../../infra/knowledge/care-v1.json): curated, knowledge_version=care-v1, SHA-256 `78361a5804b9e941f13f13e01256d742357946d7965e1b1f6a14ac7e3abeef56`.
- [Kiểm xuất](../progress/2026-11-05-d4/care-export.json): schema/coverage đúng, nội dung khoa học không đổi; hai lần export độc lập có bytes bằng catalog.
- [Smoke B0 thật](../progress/2026-11-05-d4/real-smoke.json): parity CPU/API đạt trên ba ảnh synthetic; không đánh giá accuracy hoặc TEST.
