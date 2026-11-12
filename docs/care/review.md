# Phạm vi duyệt tám thẻ chăm sóc

Chủ dự án đã chấp thuận tám thẻ thay thế biên soạn từ factsheet Plantwise/CABI ngày 05/11/2026, người duyệt Công Văn, với điều kiện đủ ghi công và chỉ dùng tham khảo offline. Xác nhận này không cấp quyền cho model hoặc dataset, không mở rộng sang các bản nguồn cũ và không phải thẩm định chuyên môn.

Tám ID: bacterial_spot, early_blight, late_blight, leaf_mold, septoria_leaf_spot, spider_mites, yellow_leaf_curl, healthy. Từng draft có SHA-256 trong [allowlist](../../infra/knowledge/allowlist.json). Catalog [care-v1](../../infra/knowledge/care-v1.json) có SHA-256 `78361a5804b9e941f13f13e01256d742357946d7965e1b1f6a14ac7e3abeef56`. Draft được khôi phục nguyên bytes trong repo public để exporter kiểm lại; không hạ gate phê duyệt hoặc thay nội dung chăm sóc.

[Bản đồ nguồn](licensed-source-map.md) và [thông báo nội dung](../../infra/knowledge/CONTENT-LICENSE.md) giữ tác giả, URL gốc, copyright, CC BY-SA 4.0, thông báo dịch/rút gọn và không chứng thực. UI/JSON giữ provenance khi dùng offline. Không kèm PDF, ảnh hoặc logo nguồn.

`expert_verified` vẫn false. Phần dễ nhầm là tổng hợp đối chiếu biểu hiện giữa tài liệu, không phải thử nghiệm chẩn đoán đã thẩm định. Giữ phạm vi Kenya, Trung Quốc hoặc Campuchia; không chuyển liều thuốc, lịch mùa vụ hoặc hiệu quả nước ngoài thành hướng dẫn cho Việt Nam. Nguồn ghi All Countries không đồng nghĩa đã được kiểm tại Việt Nam.

Các nguồn cũ UMN, NC State và UC IPM không được đưa vào các thẻ này. Không lấy giấy phép CABI gán cho nội dung cũ. Hồ sơ hội thoại/phê duyệt chi tiết được chủ dự án giữ cục bộ; repo public chỉ công bố tóm tắt phạm vi duyệt và các đầu vào có hash để tái kiểm.
