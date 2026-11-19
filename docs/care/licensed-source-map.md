# Nguồn thay thế và bản đồ claim cho tám thẻ

Đối chiếu ngày 05/11/2026 bằng bản PDF công khai tại máy chủ factsheet Plantwise và điều khoản CC. Không dùng search snippet làm bằng chứng cuối. Không có clip Web Clipper mới, không sửa vault hoặc bản clip của người dùng. Bản nháp cũ, hashes và packet được giữ tại `docs/progress/2026-11-05-d4/before-care-replacement/`.

## Nguồn đã đọc

Tất cả tài liệu dưới đây là PDF một trang; phần chân trang ghi CC BY-SA 4.0 và ©CAB International. Thông báo quyền được kiểm trên từng file, không áp giấy phép trang web chung cho tài liệu khác. Giấy phép cho phép tạo/chia sẻ bản phái sinh với điều kiện ghi công, nêu thay đổi và ShareAlike; xem [CC BY-SA 4.0, điều 2 và 3](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en). Công Văn đã chấp thuận toàn bộ tám thẻ ngày 05/11/2026 với điều kiện giữ đủ ghi công; quyền nguồn trong bản xuất được ghi permitted theo căn cứ này và giấy phép đã đối chiếu. Xem [hồ sơ duyệt](review.md).

| Mã | Tài liệu/tác giả | Phạm vi và phần dùng |
| --- | --- | --- |
| pw_bacterial | [Bacterial Spot on tomato](https://factsheetadmin.plantwise.org/Uploads/PDFs/20137804207.pdf), KENGAP Horticulture/Plantwise, 12/2012 | Kenya; nguồn nói nhiều nước có bệnh. Recognize the problem; Background; Management / Non-chemical control |
| pw_early | [Early blight on Tomato](https://factsheetadmin.plantwise.org/Uploads/PDFs/20137804212.pdf), KENGAP Horticulture/Plantwise, 12/2012 | Kenya; nhận biết đốm/vòng, phần canh tác giàn và thông khí |
| pw_late | [Late blight on Tomato](https://factsheetadmin.plantwise.org/Uploads/PDFs/20137804214.pdf), KENGAP Horticulture/Plantwise, 12/2012 | Kenya; nhận biết vết/rìa mặt dưới lá, giàn và phủ đất |
| pw_septoria | [Septoria on Tomato](https://factsheetadmin.plantwise.org/Uploads/PDFs/20137804217.pdf), KENGAP Horticulture/Plantwise, 12/2012 | Tạo ở Kenya, ghi All Countries; dùng nhận biết và lớp phủ sau trồng, không suy ra đã kiểm tại Việt Nam |
| pw_leaf_mold | [番茄叶霉病](https://factsheetadmin.plantwise.org/Uploads/PDFs/20137804306.pdf), 赵世福, 北京市顺义区植保植检站/Plantwise, 2012, sửa 2020 | Trung Quốc; 问题识别, 背景, 管理 mục 1. Tác nhân Fulvia fulva, đúng mốc lá; không nhầm black leaf mold do Pseudocercospora |
| pw_mites | [Tomato red spider mite](https://factsheetadmin.plantwise.org/Uploads/PDFs/20137804187.pdf), Faith Toroitich/James Kimani, University of Nairobi/Plantwise, 10/2012 | Kenya; Recognize, Background, Cultural. Tetranychus spp. rộng hơn nhãn nhện hai chấm, giới hạn đã ghi trong thẻ |
| pw_yellow | [Tomato Yellow Leaf Curl Virus on Tomato](https://factsheetadmin.plantwise.org/Uploads/PDFs/20137804221.pdf), KENGAP Horticulture/Plantwise, 12/2012 | Kenya; triệu chứng, bọ phấn và lựa chọn giống kháng/chống chịu |
| pw_water | [Spider mites on Tomato](https://factsheetadmin.plantwise.org/Uploads/PDFs/20177800164.pdf), General Directorate of Agriculture (GDA)/Plantwise, 11/2014 | Campuchia; chỉ Management, đoạn stress nước/tưới phù hợp cho thẻ healthy |

KENGAP factsheets dựa trên The Tomato Farming Handbook, 2011. Đầu trang “Created” và năm sửa là ngày nội dung ghi trong PDF; không dùng ngày crawler như ngày xuất bản. Không có ngày kiểm chuyên môn địa phương.

## Mapping thuộc bản đã được chủ dự án chấp thuận

| Thẻ | Triệu chứng/quan sát | So sánh dễ nhầm | Hành động |
| --- | --- | --- | --- |
| bacterial_spot | pw_bacterial: vết úng nước/chuyển nâu, quả sần; quan sát khi cây ướt | Đối chiếu triệu chứng đốm nhỏ với pw_septoria | pw_bacterial: tránh thao tác lúc tán ướt |
| early_blight | pw_early: lá thấp/vòng đồng tâm, thân/quả | Đối chiếu lá già/đốm với pw_septoria | pw_early: giàn và tỉa phù hợp để thông khí |
| late_blight | pw_late: vết không đều, rìa trắng mặt dưới khi ẩm | Đối chiếu vòng đồng tâm của pw_early | pw_late: giàn và lớp phủ giảm bắn nước |
| leaf_mold | pw_leaf_mold: hai mặt lá, lớp mốc, ẩm/thông gió | pw_mites: chấm vàng/tơ; so sánh hình thái | pw_leaf_mold: thông gió/cách tưới cho nhà kính |
| septoria_leaf_spot | pw_septoria: lá già/đốm nhỏ/chấm ở tâm | pw_early: vòng đồng tâm | pw_septoria: phủ đất sau trồng |
| spider_mites | pw_mites: mặt dưới/chấm vàng/tơ/nóng khô | pw_leaf_mold: lớp mốc thay vì tơ | pw_mites: kiểm bờ ruộng và phạm vi triệu chứng |
| yellow_leaf_curl | pw_yellow: lá nhỏ/cuốn/vàng/cây thấp, bọ phấn | pw_mites: vàng/chấm/tơ | pw_yellow: giống kháng/chống chịu cho vụ mới |
| healthy | Giới hạn nhãn từ spec/model contract của dự án; không là kết luận nông học | Ngoài tám lớp hoặc dấu hiệu không rõ là giới hạn ứng dụng | pw_water: tránh stress nước; không định mức tưới |

Các cặp “dễ nhầm” là tổng hợp đối chiếu biểu hiện giữa hai nguồn; không phải kết quả thử chẩn đoán hoặc trích nguyên văn differential diagnosis. Bản dịch từ tiếng Trung giữ đúng nhánh canh tác nhà kính; không chuyển các thuốc ở mục 2/3. Phê duyệt của chủ dự án bao gồm bản tổng hợp trong packet, không thay thẩm định chuyên môn.

## Điều không chuyển sang thẻ

- Không dùng claim “mites transmit viruses” trong factsheet Campuchia; chỉ dùng đoạn stress nước.
- Không dùng bản TYLCV Việt Nam `20147801398.pdf` cho claim vector vì có câu gộp thrips với whiteflies; không tự sửa nội dung nguồn rồi gọi là nguyên bản.
- Không dùng ảnh/bảng, tên/liều thuốc, quy tắc đốt cây, lịch mùa vụ hoặc định mức từ nước ngoài. Các hành động canh tác vẫn giữ phạm vi nơi biên soạn.
- Không dùng WorldVeg 2010 làm căn cứ quyền: bản tiếng Anh hiện trả 403, search snippet về CC chưa đủ. Không dùng NSW Keep it CLEAN: trang bản quyền PDF yêu cầu phép viết riêng. Agriculture Victoria có điều kiện tái sử dụng phi thương mại nhưng không được chọn cho bản thẻ này.

## Gate còn lại

Chủ dự án đọc [packet có hashes](draft-review-packet.md), xác nhận tên/ngày, các thẻ được chấp thuận và phạm vi biên soạn offline theo CC BY-SA 4.0. Sau xác nhận mới ghi review_status/source rights phù hợp, tạo allowlist khóa hash và export. Không coi việc người dùng sửa tên Obsidian Web Clipper là hành động duyệt.
