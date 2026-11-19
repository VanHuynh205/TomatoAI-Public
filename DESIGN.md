---
name: Tomato Desktop
description: Sổ tay đối chiếu ảnh lá cà chua và đọc hướng dẫn chăm sóc có nguồn.
colors:
  paper: "#f5f1e6"
  sheet: "#fffcf3"
  field: "#ecefe0"
  ink: "#262b20"
  muted: "#575c49"
  line: "#ccceba"
  line-strong: "#7b856c"
  green: "#314b2d"
  green-deep: "#233d23"
  tomato: "#9b392a"
  caution: "#724719"
  focus: "#126b92"
  error: "#842d29"
  error-background: "#f7e5dc"
typography:
  headline:
    fontFamily: 'Cambria, "Times New Roman", serif'
    fontSize: "clamp(2.4rem, 5vw, 4.45rem)"
    fontWeight: 400
    lineHeight: 1.08
    letterSpacing: "-0.04em"
  title:
    fontFamily: 'Cambria, "Times New Roman", serif'
    fontSize: "clamp(1.55rem, 2.6vw, 2rem)"
    fontWeight: 400
    lineHeight: 1.18
  body:
    fontFamily: '"Segoe UI", system-ui, sans-serif'
    lineHeight: 1.55
  secondary:
    fontSize: "0.875rem"
rounded:
  sheet: "0 0 13px 13px"
  picker: "8px"
  button: "7px"
components:
  button-primary:
    backgroundColor: "{colors.green}"
    textColor: "#fffdf5"
    rounded: "{rounded.button}"
    width: "100%"
  file-picker:
    backgroundColor: "#f7f7ec"
    textColor: "{colors.green-deep}"
    rounded: "{rounded.picker}"
  reading-panel:
    backgroundColor: "{colors.sheet}"
    rounded: "{rounded.sheet}"
  error-message:
    backgroundColor: "{colors.error-background}"
    textColor: "{colors.error}"
    rounded: "{rounded.button}"
---

# Sổ tay cà chua

Giao diện đặt ảnh người dùng cạnh trang đối chiếu, theo cách bố trí của một sổ tay thực địa. Nền giấy ấm và đường kẻ mảnh tách các phần. Cambria dùng cho tên trang, tên vùng và kết quả nhận diện. Segoe UI dùng cho thao tác, hướng dẫn và dữ liệu nguồn. Các phông chữ lấy từ hệ thống.

## Bố cục

Khung tối đa 74rem. Trên màn rộng, cột ảnh có nền xanh nhạt và cột đọc rộng hơn có nền giấy. Ở 760px trở xuống, hai cột xếp dọc theo thứ tự ảnh, kết quả, chăm sóc. Cả hai cột ngắt dòng với tên tệp và nội dung nguồn dài. Nút nhận diện cao tối thiểu 48px. Vùng đọc giới hạn đoạn văn và danh sách ở 69ch.

## Chọn ảnh và đọc kết quả

- Vùng nét đứt chọn ảnh có nhãn tiếng Việt và input file native. Sau khi chọn, ảnh xem trước nằm phía trên vùng chọn ảnh thu gọn. Ảnh giữ tỷ lệ gốc, tên tệp và dung lượng nằm ngay dưới vùng chọn.
- Nút xanh đậm bắt đầu nhận diện. Dòng trạng thái công bố lúc ảnh sẵn sàng, đang xử lý và có kết quả. Thông báo lỗi giữ hướng xử lý và dùng vai trò alert.
- Kết quả có tiêu đề cố định. Điểm mô hình nằm cạnh ghi chú về cách diễn giải. Chỉ kết quả được chấp nhận có thẻ chăm sóc tương ứng. Tại đó có mục lục ngắn, giới hạn áp dụng và thông tin đối chiếu nguồn. Các nút mở nguồn dùng trình duyệt hệ thống.

## Chuyển động và bàn phím

GSAP đưa ảnh xem trước vào trong 240ms. Phần tên kết quả và điểm xuất hiện theo thứ tự trong khoảng 320ms, lấy ý tưởng từ Animated Content của React Bits và kích hoạt khi có dữ liệu mới. Trạng thái xử lý có chấm nháy tối đa hai nhịp đi cùng thông báo chữ.

Khi người dùng chọn một mục hoặc nguồn trong thẻ chăm sóc, tiêu điểm chuyển đến đích và Lenis cuộn tới đó trong 320ms. GSAP ticker điều khiển vòng khung hình của Lenis trong lúc cuộn và được tháo khi hoàn tất hoặc bị ngắt. Cuộn bằng lăn chuột và chạm dùng hành vi trình duyệt. Khi bật giảm chuyển động, ảnh và kết quả hiển thị ngay, liên kết cuộn tức thì và chấm trạng thái đứng yên. Viền tiêu điểm 3px phục vụ thao tác bàn phím, kể cả trên vùng chọn ảnh.

Các thư viện chạy từ gói cục bộ. Nội dung, nguồn và trạng thái có thể đọc khi chuyển động kết thúc hoặc bị tắt.

Nguồn tham khảo về chuyển động: [React Bits Animated Content](https://reactbits.dev/animations/animated-content), [GSAP matchMedia](https://gsap.com/docs/v3/GSAP/gsap.matchMedia/), [Lenis](https://github.com/darkroomengineering/lenis#readme).
