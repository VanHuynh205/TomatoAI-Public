# Ghi công dữ liệu của DINOv2 v1

Đối chiếu ngày 13/11/2026. Đây là provenance của checkpoint v1, không phải giấy phép phát hành checkpoint. Repo không chứa ảnh dataset, archive dữ liệu hoặc weights tinh chỉnh.

## Phạm vi đã đối chiếu

Metadata checkpoint dẫn đến manifest train/validation có SHA-256 `34631a356aaaace2000799d8a55c8ae4d09c69edad5023c9a45a6197cae3c96a`; manifest test có SHA-256 `721ab09d690a3ed12aaf06c1c8c3d55e06cef6e480211e003d750b1968bbbe47`. Hai hash đã đối chiếu với bản lưu của dự án; manifest ảnh hiện không được phát hành trong repo này. Tổng hợp trực tiếp manifest xác nhận:

| Nguồn | Train | Validation | Giấy phép do nguồn công bố |
| --- | ---: | ---: | --- |
| PlantVillage | 2.800 | 600 | CC BY-SA 3.0; còn điều khoản liên quan thuật toán train |
| PlantCity v2 | 1.735 | 462 | CC BY 4.0 |
| CCMT v1 (Ghana) | 963 | 299 | CC BY 4.0 |
| Tomato Leaf Dataset v2 (Imtiaz) | 273 | 30 | CC BY 4.0 |
| PlantDoc Classification | 685 | 125 | Repo CC BY 4.0; quyền ảnh gốc chưa truy đủ |
| SoilClass v1 | 368 | 95 | CC BY 4.0 |
| Open Images V7 | 145 | 31 | Annotations CC BY 4.0; ảnh được liệt kê CC BY 2.0, cần kiểm từng ảnh |
| **Tổng** | **6.969** | **1.642** | Không áp một giấy phép chung cho toàn bộ |

Các ảnh được chọn, ánh xạ về chín nhãn của dự án, nhóm/chia tập và sàng trùng; sửa hướng EXIF, chuyển RGB, giới hạn cạnh dài 384px và mã hóa JPEG quality 73 trước tiền xử lý model. Model train ở 224px; suy luận/đánh giá v1 ở 280px. Không phát hành các ảnh đã biến đổi trong repo này. Những thay đổi này do dự án thực hiện; tác giả nguồn không chứng thực việc ánh xạ nhãn, kết quả hoặc ứng dụng.

## 1. PlantVillage

**PlantVillage Dataset**, kho `spMohanty/PlantVillage-Dataset`, revision `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`. Chọn các lớp tomato ảnh màu, nhóm/lấy mẫu/ánh xạ nhãn theo dự án.

- David P. Hughes; Marcel Salathé. *An open access repository of images on plant health to enable the development of mobile disease diagnostics*. 2015, bản v2 năm 2016. [arXiv:1511.08060v2](https://arxiv.org/abs/1511.08060v2).
- Sharada P. Mohanty; David P. Hughes; Marcel Salathé. *Using Deep Learning for Image-Based Plant Disease Detection*. 2016. [DOI:10.3389/fpls.2016.01419](https://doi.org/10.3389/fpls.2016.01419).
- [Dataset card của nhà phát hành](https://huggingface.co/datasets/mohanty/PlantVillage) khai báo [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

Bản trích nguồn sơ cấp đã lưu của dự án ghi câu “with the clarification that algorithms trained on the data fall under the same license.” Lần rà này xác minh trang abstract và dataset card; công cụ web không trích lại được nội dung PDF, nên câu trên dựa vào hồ sơ trích PDF đã lưu, không phải bản trích mới. Cần xin xác nhận tác giả hoặc rà pháp lý về phạm vi áp dụng với checkpoint cụ thể. Không kết luận mọi weights tự động là tác phẩm phái sinh CC, và cũng không bỏ qua điều khoản này để gán Apache cho model.

## 2. PlantCity v2

**Codewithsk Muhammad Sheraz Khan** (tên hiển thị của publisher). *PlantCity: A Comprehensive Image Based on Multi Crop Leaves disease in Pakistan.* 2025, version 2. [DOI:10.17632/w8kh2xkspx.2](https://doi.org/10.17632/w8kh2xkspx.2), [trang dataset](https://data.mendeley.com/datasets/w8kh2xkspx/2), [metadata đúng version](https://data.mendeley.com/public-api/datasets/w8kh2xkspx?version=2). [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Bài báo: Muhammad Sheraz Khan; Kainat Nisa; Irshad Ahmad; Muhammad Zubair; Kaznah Alshammari. *PlantCity: A comprehensive image based on multi crop leaves in Pakistan*. [DOI:10.1016/j.dib.2025.112130](https://doi.org/10.1016/j.dib.2025.112130).

Dự án dùng ảnh gốc từ archive upstream mang tên `test.zip`, không dùng archive train đã tăng cường ảnh; sau đó tạo train/validation riêng. Tên archive upstream không mô tả vai trò tập của dự án.

## 3. CCMT v1

*Dataset for Crop Pest and Disease Detection*. 2023, version 1. [DOI:10.17632/bwh3zbpkpv.1](https://doi.org/10.17632/bwh3zbpkpv.1), [trang dataset](https://data.mendeley.com/datasets/bwh3zbpkpv/1), [metadata đúng version](https://data.mendeley.com/public-api/datasets/bwh3zbpkpv?version=1). [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Metadata publisher ghi 15 contributor: Patrick Mensah Kwabena; Vivian Akoto-Adjepong; Kwabena Adu; Mighty Abra Ayidzoe; Elvis Asare Bediako; Owusu Nyarko-Boateng; Samuel Boateng; Esther Fobi Donkor; Faiza Umar Bawah; Nicodemus Songose Awarayi; Peter Nimbe; Isaac Kofi Nti; Muntala Abdulai; Remember Roger Adjei; Michael Opoku.

Bài báo *CCMT: Dataset for crop pest and disease detection*, [DOI:10.1016/j.dib.2023.109306](https://doi.org/10.1016/j.dib.2023.109306), ghi 17 tác giả: Patrick Kwabena Mensah; Vivian Akoto-Adjepong; Kwabena Adu; Mighty Abra Ayidzoe; Elvis Asare Bediako; Owusu Nyarko-Boateng; Samuel Boateng; Esther Fobi Donkor; Faiza Umar Bawah; Nicodemus Songose Awarayi; Peter Nimbe; Isaac Kofi Nti; Muntala Abdulai; Remember Roger Adjei; Michael Opoku; Suweidu Abdulai; Fred Amu-Mensah. Giữ riêng cách ghi tên và số tác giả của hai bản ghi, không coi chúng hoàn toàn giống nhau.

Dùng `Raw Data/CCMT Dataset/Tomato`, nhãn healthy và septoria. Nguồn này thực sự có trong train/validation v1, không phải held-out test.

## 4. Tomato Leaf Dataset v2 (Imtiaz)

Ahmed Imtiaz; Fahad Bin Islam Swapnil; Syed Rayhan Masud; **Debajoyti Karmaker** (cách viết trên publisher). *Tomato Leaf Dataset : A dataset for multiclass disease detection and classification*. 2025, version 2. [DOI:10.17632/bpfd9cns5g.2](https://doi.org/10.17632/bpfd9cns5g.2), [trang dataset](https://data.mendeley.com/datasets/bpfd9cns5g/2), [metadata đúng version](https://data.mendeley.com/public-api/datasets/bpfd9cns5g?version=2). [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

[Bài báo, DOI:10.1016/j.dib.2025.111520](https://doi.org/10.1016/j.dib.2025.111520) viết tên tác giả cuối là **Debajyoti Karmaker**. Giữ ghi nhận khác biệt chính tả; không tự sửa bản ghi publisher. Dự án chọn các họ annotated-export với một lớp được chấp nhận rồi chuẩn hóa/ánh xạ nhãn.

## 5. PlantDoc Classification

Davinder Singh; Naman Jain; Pranjali Jain; Pratik Kayal; Sudhakar Kumawat; Nipun Batra. *PlantDoc: A Dataset for Visual Plant Disease Detection*. Proceedings 2020, preprint 2019. [DOI:10.1145/3371158.3371196](https://doi.org/10.1145/3371158.3371196), [arXiv:1911.10317](https://arxiv.org/abs/1911.10317).

Kho dùng: [pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset), revision `5467f6012d78d1c446145d5f582da6096f852ae8`, công bố [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Train gồm 272 ảnh tomato và 413 ảnh other-leaf; validation gồm 68 và 57 ảnh tương ứng.

Bài báo mô tả ảnh thu thập từ Internet. Kiểm manifest của dự án thấy cả **810** dòng PlantDoc train/validation đều thiếu trường attribution gốc; record Git blob/path không chứng minh tác giả hoặc quyền của từng ảnh. Tác giả bài báo/dataset không được mặc nhiên coi là chủ bản quyền mọi ảnh. Chưa xác minh đầy đủ chuỗi quyền và ghi công của các ảnh này; không phát hành ảnh hoặc xem khai báo repo là xác nhận độc lập về quyền ảnh gốc.

## 6. SoilClass v1

Krishna Gopal Dhal; Arunita Das; Bishal Chakraborty; Soumik Pan. *SoilClass: A Novel Image Dataset for Soil Classification*. 2026, version 1. [DOI:10.17632/6mx34nshm4.1](https://doi.org/10.17632/6mx34nshm4.1), [trang dataset](https://data.mendeley.com/datasets/6mx34nshm4/1), [metadata đúng version](https://data.mendeley.com/public-api/datasets/6mx34nshm4?version=1). [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Dùng ảnh đất làm negative; nhóm theo block chụp hai phút, sau đó chuẩn hóa theo pipeline dự án.

## 7. Open Images V7

**Google LLC và các contributor của Open Images**, version 7, phát hành 10/2022. [Facts and figures / licenses](https://storage.googleapis.com/openimages/web/factsfigures_v7.html#licenses), [download/provenance](https://storage.googleapis.com/openimages/web/download_v7.html).

- Annotations: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Ảnh được publisher liệt kê theo [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/), với bản quyền từng photographer, không phải Google là tác giả duy nhất.
- Dự án dùng ảnh tay người làm negative; chưa kiểm thủ công toàn khung để bảo đảm không có lá cà chua.

Publisher cảnh báo không bảo đảm trạng thái giấy phép từng ảnh và yêu cầu người dùng tự kiểm. Cả **176** dòng train/validation hiện lưu JSON author, URL, license, image ID và box; chưa có đầy đủ title gốc hoặc snapshot trang giấy phép. Một URL spot-check đã trả 404; URL còn truy cập được không tự chứng minh trạng thái license tại thời điểm thu thập. Đây là khoảng trống xác minh, không phải kết luận ảnh không có giấy phép.

Trước khi phân phối ảnh phải kiểm và giữ author, title nếu có, URL gốc, URL license và thay đổi theo từng ảnh. Một dòng ghi công Google không thay thế thông tin photographer. Repo này không phát hành các ảnh đó và không tuyên bố đã hoàn tất clearance từng ảnh.

## Nguồn chỉ dùng đánh giá

**Nure Alam Siddiki Rachin; Nasim Mahmud Likhon**. *Tomato Leaf Disease Classification Dataset*. 2025, version 2. [DOI:10.17632/jttrv2w27r.2](https://doi.org/10.17632/jttrv2w27r.2), [dataset](https://data.mendeley.com/datasets/jttrv2w27r/2), [metadata](https://data.mendeley.com/public-api/datasets/jttrv2w27r?version=2). [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Dự án chọn 1.367 ảnh làm test; không đưa vào train/validation v1.

Taiwan Tomato và TOM2024 được ghi nhận là nguồn so sánh bên ngoài, không thuộc bảy nguồn fine-tuning trên. Lần rà này chưa xác minh đầy đủ quyền/ghi công của hai tập đó; không phát hành ảnh hoặc gán giấy phép cho chúng.

## Quyết định phát hành

Bốn dataset Mendeley train/validation có metadata đúng version công bố CC BY 4.0. Ghi công phải giữ tác giả, nguồn/DOI, license và thay đổi; giấy phép không cho phép giả chứng thực. Quyền của nội dung bên thứ ba trong mỗi dataset vẫn cần xét theo từng nguồn.

PlantVillage, PlantDoc và Open Images còn các khoảng trống nêu trên. Bổ sung tài liệu này không tự giải quyết chúng. **Chưa phát hành checkpoint tinh chỉnh hoặc executable chứa checkpoint.** Không gán Apache-2.0 cho weights theo giấy phép mã nguồn. Khi có xác nhận quyền hoặc kết luận pháp lý phù hợp với artifact và cách phân phối, mới quyết định giấy phép/phạm vi phát hành checkpoint; không cần train lại để hoàn thành việc rà quyền.

Tham khảo thêm [hướng dẫn CC về AI training](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/). Việc điều khoản áp dụng ra sao phụ thuộc quyền tác giả, tác phẩm được chia sẻ và khu vực pháp lý. Tài liệu này không phải tư vấn pháp lý.
