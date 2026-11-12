# Giấy phép và ghi công bên thứ ba

## DINOv2

Dự án dùng backbone pretrained DINOv2-base của Meta AI Research (FAIR), do Maxime Oquab, Timothée Darcet và các đồng tác giả phát triển. Dòng bản quyền upstream là **Copyright (c) Meta Platforms, Inc. and affiliates.** [Bài báo và danh sách đầy đủ tác giả](https://arxiv.org/abs/2304.07193).

Model gốc phát hành theo Apache License 2.0. Bản của dự án bắt đầu từ `facebook/dinov2-base` tại revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`, sau đó tinh chỉnh backbone và huấn luyện đầu phân loại lá cà chua. Bản quyền và giấy phép của DINOv2 gốc tiếp tục áp dụng cho phần upstream. [Model card gốc](https://huggingface.co/facebook/dinov2-base) · [Model card của Meta](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md) · [Bản Apache License 2.0 đi kèm](desktop/notices/dinov2/LICENSE).

## Giao diện và nội dung chăm sóc

React, React DOM và Scheduler trong bản build giao diện giữ các [giấy phép đi kèm](desktop/notices/frontend/). Catalog chăm sóc giữ [CC BY-SA 4.0 và thông báo nguồn](infra/knowledge/CONTENT-LICENSE.md).

Các thư viện Python được cài riêng theo lockfile và chịu giấy phép do nhà phát hành công bố.
