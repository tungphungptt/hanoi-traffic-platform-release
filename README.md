# hanoiFlow — dự báo ùn tắc +5/+10/+15 phút từ video giao thông

Đưa một video giao thông vào, hệ thống dự báo mức ùn tắc sau 5, 10 và 15 phút tại mọi thời điểm của
video. Mỗi dự báo chỉ dùng phần video **trước** thời điểm đó, nên có thể tua tới thời điểm được dự báo
để tự kiểm chứng.

## Chạy

Windows: bấm đúp `start.bat` (tạo `.venv`, cài thư viện, mở `http://127.0.0.1:8000`).

Hoặc thủ công (Python 3.11/3.12):

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Dùng

1. Chọn video quay liên tục dài 20–60 phút, bấm **Phân tích và dự báo** (khoảng 2,5 phút CPU cho 30 phút video).
   Video phát ngay trong trình duyệt từ file của bạn; máy chủ chỉ trả kết quả.
2. Phát video: khung bên phải là dự báo lập tại thời điểm đang xem (CI dự báo, khoảng tin cậy 80%, xác suất tắc).
   Bấm **Tua tới** để nhảy đến thời điểm được dự báo và xem thực tế. **Ẩn tương lai** giấu phần CI phía sau.
3. Mục 3 liệt kê các cảnh báo tắc trúng/sai; mục 4 so mô hình với "giữ nguyên mức hiện tại" và "kéo dài xu hướng".
4. **Tải kết quả (.json)** để lần sau mở lại cùng video mà không phải phân tích lại.

**Video nhiều camera.** Bản ghi chương trình truyền hình chuyển qua lại giữa các camera (như FM90 / Hà Nội
LIVE) được tách tự động: mỗi camera có chuỗi CI và dự báo riêng (chọn ở hàng "Camera trong video"),
đứt quãng khi camera không lên sóng; cảnh trường quay/băng rôn bị bỏ qua.

### Demo một chạm

Bấm **▶ Xem demo Hà Nội**: kết quả phân tích sẵn của bản ghi Hồ Tùng Mậu → Trần Duy Hưng (28 phút 50 giây, không có
trong dữ liệu học) mở ngay, kể cả trên Render; biểu đồ bấm được để xem dự báo ở mọi thời điểm. Muốn xem hình:
tải 6 file `traffic_016_cam1.mp4` … `traffic_021_cam1.mp4` từ thư mục Drive (link hiện trên trang) rồi chọn
**cả 6 file cùng lúc** ở mục 1 — trang tự nối chúng thành một dòng thời gian (không cần ghép video).

### Demo có sẵn (file)

| Kết quả đã lưu | Video cần mở kèm |
| --- | --- |
| `artifacts/demo_hanoi_tdh_htm.hanoiflow.json` | Thư mục Drive "Hanoi Flow" → Tran Duy Hung, ghép `traffic_016_cam1.mp4` … `traffic_021_cam1.mp4` (28 phút 50 giây; Hồ Tùng Mậu ~9 phút rồi Trần Duy Hưng ~20 phút) |
| `artifacts/demo_aic21_cam_6.hanoiflow.json` | `cam_6.mp4` của AI City Challenge 2021 Track 1, Dataset A |

Trang chủ → **Mở kết quả đã lưu** → chọn file `.json`, rồi chọn video ở mục 1. Video demo Hà Nội không
có trong dữ liệu học.

## Giao diện

Giữ giao diện "hanoiFlow · Trung tâm điều hành" (đỏ #b52226, vàng #c49a38, quốc kỳ, thanh bên trái) với ba trang:
**Dự báo & phương án** (video → dự báo → tuyến thay thế), **Bản đồ & thời tiết**, **Đánh giá mô hình**.

**Tuyến thay thế dùng dự báo.** Chọn điểm đi/đến (danh sách khu vực hoặc bấm bản đồ), hệ thống lấy tối đa 3 tuyến
từ OSRM trên OpenStreetMap. Chọn thêm vị trí camera của video (danh sách vị trí gần đúng của các camera Hà Nội hoặc
bấm bản đồ): tuyến nào đi qua camera trong 300 m mà dự báo +10 phút ở thời điểm đang xem cho CI ≥ 0,30 thì thời gian
được nhân `1 + 0,35 × CI dự báo`, rồi xếp lại. Đây là quy tắc tham khảo, không phải đo tắc thời gian thực
(đổi máy chủ chỉ đường bằng `ROUTING_API_BASE_URL`).

## Bản đồ và thời tiết Hà Nội

Bản đồ OpenStreetMap toàn Hà Nội (ranh giới hành chính OSM), nhiệt độ hiện tại của 17 khu vực từ nội thành
đến Ba Vì, Sóc Sơn, Mỹ Đức (khu vực đang/sắp mưa tô xanh). Bấm bất kỳ điểm nào trong Hà Nội để xem thời tiết
hiện tại, 24 giờ tới và 3 ngày tới tại điểm đó. Nguồn: Open-Meteo, cache 10 phút; mất kết nối thì hiện dữ
liệu cũ (tối đa 3 giờ) và ghi rõ.

## Cách hoạt động

1. **Đo** (`traffic_brain/video_analysis.py`): 1 khung hình/giây, YOLOv8n ONNX nhận diện ô tô, xe máy, buýt,
   tải; bám vết để đo chuyển động. CI (0–1) = phần diện tích xe chiếm + số xe + tỉ lệ xe đứng yên.
   CI quan sát là trung bình 60 giây, mỗi 15 giây.
2. **Dự báo** (`traffic_brain/minute_forecaster.py`): tại mỗi thời điểm t, CI sau h phút = CI hiện tại
   cộng tổ hợp ba xu hướng: kéo về mức trung bình của camera, kéo về mức trung bình 10 phút gần nhất và xu
   hướng 3 phút (giảm dần). Trọng số học bằng bình phương tối thiểu (không âm) trên thư viện huấn luyện và
   phần video đã xem xong trước t. Cảnh báo tắc khi xác suất CI ≥ 0,55 từ 40% trở lên (cố định trước khi
   chấm); mức "theo dõi" khi từ 20% đến dưới 40%.
3. **Học** (`models/forecast_library/`): chuỗi CI của 3 video AIC21 và 19 chuỗi camera Hà Nội (không lưu
   hình ảnh). Video đã phân tích trên máy bạn cũng được lưu thành chuỗi số vào `data/forecast_library/`
   để lần sau học thêm. Một video không bao giờ học từ chính nó.

Kiểm tra chéo bỏ-một-video trên 7 video (3 AIC21 + 4 bản ghi Hà Nội dài 80–109 phút, toàn bộ thư mục Drive
"Hanoi Flow" trừ phần giữ lại làm demo), so với "giữ nguyên mức hiện tại": **tốt hơn 18,3% ở +5 phút, 16,0% ở
+10 phút, 11,1% ở +15 phút** (851 / 640 / 486 lần kiểm chứng; khoảng tin cậy 95% đều dương). Riêng camera Hà
Nội: +19,2% / +19,1% / +10,1%. Cảnh báo tắc trước 5 phút: bắt 23/52 lần chuyển sang tắc, không báo nhầm lần
nào; trước 10–15 phút cảnh báo 40% hầu như không bật, mức "theo dõi" bắt 15/39 và 12/28 lần (đúng 31% và
40% số lần bật). Chi tiết: `models/forecast_library/README.md` và trang **Đánh giá mô hình**.

**Giới hạn.** "Thực tế" dùng để chấm là CI thị giác do chính hệ thống đo, không phải nhãn chuyên gia; số
lần kiểm chứng trên Hà Nội còn ít; các lần chuyển sang tắc (52 lần) tập trung ở 5 video; cấu trúc mô hình và
ngưỡng "theo dõi" được chọn sau khi xem chính các kết quả kiểm tra chéo này nên con số có thể hơi lạc quan; mô hình chưa được kiểm định trên luồng camera trực tiếp.

**Kiểm chứng CI bằng nhãn của người** (để không chỉ "tự chấm mình"):

```bash
python tools/label_frames.py sample video1.mp4 video2.mp4 --per-video 60 --out labels/   # tạo labels/label.html
# mở label.html, gán nhãn (phím 1/2/3), bấm "Lưu nhãn" -> labels_<tên>.json
python tools/label_frames.py score labels/frames.json labels_an.json labels_binh.json     # độ khớp, kappa
```

Học thêm từ video mới:

```bash
python tools/build_forecast_library.py video1.mp4 video2.mp4 --source-note "nguồn video"
```

## Đưa lên Render

Repo có `render.yaml` (Blueprint). Render → **New → Blueprint** → chọn repo, hoặc nếu service
`hanoi-traffic-platform` đã có thì Render tự deploy mỗi lần push lên `main`. Gói free chỉ có CPU chia sẻ
và 512 MB RAM: phân tích 30 phút video mất lâu hơn máy cá nhân nhiều lần, nên `render.yaml` giới hạn 500 MiB
/ 40 phút và tắt việc học từ video người khác tải lên. Mở **kết quả đã lưu** (`.json`) thì không cần máy chủ
phân tích, nên demo luôn chạy nhanh trên Render.

## Cấu hình (biến môi trường)

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `MAX_UPLOAD_MIB` | 2048 | Dung lượng video tối đa |
| `MAX_VIDEO_MINUTES` | 90 | Thời lượng video tối đa |
| `FORECAST_LIBRARY_ENABLED` | true | Lưu chuỗi CI của video đã phân tích để học thêm |
| `FORECAST_BUNDLED_LIBRARY` | true | Dùng thư viện học đi kèm |
| `MAP_TILE_URL_TEMPLATE`, `MAP_ATTRIBUTION` | OpenStreetMap | Nguồn ảnh nền bản đồ (HTTPS, có `{z}/{x}/{y}`) |
| `UPLOAD_TOKEN` | (trống) | Nếu đặt: phải nhập mã này mới tải video lên được (dùng cho bản công khai) |
| `UPLOADS_PER_HOUR` | 20 | Số lượt tải video tối đa mỗi giờ cho toàn máy chủ |

## Cấu trúc

```
app/main.py                    API (upload → job → kết quả, bản đồ, thời tiết) và trang web
app/weather.py, hanoi_scope.py thời tiết Open-Meteo, ranh giới Hà Nội
traffic_brain/detector.py      YOLOv8n ONNX + bám vết
traffic_brain/video_analysis.py  CI theo khung hình, tách camera, gọi bộ dự báo
traffic_brain/minute_forecaster.py  dự báo +5/+10/+15 phút, chấm điểm
static/                        giao diện: index.html, shell.js (điều hướng), app.js (dự báo), map.js (bản đồ, thời tiết), routes.js (tuyến)
traffic_brain/routing.py       tuyến thay thế OSRM
models/                        detector.onnx, forecast_library/
tools/build_forecast_library.py  xây thư viện học, kiểm tra chéo, xuất demo
tests/                         pytest + node --test
```

Kiểm thử: `pip install -r requirements-dev.txt && pytest -q && node --test tests/forecast_page.test.cjs`.
Giấy phép bên thứ ba: `THIRD_PARTY_NOTICES.md`.

## Camera trực tiếp và quyền dữ liệu

Bản này chỉ đọc video bạn tải lên; chưa nối vào luồng camera trực tiếp. Việc Hà Nội có hệ thống camera tập
trung (Quyết định 319/QĐ-UBND ngày 17/01/2025) không có nghĩa ứng dụng bên thứ ba được lấy luồng hình
hay API: cần xin tài liệu giao diện, điều kiện sử dụng và quyền truy cập từ chủ nguồn dữ liệu. Muốn dự
báo trực tiếp cần thêm một bộ thu nhận có xác thực (kết nối luồng được cấp phép, chạy bộ nhận diện, gửi CI
theo thời gian) rồi đưa chuỗi CI vào `forecast_series`; bộ thu nhận đó chưa có trong dự án. Mã tải lên
(`UPLOAD_TOKEN`) gửi qua tiêu đề `X-Upload-Token`, được kiểm tra cùng giới hạn dung lượng và lượt tải
trước khi máy chủ nhận video.
