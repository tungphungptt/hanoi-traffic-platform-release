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
3. **Ảnh minh chứng**: sau khi phân tích, hệ thống chụp lại khoảng 8 khung hình mỗi camera tại các thời điểm
   đổi mức, đông nhất, thông thoáng nhất và các mốc định kỳ, vẽ khung quanh xe nhận diện kèm giờ và CI. Ảnh giữ
   khoảng 1 giờ trên máy chủ và không nằm trong file kết quả tải về. Bộ nhận diện YOLOv8n còn bỏ sót nhiều xe máy
   khi đường rất đông (ảnh minh chứng cho thấy rõ điều này), nên CI có thể thấp hơn thực tế.
4. **Học** (`models/forecast_library/`): chuỗi CI của 3 video AIC21 và 19 chuỗi camera Hà Nội (không lưu
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
| `LIVE_TOKEN` | (trống) | Nếu đặt: bật nhận số liệu trực tiếp từ bộ thu nhận (`tools/live_collector.py`); để trống thì tắt |
| `LIVE_DB` | (trống) | Đường dẫn file SQLite lưu bền số liệu trực tiếp kèm thời tiết (xem mục "Lưu số liệu có dấu thời gian"); để trống thì chỉ giữ trong bộ nhớ |
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

## Đo tốc độ xe (km/h) từ video

Chỉ số tốc độ trong CI là tương đối (pixel); để có km/h phải hiệu chuẩn từng camera bằng 4 điểm trên mặt đường
mà bạn biết khoảng cách thật (góc một hình chữ nhật giữa hai vạch kẻ và hai đường ngang, đo tại chỗ hoặc bằng
thước của bản đồ vệ tinh). Công cụ đưa mỗi xe về tọa độ mét trên mặt đường, bám theo bằng bộ theo dõi vận tốc
không đổi và lấy tốc độ là độ dốc của đường thẳng khớp với vị trí theo thời gian.

```bash
# cam.json: {"image_points": [[x,y] x4], "world_points_m": [[0,0],[7,0],[7,40],[0,40]]}
python tools/measure_speed.py clip.mp4 --calib cam.json --preview check.jpg      # xem lưới 5 m phủ lên khung hình
python tools/measure_speed.py clip.mp4 --calib cam.json --fps 4 --out speeds.json # trung vị/trung bình/p85, tỉ lệ xe đứng yên theo cửa sổ 10 giây
```

- Dùng `--model yolox_s.onnx` (xem `models/README.md`) để thấy nhiều xe hơn YOLOv8n.
- Chỉ đo xe nằm trong tứ giác hiệu chuẩn; trung vị trên 120 km/h được cảnh báo là calibration sai.
- Kiểm thử dùng mặt đường tổng hợp có phối cảnh: 36 km/h đọc lại đúng trong 3%, xe đỗ đọc dưới 4 km/h, xe xa và gần cùng tốc độ cho cùng kết quả.
  **Chưa kiểm định trên video thật với tốc độ đã biết** (cần một đoạn có xe chạy đo bằng GPS hoặc đồng hồ). Với video phát sóng
  có chuyển cảnh, chỉ cắt đoạn một camera rồi mới đo, vì mỗi camera một calibration.
- Chưa nối vào trang web hay vào CI/dự báo: kết quả là file JSON.

## Dự báo trực tiếp từ camera (trang "Trực tiếp")

Máy chủ không tự mở camera. Một bộ thu nhận chạy ở nơi có quyền đọc luồng hình, đo từng khung bằng đúng bộ đo
của phần video (`FrameMeter`) và gửi chỉ số CI mỗi giây về máy chủ; máy chủ chạy bộ dự báo +5/+10/+15 phút
trên chuỗi đó và trang **Trực tiếp** hiện kết quả, tự làm mới mỗi 15 giây.

```bash
# trên máy chủ: đặt LIVE_TOKEN (chuỗi bí mật); trên Render khai báo trong dashboard
python tools/live_collector.py --source rtsp://... --camera-id tdh-216 --name "216 Trần Duy Hưng" \
    --lat 21.0075 --lng 105.7993 --server https://hanoi-traffic-platform.onrender.com --token MA_BI_MAT
```

- `--source` có thể là URL RTSP/HLS, số thứ tự webcam hoặc một file video (phát đúng tốc độ gốc, dùng để thử).
- Cần khoảng 9 phút số liệu liên tục mới có dự báo +5 phút. Nếu mất tín hiệu quá 2 phút, thẻ camera báo "Mất tín hiệu".
- Số liệu giữ trong bộ nhớ (tối đa 3 giờ và 64 camera); nếu đặt `LIVE_DB` thì còn được ghi xuống file và nạp lại sau khi khởi động lại. Mẫu cũ hơn 15 phút hoặc ở tương lai bị từ chối.
- API: `POST /api/live/<id>/samples`, `DELETE /api/live/<id>` và `GET /api/live/export` cần tiêu đề `X-Live-Token`; `GET /api/live`, `GET /api/live/<id>`, `GET /api/live/history` công khai. Mã camera `export` và `history` bị giữ chỗ.
- Chưa kiểm định trên luồng camera thật: các con số +18% ở trên chỉ đo trên video đã ghi.

### Lưu số liệu có dấu thời gian (để học giờ cao điểm và mưa)

Với `LIVE_DB=/đường/dẫn/live.sqlite`, mỗi mẫu đo được ghi kèm giờ thật, và cứ 10 phút một ảnh chụp thời tiết
(Open-Meteo, nhiệt độ, độ ẩm, lượng mưa, gió, mã thời tiết) tại tọa độ camera. `GET /api/live/export?camera_id=...&since=<epoch>`
trả về CSV, mỗi dòng ghép với ảnh chụp thời tiết gần nhất **trước** thời điểm đo (không nhìn trước tương lai); có cột giờ Hà Nội.
Mỗi lần tối đa 200.000 dòng, dùng `since` để lấy tiếp. **Gói Render miễn phí có ổ đĩa tạm: file mất khi khởi động lại**,
nên hoặc gắn ổ đĩa bền, hoặc tải CSV về định kỳ.

Chưa có camera thật thì mô phỏng bằng video ghi sẵn, phát nhanh hơn thời gian thật:

```bash
python tools/replay_video.py part000.mp4 part001.mp4 --camera-id demo-tdh --db live.sqlite \
    --start "2026-03-02 07:00" --lat 21.0075 --lng 105.7993 --forecast
```

Video ghi sẵn không có ngày thật nên bản phát lại **không** có thời tiết; nó chỉ kiểm tra đường ống lưu trữ và dự báo.

Khi đã có ít nhất 14 ngày số liệu thật, kiểm tra xem giờ-trong-tuần và mưa có cải thiện dự báo không:
`python tools/context_lab.py live_export.csv` (bỏ qua từng ngày, chỉ nhận nếu khoảng tin cậy 95% của mức cải thiện nằm trên 0).

**Quyền dữ liệu.** Việc Hà Nội có hệ thống camera tập trung (Quyết định 319/QĐ-UBND ngày 17/01/2025) không có
nghĩa ứng dụng bên thứ ba được lấy luồng hình hay API: cần xin tài liệu giao diện, điều kiện sử dụng và quyền
truy cập từ chủ nguồn dữ liệu. Mã tải lên (`UPLOAD_TOKEN`) gửi qua tiêu đề `X-Upload-Token`; cả hai mã được
kiểm tra cùng giới hạn dung lượng và lượt tải trước khi máy chủ nhận nội dung.
