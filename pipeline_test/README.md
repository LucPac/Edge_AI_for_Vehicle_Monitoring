# Pipeline Test

## Overview
`pipeline_test` là môi trường **end-to-end** để kiểm thử toàn bộ luồng AI trong hệ thống **Edge AI Vehicle Access System**. Pipeline này kết hợp:

- **YOLOv5n ONNX** để phát hiện xe và biển số.
- **ByteTrack** để theo dõi phương tiện theo thời gian thực.
- **CRNN/PaddleOCR ONNX** để nhận dạng ký tự biển số.
- **FastAPI** để stream video, xử lý RFID và phục vụ web dashboard.

Mục tiêu của thư mục này là cung cấp **một bản chạy thử hoàn chỉnh**, dễ triển khai trên edge device và có thể quan sát trực tiếp kết quả qua trình duyệt.

---

## Pipeline Architecture

```
ESP32-CAM ──> FastAPI (main.py)
               ├─ detector.py  (YOLOv5n ONNX)
               ├─ tracker.py   (ByteTrack + OCR voting)
               ├─ plate_cropper.py (crop + align)
               └─ ocr.py       (CRNN/PaddleOCR ONNX)
```

---

## File Structure

| File | Mô tả |
|------|------|
| `main.py` | FastAPI server, xử lý camera, websocket, DB, OCR pipeline |
| `detector.py` | Wrapper YOLOv5n ONNX + class-aware NMS |
| `tracker.py` | ByteTrack + chống nhấp nháy OCR, lọc hướng di chuyển |
| `ocr.py` | OCR ONNX (CTC decode) |
| `plate_cropper.py` | Cắt biển số, căn chỉnh nghiêng |
| `index.html` | Dashboard hiển thị live stream |
| `register.html` | Giao diện đăng ký RFID |
| `script.js` | WebSocket + điều khiển UI |
| `style.css` | Style UI |
| `models/` | Thư mục chứa model ONNX |
| `static/` | Ảnh chụp và ảnh crop |

---

## Quick Start

### 1. Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị model
Đặt các model vào thư mục `pipeline_test/models/`:

- `best.onnx` (YOLOv5n detect: car, motorbike, plate)
- `rec_model.onnx` (OCR CRNN/PaddleOCR)

### 3. Chạy server
```bash
cd pipeline_test
python main.py
```

### 4. Mở dashboard
Truy cập:
```
http://localhost:8000
```

---

## Notes & Best Practices

- **Không bật fliplr** khi train model biển số để tránh đảo chữ.
- Nếu biển số nhỏ, nên tăng `img_size` khi train (768 hoặc 960).
- Trong `detector.py`, threshold mặc định là `0.4` cho confidence/NMS; có thể giảm nếu biển số bị bỏ sót.
- `tracker.py` có cơ chế **OCR voting + lock** để chống nhấp nháy chữ.

---

## Dependencies (gợi ý)

- `fastapi`, `uvicorn`
- `onnxruntime`
- `opencv-python`
- `supervision`
- `psycopg2`

---

## Output

Sau khi chạy, hệ thống sẽ:
- Stream video live với bounding box + ID.
- Hiển thị kết quả OCR trực tiếp.
- Lưu ảnh chụp + crop vào `static/`.
- Ghi log vào PostgreSQL.

---

## License

Tuân thủ license chính của repo: **AGPL-3.0**.
