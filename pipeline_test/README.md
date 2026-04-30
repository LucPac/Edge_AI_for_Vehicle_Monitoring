# Pipeline Test

## Overview
`pipeline_test` is an **end-to-end validation environment** for the full AI workflow in the **Edge AI Vehicle Access System**. It integrates:

- **YOLOv5n ONNX** for vehicle and license plate detection.
- **ByteTrack** for real-time tracking.
- **CRNN/PaddleOCR ONNX** for license plate recognition.
- **FastAPI** for video streaming, RFID handling, and web dashboard services.

The goal of this folder is to provide a **complete, runnable pipeline** that is easy to deploy on edge devices while allowing live monitoring through a browser.

---

## Pipeline Architecture

```
ESP32-CAM ──> FastAPI (main.py)
               ├─ detector.py      (YOLOv5n ONNX)
               ├─ tracker.py       (ByteTrack + OCR voting)
               ├─ plate_cropper.py (crop + alignment)
               └─ ocr.py           (CRNN/PaddleOCR ONNX)
```

---

## File Structure

| File | Description |
|------|-------------|
| `main.py` | FastAPI server, camera loop, websocket, DB, OCR pipeline |
| `detector.py` | YOLOv5n ONNX wrapper with class-aware NMS |
| `tracker.py` | ByteTrack + OCR de-flicker voting + direction filtering |
| `ocr.py` | OCR ONNX (CTC decode) |
| `plate_cropper.py` | Plate crop + skew alignment |
| `index.html` | Live dashboard UI |
| `register.html` | RFID registration UI |
| `script.js` | WebSocket + UI logic |
| `style.css` | UI styling |
| `models/` | ONNX model directory |
| `static/` | Snapshots and cropped plate images |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare models
Place the following files in `pipeline_test/models/`:

- `best.onnx` (YOLOv5n detector: car, motorbike, plate)
- `rec_model.onnx` (OCR CRNN/PaddleOCR)

### 3. Run the server
```bash
cd pipeline_test
python main.py
```

### 4. Open dashboard
Visit:
```
http://localhost:8000
```

---

## Notes & Best Practices

- **Disable fliplr** when training plate detectors to avoid inverted characters.
- If plates are small, increase training `img_size` (768 or 960).
- In `detector.py`, the default confidence/NMS threshold is `0.4`; lower it if plates are missed.
- `tracker.py` uses **OCR voting + lock** to prevent text flicker.

---

## Dependencies (suggested)

- `fastapi`, `uvicorn`
- `onnxruntime`
- `opencv-python`
- `supervision`
- `psycopg2`

---

## Output

When running, the system will:
- Stream live video with bounding boxes + IDs.
- Display OCR results in real time.
- Save snapshots and crops to `static/`.
- Write logs into PostgreSQL.

---

## License

Follows the repository license: **AGPL-3.0**.
