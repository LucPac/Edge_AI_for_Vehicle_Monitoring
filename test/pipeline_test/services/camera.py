# services/camera.py
import cv2
import time
import threading
import pathlib
import numpy as np
import supervision as sv

from config import YOLO_MODEL_PATH
from ai.detection import YOLODetector

# Fix lỗi PosixPath khi chạy model ONNX trên Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

CURRENT_FRAME_IN, DISPLAY_FRAME_IN = None, None
CURRENT_FRAME_OUT, DISPLAY_FRAME_OUT = None, None
ai_lock = threading.Lock()

print("[*] Đang load model YOLO ...")
detector = YOLODetector(YOLO_MODEL_PATH)

box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

CLASS_NAMES = {0: "Car", 1: "Motorcycle", 2: "Plate"}

def process_camera(cap, is_in_gate, skip_frames=15):
    """
    Luồng xử lý camera độc lập.
    skip_frames: Số lượng khung hình bỏ qua trước khi cho AI quét 1 lần.
    """
    global CURRENT_FRAME_IN, DISPLAY_FRAME_IN, CURRENT_FRAME_OUT, DISPLAY_FRAME_OUT
    frame_count = 0
    last_det = None

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
            
        disp = frame.copy()
        frame_count += 1

        # Kỹ thuật bóp FPS để giảm tải CPU
        if frame_count % skip_frames == 0:
            with ai_lock:
                last_det = detector.detect(disp)

        # Vẽ Bounding Box nếu có dữ liệu
        if last_det is not None:
            disp = box_annotator.annotate(scene=disp, detections=last_det)
            labels = [CLASS_NAMES.get(c, f"ID: {c}") for c in last_det.class_id]
            disp = label_annotator.annotate(scene=disp, detections=last_det, labels=labels)
            
        # Cập nhật ảnh Raw (để chụp Burst) và ảnh Disp (để hiện lên Web)
        if is_in_gate:
            CURRENT_FRAME_IN, DISPLAY_FRAME_IN = frame.copy(), disp
        else:
            CURRENT_FRAME_OUT, DISPLAY_FRAME_OUT = frame.copy(), disp

def camera_loop():
    """Khởi động 2 camera song song với cấu hình FPS riêng biệt"""
    
    # Cam Lối Vào
    cap_in = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap_in.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_in.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap_in.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    
    # Cam Lối Ra 
    cap_out = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap_out.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_out.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap_out.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    # ==========================================
    # ⚙️ CẤU HÌNH TỐC ĐỘ QUÉT AI (CHỐNG LAG CPU)
    # Số càng lớn -> AI quét càng ít -> CPU càng mát (nhưng khung hình bám xe hơi giật)
    # ==========================================
    SKIP_IN = 10  # Quét ~1.5 lần/giây (Tối ưu cho cam FHD Lối Vào)
    SKIP_OUT = 20 # Quét ~3 lần/giây (Tối ưu cho cam VGA Lối Ra)

    # Khởi động 2 luồng độc lập, truyền tham số skip_frames vào
    threading.Thread(target=process_camera, args=(cap_in, True, SKIP_IN), daemon=True).start()
    threading.Thread(target=process_camera, args=(cap_out, False, SKIP_OUT), daemon=True).start()
    
    while True: 
        time.sleep(1)

# Các hàm Getter cho Burst Capture
def get_current_frame_in(): return CURRENT_FRAME_IN
def get_current_frame_out(): return CURRENT_FRAME_OUT

# Các hàm Streaming cho Web
def gen_frames_in():
    global DISPLAY_FRAME_IN
    while True:
        if DISPLAY_FRAME_IN is not None: 
            ret, buffer = cv2.imencode('.jpg', DISPLAY_FRAME_IN)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else: 
            time.sleep(0.01)

def gen_frames_out():
    global DISPLAY_FRAME_OUT
    while True:
        if DISPLAY_FRAME_OUT is not None: 
            ret, buffer = cv2.imencode('.jpg', DISPLAY_FRAME_OUT)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else: 
            time.sleep(0.01)