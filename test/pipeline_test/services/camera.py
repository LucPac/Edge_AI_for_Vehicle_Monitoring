# services/camera.py
import cv2
import time
import re
import threading
import pathlib
import numpy as np
from config import ESP32_CAM_URL, CLASS_MAP, COLOR_MAP, YOLO_MODEL_PATH
from ai.detection import YOLODetector
from ai.recognition import recognize_onnx

# Nạp "Bảo bối" Tracker của bạn vào đây
from services.tracking_service import TrafficTracker

# Fix lỗi PosixPath khi chạy model trên Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

CURRENT_FRAME = None   
DISPLAY_FRAME = None   
ai_lock = threading.Lock()

print("[*] Đang load model YOLO (ONNX) qua ai/detection.py...")
detector = YOLODetector(YOLO_MODEL_PATH)

# --- KHỞI TẠO TRACKER VÀ VÙNG DI CHUYỂN ---
# Chia đôi màn hình camera: Nửa trên là Lối Vào (Green), nửa dưới là Lối Ra (Red)
# (Bạn có thể điều chỉnh tọa độ này sau khi đặt camera lên sa bàn thực tế)
GREEN_ZONES = np.array([[0, 0], [1280, 0], [1280, 360], [0, 360]]) 
RED_ZONES = np.array([[0, 360], [1280, 360], [1280, 720], [0, 720]]) 

tracker = TrafficTracker(green_points=GREEN_ZONES, red_points=RED_ZONES)

def extract_plate_text_wrapper(p_box):
    """Hàm bọc để cắt ảnh biển số và đưa cho CRNN (Dùng cho Tracker)"""
    global CURRENT_FRAME
    if CURRENT_FRAME is None: return ""
    
    x1, y1, x2, y2 = map(int, p_box)
    p = 2
    h_orig, w_orig, _ = CURRENT_FRAME.shape
    xmin, ymin = max(0, x1 - p), max(0, y1 - p)
    xmax, ymax = min(w_orig, x2 + p), min(h_orig, y2 + p)
    
    crop_img = CURRENT_FRAME[ymin:ymax, xmin:xmax]
    if crop_img.size == 0: return ""
    
    h, w = crop_img.shape[:2]
    ratio = w / h if h > 0 else 0
    raw_text = ""
    if 0 < ratio < 1.9: # Biển vuông (2 dòng)
        text1 = recognize_onnx(crop_img[:int(h*0.55), :])
        text2 = recognize_onnx(crop_img[int(h*0.45):, :])
        raw_text = text1 + text2
    else: # Biển dài (1 dòng)
        raw_text = recognize_onnx(crop_img)
        
    clean_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    return clean_text

def camera_loop():
    global CURRENT_FRAME, DISPLAY_FRAME
    cap = cv2.VideoCapture(0)    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)       
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    while True:
        ret, frame = cap.read()
        if ret:
            CURRENT_FRAME = frame.copy() 
            display_img = frame.copy()
            
            with ai_lock:
                detections = detector.detect(display_img)
            
            if detections is not None:
                # 1. Tách riêng Detections của Xe (ID 0,1) và Biển số (ID 2)
                # Lưu ý: Sửa lại ID 0, 1, 2 này cho khớp với file config.py của bạn nhé!
                vehicle_detections = detections[(detections.class_id == 0) | (detections.class_id == 1)]
                plate_detections = detections[detections.class_id == 2]

                # 2. Đưa vào Tracker xử lý toàn bộ logic vẽ và bắt chữ
                display_img = tracker.update_and_draw(
                    frame=display_img,
                    vehicle_detections=vehicle_detections,
                    plate_detections=plate_detections,
                    extract_plate_text_func=extract_plate_text_wrapper
                )
            
            DISPLAY_FRAME = display_img
        else:
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(0)

def gen_frames():
    global DISPLAY_FRAME
    while True:
        if DISPLAY_FRAME is not None:
            ret, buffer = cv2.imencode('.jpg', DISPLAY_FRAME)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.01)

def get_current_frame():
    global CURRENT_FRAME
    return CURRENT_FRAME