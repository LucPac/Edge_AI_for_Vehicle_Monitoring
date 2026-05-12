# services/camera.py
import cv2
import time
import threading
import pathlib
import numpy as np
import re
import supervision as sv
from collections import Counter

from config import YOLO_MODEL_PATH
from ai.detection import YOLODetector
from ai.recognition import recognize_onnx

# Fix lỗi PosixPath trên Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

CURRENT_FRAME_IN, DISPLAY_FRAME_IN = None, None
CURRENT_FRAME_OUT, DISPLAY_FRAME_OUT = None, None
ai_lock = threading.Lock()

print("[*] Đang load model YOLO (ONNX)...")
detector = YOLODetector(YOLO_MODEL_PATH)

box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

def extract_plate_text(frame, p_box):
    x1, y1, x2, y2 = map(int, p_box)
    h_orig, w_orig, _ = frame.shape
    p = 2
    crop_img = frame[max(0, y1-p):min(h_orig, y2+p), max(0, x1-p):min(w_orig, x2+p)]
    if crop_img.size == 0: return ""
    h, w = crop_img.shape[:2]
    ratio = w / h if h > 0 else 0
    if 0 < ratio < 1.9:
        raw_text = recognize_onnx(crop_img[:int(h*0.55), :]) + recognize_onnx(crop_img[int(h*0.45):, :])
    else:
        raw_text = recognize_onnx(crop_img)
    return re.sub(r'[^A-Z0-9]', '', raw_text.upper())

def process_camera(cap, is_in_gate):
    global CURRENT_FRAME_IN, DISPLAY_FRAME_IN, CURRENT_FRAME_OUT, DISPLAY_FRAME_OUT
    gate_name = "VÀO" if is_in_gate else "RA"
    frame_count = 0
    last_det = None
    plate_text_cache = {} 
    plate_history = [] 

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
            
        disp = frame.copy()
        frame_count += 1

        if frame_count % 3 == 0:
            with ai_lock:
                last_det = detector.detect(disp)
            
            if last_det is not None:
                has_plate = False
                for i, class_id in enumerate(last_det.class_id):
                    if class_id == 2: # Class 2 là Biển số
                        has_plate = True
                        text = extract_plate_text(frame, last_det.xyxy[i])
                        
                        if text:
                            plate_history.append(text)
                            if len(plate_history) > 10: plate_history.pop(0)

                            # BẦU CHỌN (VOTING)
                            most_common_text, count = Counter(plate_history).most_common(1)[0]

                            # CHỈ HIỂN THỊ KHI ĐÃ CHỐT (>= 6 phiếu)
                            if count >= 6:
                                if i not in plate_text_cache or plate_text_cache[i] != most_common_text:
                                    print(f"✅ [CAM {gate_name}] ĐÃ CHỐT BIỂN SỐ: {most_common_text}")
                                plate_text_cache[i] = most_common_text
                            else:
                                # Nếu chưa đủ phiếu, xóa khỏi cache để không in lên màn hình
                                if i in plate_text_cache: del plate_text_cache[i]
                
                # Nếu không thấy biển số nào, clear cache
                if not has_plate:
                    plate_history.clear()
                    plate_text_cache.clear()

        if last_det is not None:
            disp = box_annotator.annotate(scene=disp, detections=last_det)
            labels = []
            
            # GÁN NHÃN CHUẨN XÁC THEO ID
            for i, class_id in enumerate(last_det.class_id):
                if class_id == 2: # Biển số
                    if i in plate_text_cache:
                        labels.append(f"Plate: {plate_text_cache[i]}")
                    else:
                        labels.append("Plate")
                elif class_id == 0: # Ô tô
                    labels.append("Car")
                elif class_id == 1: # Xe máy
                    labels.append("Motorcycle")
                else: # Đề phòng ID lạ
                    labels.append(f"ID: {class_id}")
                    
            disp = label_annotator.annotate(scene=disp, detections=last_det, labels=labels)
            
        if is_in_gate:
            CURRENT_FRAME_IN, DISPLAY_FRAME_IN = frame.copy(), disp
        else:
            CURRENT_FRAME_OUT, DISPLAY_FRAME_OUT = frame.copy(), disp

def camera_loop():
    cap_in = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap_in.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap_in.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); cap_in.set(cv2.CAP_PROP_BUFFERSIZE, 10)
    
    cap_out = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap_out.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap_out.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); cap_out.set(cv2.CAP_PROP_BUFFERSIZE, 5)

    threading.Thread(target=process_camera, args=(cap_in, True), daemon=True).start()
    threading.Thread(target=process_camera, args=(cap_out, False), daemon=True).start()
    while True: time.sleep(1)

def get_current_frame_in(): return CURRENT_FRAME_IN
def get_current_frame_out(): return CURRENT_FRAME_OUT
def gen_frames_in():
    global DISPLAY_FRAME_IN
    while True:
        if DISPLAY_FRAME_IN is not None: yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', DISPLAY_FRAME_IN)[1].tobytes() + b'\r\n')
        else: time.sleep(0.01)
def gen_frames_out():
    global DISPLAY_FRAME_OUT
    while True:
        if DISPLAY_FRAME_OUT is not None: yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', DISPLAY_FRAME_OUT)[1].tobytes() + b'\r\n')
        else: time.sleep(0.01)