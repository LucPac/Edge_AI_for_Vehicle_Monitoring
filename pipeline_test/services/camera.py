# services/camera.py
import cv2
import time
import re
import threading
import pathlib
from config import ESP32_CAM_URL, CLASS_MAP, COLOR_MAP, YOLO_MODEL_PATH
from ai.detection import YOLODetector
from ai.recognition import recognize_onnx

# Fix lỗi PosixPath khi chạy model trên Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

CURRENT_FRAME = None   
DISPLAY_FRAME = None   

# Khóa luồng an toàn chống đụng độ AI
ai_lock = threading.Lock()

print("[*] Đang load model YOLO (ONNX) qua ai/detection.py...")
detector = YOLODetector(YOLO_MODEL_PATH)

def camera_loop():
    global CURRENT_FRAME, DISPLAY_FRAME
    cap = cv2.VideoCapture(0)    
    
    # (Tùy chọn) Ép độ phân giải HD để AI dễ bắt nét và chạy mượt trên máy tính
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)       
                               
    cap = cv2.VideoCapture(ESP32_CAM_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_count = 0
    PROCESS_EVERY_N_FRAMES = 5
    
    cached_boxes, cached_labels, cached_colors = [], [], []
    
    while True:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            CURRENT_FRAME = frame.copy() 
            display_img = frame.copy()
            
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                with ai_lock:
                    detections = detector.detect(display_img)
                
                new_boxes, new_labels, new_colors = [], [], []
                
                if detections is not None:
                    try:
                        boxes = detections.xyxy
                        class_ids = detections.class_id

                        for i in range(len(boxes)):
                            x1, y1, x2, y2 = map(int, boxes[i])
                            cls_id = int(class_ids[i])
                            
                            class_name = CLASS_MAP.get(cls_id, f"Obj {cls_id}")
                            color = COLOR_MAP.get(cls_id, (255, 255, 255))
                            label_text = f"{class_name} ID: {cls_id}"
                            
                            if cls_id == 2:
                                p = 2
                                h_orig, w_orig, _ = CURRENT_FRAME.shape
                                xmin, ymin = max(0, x1 - p), max(0, y1 - p)
                                xmax, ymax = min(w_orig, x2 + p), min(h_orig, y2 + p)
                                
                                crop_img = CURRENT_FRAME[ymin:ymax, xmin:xmax]
                                if crop_img.size > 0:
                                    h, w = crop_img.shape[:2]
                                    ratio = w / h if h > 0 else 0
                                    raw_text = ""
                                    if 0 < ratio < 1.9: 
                                        text1 = recognize_onnx(crop_img[:int(h*0.55), :])
                                        text2 = recognize_onnx(crop_img[int(h*0.45):, :])
                                        raw_text = text1 + text2
                                    else: 
                                        raw_text = recognize_onnx(crop_img)
                                        
                                    clean_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
                                    if clean_text:
                                        label_text = clean_text 

                            new_boxes.append((x1, y1, x2, y2))
                            new_labels.append(label_text)
                            new_colors.append(color)
                    except Exception: 
                        pass
                
                cached_boxes = new_boxes
                cached_labels = new_labels
                cached_colors = new_colors

            for i in range(len(cached_boxes)):
                x1, y1, x2, y2 = cached_boxes[i]
                label_text = cached_labels[i]
                color = cached_colors[i]
                
                cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2 if color == (0, 255, 0) else 1
                
                (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
                label_bg_x1, label_bg_y1 = x1, y1 - text_height - 10 
                label_bg_x2, label_bg_y2 = x1 + text_width + 10, y1
                if label_bg_y1 < 0: label_bg_y1, label_bg_y2 = 0, text_height + 10
                
                cv2.rectangle(display_img, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, cv2.FILLED)
                cv2.putText(display_img, label_text, (label_bg_x1 + 5, label_bg_y2 - 5), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
            
            DISPLAY_FRAME = display_img
        else:
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(ESP32_CAM_URL)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
    """Hàm hỗ trợ lấy frame hiện tại để API xử lý chụp ảnh lúc quẹt thẻ"""
    global CURRENT_FRAME
    return CURRENT_FRAME