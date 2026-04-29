from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import psycopg2
from datetime import datetime
import json
import cv2
import numpy as np
import os
import time
import pathlib
import glob
import re
import onnxruntime as ort
import threading

# Import module YOLO siêu nhẹ của bạn
from detector import YOLODetector

app = FastAPI()
ai_lock = threading.Lock() # Khóa an toàn chống đụng độ AI

# --- 1. CẤU HÌNH BẢO MẬT CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. CẤU HÌNH THƯ MỤC TĨNH ---
os.makedirs("static/images", exist_ok=True)
os.makedirs("static/crops", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# =========================================================
# THIẾT LẬP MÀU SẮC VÀ TÊN CLASS CHO TRACKING LIVE
# =========================================================
CLASS_MAP = {
    0: "Car",
    1: "Motorbike",
    2: "Plate"
}

COLOR_MAP = {
    0: (255, 0, 255), # Car: Màu Tím
    1: (255, 255, 0), # Motorbike: Màu Xanh Lơ
    2: (0, 255, 0)    # Plate: Màu Xanh Lá
}

# --- 3. LOAD AI MODELS ---
print("[*] Đang load model YOLO (ONNX) qua detector.py...")
detector = YOLODetector("models/best.onnx")

print("[*] Đang load model CRNN ONNX...")
ort_session = ort.InferenceSession('models/rec_model.onnx', providers=['CPUExecutionProvider'])
input_name = ort_session.get_inputs()[0].name

chars_list = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 
    'M', 'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z' ,'R'
]
id2char = {0: ''} 
for idx, char in enumerate(chars_list):
    id2char[idx + 1] = char

def decode_onnx_predictions(preds_numpy):
    preds_seq = preds_numpy[0] 
    preds_index = np.argmax(preds_seq, axis=1)
    char_list = []
    for i in range(len(preds_index)):
        if preds_index[i] != 0 and (not (i > 0 and preds_index[i - 1] == preds_index[i])):
            if preds_index[i] in id2char:
                char_list.append(id2char[preds_index[i]])
    return ''.join(char_list)

# ĐƯA HÀM OCR RA NGOÀI ĐỂ DÙNG CHUNG (Cả Live và lúc chụp)
def recognize_onnx(img_bgr):
    try:
        img_resized = cv2.resize(img_bgr, (320, 48))
        img_float = img_resized.astype(np.float32) / 255.0
        img_float = (img_float - 0.5) / 0.5
        img_tensor = np.transpose(img_float, (2, 0, 1))
        img_tensor = np.expand_dims(img_tensor, axis=0)
        ort_inputs = {input_name: img_tensor}
        preds_numpy = ort_session.run(None, ort_inputs)[0]
        return decode_onnx_predictions(preds_numpy)
    except Exception as e:
        return ""

print("[+] Load tất cả Model thành công!")

# --- 4. CẤU HÌNH DATABASE ---
DB_CONFIG = {
    "dbname": "P_Dashboard",
    "user": "postgres",
    "password": "241120", 
    "host": "localhost",
    "port": "5432"
}

# =========================================================
# LUỒNG BACKGROUND CAMERA: TỐI ƯU HÓA CHỐNG LAG (FRAME SKIPPING)
# =========================================================
CURRENT_FRAME = None   
DISPLAY_FRAME = None   
ESP32_CAM_URL = "http://192.168.1.254/" 

def camera_loop():
    global CURRENT_FRAME, DISPLAY_FRAME
    cap = cv2.VideoCapture(ESP32_CAM_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_count = 0
    PROCESS_EVERY_N_FRAMES = 5  # Cứ 5 frame mới gọi AI 1 lần để chống Lag
    
    # Bộ nhớ đệm giữ khung và nhãn
    cached_boxes = []
    cached_labels = []
    cached_colors = []
    
    while True:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            CURRENT_FRAME = frame.copy() 
            display_img = frame.copy()
            
            # CHỈ CHẠY AI VÀ OCR MỖI N FRAME
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                with ai_lock:
                    detections = detector.detect(display_img)
                
                new_boxes = []
                new_labels = []
                new_colors = []
                
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
                            
                            # NẾU LÀ BIỂN SỐ -> CẮT VÀ ĐỌC CHỮ NGAY LẬP TỨC
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
                    except Exception as e: 
                        pass
                
                cached_boxes = new_boxes
                cached_labels = new_labels
                cached_colors = new_colors

            # VẼ KHUNG TỪ BỘ NHỚ ĐỆM LÊN FRAME CHẠY LIVE
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
                if label_bg_y1 < 0: 
                    label_bg_y1, label_bg_y2 = 0, text_height + 10
                
                cv2.rectangle(display_img, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, cv2.FILLED)
                cv2.putText(display_img, label_text, (label_bg_x1 + 5, label_bg_y2 - 5), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
            
            DISPLAY_FRAME = display_img
        else:
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(ESP32_CAM_URL)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

@app.on_event("startup")
def startup_event():
    print("[*] Khởi động luồng Camera ngầm...")
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

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

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

# --- 6. QUẢN LÝ WEBSOCKET ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()
class RFIDData(BaseModel):
    rfid_code: str 

class RegisterData(BaseModel):
    rfid_code: str
    owner_name: str
    plate_number: str
    phone: str

# THÊM MỚI: API Xử lý Lưu thẻ vào Database
@app.post("/api/register")
async def register_card(data: RegisterData):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    try:
        # 1. Kiểm tra xem mã thẻ này đã được đăng ký cho ai chưa
        cur.execute("SELECT id FROM registered_cards WHERE rfid_code = %s", (data.rfid_code,))
        if cur.fetchone():
            return {"status": "error", "message": "Mã thẻ RFID này đã tồn tại trong hệ thống!"}

        # 2. Xóa các ký tự đặc biệt/khoảng trắng của biển số cho chuẩn format
        clean_plate = re.sub(r'[^A-Za-z0-9]', '', data.plate_number).upper()

        # 3. Lưu vào bảng registered_cards
        cur.execute("""
            INSERT INTO registered_cards (rfid_code, owner_name, plate_number, phone) 
            VALUES (%s, %s, %s, %s)
        """, (data.rfid_code, data.owner_name, clean_plate, data.phone))
        
        conn.commit()
        return {"status": "success", "message": "Đăng ký thẻ thành công!"}
    except Exception as e:
        conn.rollback()
        return {"status": "error", "message": f"Lỗi cơ sở dữ liệu: {e}"}
    finally:
        cur.close()
        conn.close()

# --- 7. HÀM XỬ LÝ ẢNH CHỤP KHI QUẸT THẺ (SỬ DỤNG ẢNH GỐC SẠCH) ---
def align_plate_image(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=int(img.shape[1]*0.4), maxLineGap=20)
        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if -30 < angle < 30 and abs(angle) > 1.0:
                    angles.append(angle)
        if len(angles) > 0:
            median_angle = np.median(angles)
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
        return img 
    except Exception:
        return img 

def process_vehicle_image(rfid_code, action_type):
    base_img_path = f"static/images/{rfid_code}.jpg"
    plate_text = f"LOI-OCR-{rfid_code[-4:]}" 
    
    if not os.path.exists(base_img_path):
        return "https://placehold.co/600x300/1e293b/475569?text=No+Image", "https://placehold.co/200x80/1a1a1a/475569?text=No+Crop", plate_text

    img = cv2.imread(base_img_path)
    crop_url = "https://placehold.co/200x80/1a1a1a/475569?text=No+Plate+Found"

    with ai_lock:
        detections = detector.detect(img)
    
    if detections is not None:
        plate_detections = detections[detections.class_id == 2]

        if len(plate_detections) > 0:
            p_box = plate_detections.xyxy[0]
            px1, py1, px2, py2 = map(int, p_box)

            p = 2
            h_orig, w_orig, _ = img.shape
            xmin = max(0, px1 - p)
            ymin = max(0, py1 - p)
            xmax = min(w_orig, px2 + p)
            ymax = min(h_orig, py2 + p)
            
            cropped_img = img[ymin:ymax, xmin:xmax]
            cropped_img = align_plate_image(cropped_img)
            
            timestamp = int(time.time())
            full_crop_filename = f"static/crops/{rfid_code}_{action_type}_full_{timestamp}.jpg"
            cv2.imwrite(full_crop_filename, cropped_img)
            crop_url = f"http://localhost:8000/{full_crop_filename}"

            h, w, _ = cropped_img.shape
            ratio = w / h if h > 0 else 0
            
            if 0 < ratio < 1.9:
                text1 = recognize_onnx(cropped_img[:int(h*0.55), :])
                text2 = recognize_onnx(cropped_img[int(h*0.45):, :])
                raw_text = text1 + text2
            else:
                raw_text = recognize_onnx(cropped_img)
                
            clean_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
            if clean_text: plate_text = clean_text
    
    timestamp_now = int(time.time())
    full_img_url = f"http://localhost:8000/{base_img_path}?t={timestamp_now}"
    return full_img_url, crop_url, plate_text

# --- 8. API NHẬN THẺ TỪ ESP32 VÀ CHECK DB ---
@app.post("/api/swipe")
async def handle_rfid_swipe(data: RFIDData):
    global CURRENT_FRAME
    rfid = data.rfid_code
    
    if CURRENT_FRAME is not None:
        cv2.imwrite(f"static/images/{rfid}.jpg", CURRENT_FRAME)
    else:
        blank_img = np.zeros((720, 1280, 3), np.uint8)
        cv2.imwrite(f"static/images/{rfid}.jpg", blank_img)

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    customer_type = "Khách Vãng Lai" 
    is_registered = False
    reg_plate = ""
    warning_msg = None
    
    try:
        cur.execute("SELECT owner_name, phone, plate_number FROM registered_cards WHERE rfid_code = %s", (rfid,))
        reg_info = cur.fetchone()
        if reg_info:
            is_registered = True
            reg_name, reg_phone, reg_plate = reg_info
            
            # Xuống dòng bằng <br>, thu nhỏ chữ và làm mờ một chút cho đẹp
            details = f"{reg_name} - {reg_phone}"
            customer_type = f"Khách Đăng Ký<br><span style='font-size: 0.9em; font-weight: normal; opacity: 0.8;'>{details}</span>"
    except Exception as e:
        conn.rollback() 
        print(f"Lỗi check thẻ đăng ký: {e}")
    
    cur.execute("SELECT id, plate_in, image_in_url, time_in FROM parking_logs WHERE rfid_code = %s AND status IN ('PARKING', 'ERROR_IN')", (rfid,))
    record = cur.fetchone()
    response_data = {}

    if record:
        # ==========================
        # LOGIC LÚC XE RA
        # ==========================
        log_id, plate_in, image_in_url, time_in = record
        time_out = datetime.now()
        duration = time_out - time_in
        duration_str = f"{int(duration.total_seconds()//3600):02d}:{int((duration.total_seconds()%3600)//60):02d}:{int(duration.total_seconds()%60):02d}"
        
        full_img_url, crop_img_url, plate_out = process_vehicle_image(rfid, "out")
        
        status_db = "COMPLETED" 
        
        # 🌟 LOGIC BẮT LỖI MỚI (Cho cả Vé tháng và Vãng lai)
        clean_out = re.sub(r'[^A-Z0-9]', '', plate_out.upper())
        clean_in = re.sub(r'[^A-Z0-9]', '', plate_in.upper())
        
        if is_registered:
            clean_reg = re.sub(r'[^A-Z0-9]', '', reg_plate.upper()) if reg_plate else ""
            if clean_out != clean_reg:
                warning_msg = "THẺ VÀ BIỂN SỐ KHÔNG KHỚP (SAI ĐĂNG KÝ)!"
                status_db = "ERROR_OUT"
            elif clean_out != clean_in:
                warning_msg = "BIỂN SỐ VÀO VÀ RA KHÔNG KHỚP NHAU!"
                status_db = "ERROR_OUT"
        else:
            # Dành cho Khách Vãng Lai: Ép buộc Biển Ra phải giống Biển Vào
            if clean_out != clean_in:
                warning_msg = "BIỂN SỐ VÀO VÀ RA KHÔNG KHỚP NHAU!"
                status_db = "ERROR_OUT"

        cur.execute("UPDATE parking_logs SET plate_out = %s, image_out_url = %s, time_out = %s, status = %s WHERE id = %s", 
                    (plate_out, full_img_url, time_out, status_db, log_id))
        
        crop_in_url = "https://placehold.co/200x80/1a1a1a/475569?text=No+Crop"
        try:
            list_of_files = glob.glob(os.path.join("static", "crops", f"{rfid}_in_full_*.jpg"))
            if list_of_files:
                crop_in_url = f"http://localhost:8000/{max(list_of_files, key=os.path.getctime).replace(os.sep, '/')}"
        except: pass

        response_data = {
            "action": "OUT", "rfid": rfid, "plate_in": plate_in, "plate_out": plate_out,
            "img_in": image_in_url, "img_out": full_img_url, "img_crop_in": crop_in_url, "img_crop_out": crop_img_url,
            "time_in": time_in.strftime("%H:%M:%S"), "time_out": time_out.strftime("%H:%M:%S"), "duration": duration_str,
            "customer_type": customer_type,
            "warning": warning_msg
        }
    else:
        # ==========================
        # LOGIC LÚC XE VÀO
        # ==========================
        full_img_url, crop_img_url, plate_in = process_vehicle_image(rfid, "in")
        status_db = "PARKING"
        
        # Kiểm tra khớp biển số đăng ký (lúc vào)
        if is_registered:
            clean_in = re.sub(r'[^A-Z0-9]', '', plate_in.upper())
            clean_reg = re.sub(r'[^A-Z0-9]', '', reg_plate.upper()) if reg_plate else ""
            if clean_in != clean_reg:
                warning_msg = "THẺ VÀ BIỂN SỐ KHÔNG KHỚP (SAI ĐĂNG KÝ)!"
                status_db = "ERROR_IN" 

        cur.execute("INSERT INTO parking_logs (rfid_code, plate_in, image_in_url, status) VALUES (%s, %s, %s, %s) RETURNING time_in", 
                    (rfid, plate_in, full_img_url, status_db))
        time_in = cur.fetchone()[0]
        
        response_data = {
            "action": "IN", "rfid": rfid, "plate_in": plate_in,
            "img_in": full_img_url, "img_crop_in": crop_img_url, "time_in": time_in.strftime("%H:%M:%S"),
            "customer_type": customer_type,
            "warning": warning_msg
        }
        
    conn.commit()
    cur.close()
    conn.close()
    
    await manager.broadcast(json.dumps(response_data))
    return {"status": "success"}

# --- 9. API DỮ LIỆU BẢNG (TÍNH PHÍ CHO CẢ TRƯỜNG HỢP LỖI LÚC RA) ---
@app.get("/api/logs")
async def get_parking_logs():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    query = """
        SELECT p.id, p.rfid_code, p.plate_in, p.time_in, p.time_out, p.status, p.plate_out, r.owner_name, r.phone
        FROM parking_logs p
        LEFT JOIN registered_cards r ON p.rfid_code = r.rfid_code
        ORDER BY p.time_in DESC LIMIT 30
    """
    cur.execute(query)
    rows = cur.fetchall()
    
    logs = []
    for r in rows:
        is_registered = (r[7] is not None)
        
        # Cập nhật hiển thị cho trang Dữ Liệu
        if is_registered:
            details = f"{r[7]} - {r[8]}"
            customer_type = f"Khách Đăng Ký<br><span style='font-size: 0.9em; font-weight: normal; color: #94a3b8;'>{details}</span>"
        else:
            customer_type = "Khách Vãng Lai"
        
        fee = "-"
        # Tính phí cho xe đã ra khỏi bãi (Dù Hoàn thành hay Ra bị lỗi)
        if r[5] in ["COMPLETED", "ERROR_OUT"]:
            fee = "0 đ" if is_registered else "5,000 đ"
            
        logs.append({
            "id": r[0],
            "ticket": r[1],
            "plate": r[2], 
            "time_in": r[3].strftime("%d/%m/%Y - %H:%M:%S") if r[3] else "--",
            "time_out": r[4].strftime("%d/%m/%Y - %H:%M:%S") if r[4] else "--",
            "status": r[5],
            "customer_type": customer_type,
            "fee": fee
        })
        
    cur.close()
    conn.close()
    return logs

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)