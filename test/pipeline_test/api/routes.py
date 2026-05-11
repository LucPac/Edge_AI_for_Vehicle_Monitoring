# api/routes.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
import os
import json
import glob
import re
from datetime import datetime

from database import get_db_connection
from models import RFIDData
from services.websocket import manager
from services.vehicle import process_vehicle_image
from services.camera import get_current_frame

router = APIRouter()

# Hàm phụ trợ chuyển đổi chuỗi thời gian của SQLite thành Datetime
def parse_sqlite_time(time_val):
    if not time_val: return None
    if isinstance(time_val, str):
        try:
            return datetime.strptime(time_val, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            return datetime.strptime(time_val, "%Y-%m-%d %H:%M:%S")
    return time_val

@router.post("/api/swipe")
async def handle_rfid_swipe(data: RFIDData):
    rfid = data.rfid_code
    current_frame = get_current_frame()
    
    if current_frame is not None:
        cv2.imwrite(f"static/images/{rfid}.jpg", current_frame)
    else:
        blank_img = np.zeros((720, 1280, 3), np.uint8)
        cv2.imwrite(f"static/images/{rfid}.jpg", blank_img)

    conn = get_db_connection()
    cur = conn.cursor()
    
    customer_type = "Khách Vãng Lai"
    warning_msg = None
    
    try:
        # THAY %s BẰNG ?
        cur.execute("""
            SELECT id, plate_in, image_in_url, time_in, plate_out 
            FROM parking_logs 
            WHERE rfid_code = ? 
            ORDER BY time_in DESC LIMIT 1
        """, (rfid,))
        record = cur.fetchone()
        response_data = {}

        if record:
            log_id, plate_in, image_in_url, time_in_raw, plate_out = record
            time_in = parse_sqlite_time(time_in_raw)
            
            # Nếu plate_out chưa có = xe đang trong bãi → XE RA
            if plate_out is None:
                time_out = datetime.now()
                duration = time_out - time_in
                duration_str = f"{int(duration.total_seconds()//3600):02d}:{int((duration.total_seconds()%3600)//60):02d}:{int(duration.total_seconds()%60):02d}"
                
                full_img_url, crop_img_url, plate_out_new = process_vehicle_image(rfid, "out")
                
                clean_out = re.sub(r'[^A-Z0-9]', '', plate_out_new.upper())
                clean_in = re.sub(r'[^A-Z0-9]', '', plate_in.upper())
                
                if clean_out != clean_in:
                    warning_msg = "BIỂN SỐ VÀO VÀ RA KHÔNG KHỚP NHAU!"

                # THAY %s BẰNG ?
                cur.execute("""
                    UPDATE parking_logs 
                    SET plate_out = ?, image_out_url = ?, time_out = ? 
                    WHERE id = ?
                """, (plate_out_new, full_img_url, time_out.strftime("%Y-%m-%d %H:%M:%S"), log_id))
                
                crop_in_url = "https://placehold.co/200x80/1a1a1a/475569?text=No+Crop"
                try:
                    list_of_files = glob.glob(os.path.join("static", "crops", f"{rfid}_in_full_*.jpg"))
                    if list_of_files:
                        crop_in_url = f"http://localhost:8000/{max(list_of_files, key=os.path.getctime).replace(os.sep, '/')}"
                except: 
                    pass

                response_data = {
                    "action": "OUT", "rfid": rfid, "plate_in": plate_in, "plate_out": plate_out_new,
                    "img_in": image_in_url, "img_out": full_img_url, "img_crop_in": crop_in_url, "img_crop_out": crop_img_url,
                    "time_in": time_in.strftime("%H:%M:%S"), "time_out": time_out.strftime("%H:%M:%S"), "duration": duration_str,
                    "customer_type": customer_type,
                    "warning": warning_msg
                }
            else:
                # Xe đã ra rồi → Tạo record mới cho xe vào
                full_img_url, crop_img_url, plate_in_new = process_vehicle_image(rfid, "in")
                time_in_new = datetime.now()

                cur.execute("""
                    INSERT INTO parking_logs (rfid_code, plate_in, image_in_url, time_in) 
                    VALUES (?, ?, ?, ?) 
                """, (rfid, plate_in_new, full_img_url, time_in_new.strftime("%Y-%m-%d %H:%M:%S")))
                
                response_data = {
                    "action": "IN", "rfid": rfid, "plate_in": plate_in_new,
                    "img_in": full_img_url, "img_crop_in": crop_img_url, "time_in": time_in_new.strftime("%H:%M:%S"),
                    "customer_type": customer_type,
                    "warning": warning_msg
                }
        else:
            # Record mới → XE VÀO
            full_img_url, crop_img_url, plate_in = process_vehicle_image(rfid, "in")
            time_in_new = datetime.now()

            cur.execute("""
                INSERT INTO parking_logs (rfid_code, plate_in, image_in_url, time_in) 
                VALUES (?, ?, ?, ?) 
            """, (rfid, plate_in, full_img_url, time_in_new.strftime("%Y-%m-%d %H:%M:%S")))
            
            response_data = {
                "action": "IN", "rfid": rfid, "plate_in": plate_in,
                "img_in": full_img_url, "img_crop_in": crop_img_url, "time_in": time_in_new.strftime("%H:%M:%S"),
                "customer_type": customer_type,
                "warning": warning_msg
            }
            
        conn.commit()
        cur.close()
        conn.close()
        
        await manager.broadcast(json.dumps(response_data))
        return {"status": "success"}
        
    except Exception as e:
        print(f"[ERROR] Lỗi xử lý swipe: {str(e)}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        cur.close()
        conn.close()
        return {"status": "error", "message": str(e)}

@router.get("/api/logs")
async def get_parking_logs():
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            SELECT id, rfid_code, plate_in, time_in, time_out, plate_out
            FROM parking_logs
            ORDER BY time_in DESC LIMIT 30
        """)
        rows = cur.fetchall()
        
        logs = []
        for r in rows:
            time_in = parse_sqlite_time(r[3])
            time_out = parse_sqlite_time(r[4])
            
            fee = "-"
            if time_out is not None:  
                fee = "5,000 đ"
                
            logs.append({
                "id": r[0], "ticket": r[1], "plate": r[2], 
                "time_in": time_in.strftime("%d/%m/%Y - %H:%M:%S") if time_in else "--",
                "time_out": time_out.strftime("%d/%m/%Y - %H:%M:%S") if time_out else "--",
                "status": "Hoàn thành" if time_out else "Đang gửi", 
                "customer_type": "Khách Vãng Lai", 
                "fee": fee
            })
            
        cur.close()
        conn.close()
        return logs
    except Exception as e:
        print(f"[ERROR] Lỗi lấy logs: {str(e)}")
        import traceback
        traceback.print_exc()
        cur.close()
        conn.close()
        return []

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True: 
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)