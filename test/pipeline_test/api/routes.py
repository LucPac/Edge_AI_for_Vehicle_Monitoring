# api/routes.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
import os
import json
import glob
import re
import asyncio
from datetime import datetime

from database import get_db_connection
from models import RFIDData
from services.websocket import manager

# Import hàm Burst Voting mới
from services.vehicle import process_vehicle_burst
# Import 2 hàm lấy ảnh từ 2 camera riêng biệt
from services.camera import get_current_frame_in, get_current_frame_out

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
    conn = get_db_connection()
    cur = conn.cursor()
    
    customer_type = "Khách Vãng Lai"
    warning_msg = None
    
    try:
        # Kiểm tra trạng thái xe trong CSDL (Vào hay Ra)
        cur.execute("""
            SELECT id, plate_in, image_in_url, time_in, plate_out 
            FROM parking_logs 
            WHERE rfid_code = ? 
            ORDER BY time_in DESC LIMIT 1
        """, (rfid,))
        record = cur.fetchone()
        response_data = {}

        if record and record[4] is None:
            # ==========================================
            # TRƯỜNG HỢP: XE RA KHỎI BÃI
            # ==========================================
            log_id, plate_in, image_in_url, time_in_raw, plate_out = record
            time_in = parse_sqlite_time(time_in_raw)
            time_out = datetime.now()
            
            print(f"\n📸 [CAMERA] Bắt đầu chụp Burst LỐI RA cho thẻ {rfid}...")
            frames_list = []
            for _ in range(5):
                f = get_current_frame_out()
                if f is not None: 
                    frames_list.append(f.copy())
                await asyncio.sleep(0.1) # Khoảng cách 0.1s mỗi tấm
            
            # Đưa 5 tấm ảnh cho AI đọc và chốt biển số
            full_img_url, crop_img_url, plate_out_new = process_vehicle_burst(rfid, "out", frames_list)
            
            # Cảnh báo nếu biển số Không khớp
            clean_out = re.sub(r'[^A-Z0-9]', '', plate_out_new.upper())
            clean_in = re.sub(r'[^A-Z0-9]', '', plate_in.upper())
            if clean_out != clean_in:
                warning_msg = "BIỂN SỐ VÀO VÀ RA KHÔNG KHỚP NHAU!"

            # Tính toán thời gian gửi
            duration = time_out - time_in
            duration_str = f"{int(duration.total_seconds()//3600):02d}:{int((duration.total_seconds()%3600)//60):02d}:{int(duration.total_seconds()%60):02d}"
            
            cur.execute("""
                UPDATE parking_logs 
                SET plate_out = ?, image_out_url = ?, time_out = ? 
                WHERE id = ?
            """, (plate_out_new, full_img_url, time_out.strftime("%Y-%m-%d %H:%M:%S"), log_id))
            
            # Tìm lại ảnh crop biển số lúc vào để hiển thị so sánh
            crop_in_url = "https://placehold.co/200x80/1a1a1a/475569?text=No+Crop"
            try:
                # Tìm file có chữ 'burst' để tương thích với cơ chế mới
                list_of_files = glob.glob(os.path.join("static", "crops", f"{rfid}_in_burst_*.jpg"))
                if not list_of_files: # Fallback tìm file cơ chế cũ (nếu có)
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
            # ==========================================
            # TRƯỜNG HỢP: XE VÀO BÃI (Tạo mới)
            # ==========================================
            print(f"\n📸 [CAMERA] Bắt đầu chụp Burst LỐI VÀO cho thẻ {rfid}...")
            frames_list = []
            for _ in range(5):
                f = get_current_frame_in()
                if f is not None: 
                    frames_list.append(f.copy())
                await asyncio.sleep(0.1) # Khoảng cách 0.1s mỗi tấm

            # Đưa 5 tấm ảnh cho AI đọc và chốt biển số
            full_img_url, crop_img_url, plate_in_new = process_vehicle_burst(rfid, "in", frames_list)
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
            
        conn.commit()
        cur.close()
        conn.close()
        
        # Bắn dữ liệu về giao diện Web thông qua Websocket
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