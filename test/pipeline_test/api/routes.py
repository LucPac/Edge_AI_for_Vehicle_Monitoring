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
from models import RFIDData, RegisterData
from services.websocket import manager
from services.vehicle import process_vehicle_image
from services.camera import get_current_frame

router = APIRouter()

@router.post("/api/register")
async def register_card(data: RegisterData):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id FROM registered_cards WHERE rfid_code = %s", (data.rfid_code,))
        if cur.fetchone():
            return {"status": "error", "message": "Mã thẻ RFID này đã tồn tại trong hệ thống!"}

        clean_plate = re.sub(r'[^A-Za-z0-9]', '', data.plate_number).upper()
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
    is_registered = False
    reg_plate = ""
    warning_msg = None
    
    try:
        cur.execute("SELECT owner_name, phone, plate_number FROM registered_cards WHERE rfid_code = %s", (rfid,))
        reg_info = cur.fetchone()
        if reg_info:
            is_registered = True
            reg_name, reg_phone, reg_plate = reg_info
            details = f"{reg_name} - {reg_phone}"
            customer_type = f"Khách Đăng Ký<br><span style='font-size: 0.9em; font-weight: normal; opacity: 0.8;'>{details}</span>"
    except Exception as e:
        conn.rollback() 
        print(f"Lỗi check thẻ đăng ký: {e}")
    
    cur.execute("SELECT id, plate_in, image_in_url, time_in FROM parking_logs WHERE rfid_code = %s AND status IN ('PARKING', 'ERROR_IN')", (rfid,))
    record = cur.fetchone()
    response_data = {}

    if record:
        # LOGIC XUẤT BẾN
        log_id, plate_in, image_in_url, time_in = record
        time_out = datetime.now()
        duration = time_out - time_in
        duration_str = f"{int(duration.total_seconds()//3600):02d}:{int((duration.total_seconds()%3600)//60):02d}:{int(duration.total_seconds()%60):02d}"
        
        full_img_url, crop_img_url, plate_out = process_vehicle_image(rfid, "out")
        status_db = "COMPLETED" 
        
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
        # LOGIC VÀO BẾN
        full_img_url, crop_img_url, plate_in = process_vehicle_image(rfid, "in")
        status_db = "PARKING"
        
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

@router.get("/api/logs")
async def get_parking_logs():
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT p.id, p.rfid_code, p.plate_in, p.time_in, p.time_out, p.status, p.plate_out, r.owner_name, r.phone
        FROM parking_logs p
        LEFT JOIN registered_cards r ON p.rfid_code = r.rfid_code
        ORDER BY p.time_in DESC LIMIT 30
    """)
    rows = cur.fetchall()
    
    logs = []
    for r in rows:
        is_registered = (r[7] is not None)
        if is_registered:
            details = f"{r[7]} - {r[8]}"
            customer_type = f"Khách Đăng Ký<br><span style='font-size: 0.9em; font-weight: normal; color: #94a3b8;'>{details}</span>"
        else:
            customer_type = "Khách Vãng Lai"
        
        fee = "-"
        if r[5] in ["COMPLETED", "ERROR_OUT"]:
            fee = "0 đ" if is_registered else "5,000 đ"
            
        logs.append({
            "id": r[0], "ticket": r[1], "plate": r[2], 
            "time_in": r[3].strftime("%d/%m/%Y - %H:%M:%S") if r[3] else "--",
            "time_out": r[4].strftime("%d/%m/%Y - %H:%M:%S") if r[4] else "--",
            "status": r[5], "customer_type": customer_type, "fee": fee
        })
        
    cur.close()
    conn.close()
    return logs

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)