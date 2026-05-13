from collections import Counter
import random
import cv2
import time
import numpy as np
import re
from ai.recognition import recognize_onnx
from services.camera import detector, ai_lock # Dùng chung model với camera cho nhẹ RAM

def process_vehicle_burst(rfid_code, action_type, frames_list):
    """Hàm xử lý Burst: Nhận 5 ảnh, OCR cả 5 và Bầu chọn (Voting)"""
    all_texts = []
    best_img = None
    best_crop = None
    
    print(f"\n[AI] Đang phân tích {len(frames_list)} khung hình...")
    
    # Duyệt qua từng tấm ảnh chụp được
    for img in frames_list:
        if img is None: continue
        
        # Nhờ YOLO tìm biển số (dùng Lock để không đụng chạm với luồng Camera)
        with ai_lock:
            detections = detector.detect(img)
            
        if detections is not None:
            for i, class_id in enumerate(detections.class_id):
                if class_id == 2: # Nếu thấy Plate
                    best_img = img # Lưu lại ảnh nét nhất
                    p_box = detections.xyxy[i]
                    px1, py1, px2, py2 = map(int, p_box)
                    h_orig, w_orig, _ = img.shape
                    p = 2
                    
                    # Cắt ảnh biển số
                    cropped_img = img[max(0, py1-p):min(h_orig, py2+p), max(0, px1-p):min(w_orig, px2+p)]
                    
                    if cropped_img.size > 0:
                        best_crop = cropped_img
                        h, w = cropped_img.shape[:2]
                        ratio = w / h if h > 0 else 0
                        
                        # Đưa vào CRNN đọc chữ
                        if 0 < ratio < 1.9: # Biển vuông (2 dòng)
                            raw_text = recognize_onnx(cropped_img[:int(h*0.55), :]) + recognize_onnx(cropped_img[int(h*0.45):, :])
                        else: # Biển dài (1 dòng)
                            raw_text = recognize_onnx(cropped_img)
                        
                        clean_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
                        if len(clean_text) >= 5: # Chỉ lấy chữ có ý nghĩa
                            all_texts.append(clean_text)

    # ================== BẦU CHỌN (VOTING) ==================
    plate_text = f"LOI-OCR-{rfid_code[-4:]}" # Default nếu mù hẳn
    if len(all_texts) > 0:
        most_common_text, count = Counter(all_texts).most_common(1)[0]
        plate_text = most_common_text
        print(f"🎯 [BURST CHỐT ĐƠN] Biển số: {plate_text} (Đồng thuận: {count}/{len(all_texts)} phiếu)")
    else:
        print("❌ [BURST THẤT BẠI] Không đọc được biển số nào trong loạt ảnh!")

    # ================== LƯU ẢNH RA Ổ CỨNG ==================
    base_img_path = f"static/images/{rfid_code}_{action_type}.jpg"
    crop_url = "https://placehold.co/200x80/1a1a1a/475569?text=No+Crop"
    
    if best_img is not None:
        cv2.imwrite(base_img_path, best_img)
        if best_crop is not None:
            timestamp = int(time.time()) + random.randint(1, 100)
            full_crop_filename = f"static/crops/{rfid_code}_{action_type}_burst_{timestamp}.jpg"
            cv2.imwrite(full_crop_filename, best_crop)
            crop_url = f"http://localhost:8000/{full_crop_filename}"
    else:
        blank = np.zeros((480, 640, 3), np.uint8)
        cv2.imwrite(base_img_path, blank)
        
    full_img_url = f"http://localhost:8000/{base_img_path}?t={int(time.time())}"
    return full_img_url, crop_url, plate_text