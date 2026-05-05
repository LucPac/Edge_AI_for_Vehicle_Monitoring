# services/vehicle.py
import cv2
import os
import time
import re
from ai.image_processing import align_plate_image
from ai.recognition import recognize_onnx
from services.camera import detector, ai_lock

def process_vehicle_image(rfid_code, action_type):
    """Xử lý ảnh khi quẹt thẻ: tìm biển số, cắt ảnh, làm thẳng và đọc chữ"""
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