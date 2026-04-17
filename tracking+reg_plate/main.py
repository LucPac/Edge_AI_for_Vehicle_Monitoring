import cv2
import numpy as np
from detector import YOLODetector
from plate_cropper import align_plate
from ocr import PlateOCR
from tracker import TrafficTracker, is_valid_plate

# Khởi tạo
detector = YOLODetector("models/best.onnx")
ocr = PlateOCR("models/rec_model.onnx", dict_path="models/plate_dict.txt")
green_points = np.array([[470, 302], [566, 297], [603, 381], [479, 384]], np.int32)
red_points = np.array([[200, 300], [450, 300], [450, 400], [200, 400]], np.int32)
traffic_tracker = TrafficTracker(green_points, red_points)

cap = cv2.VideoCapture("D:/Dropbox/GitHub/RASPBERRY_Parking_lot/license_plate/videos_test/camera_duong_pho1.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    detections = detector.detect(frame)
    if detections is not None:
        # Tách xe và biển số để xử lý riêng
        vehicle_detections = detections[(detections.class_id == 0) | (detections.class_id == 1)]
        plate_detections = detections[detections.class_id == 2]

        def read_plate_text(p_box):
            px1, py1, px2, py2 = map(int, p_box)

            # 1. Tính toán kích thước và tỷ lệ
            w = px2 - px1
            h = py2 - py1
            aspect_ratio = w / h

            # Cắt lấy ảnh biển số gốc
            plate_img = frame[max(0, py1-5):py2+5, max(0, px1-5):px2+5]

            final_text = ""

            # 2. PHÂN LOẠI BIỂN
            if aspect_ratio < 2.2: # Ngưỡng biển vuông (2 hàng)
                # --- XỬ LÝ BIỂN 2 HÀNG ---
                height, width = plate_img.shape[:2]
                mid = height // 2

                # Cắt làm 2 nửa: Trên và Dưới
                top_half = plate_img[0:mid+5, 0:width]
                bottom_half = plate_img[mid-5:height, 0:width]

                # Đọc OCR từng nửa
                text_top = ocr.infer(align_plate(top_half))
                text_bottom = ocr.infer(align_plate(bottom_half))

                # Ghép lại (nhớ xóa khoảng trắng)
                final_text = (text_top if text_top else "") + (text_bottom if text_bottom else "")
                # print(f"--- Biển Vuông detected! Ghép: {final_text} ---")

            else:
                # --- XỬ LÝ BIỂN DÀI (1 hàng) ---
                raw_text = ocr.infer(align_plate(plate_img))
                final_text = raw_text if raw_text else ""

            # 3. LÀM SẠCH VÀ KIỂM TRA REGEX
            if final_text:
                final_text = final_text.replace(" ", "").upper()
                # Gọi hàm kiểm tra Regex mà anh em mình làm hôm trước
                if is_valid_plate(final_text):
                    return final_text

            return None

        frame = traffic_tracker.update_and_draw(frame, vehicle_detections, plate_detections, read_plate_text)

    cv2.imshow("He thong quan ly phuong tien", cv2.resize(frame, (1280, 720)))
    if cv2.waitKey(200) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
