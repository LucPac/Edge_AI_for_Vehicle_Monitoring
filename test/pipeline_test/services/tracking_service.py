import cv2
import supervision as sv
from collections import Counter
import re

def is_valid_plate(text):
    if not text: return False
    clean = re.sub(r'[^A-Z0-9]', '', text)
    pattern = r'^[1-9][0-9][A-Z][A-Z0-9]?[0-9]{4,5}$'
    if re.match(pattern, clean):
        return True
    return False

class TrafficTracker:
    def __init__(self, green_points, red_points):
        # HẠ NGƯỠNG XUỐNG CỰC ĐẠI: Chỉ cần YOLO tự tin 15% là bắt luôn! (Bắt cực xa)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.15, 
            lost_track_buffer=60,
            minimum_matching_threshold=0.3  
        )

        self.green_zone = sv.PolygonZone(polygon=green_points) # xe vào
        self.red_zone = sv.PolygonZone(polygon=red_points) # xe ra

        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

        self.vehicle_history = {}
        self.vehicle_ocr_cache = {}    
        self.vehicle_ocr_history = {}  
        self.vehicle_is_locked = {}    

    def get_direction(self, tracker_id, cy):
        if tracker_id not in self.vehicle_history:
            self.vehicle_history[tracker_id] = []

        self.vehicle_history[tracker_id].append(cy)
        if len(self.vehicle_history[tracker_id]) > 15:
            self.vehicle_history[tracker_id].pop(0)

        direction = "Stopped"
        if len(self.vehicle_history[tracker_id]) >= 5:
            dy = self.vehicle_history[tracker_id][-1] - self.vehicle_history[tracker_id][0]
            if dy < -5: direction = "CheckIn"
            elif dy > 5: direction = "CheckOut"

        return direction

    def update_and_draw(self, frame, vehicle_detections, plate_detections, extract_plate_text_func):
        if len(vehicle_detections) == 0:
            return frame

        tracked_vehicles = self.tracker.update_with_detections(detections=vehicle_detections)
        is_in_green = self.green_zone.trigger(detections=tracked_vehicles)
        is_in_red = self.red_zone.trigger(detections=tracked_vehicles)

        labels = []
        warning_flag = False

        # ĐÃ SỬA: Lấy thêm class_id từ tracked_vehicles
        for i, (tracker_id, v_box, class_id) in enumerate(zip(tracked_vehicles.tracker_id, tracked_vehicles.xyxy, tracked_vehicles.class_id)):
            cy = (v_box[1] + v_box[3]) / 2
            direction = self.get_direction(tracker_id, cy)
            
            # XÁC ĐỊNH TÊN LOẠI XE DỰA VÀO CLASS_ID CỦA YOLO (0: Car, 1: Motorcycle)
            # Lưu ý: Cần đảm bảo ID này khớp với file config.py của bạn
            vehicle_type = "Car" if class_id == 0 else ("Motorcycle" if class_id == 1 else "Vehicle")

            if tracker_id not in self.vehicle_ocr_cache:
                self.vehicle_ocr_cache[tracker_id] = ""
                self.vehicle_ocr_history[tracker_id] = []
                self.vehicle_is_locked[tracker_id] = False 

            if len(plate_detections) > 0:
                vx1, vy1, vx2, vy2 = v_box
                margin = 200

                for p_box in plate_detections.xyxy:
                    px1, py1, px2, py2 = map(int, p_box)
                    p_cx, p_cy = (px1 + px2) / 2, (py1 + py2) / 2

                    if (vx1 - margin) <= p_cx <= (vx2 + margin) and (vy1 - margin) <= p_cy <= (vy2 + margin):
                        text = extract_plate_text_func(p_box)

                        if text:
                            current_best = self.vehicle_ocr_cache.get(tracker_id, "")

                            if len(text) > len(current_best):
                                self.vehicle_ocr_history[tracker_id] = []  
                                self.vehicle_is_locked[tracker_id] = False 

                            if not self.vehicle_is_locked.get(tracker_id, False):
                                self.vehicle_ocr_history[tracker_id].append(text)

                                if len(self.vehicle_ocr_history[tracker_id]) > 3:
                                    self.vehicle_ocr_history[tracker_id].pop(0)

                                most_common_text, count = Counter(self.vehicle_ocr_history[tracker_id]).most_common(1)[0]
                                self.vehicle_ocr_cache[tracker_id] = most_common_text

                                if count >= 2:
                                    self.vehicle_is_locked[tracker_id] = True
                        break 

            # ĐÃ SỬA: Thay thế chữ ID thành tên xe (Car / Motorcycle)
            plate_text = self.vehicle_ocr_cache[tracker_id]
            display_text = f"{vehicle_type} | {plate_text}" if plate_text else f"{vehicle_type}"

            if is_in_green[i]:
                if direction == "CheckIn": display_text += " | IN"
                elif direction == "CheckOut": display_text += " | OUT"
            elif is_in_red[i]:
                if direction == "CheckOut": display_text += " | OUT"
                elif direction == "CheckIn":
                    display_text += " | IN"
                    warning_flag = True

            labels.append(display_text)

        frame = self.box_annotator.annotate(scene=frame, detections=tracked_vehicles)
        frame = self.label_annotator.annotate(scene=frame, detections=tracked_vehicles, labels=labels)

        if warning_flag:
            cv2.putText(frame, "CANH BAO: NGUOC CHIEU", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

        return frame