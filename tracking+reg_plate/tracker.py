import cv2
import supervision as sv
from collections import Counter
import re

def is_valid_plate(text):
    if not text: return False
    # Xóa ký tự lạ
    clean = re.sub(r'[^A-Z0-9]', '', text)

    # Luật chuẩn Việt Nam:
    # Bắt đầu bằng 2 số + 1 chữ cái + kết thúc bằng 4 hoặc 5 số
    pattern = r'^[1-9][0-9][A-Z][A-Z0-9]?[0-9]{4,5}$'

    if re.match(pattern, clean):
        return True
    return False

class TrafficTracker:
    def __init__(self, green_points, red_points):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.5,
            lost_track_buffer=60,
            minimum_matching_threshold=0.6
        )

        self.green_zone = sv.PolygonZone(polygon=green_points) # xe vào
        self.red_zone = sv.PolygonZone(polygon=red_points) # xe ra

        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

        # HỆ THỐNG CHỐNG NHẤP NHÁY
        self.vehicle_history = {}
        self.vehicle_ocr_cache = {}    # Kết quả đang hiển thị
        self.vehicle_ocr_history = {}  # Rổ phiếu bầu
        self.vehicle_is_locked = {}    # Trạng thái Khóa (True/False)

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

        for i, (tracker_id, v_box) in enumerate(zip(tracked_vehicles.tracker_id, tracked_vehicles.xyxy)):
            cy = (v_box[1] + v_box[3]) / 2
            direction = self.get_direction(tracker_id, cy)

            # Cấp rổ phiếu cho xe mới
            if tracker_id not in self.vehicle_ocr_cache:
                self.vehicle_ocr_cache[tracker_id] = ""
                self.vehicle_ocr_history[tracker_id] = []
                self.vehicle_is_locked[tracker_id] = False # Mặc định là chưa khóa

            # ---> CHỈ ĐỌC KHI CHƯA BỊ KHÓA <---
            #if not self.vehicle_is_locked[tracker_id] and len(plate_detections) > 0:
            if len(plate_detections) > 0:
                vx1, vy1, vx2, vy2 = v_box
                margin = 50

                for p_box in plate_detections.xyxy:
                    px1, py1, px2, py2 = map(int, p_box)
                    p_cx, p_cy = (px1 + px2) / 2, (py1 + py2) / 2

                    if (vx1 - margin) <= p_cx <= (vx2 + margin) and (vy1 - margin) <= p_cy <= (vy2 + margin):
                        text = extract_plate_text_func(p_box)

                        if text:
                            current_best = self.vehicle_ocr_cache.get(tracker_id, "")

                            # So sánh: Nếu chữ MỚI đọc được dài hơn chữ CŨ đang hiện trên màn hình
                            if len(text) > len(current_best):
                                self.vehicle_ocr_history[tracker_id] = []  # Đổ rác, làm lại từ đầu
                                self.vehicle_is_locked[tracker_id] = False # Mở khóa ngay lập tức!
                                print(f"🔓 [LẬT KÈO] ID {tracker_id}: Rõ hơn -> Cập nhật {current_best} thành {text}")

                            if not self.vehicle_is_locked.get(tracker_id, False):
                                # 1. Bỏ kết quả vào rổ
                                self.vehicle_ocr_history[tracker_id].append(text)

                                # 2. Rổ chỉ chứa tối đa 5 phiếu gần nhất (Bỏ phiếu cũ đi)
                                if len(self.vehicle_ocr_history[tracker_id]) > 5:
                                    self.vehicle_ocr_history[tracker_id].pop(0)

                                # 3. Tìm xem chữ nào xuất hiện nhiều nhất trong rổ hiện tại
                                most_common_text, count = Counter(self.vehicle_ocr_history[tracker_id]).most_common(1)[0]

                                # 4. LUÔN HIỂN THỊ CHỮ PHỔ BIẾN NHẤT (Chống nhấp nháy)
                                self.vehicle_ocr_cache[tracker_id] = most_common_text

                                # 5. NẾU ĐẠT 3 PHIẾU GIỐNG NHAU -> KHÓA LẠI!
                                if count >= 3:
                                    self.vehicle_is_locked[tracker_id] = True
                                    print(f"🔒 [ĐÃ KHÓA] ID {tracker_id}: {most_common_text} (Tin cậy cao)")
                        break # Dừng tìm kiếm biển số cho xe này ở frame hiện tại

            # Lấy kết quả ra hiển thị
            plate_text = self.vehicle_ocr_cache[tracker_id]
            display_text = f"ID:{tracker_id} | {plate_text}" if plate_text else f"ID:{tracker_id}"

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
