import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv

class YOLODetector:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        # Lưu kích thước model để dùng cho padded
        self.input_shape = self.session.get_inputs()[0].shape[2:]

    def detect(self, img):
        height, width = img.shape[:2]
        max_dim = max(width, height)

        # --- TIỀN XỬ LÝ ---
        base_padded = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        base_padded[0:height, 0:width] = img

        img_rgb = cv2.cvtColor(base_padded, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_shape[1], self.input_shape[0]))
        img_chw = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(img_chw, axis=0)

        # --- CHẠY YOLO ---
        preds = self.session.run(None, {self.input_name: input_tensor})[0]

        factor = max_dim / float(self.input_shape[1])
        TARGET_CLASSES = np.array([0, 1, 2])

        predictions = preds[0]
        # Vẫn giữ ngưỡng 15% để hệ thống nhìn xa hết cỡ
        valid_preds = predictions[predictions[:, 4] > 0.15]

        if len(valid_preds) == 0:
            return sv.Detections.empty()

        class_scores_matrix = valid_preds[:, 5:]
        class_ids_array = np.argmax(class_scores_matrix, axis=1)
        max_class_scores = np.max(class_scores_matrix, axis=1)
        confidences_array = valid_preds[:, 4] * max_class_scores

        mask = (confidences_array > 0.15) & np.isin(class_ids_array, TARGET_CLASSES)

        final_preds = valid_preds[mask]
        final_confs = confidences_array[mask]
        final_class_ids = class_ids_array[mask]

        if len(final_preds) == 0:
            return sv.Detections.empty()

        cx, cy, w, h = final_preds[:, 0], final_preds[:, 1], final_preds[:, 2], final_preds[:, 3]

        left = ((cx - w / 2) * factor).astype(int)
        top = ((cy - h / 2) * factor).astype(int)
        width_box = (w * factor).astype(int)
        height_box = (h * factor).astype(int)

        boxes = np.column_stack((left, top, width_box, height_box)).tolist()
        confidences = final_confs.tolist()
        class_ids = final_class_ids.tolist()

        # --- CLASS-AWARE NMS ---
        shifted_boxes = []
        max_wh = 4096
        for i in range(len(boxes)):
            cls_id = class_ids[i]
            shifted_boxes.append([boxes[i][0] + cls_id * max_wh, boxes[i][1] + cls_id * max_wh, boxes[i][2], boxes[i][3]])

        indices = cv2.dnn.NMSBoxes(shifted_boxes, confidences, 0.15, 0.4)

        if len(indices) == 0:
            return sv.Detections.empty()

        idx = indices.flatten()
        final_boxes = np.array(boxes)[idx]
        final_confs = np.array(confidences)[idx]
        final_class_ids = np.array(class_ids)[idx]

        xyxy = final_boxes.copy()
        xyxy[:, 2] += xyxy[:, 0]
        xyxy[:, 3] += xyxy[:, 1]

        # ========================================================
        # --- BỘ LỌC XÓA BOX LỒNG NHAU (CHỐNG ẢO GIÁC 2 KHUNG) ---
        # ========================================================
        final_keep = []
        for i in range(len(xyxy)):
            keep = True
            box1_area = (xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])
            
            for j in range(len(xyxy)):
                if i == j: continue
                
                # Bỏ qua không xét lồng nhau với Biển số (Class ID = 2)
                if final_class_ids[i] == 2 or final_class_ids[j] == 2:
                    continue

                box2_area = (xyxy[j, 2] - xyxy[j, 0]) * (xyxy[j, 3] - xyxy[j, 1])
                
                # Nếu box(i) nhỏ hơn box(j), kiểm tra xem i có nằm trong j không
                if box1_area < box2_area:
                    ix1 = max(xyxy[i, 0], xyxy[j, 0])
                    iy1 = max(xyxy[i, 1], xyxy[j, 1])
                    ix2 = min(xyxy[i, 2], xyxy[j, 2])
                    iy2 = min(xyxy[i, 3], xyxy[j, 3])

                    inter_w = max(0, ix2 - ix1)
                    inter_h = max(0, iy2 - iy1)
                    
                    if inter_w > 0 and inter_h > 0:
                        inter_area = inter_w * inter_h
                        # Nếu bị nuốt hơn 70% diện tích -> Loại bỏ box nhỏ
                        if inter_area / box1_area > 0.7:
                            keep = False
                            break
                            
            if keep:
                final_keep.append(i)

        # Lọc lại mảng kết quả cuối cùng
        xyxy = xyxy[final_keep]
        final_confs = final_confs[final_keep]
        final_class_ids = final_class_ids[final_keep]

        return sv.Detections(xyxy=xyxy, confidence=final_confs, class_id=final_class_ids)