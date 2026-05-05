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

        # --- TIỀN XỬ LÝ (Padding vuông giữ đúng tỷ lệ) ---
        base_padded = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        base_padded[0:height, 0:width] = img

        img_rgb = cv2.cvtColor(base_padded, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_shape[1], self.input_shape[0]))
        img_chw = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(img_chw, axis=0)

        # --- CHẠY YOLO ---
        preds = self.session.run(None, {self.input_name: input_tensor})[0]

        # --- TỐI ƯU HÓA MA TRẬN NUMPY (Tuyệt chiêu của ông) ---
        factor = max_dim / float(self.input_shape[1])
        TARGET_CLASSES = np.array([0, 1, 2])

        predictions = preds[0]
        valid_preds = predictions[predictions[:, 4] > 0.4]

        if len(valid_preds) == 0:
            return sv.Detections.empty()

        class_scores_matrix = valid_preds[:, 5:]
        class_ids_array = np.argmax(class_scores_matrix, axis=1)
        max_class_scores = np.max(class_scores_matrix, axis=1)
        confidences_array = valid_preds[:, 4] * max_class_scores

        mask = (confidences_array > 0.4) & np.isin(class_ids_array, TARGET_CLASSES)

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

        indices = cv2.dnn.NMSBoxes(shifted_boxes, confidences, 0.4, 0.4)

        if len(indices) == 0:
            return sv.Detections.empty()

        idx = indices.flatten()
        final_boxes = np.array(boxes)[idx]
        final_confs = np.array(confidences)[idx]
        final_class_ids = np.array(class_ids)[idx]

        # Đổi [x, y, w, h] sang [x1, y1, x2, y2]
        xyxy = final_boxes.copy()
        xyxy[:, 2] += xyxy[:, 0]
        xyxy[:, 3] += xyxy[:, 1]

        return sv.Detections(xyxy=xyxy, confidence=final_confs, class_id=final_class_ids)
