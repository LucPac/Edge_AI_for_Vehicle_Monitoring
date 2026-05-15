import cv2
import numpy as np
import supervision as sv
import onnxruntime as ort

# --- TÍNH NĂNG TỰ ĐỘNG NHẬN DIỆN MÔI TRƯỜNG TFLITE ---
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False
        print("[!] Không tìm thấy thư viện tflite. Chỉ có thể chạy ONNX.")

class YOLODetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.is_tflite = model_path.endswith('.tflite')

        if self.is_tflite:
            if not TFLITE_AVAILABLE:
                raise ImportError("Bạn chưa cài TensorFlow hoặc tflite_runtime để chạy model .tflite!")
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            input_shape = self.input_details[0]['shape']
            self.input_height = input_shape[1]
            self.input_width = input_shape[2]
            
        else:
            print(f"[*] Khởi tạo ONNX Engine với model: {model_path}")
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            
            self.input_shape = self.session.get_inputs()[0].shape[2:]
            self.input_height = self.input_shape[0]
            self.input_width = self.input_shape[1]

    def detect(self, img):
        height, width = img.shape[:2]
        max_dim = max(width, height)

        # --- TIỀN XỬ LÝ CHUNG ---
        base_padded = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        base_padded[0:height, 0:width] = img
        img_rgb = cv2.cvtColor(base_padded, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_width, self.input_height))

        # --- RẼ NHÁNH XỬ LÝ THEO ĐỊNH DẠNG MODEL ---
        # --- RẼ NHÁNH XỬ LÝ THEO ĐỊNH DẠNG MODEL ---
        if self.is_tflite:
            # 1. BẮT BUỘC chia 255 để đưa ảnh về [0, 1]
            img_hwc = img_resized.astype(np.float32) / 255.0
            input_tensor = np.expand_dims(img_hwc, axis=0)

            # 2. Chạy AI
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            preds = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            predictions = preds[0]
            
            # 3. Lật ma trận (nếu model xuất ra [8, 25200])
            if predictions.shape[0] < predictions.shape[1]:
                predictions = predictions.transpose()
                
            # 4. [GIẢI MÃ TỌA ĐỘ]: Phóng to tọa độ từ [0, 1] về kích thước pixel thật
            predictions[:, 0] *= self.input_width   # cx (Tâm X)
            predictions[:, 1] *= self.input_height  # cy (Tâm Y)
            predictions[:, 2] *= self.input_width   # w  (Chiều rộng)
            predictions[:, 3] *= self.input_height  # h  (Chiều cao)
                
        else:
            # Chuẩn CHW cho ONNX
            img_chw = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
            input_tensor = np.expand_dims(img_chw, axis=0)

            preds = self.session.run(None, {self.input_name: input_tensor})[0]
            predictions = preds[0]

        # ========================================================
        # --- HẬU XỬ LÝ (CHUNG CHO CẢ 2 BỘ MÁY) ---
        # ========================================================
        # (Lưu ý: Phải đảm bảo bạn đã xóa dòng predictions = preds[0] cũ ở dưới này đi rồi nhé)
        
        factor = max_dim / float(self.input_width)
        TARGET_CLASSES = np.array([0, 1, 2])

        valid_preds = predictions[predictions[:, 4] > 0.35]

        if len(valid_preds) == 0:
            return sv.Detections.empty()

        class_scores_matrix = valid_preds[:, 5:]
        class_ids_array = np.argmax(class_scores_matrix, axis=1)
        max_class_scores = np.max(class_scores_matrix, axis=1)
        confidences_array = valid_preds[:, 4] * max_class_scores

        mask = (confidences_array > 0.35) & np.isin(class_ids_array, TARGET_CLASSES)

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

        indices = cv2.dnn.NMSBoxes(shifted_boxes, confidences, 0.35, 0.4)

        if len(indices) == 0:
            return sv.Detections.empty()

        idx = indices.flatten()
        final_boxes = np.array(boxes)[idx]
        final_confs = np.array(confidences)[idx]
        final_class_ids = np.array(class_ids)[idx]

        xyxy = final_boxes.copy()
        xyxy[:, 2] += xyxy[:, 0]
        xyxy[:, 3] += xyxy[:, 1]

        # --- BỘ LỌC XÓA BOX LỒNG NHAU ---
        final_keep = []
        for i in range(len(xyxy)):
            keep = True
            box1_area = (xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])
            
            for j in range(len(xyxy)):
                if i == j: continue
                
                if final_class_ids[i] == 2 or final_class_ids[j] == 2:
                    continue

                box2_area = (xyxy[j, 2] - xyxy[j, 0]) * (xyxy[j, 3] - xyxy[j, 1])
                
                if box1_area < box2_area:
                    ix1 = max(xyxy[i, 0], xyxy[j, 0])
                    iy1 = max(xyxy[i, 1], xyxy[j, 1])
                    ix2 = min(xyxy[i, 2], xyxy[j, 2])
                    iy2 = min(xyxy[i, 3], xyxy[j, 3])

                    inter_w = max(0, ix2 - ix1)
                    inter_h = max(0, iy2 - iy1)
                    
                    if inter_w > 0 and inter_h > 0:
                        inter_area = inter_w * inter_h
                        if inter_area / box1_area > 0.7:
                            keep = False
                            break
                            
            if keep:
                final_keep.append(i)

        xyxy = xyxy[final_keep]
        final_confs = final_confs[final_keep]
        final_class_ids = final_class_ids[final_keep]

        return sv.Detections(xyxy=xyxy, confidence=final_confs, class_id=final_class_ids)