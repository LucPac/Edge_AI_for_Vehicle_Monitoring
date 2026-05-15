# ai/recognition.py
import cv2
import numpy as np
import onnxruntime as ort
from config import CRNN_MODEL_PATH, ID2CHAR

# --- KIỂM TRA THƯ VIỆN TFLITE ---
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

class CRNNRecognizer:
    def __init__(self, model_path):
        self.is_tflite = model_path.endswith('.tflite')
        
        if self.is_tflite:
            if not TFLITE_AVAILABLE:
                raise ImportError("Bạn chưa cài TensorFlow/tflite_runtime để chạy model .tflite!")
            print(f"[*] Khởi tạo TFLite Engine (CRNN) với model: {model_path}")
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            print(f"[*] Khởi tạo ONNX Engine (CRNN) với model: {model_path}")
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name

    def decode_predictions(self, preds_numpy):
        preds_seq = preds_numpy[0] 
        preds_index = np.argmax(preds_seq, axis=1)
        char_list = []
        for i in range(len(preds_index)):
            if preds_index[i] != 0 and (not (i > 0 and preds_index[i - 1] == preds_index[i])):
                if preds_index[i] in ID2CHAR:
                    char_list.append(ID2CHAR[preds_index[i]])
        return ''.join(char_list)

    def recognize(self, img_bgr):
        """Nhận ảnh biển số đã cắt và trả về chuỗi ký tự"""
        try:
            img_resized = cv2.resize(img_bgr, (320, 48))
            img_float = img_resized.astype(np.float32) / 255.0
            img_float = (img_float - 0.5) / 0.5
            
            # Giữ nguyên chuẩn CHW cho cả ONNX và TFLite (do onnx-tf tự map)
            img_tensor = np.transpose(img_float, (2, 0, 1))
            img_tensor = np.expand_dims(img_tensor, axis=0)
            
            if self.is_tflite:
                self.interpreter.set_tensor(self.input_details[0]['index'], img_tensor)
                self.interpreter.invoke()
                preds_numpy = self.interpreter.get_tensor(self.output_details[0]['index'])
            else:
                ort_inputs = {self.input_name: img_tensor}
                preds_numpy = self.session.run(None, ort_inputs)[0]
                
            return self.decode_predictions(preds_numpy)
        except Exception as e:
            print(f"[!] Lỗi OCR: {e}")
            return ""

# Khởi tạo một đối tượng toàn cục duy nhất
recognizer = CRNNRecognizer(CRNN_MODEL_PATH)

# Giữ nguyên tên hàm này để các file khác gọi không bị lỗi
def recognize_onnx(img_bgr):
    return recognizer.recognize(img_bgr)