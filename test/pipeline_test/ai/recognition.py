# ai/recognition.py
import cv2
import numpy as np
import onnxruntime as ort
from config import CRNN_MODEL_PATH, ID2CHAR

print("[*] Đang load model CRNN ONNX...")
# Khởi tạo session một lần duy nhất khi import file này
ort_session = ort.InferenceSession(CRNN_MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = ort_session.get_inputs()[0].name

def decode_onnx_predictions(preds_numpy):
    preds_seq = preds_numpy[0] 
    preds_index = np.argmax(preds_seq, axis=1)
    char_list = []
    for i in range(len(preds_index)):
        if preds_index[i] != 0 and (not (i > 0 and preds_index[i - 1] == preds_index[i])):
            if preds_index[i] in ID2CHAR:
                char_list.append(ID2CHAR[preds_index[i]])
    return ''.join(char_list)

def recognize_onnx(img_bgr):
    """Nhận ảnh biển số đã cắt và trả về chuỗi ký tự"""
    try:
        img_resized = cv2.resize(img_bgr, (320, 48))
        img_float = img_resized.astype(np.float32) / 255.0
        img_float = (img_float - 0.5) / 0.5
        img_tensor = np.transpose(img_float, (2, 0, 1))
        img_tensor = np.expand_dims(img_tensor, axis=0)
        
        ort_inputs = {input_name: img_tensor}
        preds_numpy = ort_session.run(None, ort_inputs)[0]
        return decode_onnx_predictions(preds_numpy)
    except Exception as e:
        print(f"[!] Lỗi OCR: {e}")
        return ""