import onnxruntime as ort
import cv2
import numpy as np

class PlateOCR:
    def __init__(self, model_path, dict_path):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

        # Load từ điển tự động
        with open(dict_path, 'r', encoding='utf-8') as f:
            self.alphabet = [line.strip() for line in f.readlines()]

        self.alphabet.insert(0, ' ')
        self.blank_idx = 0

        # PADDLEOCR LƯU Ý: Thuật toán CTC luôn có 1 ký tự "Khoảng trắng" (Blank).
        # Nó thường được Baidu nhét vào vị trí CUỐI CÙNG của mảng.
        self.blank_idx = len(self.alphabet)
        self.alphabet.append(' ') # Thêm blank vào để mảng không bị out-of-index

    def preprocess(self, img):
        # 1. PP-OCRv4 yêu cầu chiều cao CỐ ĐỊNH là 48 (bản v3 là 32), chiều rộng co giãn theo tỷ lệ
        h, w = img.shape[:2]
        img_h = 48
        img_w = int(img_h * (w / h))
        img = cv2.resize(img, (img_w, img_h))

        # 2. GIỮ NGUYÊN 3 KÊNH MÀU, chỉ đổi BGR (OpenCV) sang RGB (AI)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. CHUẨN HÓA PIXEL (Cực kỳ quan trọng)
        img = img.astype(np.float32) / 255.0
        img -= 0.5
        img /= 0.5

        # Chuyển kênh màu lên đầu (HWC -> CHW) và nạp thêm Batch size
        img = img.transpose(2, 0, 1)
        return np.expand_dims(img, axis=0)

    def decode(self, preds):
        # Lấy class có xác suất cao nhất tại mỗi bước
        pred_idx = np.argmax(preds, axis=2)[0]

        text = ""
        # 4. GIẢI MÃ CTC CHUẨN MỰC
        for i in range(len(pred_idx)):
            # Nếu không phải là "Blank" (khoảng trắng)
            if pred_idx[i] != self.blank_idx:
                # VÀ không bị trùng lặp với ký tự liền trước nó
                if i == 0 or pred_idx[i] != pred_idx[i - 1]:
                    text += self.alphabet[pred_idx[i]]
        return text

    def infer(self, img):
        # Băm ảnh
        input_tensor = self.preprocess(img)
        # Chạy ONNX
        outputs = self.session.run(None, {self.input_name: input_tensor})
        # Dịch mã CTC
        final_text = self.decode(outputs[0])

        return final_text
