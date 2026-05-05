# config.py

# ==========================================
# 1. CẤU HÌNH CAMERA & ĐƯỜNG DẪN MODEL
# ==========================================
ESP32_CAM_URL = "http://192.168.1.15/"
YOLO_MODEL_PATH = "models/best.onnx"
CRNN_MODEL_PATH = "models/rec_model.onnx"

# ==========================================
# 2. CẤU HÌNH NHẬN DIỆN (YOLO)
# ==========================================
CLASS_MAP = {
    0: "Car",
    1: "Motorbike",
    2: "Plate"
}

COLOR_MAP = {
    0: (255, 0, 255), # Car: Màu Tím
    1: (255, 255, 0), # Motorbike: Màu Xanh Lơ
    2: (0, 255, 0)    # Plate: Màu Xanh Lá
}

# ==========================================
# 3. CẤU HÌNH ĐỌC CHỮ (CRNN OCR)
# ==========================================
CHARS_LIST = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 
    'M', 'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z' ,'R'
]

ID2CHAR = {0: ''} 
for idx, char in enumerate(CHARS_LIST):
    ID2CHAR[idx + 1] = char

# ==========================================
# 4. CẤU HÌNH DATABASE
# ==========================================
DB_CONFIG = {
    "dbname": "P_Dashboard",
    "user": "postgres",
    "password": "241120", 
    "host": "localhost",
    "port": "5432"
}