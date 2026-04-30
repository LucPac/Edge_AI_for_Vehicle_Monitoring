import warnings
warnings.filterwarnings("ignore")
import onnxruntime as ort
import os
import time
import cv2
import numpy as np

# ===================== PATH =====================
IMG_PATH = "test/images"
LABELS_PATH = "test/labels.txt"
ONNX_PATH = "models/r34_vd_none_bilstm_ctc/model.onnx"
DICT_PATH = "models/plate_dict.txt"

# ===================== LOAD ONNX =====================
session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ===================== LOAD DICT =====================
def load_dict(dict_path):
    with open(dict_path, "r", encoding="utf-8") as f:
        chars = [line.strip() for line in f.readlines()]
    return ["blank"] + chars  # CTC blank = 0

char_list = load_dict(DICT_PATH)

# ===================== PREPROCESS =====================
def preprocess(img_path):
    img = cv2.imread(img_path)

    h, w = img.shape[:2] # [C, H, W]
    ratio = w / float(h)
    new_w = min(int(32 * ratio), 100)

    img = cv2.resize(img, (new_w, 32))

    padded = np.zeros((32, 100, 3), dtype=np.uint8) # [H, W, C]
    padded[:, :new_w, :] = img

    img = padded.astype("float32") / 255.0

    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)

    return img

# ===================== DECODE (CTC) =====================
def decode(preds):
    preds_idx = preds.argmax(axis=2)[0]

    last = -1
    text = ""
    for idx in preds_idx:
        if idx != 0 and idx != last:
            if idx < len(char_list):
                text += char_list[idx]
        last = idx

    return text

# ===================== PARSER =====================
def parser_imgname_imglabel(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    imgname_label_dict = {}
    for line in lines:
        if not line.strip():
            continue
        imgname, label = line.strip().split("\t")
        imgname_label_dict[imgname] = label

    return imgname_label_dict

# ===================== EVALUATE =====================
def evaluate(img_dir, labels_path):
    imgname_label_dict = parser_imgname_imglabel(labels_path)

    total_samples = len(imgname_label_dict)
    correct_predictions = 0
    total_inference_time = 0

    for i, (imgname, true_label) in enumerate(imgname_label_dict.items()):
        img_path = os.path.join(img_dir, imgname)

        start_time = time.perf_counter()

        # ===== ONNX INFERENCE =====
        img = preprocess(img_path)
        preds = session.run([output_name], {input_name: img})[0]
        predicted_label = decode(preds)

        end_time = time.perf_counter()

        if i > 0:  # bỏ warmup
            total_inference_time += (end_time - start_time)

        if predicted_label == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_samples
    valid_samples = total_samples - 1 if total_samples > 1 else 1
    avg_time_ms = (total_inference_time / valid_samples) * 1000 # time(ms) for rec 1 img
    frame_rate = 1.0 / (avg_time_ms / 1000) if avg_time_ms > 0 else 0 # fps

    return accuracy, avg_time_ms, frame_rate

# ===================== RUN =====================
accuracy, avg_time_ms, fps = evaluate(IMG_PATH, LABELS_PATH)

print(f"Accuracy: {accuracy:.4f}")
print(f"Inference Time: {avg_time_ms:.2f} ms")
print(f"FPS: {fps:.2f}")
