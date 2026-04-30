import warnings
warnings.filterwarnings("ignore")
from paddleocr import PaddleOCR
import os
import time

# PATH
IMG_PATH_ppocrv3 = "test/images"
LABELS_PATH_ppocrv3 = "test/labels.txt"
MODEL_DIR_ppocrv3 = "models/latinppocrv3"
DICT_DIR_ppocrv3 = "models/plate_dict.txt"

# Load OCR model
ocr_ppocrv3 = PaddleOCR(
    det_model_dir=MODEL_DIR_ppocrv3,
    cls_model_dir=MODEL_DIR_ppocrv3,
    det=False,
    cls=False,
    rec_model_dir=MODEL_DIR_ppocrv3,
    rec_char_dict_path=DICT_DIR_ppocrv3,
    rec_algorithm="SVTR_LCNet",
    rec_image_shape="3, 48, 320",
    use_angle_cls=False,
    use_gpu=False,
    show_log=False
)

# Split img name - label
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

# Evaluate OCR model
def evaluate(ocr, img_dir, labels_path):
    imgname_label_dict = parser_imgname_imglabel(labels_path)
    total_samples = len(imgname_label_dict)
    correct_predictions = 0
    total_inference_time = 0

    for i, (imgname, true_label) in enumerate(imgname_label_dict.items()):
        img_path = os.path.join(img_dir, imgname)

        start_time = time.perf_counter()
        result = ocr.ocr(img_path, det=False, cls=False)
        end_time = time.perf_counter()

        if i > 0:
            total_inference_time += (end_time - start_time)

        predicted_label = ""
        if result and result[0]:
            try:
                predicted_label = result[0][0][0]
            except IndexError:
                predicted_label = ""

        if predicted_label == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_samples
    valid_samples = total_samples - 1 if total_samples > 1 else 1
    avg_time_ms = (total_inference_time / valid_samples) * 1000 # time(ms) for rec 1 img
    frame_rate = 1.0 / (avg_time_ms / 1000) if avg_time_ms > 0 else 0 # fps

    return accuracy, avg_time_ms, frame_rate

accuracy_ppocrv3, avg_time_ms_ppocrv3, frame_rate_ppocrv3 = evaluate(ocr_ppocrv3, IMG_PATH_ppocrv3, LABELS_PATH_ppocrv3)
print(f"Accuracy: {accuracy_ppocrv3}")
print(f"Inference Time: {avg_time_ms_ppocrv3:.2f} ms")
print(f"FPS: {frame_rate_ppocrv3:.2f}")
