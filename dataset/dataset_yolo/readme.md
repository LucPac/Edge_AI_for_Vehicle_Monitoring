# Dataset & Training Strategy

This project strictly follows the official Ultralytics YOLO guidelines and utilizes **Real-time Data Augmentation** instead of static data tools (like Roboflow).

**Why do we use this method?**
1. **Saves Storage:** We do not need to save thousands of modified images on the hard drive.
2. **Prevents Overfitting:** Images are randomly changed in the computer's memory (RAM) during training. The model never sees the exact same image twice. This forces the AI to learn the real shapes of a license plate, rather than just memorizing the pictures.
3. **Simulates Real-world Conditions:** We can adjust the settings to match the low-quality video from cheap cameras (like the ESP32-CAM).

---

## 1. Dataset Preparation
A high-quality dataset is the most important part of object detection.
* **Volume:** Aim for at least **1,500 images** and **10,000 labeled objects** per class.
* **Real-world Variety:** Images must match the actual deployed environment (different times of day, weather, lighting, and camera angles).
* **Label Quality:** Bounding boxes must tightly enclose the object with no gaps. Every single object must be labeled.
* **Background Images:** Add **0-10% images with no objects** (background only). This forces the AI to learn what *not* to detect, heavily reducing False Positives.
* **Strict Splits:** Make sure validation and test images NEVER leak into the training set.

---

## 2. Training Configurations
**Golden Rule:** Always train with the default settings (`hyp.scratch-low.yaml`) to establish a baseline before modifying any parameters.

* **Epochs & Early Stopping:** Start with **300 epochs**. Use `--patience 50` so the training stops automatically if the model stops getting smarter. This saves time and prevents overfitting.
* **Warmup Epochs:** We set `warmup_epochs=3.0`. The AI learns slowly at the start so it does not make big errors.
* **Image Size (`--img`):** Train at the native resolution of your camera (e.g., `640`). *Note: If detecting very small objects, training at a higher resolution (like `1280`) yields better results.* Always run inference at the same size you trained on.
* **Hardware Optimization:** Maximize your `--batch-size` (or use `-1` for auto-size) to keep training stable.
  * Enable Mixed Precision (`--amp`) to speed up training and save GPU memory without losing accuracy.
  * If you have multiple GPUs, use `--device 0,1,2,3` to distribute the workload.

---

## 3. Training Settings: Data Augmentation

By default, YOLOv5 has its own built-in data augmentation. However, you can add more advanced effects by using the `albumentations` library.

* **Option 1: Default YOLOv5 Augmentation (Without `albumentations`)**
  If you do not install the `albumentations` library, YOLOv5 will automatically use the standard augmentation settings defined in the config file:
  `YOLOV5/data/hyps/hyp.scratch-low.yaml`

```yaml
  # Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Hyperparameters for low-augmentation COCO training from scratch
# python train.py --batch 64 --cfg yolov5n6.yaml --weights '' --data coco.yaml --img 640 --epochs 300 --linear
# See tutorials for hyperparameter evolution https://github.com/ultralytics/yolov5#tutorials

lr0: 0.01 # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.01 # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937 # SGD momentum/Adam beta1
weight_decay: 0.0005 # optimizer weight decay 5e-4
warmup_epochs: 3.0 # warmup epochs (fractions ok)
warmup_momentum: 0.8 # warmup initial momentum
warmup_bias_lr: 0.1 # warmup initial bias lr
box: 0.05 # box loss gain
cls: 0.5 # cls loss gain
cls_pw: 1.0 # cls BCELoss positive_weight
obj: 1.0 # obj loss gain (scale with pixels)
obj_pw: 1.0 # obj BCELoss positive_weight
iou_t: 0.20 # IoU training threshold
anchor_t: 4.0 # anchor-multiple threshold
# anchors: 3  # anchors per output layer (0 to ignore)
fl_gamma: 0.0 # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: 0.015 # image HSV-Hue augmentation (fraction)
hsv_s: 0.7 # image HSV-Saturation augmentation (fraction)
hsv_v: 0.4 # image HSV-Value augmentation (fraction)
degrees: 0.0 # image rotation (+/- deg)
translate: 0.1 # image translation (+/- fraction)
scale: 0.5 # image scale (+/- gain)
shear: 0.0 # image shear (+/- deg)
perspective: 0.0 # image perspective (+/- fraction), range 0-0.001
flipud: 0.0 # image flip up-down (probability)
fliplr: 0.5 # image flip left-right (probability)
mosaic: 1.0 # image mosaic (probability)
mixup: 0.0 # image mixup (probability)
copy_paste: 0.0 # segment copy-paste (probability)
```

* **Option 2: Advanced Augmentation (With `albumentations`)**
  To unlock more effects, simply uncomment or add `albumentations` in your `YOLOV5/requirements.txt` file.

```text
# Extras ----------------------------------------------------------------------
# ipython  # interactive notebook
# mss  # screenshots
# albumentations>=1.0.3
# pycocotools>=2.0.6  # COCO mAP
urllib3>=2.6.0 ; python_version > "3.8" # not directly required, pinned by Snyk to avoid a vulnerability
```

  When installed, `train.py` will combine the base settings from `hyp.scratch-low.yaml` **AND** the extra effects defined in this file:
  `YOLOV5/utils/augmentations.py`

```python
try:
    import albumentations as A

    check_version(A.__version__, "1.0.3", hard=True)  # version requirement

    T = [
        A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
        A.Blur(p=0.01),
        A.MedianBlur(p=0.01),
        A.ToGray(p=0.01),
        A.CLAHE(p=0.01),
        A.RandomBrightnessContrast(p=0.0),
        A.RandomGamma(p=0.0),
        A.ImageCompression(quality_lower=75, p=0.0),
    ]  # transforms
    self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

    LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
except ImportError:  # package not installed, skip
    pass
```
