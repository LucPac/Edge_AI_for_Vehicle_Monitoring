## Model Benchmark & Performance

### 1. Local Development Benchmark

Initial testing to evaluate model architectures before edge optimization.

> **Hardware:** Local PC (CPU: *AMD Ryzen AI 7 350*, RAM: *32GB*)  
> **Format:** `.pdmodel` (Original Paddle format)  
> **Engine:** PaddleOCR Inference (CPU)  

| Model Architecture | Accuracy (%) | Inference Time (ms/img) | Speed (FPS) | Size (MB) | Note |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **[MobileNet (CRNN)](models/crnn/trainCRNN_paddle_scratch_mv3large_none_bilstm_ctc)** | 81.6% | 16.9 ms | 59.2 | 5.78 MB | Lightweight baseline. |
| **[ResNet (CRNN)](models/crnn/trainCRNN_paddle_scratch_r34_vd_none_bilstm_ctc)** | 24.5% | 85.5 ms | 11.7 | 93.5 MB | Poor convergence, heavy. |
| **[SVTR_LCNet (PP-OCRv3)](models/svtr/trainSVTRLCNet_paddle_latinppocrv3)** | **97.8%** | **11.5 ms** | **86.9** | **8.61 MB** | Best accuracy & speed balance. |

---

### 2. Edge Deployment Benchmark (Raspberry Pi)

The most best models from local testing are converted to **ONNX** format to achieve real-time performance on resource-constrained edge hardware.

> **Hardware:** Raspberry Pi 4 Model B (RAM: *2GB*)  
> **Format:** `.onnx` (Optimized Graph)  
> **Engine:** ONNX Runtime (CPU)  

| Model Architecture | Accuracy (%) | Inference Time (ms/img) | Speed (FPS) | Size (MB) | Note |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **[MobileNet (CRNN)](models/crnn/trainCRNN_paddle_scratch_mv3large_none_bilstm_ctc)** | TBD | TBD ms | TBD | ~TBD MB | Awaiting edge testing. |
| **[ResNet (CRNN)](models/crnn/trainCRNN_paddle_scratch_r34_vd_none_bilstm_ctc)** | *Skipped* | *Skipped* | *Skipped* | *Skipped* | Dropped due to low local accuracy. |
| **[SVTR_LCNet (PP-OCRv3)](models/svtr/trainSVTRLCNet_paddle_latinppocrv3)** | TBD | TBD ms | TBD | ~TBD MB | Target model for final deployment. |
