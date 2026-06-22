# CIFAR-10 Image Classifier: Transfer Learning + ONNX Deployment

ResNet18 fine-tuned on CIFAR-10 using progressive layer unfreezing, reaching **95% test accuracy**, then exported and quantized to a **10.7 MB INT8 ONNX model** — a 4x size reduction with no accuracy loss.

---

## Results

| Stage | Unfrozen Layers | Best Val Loss | Test Accuracy |
|-------|----------------|---------------|---------------|
| Stage 1 | Classifier head only | 0.5691 | — |
| Stage 2 | + layer4, layer3 | 0.2122 | 93.29% |
| Stage 3 | All layers | 0.0635 | **95.00%** |

| Model Format | Size | Notes |
|---|---|---|
| PyTorch FP32 | 44.8 MB | Training checkpoint |
| ONNX FP32 | 42.65 MB | Platform-independent export |
| ONNX INT8 | **10.71 MB** | **3.98x smaller, same accuracy** |

---

## Approach

### Progressive Layer Unfreezing
Rather than fine-tuning all layers at once, training proceeds in three stages of increasing depth. Each stage builds on the troubleshooting from the last one, only with more and more regularization.

1. **Stage 1** — Freeze the entire ResNet18 backbone. Train only the replaced FC head (10 classes). Adam, lr=1e-4.
2. **Stage 2** — Load Stage 1 weights. Unfreeze `layer4` and `layer3`. AdamW, lr=1e-4, weight_decay=1e-5.
3. **Stage 3** — Load Stage 2 weights. Unfreeze `layer2` and `layer1` (full network). AdamW, lr=1e-5, weight_decay=1e-7.

---

## Project Structure

```
cifar10_optimized/
├── train.py            # Stage 1 and Stage 2 training
├── train_s3.py         # Stage 3 training (full network fine-tune)
├── utils.py            # Shared: EarlyStopping, train loop, test loop, plotting
├── export_onnx.py      # PyTorch → ONNX export + ONNX INT8 quantization
├── quantize.py         # PyTorch native dynamic quantization
├── inference.py        # Run predictions with the INT8 ONNX model
├── requirements.txt
├── test_example.png    # Sample inference image (airplane)
└── results/
    ├── training_results.json   # All metrics across stages + ONNX export
    └── losses_s3.png           # Stage 3 training/validation loss curves
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

**Train (run in THIS SPECIFIC order):**
```bash
python train.py       # Runs Stage 1 then Stage 2, saves s1/s2 checkpoints
python train_s3.py    # Runs Stage 3, saves s3 checkpoint + results JSON
```

**Export and quantize:**
```bash
python export_onnx.py   # Exports ONNX FP32 + INT8, updates results JSON
python quantize.py      # PyTorch native INT8 quantization (standalone comparison)
```

**Run inference:**
```bash
python inference.py test_example.png
# Predicted: airplane
# Confidence: 94.32%
```

---

## Key Takeaways

- Progressive unfreezing improved val loss by **9x** from Stage 1 to Stage 3 (0.5691 → 0.0635)
- ONNX INT8 quantization cuts model size by ~4x with no measurable accuracy drop, making the model viable for edge and mobile deployment
- Separating PyTorch and ONNX quantization paths illustrates the trade-off between framework-native tooling and portable inference formats
