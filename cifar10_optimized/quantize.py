# Dynamic Quantization --> Quantize weights and biases, and then activations are quantized during training
import torch 
import torch.nn as nn
from torchvision import models
import torch.quantization
import os

main_num_classes = 10

# part 1. imports and load best model (s3) into ResNet18 architecture

# load with the same structure so that it matches up with s3 when we load it back in 
model_o = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)

# no need for freeze and unfreeze because we aren't training

orig_num_feats = model_o.fc.in_features
model_o.fc = nn.Linear(orig_num_feats, main_num_classes)

model_o.load_state_dict(torch.load("s3_best_model.pth",weights_only=True))
model_o.eval()
print("checkpoint: loaded original best model. ready to quantize")

# REMEMBER: Train on GPU. Quantize on CPU.
# part 2. QUANTIZE
torch.backends.quantized.engine = "qnnpack"

model_quant = torch.quantization.quantize_dynamic(
    model_o,
    {nn.Linear},
    dtype=torch.qint8
)

# part 3. Size and Speed comparison
torch.save(model_o.state_dict(),"results/model_fp32.pth")
torch.save(model_quant.state_dict(),"results/model_int8.pth")

fp32_size = os.path.getsize("results/model_fp32.pth")/(1024*1024)
int8_size = os.path.getsize("results/model_int8.pth")/(1024*1024)

print(f"FP32: {fp32_size:.2f} MB")
print(f"INT8: {int8_size:.2f} MB")
print(f"Reduction: {fp32_size/int8_size:.2f}x smaller")






