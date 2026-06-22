import torch 
import torch.nn as nn 
import torch.onnx as onnx
import onnxruntime as ort 
from onnxruntime.quantization import quantize_dynamic, QuantType
from torchvision import models
import numpy as np
import json
import os

# we need to load the model architecture and best model weights in pytorch first.
main_num_classes = 10
model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
orig_num_feats = model.fc.in_features
model.fc = nn.Linear(orig_num_feats, main_num_classes)
model.load_state_dict(torch.load("s3_best_model.pth",weights_only=True))
model.eval()
print("checkpoint: loaded original model")

########################### PyTorch --> ONNX export ###########################
dummy_input = torch.randn(1,3,224,224)
torch.onnx.export(
    model,
    dummy_input,
    "results/cifar_classifier.onnx",
    input_names = ["input"],
    output_names = ["output"],
    dynamic_axes = {"input":{0:"batch_size"}, "output":{0:"batch_size"}}
)

print("checkpoint: exported to onnx")

########################### Load onnx model ###########################
session = ort.InferenceSession("results/cifar_classifier.onnx")

########################### Inference ###########################
test_input = np.random.randn(1,3,224,224).astype(np.float32)
result = session.run(
    ["output"],
    {"input":test_input}
)
print(f"Logits: {result[0]}")

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
predicted_class = np.argmax(result[0], axis=1)
print(f"Predicted class: {classes[predicted_class[0]]}")

import json

########################### LEt ONNX handle its own quantization ##########################
quantize_dynamic(
    "results/cifar_classifier.onnx", 
    "results/cifar_classifier_int8.onnx",
    weight_type = QuantType.QInt8
)

fp32_size = os.path.getsize("results/cifar_classifier.onnx") / (1024*1024)
int8_size = os.path.getsize("results/cifar_classifier_int8.onnx") / (1024*1024)

print(f"FP32 ONNX: {fp32_size:.2f} MB")
print(f"INT8 ONNX: {int8_size:.2f} MB")
print(f"Reduction: {fp32_size/int8_size:.2f}x smaller!")

########################### save to json ##########################

with open("results/training_results.json", "r") as f:
    results = json.load(f)

results["onnx_export"] = "results/cifar_classifier.onnx"
results["onnx_input_shape"] = [1, 3, 224, 224]
results["onnx_fp32_size_mb"] = round(fp32_size, 2)
results["onnx_int8_size_mb"] = round(int8_size, 2)
results["onnx_size_reduction"] = round(fp32_size/int8_size, 2)

with open("results/training_results.json", "w") as f:
    json.dump(results, f, indent=2)