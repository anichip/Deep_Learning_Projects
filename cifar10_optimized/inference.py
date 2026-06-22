import torch 
import torch.nn as nn 
import torch.onnx as onnx
import onnxruntime as ort 
import numpy as np
import json
import os
from PIL import Image
import torchvision.transforms.v2 as transforms

def predict(image_path,model_path="results/cifar_classifier_int8.onnx"):

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

    #1. load int8 onnx model 
    session = ort.InferenceSession("results/cifar_classifier_int8.onnx")

    #2. take image path as input
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image)
    img_numpy = img_tensor.unsqueeze(0).numpy() #make it (1,3,224,224)

    #3. run inference
    logits = session.run(["output"], {"input": img_numpy})[0] #[0] gets the array of logits, how many ever output neurons

    #softmax converts logits to probabilities
    probabilities = np.exp(logits) / np.sum(np.exp(logits))
    confidence = np.max(probabilities)

    predicted_idx = np.argmax(probabilities)
    predicted_class = classes[predicted_idx]

    return predicted_class, confidence

#########################################
if __name__=="__main__":
    predicted_class,confidence = predict("test_example.png","results/cifar_classifier_int8.onnx")
    print(f"Predicted: {predicted_class}")
    print(f"Confidence: {confidence*100:.2f}%")