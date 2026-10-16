import os

import torch
from torchvision import transforms
from PIL import Image, ImageOps

from model.cnn_model import AssignmentCNN, SimpleCNN
from model.cifar_data import CLASSES

# Resolve weight files relative to this file so the API works from any working directory
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- CIFAR-10 classifier (Assignment 2) ----------
cifar_model = AssignmentCNN(num_classes=10)
cifar_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "cifar10_cnn.pth"), map_location="cpu"))
cifar_model.eval()

# Must match the test-time preprocessing used during training
cifar_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def predict_image(image: Image.Image, top_k=3):
    tensor = cifar_transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(cifar_model(tensor), dim=1)[0]
    top_probs, top_idx = probs.topk(top_k)
    return [(CLASSES[i], int(i), float(p)) for p, i in zip(top_probs, top_idx)]

# ---------- MNIST digit classifier (earlier class activity) ----------
digit_model = SimpleCNN()
digit_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "mnist_cnn.pth"), map_location="cpu"))
digit_model.eval()

digit_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_digit(image: Image.Image):
    image = image.convert("L")
    # MNIST digits are white on a black background; invert images with a light background
    if sum(image.resize((28, 28)).getdata()) / (28 * 28) > 127:
        image = ImageOps.invert(image)

    tensor = digit_transform(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(digit_model(tensor), dim=1)[0]
        predicted = int(torch.argmax(probs).item())
    return predicted, float(probs[predicted].item())
