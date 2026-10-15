import os

import torch
from torchvision import transforms
from PIL import Image, ImageOps

from model.cnn_model import SimpleCNN

# Resolve model.pth relative to this file so the API works from any working directory
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pth")

model = SimpleCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Must match the preprocessing used during training (ToTensor + Normalize(0.5, 0.5))
transform = transforms.Compose([
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

    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        predicted = int(torch.argmax(probs).item())
    return predicted, float(probs[predicted].item())
