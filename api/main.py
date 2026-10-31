import io
import os
import sys

# Make the project root importable so `model` is found no matter where uvicorn is started
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import helper_lib  # noqa: F401  (sets the OpenMP workaround before torch is imported)

import math
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from torchvision.utils import make_grid

from helper_lib.generator import generate_images
from helper_lib.model import get_model
from model.cifar_data import CLASSES
from model.predict import predict_digit, predict_image

app = FastAPI(title="CNN Image Classifier and GAN Generator API")

# GAN generator trained on MNIST (Assignment 3)
GAN_WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "gan_generator.pth")
gan = get_model("GAN")
gan.generator.load_state_dict(torch.load(GAN_WEIGHTS, map_location="cpu"))
gan.generator.eval()

def read_image(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

@app.get("/")
def home():
    return {"message": "CNN Image Classifier API is running"}

@app.get("/classes")
def classes():
    return {"classes": CLASSES}

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    """Classify an image into one of the 10 CIFAR-10 classes."""
    image = read_image(await file.read())
    top = predict_image(image)
    label, index, confidence = top[0]
    return {
        "filename": file.filename,
        "prediction": label,
        "class_index": index,
        "confidence": round(confidence, 4),
        "top_3": [{"class": c, "confidence": round(p, 4)} for c, _, p in top],
    }

@app.post("/predict-digit")
async def predict_digit_api(file: UploadFile = File(...)):
    """Classify a handwritten digit (0-9) using the MNIST CNN."""
    image = read_image(await file.read())
    prediction, confidence = predict_digit(image)
    return {
        "filename": file.filename,
        "prediction": prediction,
        "confidence": round(confidence, 4),
    }

@app.get("/generate", responses={200: {"content": {"image/png": {}}}})
def generate(num_samples: int = Query(16, ge=1, le=100), seed: int | None = None):
    """Generate handwritten digits with the GAN and return them as a PNG grid."""
    images = generate_images(gan, num_samples=num_samples, seed=seed)
    grid = make_grid(images, nrow=math.ceil(math.sqrt(num_samples)), padding=2, pad_value=1)
    array = (grid[0].numpy() * 255).clip(0, 255).astype("uint8")
    # Upscale 4x so the 28x28 digits are easy to see in a browser
    image = Image.fromarray(array)
    image = image.resize((image.width * 4, image.height * 4), Image.NEAREST)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
