import io
import os
import sys

# Make the project root importable so `model` is found no matter where uvicorn is started
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from model.cifar_data import CLASSES
from model.predict import predict_digit, predict_image

app = FastAPI(title="CNN Image Classifier API")

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
