import io
import os
import sys

# Make the project root importable so `model` is found no matter where uvicorn is started
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from model.predict import predict_digit

app = FastAPI(title="MNIST CNN Image Classifier API")

@app.get("/")
def home():
    return {"message": "MNIST CNN API is running"}

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    prediction, confidence = predict_digit(image)
    return {
        "filename": file.filename,
        "prediction": prediction,
        "confidence": round(confidence, 4),
    }
