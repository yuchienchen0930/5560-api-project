# 5560 Assignment 2 – CNN Image Classifier API

A Convolutional Neural Network trained on CIFAR-10 and served through a FastAPI endpoint that returns the class of an uploaded image.

## Model architecture (`model/cnn_model.py` → `AssignmentCNN`)

| Layer | Output shape |
|-------|--------------|
| Input: RGB image | 64 × 64 × 3 |
| Conv2D 16 filters, 3×3, stride 1, padding 1 + ReLU | 64 × 64 × 16 |
| MaxPool2D 2×2, stride 2 | 32 × 32 × 16 |
| Conv2D 32 filters, 3×3, stride 1, padding 1 + ReLU | 32 × 32 × 32 |
| MaxPool2D 2×2, stride 2 | 16 × 16 × 32 |
| Flatten | 8192 |
| Fully connected 100 + ReLU | 100 |
| Fully connected 10 | 10 |

Total trainable parameters: 825,398.

## Project structure

```
api/main.py             FastAPI app
model/cnn_model.py      AssignmentCNN (CIFAR-10) and SimpleCNN (MNIST)
model/cifar_data.py     CIFAR-10 dataset loader
model/predict.py        Preprocessing and inference
model/cifar10_cnn.pth   Trained CIFAR-10 weights
model/mnist_cnn.pth     Trained MNIST weights
5560 Assignment 2.ipynb Training, evaluation and theory answers
```

## Setup

```
pip install -r requirements.txt
```

Train the model by running `5560 Assignment 2.ipynb` (CIFAR-10 is downloaded automatically to `data/cifar10/`).
The trained weights are already included in `model/`, so this step is optional for running the API.

## Run the API

From the project root:

```
python -m uvicorn api.main:app --reload
```

Then open http://127.0.0.1:8000/docs to try it in the browser.

| Method | Endpoint | Input | Output |
|--------|----------|-------|--------|
| GET | `/` | – | health check |
| GET | `/classes` | – | the 10 CIFAR-10 classes |
| POST | `/predict` | image file (form field `file`) | predicted CIFAR-10 class, confidence, top 3 |
| POST | `/predict-digit` | image file (form field `file`) | predicted MNIST digit and confidence |

Example:

```
curl -X POST -F "file=@cat.png" http://127.0.0.1:8000/predict
```

```json
{
  "filename": "cat.png",
  "prediction": "cat",
  "class_index": 3,
  "confidence": 0.9689,
  "top_3": [
    {"class": "cat", "confidence": 0.9689},
    {"class": "dog", "confidence": 0.0299},
    {"class": "ship", "confidence": 0.0004}
  ]
}
```

## Results

Trained for 10 epochs with Adam (lr 0.001) on CIFAR-10 resized to 64×64: **68.95% test accuracy**.
See the notebook for the training curves, per-class report and confusion matrix.

CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## Notes for Windows / Anaconda

`model/cnn_model.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` before importing PyTorch, to avoid the
`OMP: Error #15` crash caused by Anaconda and PyTorch each loading their own OpenMP runtime.
