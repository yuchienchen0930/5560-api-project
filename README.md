# 5560 Applied Generative AI – CNN Classifier and GAN Generator API

A FastAPI app that serves:

- **Assignment 2:** a CNN trained on CIFAR-10 that returns the class of an uploaded image.
- **Assignment 3:** a GAN trained on MNIST that generates handwritten digits.

## Project structure

```
api/main.py                FastAPI app
helper_lib/                Helper library from the class activities
  model.py                 FCNN, CNN, EnhancedCNN, Generator, Discriminator, GAN, get_model()
  trainer.py               train_model(), train_gan()
  generator.py             generate_samples(), generate_images()
  data_loader.py           MNIST data loader (normalized to [-1, 1])
  evaluator.py             evaluate_model()
model/cnn_model.py         AssignmentCNN (CIFAR-10) and SimpleCNN (MNIST)
model/cifar_data.py        CIFAR-10 dataset loader
model/predict.py           Classifier preprocessing and inference
model/cifar10_cnn.pth      Trained CIFAR-10 CNN weights
model/mnist_cnn.pth        Trained MNIST CNN weights
model/gan_generator.pth    Trained GAN generator weights
5560 Assignment 2.ipynb    CNN: architecture, training, evaluation, theory answers
5560 Assignment 3.ipynb    GAN: architecture, training, generated samples, theory answers
```

## Setup

```
pip install -r requirements.txt
```

All trained weights are included in `model/`, so the notebooks only need to be re-run to retrain.
Datasets are downloaded automatically to `data/`.

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
| GET | `/generate` | `num_samples` (1–100, default 16), optional `seed` | PNG grid of GAN-generated digits |

### Classify an image

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

### Generate digits

Open http://127.0.0.1:8000/generate?num_samples=16 in a browser, or:

```
curl -o digits.png "http://127.0.0.1:8000/generate?num_samples=16&seed=7"
```

## Assignment 2: CNN (`AssignmentCNN`)

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

825,398 trainable parameters. Trained for 10 epochs with Adam (lr 0.001) on CIFAR-10 resized to 64×64:
**68.95% test accuracy**.

## Assignment 3: GAN (`helper_lib/model.py`)

**Generator** – noise (N, 100) → Linear to 7×7×128 → reshape → ConvTranspose2d 128→64 (k4, s2, p1) + BatchNorm + ReLU (14×14)
→ ConvTranspose2d 64→1 (k4, s2, p1) + Tanh (28×28).

**Discriminator** – image (1, 28, 28) → Conv2d 1→64 (k4, s2, p1) + LeakyReLU(0.2) (14×14)
→ Conv2d 64→128 (k4, s2, p1) + BatchNorm + LeakyReLU(0.2) (7×7) → Flatten → Linear → 1 logit.

Trained on MNIST for 15 epochs with BCE loss and Adam (lr 0.0002, β₁ 0.5).

## Notes for Windows / Anaconda

`helper_lib/__init__.py` and `model/cnn_model.py` set `KMP_DUPLICATE_LIB_OK=TRUE` before importing PyTorch, to avoid
the `OMP: Error #15` crash caused by Anaconda and PyTorch each loading their own OpenMP runtime.
