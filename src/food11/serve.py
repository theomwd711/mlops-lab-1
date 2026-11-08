import io
import os

import mlflow
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

# Models logged from a GPU run hold CUDA tensors; remap to CPU so CPU-only containers can load them.
_original_torch_load = torch.load


def _torch_load_cpu(f, *args, **kwargs):
    kwargs.setdefault("map_location", torch.device("cpu"))
    return _original_torch_load(f, *args, **kwargs)


torch.load = _torch_load_cpu

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

CATEGORIES = [
    "Bread", "Dairy product", "Dessert", "Egg", "Fried food",
    "Meat", "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit",
]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

app = FastAPI()
model = None


@app.on_event("startup")
def load_model():
    global model
    if os.environ.get("SKIP_MODEL"):
        return
    model = mlflow.pyfunc.load_model("models:/food11@champion")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # add batch dim

    output = model.predict(input_tensor.numpy())
    logits = torch.tensor(output)
    probs = torch.softmax(logits, dim=1)
    confidence, predicted_idx = torch.max(probs, dim=1)

    return {
        "category": CATEGORIES[predicted_idx.item()],
        "confidence": round(confidence.item(), 4),
    }