"""
FastAPI Backend for Real-Time Dermatological Screening
Endpoints:
  POST /api/analyze       - Analyze a skin lesion image
  POST /api/augment       - Apply 3D augmentation and return steps
  GET  /api/health        - Health check
  GET  /api/model-info    - Model metadata
"""

import io
import base64
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model import load_model, predict, get_preprocessing_transform, SKIN_CLASSES, RISK_LEVELS
from augmentation import DermatologyAugmentor3D

# ── Global state ──────────────────────────────────────────────────────────────
model = None
transform = None
augmentor = None
device = "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model, transform, augmentor
    print("Loading DermClassifier model...")
    model = load_model(weights_path=None, device=device)
    transform = get_preprocessing_transform()
    augmentor = DermatologyAugmentor3D(severity="medium")
    print(f"Model loaded on {device} — ready for inference.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="DermScreen AI",
    description="Real-time dermatological screening with 3D augmentation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def pil_to_base64(img: Image.Image, fmt="PNG") -> str:
    """Convert PIL image to base64 data URI."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


def read_upload_image(file_bytes: bytes) -> Image.Image:
    """Read uploaded bytes into RGB PIL Image."""
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return img
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/api/model-info")
async def model_info():
    return {
        "model_name": "DermClassifier v1.0",
        "backbone": "MobileNetV2 (ImageNet pretrained)",
        "classes": SKIN_CLASSES,
        "risk_levels": RISK_LEVELS,
        "input_size": "224x224",
        "augmentation": "3D perspective, lighting, elastic, color",
        "note": "Demo model — not for clinical use. Always consult a dermatologist.",
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Analyze a skin lesion image.
    Returns classification results with probabilities and risk assessment.
    """
    start = time.time()
    file_bytes = await file.read()
    image = read_upload_image(file_bytes)

    # Run inference
    result = predict(model, image, transform=transform, device=device)
    elapsed = time.time() - start

    return {
        "id": str(uuid.uuid4()),
        "prediction": result["prediction"],
        "confidence": round(result["confidence"], 4),
        "top_probability": round(result["top_probability"], 4),
        "risk_level": result["risk_level"],
        "description": result["description"],
        "all_predictions": [
            {
                "class": p["class"],
                "probability": round(p["probability"], 4),
                "risk_level": p["risk_level"],
                "description": p["description"],
            }
            for p in result["all_predictions"]
        ],
        "inference_time_ms": round(elapsed * 1000, 1),
        "disclaimer": "This is a screening tool only. Not a medical diagnosis. Please consult a healthcare professional.",
    }


@app.post("/api/augment")
async def augment(file: UploadFile = File(...)):
    """
    Apply 3D augmentation pipeline and return each step as base64 images.
    Useful for visualizing the augmentation process.
    """
    file_bytes = await file.read()
    image = read_upload_image(file_bytes)

    # Resize for faster processing
    image = image.resize((256, 256))

    augmented, steps = augmentor.augment(image, return_steps=True)

    step_images = {}
    for name, img in steps.items():
        step_images[name] = pil_to_base64(img)

    step_images["augmented"] = pil_to_base64(augmented)

    return {
        "steps": step_images,
        "step_order": ["original", "normals", "perspective", "lighting", "elastic", "color", "augmented"],
        "step_labels": {
            "original": "Original Image",
            "normals": "Surface Normals",
            "perspective": "3D Perspective Warp",
            "lighting": "Lighting Simulation",
            "elastic": "Elastic Deformation",
            "color": "Color Augmentation",
            "augmented": "Final Augmented",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
