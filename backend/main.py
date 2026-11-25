"""Local API for the DermScreen computer-vision demo."""

import base64
import io
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from augmentation import DermatologyAugmentor3D
from model import (
    SKIN_CLASSES,
    NonFiniteInferenceError,
    get_preprocessing_transform,
    load_model,
    predict,
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# Allow multipart headers in addition to the file itself.
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 64 * 1024
MAX_IMAGE_PIXELS = 25_000_000
ALLOWED_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")
NO_MODEL_MESSAGE = "No trained model is loaded. Classification is unavailable."
DISCLAIMER = "Educational model output only; not clinically validated or a medical risk estimate."


class UploadLimitMiddleware:
    """Reject disallowed browser origins and oversized upload bodies before processing."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] not in ("/api/analyze", "/api/augment"):
            await self.app(scope, receive, send)
            return

        origins = [value.decode("latin-1") for key, value in scope["headers"] if key.lower() == b"origin"]
        if origins and (len(origins) != 1 or origins[0] not in ALLOWED_ORIGINS):
            response = JSONResponse({"detail": "Upload origin is not allowed"}, status_code=403)
            await response(scope, receive, send)
            return

        headers = dict(scope["headers"])
        length = headers.get(b"content-length")
        if length is not None:
            try:
                size = int(length)
            except ValueError:
                size = -1
            if size < 0 or size > MAX_REQUEST_BYTES:
                response = JSONResponse({"detail": "Upload exceeds the 10 MiB limit"}, status_code=413)
                await response(scope, receive, send)
                return

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > MAX_REQUEST_BYTES:
                raise HTTPException(status_code=413, detail="Upload exceeds the 10 MiB limit")
            return message

        await self.app(scope, limited_receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enable inference only when an explicit checkpoint loads successfully."""
    weights_path = os.environ.get("DERMSCREEN_WEIGHTS")
    app.state.model = None
    app.state.transform = None
    app.state.augmentor = DermatologyAugmentor3D(severity="medium")
    if weights_path:
        app.state.model = load_model(weights_path, device="cpu")
        app.state.transform = get_preprocessing_transform()
    yield
    app.state.model = None
    app.state.transform = None


app = FastAPI(
    title="DermScreen AI",
    description="Educational computer vision with 3D-aware augmentation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(UploadLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def pil_to_base64(img: Image.Image, fmt="PNG") -> str:
    """Convert PIL image to base64 data URI."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


async def read_upload_image(file: UploadFile) -> Image.Image:
    """Limit file size and validate decoded image content, not the filename."""
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload exceeds the 10 MiB limit")
    file_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload exceeds the 10 MiB limit")
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            if img.width * img.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=400, detail="Image exceeds the 25 megapixel limit")
            img.verify()
        with Image.open(io.BytesIO(file_bytes)) as img:
            return img.convert("RGB")
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc


@app.get("/api/health")
async def health():
    trained = app.state.model is not None
    return {"status": "ok", "model_loaded": trained, "model_trained": trained}


@app.get("/api/model-info")
async def model_info():
    return {
        "model_name": "DermClassifier v1.0",
        "model_trained": app.state.model is not None,
        "backbone": "MobileNetV2",
        "classes": SKIN_CLASSES,
        "input_size": "224x224",
        "augmentation": "3D perspective, lighting, elastic, color",
        "note": DISCLAIMER,
    }


@app.post("/api/analyze")
async def analyze(file: Annotated[UploadFile, File()]):
    """Return experimental classification only with a loaded checkpoint."""
    image = await read_upload_image(file)
    if app.state.model is None:
        return {"model_trained": False, "message": NO_MODEL_MESSAGE}

    start = time.perf_counter()
    try:
        result = predict(app.state.model, image, transform=app.state.transform, device="cpu")
    except NonFiniteInferenceError as exc:
        raise HTTPException(status_code=500, detail="Model produced non-finite class scores") from exc
    return {
        "id": str(uuid.uuid4()),
        "model_trained": True,
        **result,
        "inference_time_ms": round((time.perf_counter() - start) * 1000, 1),
        "disclaimer": DISCLAIMER,
    }


@app.post("/api/augment")
async def augment(file: Annotated[UploadFile, File()]):
    """Apply the augmentation pipeline independently of classification."""
    image = (await read_upload_image(file)).resize((256, 256))
    augmented, steps = app.state.augmentor.augment(image, return_steps=True)
    step_images = {name: pil_to_base64(img) for name, img in steps.items()}
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

    uvicorn.run("main:app", host="127.0.0.1", port=8000)
