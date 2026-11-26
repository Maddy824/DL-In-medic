import asyncio
import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main
from model import SKIN_CLASSES


@pytest.fixture
def client(monkeypatch):
    def unexpected_load(*args, **kwargs):
        pytest.fail("The model must not be constructed without configured weights")

    monkeypatch.setattr(main, "load_model", unexpected_load)
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (180, 100, 50)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health(client):
    assert client.get("/api/health").json() == {
        "status": "ok", "model_loaded": False, "model_trained": False,
    }


def test_model_info(client):
    response = client.get("/api/model-info")
    assert response.status_code == 200
    info = response.json()
    assert info["model_trained"] is False
    assert info["classes"] == SKIN_CLASSES
    assert "risk_levels" not in info


def test_analyze_without_weights(client, png_bytes, monkeypatch):
    def unexpected_predict(*args, **kwargs):
        pytest.fail("The untrained model must not run inference")

    monkeypatch.setattr(main, "predict", unexpected_predict)
    response = client.post("/api/analyze", files={"file": ("image.png", png_bytes, "image/png")})
    assert response.status_code == 200
    assert response.json() == {"model_trained": False, "message": main.NO_MODEL_MESSAGE}


def test_augment_without_weights(client, png_bytes):
    response = client.post("/api/augment", files={"file": ("image.png", png_bytes, "image/png")})
    assert response.status_code == 200
    data = response.json()
    assert data["step_order"] == ["original", "normals", "perspective", "lighting", "elastic", "color", "augmented"]
    assert set(data["steps"]) == set(data["step_order"]) == set(data["step_labels"])
    for uri in data["steps"].values():
        prefix, encoded = uri.split(",", 1)
        assert prefix == "data:image/png;base64"
        with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
            assert image.size == (256, 256)
            image.verify()


@pytest.mark.parametrize("endpoint", ["analyze", "augment"])
def test_reject_non_image(client, endpoint):
    response = client.post(f"/api/{endpoint}", files={"file": ("fake.png", b"not an image", "image/png")})
    assert response.status_code == 400


@pytest.mark.parametrize("endpoint", ["analyze", "augment"])
@pytest.mark.parametrize("excess", [1, main.MAX_REQUEST_BYTES])
def test_upload_limit(client, endpoint, excess):
    response = client.post(f"/api/{endpoint}", files={
        "file": ("large.png", b"x" * (main.MAX_UPLOAD_BYTES + excess), "image/png"),
    })
    assert response.status_code == 413


def asgi_upload(endpoint, headers, chunks):
    """Deliver distinct ASGI messages without buffering through an HTTP transport."""
    received = []
    sent = []
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "scheme": "http", "method": "POST", "path": f"/api/{endpoint}",
        "query_string": b"", "headers": headers,
    }

    async def receive():
        index = len(received)
        assert index < len(chunks), "Request body was read after its final message"
        received.append(index)
        return {"type": "http.request", "body": chunks[index], "more_body": index < len(chunks) - 1}

    async def send(message):
        sent.append(message)

    asyncio.run(main.app(scope, receive, send))
    status = next(message["status"] for message in sent if message["type"] == "http.response.start")
    return status, received


@pytest.mark.parametrize("endpoint", ["analyze", "augment"])
def test_streamed_upload_limit(client, endpoint):
    boundary = "test-boundary"
    prefix = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="large.png"\r\n\r\n'.encode()
    suffix = f"\r\n--{boundary}--\r\n".encode()
    # Each message is within the limit; only their cumulative size exceeds it.
    chunk = b"x" * (1024 * 1024)
    chunks = [prefix, *([chunk] * 12), suffix]
    status, received = asgi_upload(endpoint, [(b"content-type", f"multipart/form-data; boundary={boundary}".encode())], chunks)
    assert status == 413
    assert received == list(range(12))  # Reject the 11th MiB without reading the rest.


def test_streamed_upload_within_limit(client, png_bytes):
    boundary = "test-boundary"
    prefix = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="image.png"\r\n\r\n'.encode()
    chunks = [prefix, png_bytes[:16], png_bytes[16:], f"\r\n--{boundary}--\r\n".encode()]
    status, received = asgi_upload("analyze", [(b"content-type", f"multipart/form-data; boundary={boundary}".encode())], chunks)
    assert status == 200
    assert received == list(range(len(chunks)))


@pytest.mark.parametrize("origin,allowed", [
    ("http://localhost:5173", True),
    ("http://127.0.0.1:5173", True),
    ("https://example.com", False),
])
def test_cors(client, origin, allowed):
    response = client.get("/api/health", headers={"Origin": origin})
    assert (response.headers.get("access-control-allow-origin") == origin) is allowed


@pytest.mark.parametrize("endpoint", ["analyze", "augment"])
@pytest.mark.parametrize("origin", [None, "http://localhost:5173", "http://127.0.0.1:5173"])
def test_upload_allowed_origins(client, png_bytes, endpoint, origin):
    response = client.post(f"/api/{endpoint}", files={"file": ("image.png", png_bytes, "image/png")},
                           headers={"Origin": origin} if origin is not None else {})
    assert response.status_code == 200


@pytest.mark.parametrize("endpoint", ["analyze", "augment"])
@pytest.mark.parametrize("origins", [
    [b"https://example.com"], [b"null"], [b""], [b"http://localhost:5174"],
    [b"http://localhost:5173", b"https://example.com"],
])
def test_upload_rejects_disallowed_origins_before_reading(endpoint, origins):
    headers = [(b"origin", origin) for origin in origins]
    headers.append((b"content-type", b"multipart/form-data; boundary=test-boundary"))
    status, received = asgi_upload(endpoint, headers, [])
    assert status == 403
    assert not received


def test_invalid_weights_fail_startup(monkeypatch, tmp_path):
    monkeypatch.setenv("DERMSCREEN_WEIGHTS", str(tmp_path / "missing.pt"))
    with pytest.raises(FileNotFoundError), TestClient(main.app):
        pytest.fail("Startup must fail with a missing checkpoint")


def test_upload_at_limit(client, png_bytes):
    padded = png_bytes.ljust(main.MAX_UPLOAD_BYTES, b"\x00")
    response = client.post("/api/analyze", files={"file": ("image.png", padded, "image/png")})
    assert response.status_code == 200
    assert response.json()["model_trained"] is False


@pytest.mark.parametrize("endpoint", ["analyze", "augment"])
def test_decoded_image_size_limit(client, png_bytes, endpoint, monkeypatch):
    monkeypatch.setattr(main, "MAX_IMAGE_PIXELS", 100)
    response = client.post(f"/api/{endpoint}", files={"file": ("image.png", png_bytes, "image/png")})
    assert response.status_code == 400
    assert "megapixel limit" in response.json()["detail"]
