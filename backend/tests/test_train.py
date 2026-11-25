import io
import json

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from main import app
from model import (
    SKIN_CLASSES,
    DermClassifier,
    NonFiniteInferenceError,
    load_model,
    predict,
)
from train import evaluate, make_dataset, train


def test_one_training_step_and_checkpoint(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "dataset"
    for split in ("train", "val"):
        for index, name in enumerate(SKIN_CLASSES):
            folder = data_dir / split / name
            folder.mkdir(parents=True)
            Image.new("RGB", (32, 32), (index * 30, 120 if split == "train" else 140, 80)).save(folder / "example.png")

    assert make_dataset(data_dir / "train").class_to_idx == {
        name: index for index, name in enumerate(SKIN_CLASSES)
    }

    steps = 0
    original_step = torch.optim.AdamW.step

    def counted_step(self, *args, **kwargs):
        nonlocal steps
        steps += 1
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", counted_step)
    output = tmp_path / "checkpoint.pt"
    metrics = train(data_dir, output, epochs=1, batch_size=7, pretrained=False)
    assert steps == 1
    assert output.is_file()
    assert 0 <= metrics["validation_accuracy"] <= 1
    recalls = metrics["per_class_recall"]
    assert list(recalls) == SKIN_CLASSES
    assert all(0 <= value <= 1 for value in recalls.values())
    assert metrics["balanced_accuracy"] == pytest.approx(sum(recalls.values()) / len(recalls))
    assert metrics["validation_accuracy"] == pytest.approx(metrics["balanced_accuracy"])
    assert json.loads(capsys.readouterr().out)["epoch"] == 1
    reloaded = load_model(output)
    assert not reloaded.training
    monkeypatch.setenv("DERMSCREEN_WEIGHTS", str(output))
    with TestClient(app) as client:
        assert client.get("/api/health").json()["model_trained"] is True
        assert client.get("/api/model-info").json()["model_trained"] is True
        buffer = io.BytesIO()
        Image.new("RGB", (32, 32)).save(buffer, format="PNG")
        response = client.post("/api/analyze", files={"file": ("image.png", buffer.getvalue(), "image/png")})
        assert response.status_code == 200
        result = response.json()
        assert result["model_trained"] is True
        assert result["prediction"] in SKIN_CLASSES
        assert {item["class"] for item in result["all_predictions"]} == set(SKIN_CLASSES)
        assert sum(item["probability"] for item in result["all_predictions"]) == pytest.approx(1, abs=0.001)
        assert "risk_level" not in result
        assert "confidence" not in result
        assert all("risk_level" not in item for item in result["all_predictions"])


def test_wrong_class_folders_rejected(tmp_path):
    folder = tmp_path / "wrong_class"
    folder.mkdir()
    Image.new("RGB", (32, 32)).save(folder / "example.png")
    with pytest.raises(ValueError, match="exactly these class folders"):
        make_dataset(tmp_path)



def test_metrics_on_imbalanced_validation_examples():
    # Predict the majority class for every example. Raw accuracy is high, but
    # balanced accuracy gives every class equal weight.
    class MajorityClassifier(torch.nn.Module):
        def forward(self, images):
            return torch.zeros((len(images), len(SKIN_CLASSES))), None

    labels = torch.tensor([0] * 10 + list(range(1, len(SKIN_CLASSES))))
    metrics = evaluate(MajorityClassifier(), [(torch.zeros(len(labels), 1), labels)], "cpu")
    assert metrics["validation_accuracy"] == pytest.approx(10 / 16)
    assert metrics["balanced_accuracy"] == pytest.approx(1 / 7)
    assert metrics["per_class_recall"][SKIN_CLASSES[0]] == 1
    assert all(metrics["per_class_recall"][name] == 0 for name in SKIN_CLASSES[1:])


def test_incompatible_checkpoint_rejected(tmp_path):
    path = tmp_path / "invalid.pt"
    torch.save({"unrelated": torch.zeros(2)}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_model(path)


@pytest.mark.parametrize("name,value", [
    ("classifier.9.weight", float("nan")),
    ("classifier.9.bias", float("inf")),
    ("classifier.2.running_mean", float("-inf")),
    ("classifier.9.bias", 1e100),  # Finite float64 overflows when copied to float32.
])
def test_nonfinite_checkpoint_fails_startup(tmp_path, monkeypatch, name, value):
    state_dict = DermClassifier(pretrained=False).state_dict()
    state_dict[name] = state_dict[name].double()
    state_dict[name].view(-1)[0] = value
    path = tmp_path / "nonfinite.pt"
    torch.save(state_dict, path)
    monkeypatch.setenv("DERMSCREEN_WEIGHTS", str(path))
    with pytest.raises(ValueError, match="non-finite"), TestClient(app):
        pytest.fail("Startup must reject non-finite checkpoint tensors")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_inference_rejected(monkeypatch, value):
    class InvalidClassifier(torch.nn.Module):
        def forward(self, images):
            logits = torch.zeros((len(images), len(SKIN_CLASSES)))
            logits[:, 0] = value
            return logits, None

    model = InvalidClassifier()
    with pytest.raises(NonFiniteInferenceError, match="logits"):
        predict(model, Image.new("RGB", (32, 32)))

    with TestClient(app) as client:
        monkeypatch.setattr(app.state, "model", model)
        buffer = io.BytesIO()
        Image.new("RGB", (32, 32)).save(buffer, format="PNG")
        response = client.post("/api/analyze", files={"file": ("image.png", buffer.getvalue(), "image/png")})
        assert response.status_code == 500
        assert response.json() == {"detail": "Model produced non-finite class scores"}


def test_nonfinite_softmax_scores_rejected(monkeypatch):
    class FiniteClassifier(torch.nn.Module):
        def forward(self, images):
            return torch.zeros((len(images), len(SKIN_CLASSES))), None

    monkeypatch.setattr(torch.nn.functional, "softmax", lambda logits, **kwargs: torch.full_like(logits, float("nan")))
    with pytest.raises(NonFiniteInferenceError, match="probabilities"):
        predict(FiniteClassifier(), Image.new("RGB", (32, 32)))


@pytest.mark.parametrize("lr", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_learning_rate_rejected(tmp_path, lr):
    with pytest.raises(ValueError, match="finite positive learning rate"):
        train(tmp_path / "not-read", tmp_path / "checkpoint.pt", lr=lr, pretrained=False)
