import sys
from pathlib import Path

import pytest
import torch
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def no_weight_downloads(monkeypatch):
    monkeypatch.delenv("DERMSCREEN_WEIGHTS", raising=False)
    original = models.mobilenet_v2

    def offline_mobilenet(*args, **kwargs):
        assert kwargs.get("weights") is None, "Tests must not request ImageNet weights"
        return original(*args, **kwargs)

    monkeypatch.setattr(models, "mobilenet_v2", offline_mobilenet)
