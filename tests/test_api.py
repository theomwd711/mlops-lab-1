
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.food11.serve as serve
from src.food11.data import CLASSES

FIXTURE = Path(__file__).parent / "fixtures" / "bread.jpg"


class FakeModel:
    """Returns fixed logits where class 0 ('Bread') wins."""

    def predict(self, x):
        return [[5.0] + [0.0] * 10]


def post_image(client):
    with open(FIXTURE, "rb") as f:
        return client.post("/predict", files={"file": ("bread.jpg", f, "image/jpeg")})


def test_health(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as client:
        assert client.get("/health").status_code == 200


def test_predict_returns_valid_json(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as client:
        monkeypatch.setattr(serve, "model", FakeModel())
        r = post_image(client)
    assert r.status_code == 200
    body = r.json()
    assert body["category"] == "Bread"
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["confidence"] == pytest.approx(0.9369, abs=1e-3)


def test_predict_without_model_returns_503(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as client:
        monkeypatch.setattr(serve, "model", None)
        r = post_image(client)
    assert r.status_code == 503


def test_serve_categories_match_data_classes():
    assert serve.CATEGORIES == CLASSES
