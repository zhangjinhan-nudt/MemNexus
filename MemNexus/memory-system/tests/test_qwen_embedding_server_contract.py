"""Service contract tests; real GPU backend comparisons are archived separately."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient


@pytest.fixture
def service(monkeypatch):
    constructed = []
    encoded = []

    class FakeModel:
        device = torch.device("cpu")
        fail = False

        def __init__(self, *args, **kwargs):
            constructed.append((args, kwargs))
            self.evaluated = False

        def eval(self):
            self.evaluated = True
            return self

        def get_sentence_embedding_dimension(self):
            return 2

        def parameters(self):
            return iter([torch.zeros(2, dtype=torch.float32)])

        def __getitem__(self, index):
            assert index == 0
            return SimpleNamespace(auto_model=SimpleNamespace(config=SimpleNamespace(_attn_implementation="eager")))

        def encode(self, texts, **kwargs):
            encoded.append((texts, kwargs))
            if self.fail:
                raise RuntimeError("private inference diagnostic")
            return torch.tensor([[float(i == 0), float(i != 0)] for i in range(len(texts))]).numpy()

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeModel))
    path = Path(__file__).resolve().parents[1] / "scripts/qwen_embedding_server.py"
    spec = importlib.util.spec_from_file_location("qwen_embedding_server_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, TestClient(module.app), constructed, encoded


def test_model_configuration_and_cached_load(service):
    module, _, constructed, _ = service
    model = module.get_model()
    assert module.get_model() is model
    assert model.evaluated
    assert constructed == [((module.MODEL_NAME,), {"trust_remote_code": True,
        "model_kwargs": {"attn_implementation": "eager", "dtype": torch.float32}})]


def test_health_reports_actual_configuration_without_claiming_inference(service):
    _, client, _, encoded = service
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "readiness": "model_loaded_only",
                              "model": "Qwen/Qwen3-Embedding-0.6B", "dimension": 2,
                              "device": "cpu", "dtype": "torch.float32", "attention_implementation": "eager"}
    assert not encoded


@pytest.mark.parametrize("input_type", ["query", "document"])
def test_existing_role_prompts_and_response_order_preserved(service, input_type):
    _, client, _, encoded = service
    response = client.post("/v1/embeddings", json={"input": ["one", "two"], "input_type": input_type})
    assert response.status_code == 200
    assert [row["index"] for row in response.json()["data"]] == [0, 1]
    assert [row["embedding"] for row in response.json()["data"]] == [[1.0, 0.0], [0.0, 1.0]]
    expected = {"normalize_embeddings": True, "show_progress_bar": False}
    if input_type == "query":
        expected["prompt_name"] = "query"
    assert encoded == [(["one", "two"], expected)]


def test_inference_failure_logged_with_safe_http_error(service, caplog):
    module, client, _, _ = service
    module.get_model().fail = True
    response = client.post("/v1/embeddings", json={"input": "content must not appear in HTTP error"})
    assert response.status_code == 500
    assert response.json() == {"detail": "embedding generation failed"}
    assert "private inference diagnostic" not in response.text
    record = next(r for r in caplog.records if r.getMessage() == "Embedding inference failed")
    assert record.exc_info[0] is RuntimeError


@pytest.mark.parametrize("payload", [{"input": " "}, {"input": "valid", "model": "different-model"}])
def test_invalid_request_still_rejected_before_inference(service, payload):
    _, client, _, encoded = service
    response = client.post("/v1/embeddings", json=payload)
    assert response.status_code == 400
    assert not encoded
