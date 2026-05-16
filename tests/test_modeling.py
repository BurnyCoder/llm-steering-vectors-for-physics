import torch
from torch import nn

from physics_steering_vectors import modeling
from physics_steering_vectors.config import ExperimentConfig
from physics_steering_vectors.modeling import load_qwen35_bundle, model_device


def test_model_device_uses_explicit_device_without_requiring_parameters() -> None:
    class DeviceOnlyModel:
        device = "cpu"

        def parameters(self) -> object:
            raise AssertionError("parameters should not be inspected when device is present")

    assert model_device(DeviceOnlyModel()) == torch.device("cpu")


def test_model_device_falls_back_to_first_parameter_device() -> None:
    model = nn.Linear(1, 1)

    assert model_device(model) == next(model.parameters()).device


def test_load_qwen35_bundle_configures_model_tokenizer_and_layer_config(monkeypatch) -> None:
    class FakeTokenizer:
        pad_token_id = None
        eos_token = "<eos>"
        pad_token = None
        padding_side = "right"

    class FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = FakeTokenizer()

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_id: str) -> FakeProcessor:
            calls["processor_model_id"] = model_id
            return FakeProcessor()

    class FakeModel:
        def eval(self) -> "FakeModel":
            calls["eval_called"] = True
            return self

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeModel:
            calls["model_model_id"] = model_id
            calls["model_kwargs"] = kwargs
            return FakeModel()

    calls: dict[str, object] = {}
    monkeypatch.setattr(modeling, "AutoProcessor", FakeAutoProcessor)
    monkeypatch.setattr(modeling, "AutoModelForImageTextToText", FakeAutoModel)
    monkeypatch.setattr(modeling.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(modeling, "infer_decoder_block_template", lambda model: "model.layers.{num}")

    bundle = load_qwen35_bundle(ExperimentConfig(model_id="fake/model"))

    assert calls["processor_model_id"] == "fake/model"
    assert calls["model_model_id"] == "fake/model"
    assert calls["model_kwargs"] == {"torch_dtype": "auto", "device_map": "auto"}
    assert calls["eval_called"] is True
    assert bundle.tokenizer.pad_token == "<eos>"
    assert bundle.tokenizer.padding_side == "left"
    assert bundle.layer_config == {"decoder_block": "model.layers.{num}"}
