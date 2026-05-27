import torch
from steering_vectors import SteeringVector

from physics_steering_vectors import steering
from physics_steering_vectors.config import ExperimentConfig
from physics_steering_vectors.logging_utils import configure_logging
from physics_steering_vectors.schemas import ModelBundle
from physics_steering_vectors.steering import load_steering_vector, save_steering_vector, steering_vector_path, train_vector_for_layer


def test_train_vector_for_layer_calls_library_with_project_defaults(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_train_steering_vector(model: object, tokenizer: object, training_pairs: object, **kwargs: object) -> str:
        calls["model"] = model
        calls["tokenizer"] = tokenizer
        calls["training_pairs"] = training_pairs
        calls["kwargs"] = kwargs
        return "vector"

    monkeypatch.setattr(steering, "train_steering_vector", fake_train_steering_vector)
    bundle = ModelBundle(
        model="model",
        processor=None,
        tokenizer="tokenizer",
        layer_config={"decoder_block": "model.layers.{num}"},
    )
    pairs = [("positive", "negative")]

    result = train_vector_for_layer(ExperimentConfig(train_batch_size=3), bundle, pairs, layer=8)

    assert result == "vector"
    assert calls["model"] == "model"
    assert calls["tokenizer"] == "tokenizer"
    assert calls["training_pairs"] == pairs
    assert calls["kwargs"] == {
        "layers": [8],
        "layer_type": "decoder_block",
        "layer_config": {"decoder_block": "model.layers.{num}"},
        "read_token_index": -1,
        "batch_size": 3,
        "show_progress": True,
    }


def test_train_vector_for_layer_logs_full_library_inputs(monkeypatch, capsys) -> None:
    monkeypatch.setattr(steering, "train_steering_vector", lambda *args, **kwargs: "vector")
    bundle = ModelBundle(
        model="model",
        processor=None,
        tokenizer="tokenizer",
        layer_config={"decoder_block": "model.layers.{num}"},
    )
    config = ExperimentConfig(log_level="DEBUG", log_full_text=True)
    configure_logging(config)

    train_vector_for_layer(
        config,
        bundle,
        [("FULL POSITIVE TEXT\nanswer is (A)", "FULL NEGATIVE TEXT\nanswer is (B)")],
        layer=8,
    )

    output = capsys.readouterr().out
    assert "steering_vector_library_input layer=8 pair_index=0 side=positive" in output
    assert "FULL POSITIVE TEXT\nanswer is (A)" in output
    assert "steering_vector_library_input layer=8 pair_index=0 side=negative" in output
    assert "FULL NEGATIVE TEXT\nanswer is (B)" in output


def test_steering_vector_path_uses_configured_artifact_directory() -> None:
    config = ExperimentConfig(model_id="org/model", subject="physics", seed=7, steering_vector_dir="custom/vectors")

    assert steering_vector_path(config, layer=12).as_posix() == "custom/vectors/org__model_physics_seed_7_layer_12.pt"


def test_save_and_load_steering_vector_round_trips_tensor_state(tmp_path) -> None:
    vector = SteeringVector(
        layer_activations={
            8: torch.tensor([1.0, 2.0]),
            9: torch.tensor([3.0, 4.0]),
        },
        layer_type="decoder_block",
    )
    path = tmp_path / "nested" / "vector.pt"

    saved_path = save_steering_vector(vector, path, metadata={"layer": 8})
    loaded = load_steering_vector(saved_path)
    payload = torch.load(saved_path, map_location="cpu", weights_only=True)

    assert saved_path == path
    assert path.exists()
    assert payload["format_version"] == steering.VECTOR_FILE_VERSION
    assert payload["metadata"] == {"layer": 8}
    assert loaded.layer_type == vector.layer_type
    assert loaded.layer_activations.keys() == vector.layer_activations.keys()
    assert torch.equal(loaded.layer_activations[8], vector.layer_activations[8])
    assert torch.equal(loaded.layer_activations[9], vector.layer_activations[9])
