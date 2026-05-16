from physics_steering_vectors import steering
from physics_steering_vectors.config import ExperimentConfig
from physics_steering_vectors.schemas import ModelBundle
from physics_steering_vectors.steering import train_vector_for_layer


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
