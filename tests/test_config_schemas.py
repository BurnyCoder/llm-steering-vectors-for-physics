from dataclasses import FrozenInstanceError

import pytest

from physics_steering_vectors.config import ExperimentConfig
from physics_steering_vectors.schemas import BenchmarkSplits, EvaluationRecord, EvaluationResult, ModelBundle


def test_experiment_config_defaults_capture_protocol_settings() -> None:
    config = ExperimentConfig()

    assert config.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config.dataset_id == "TIGER-Lab/MMLU-Pro"
    assert config.subject == "physics"
    assert config.layer_sweep == (6, 12, 18)
    assert config.multipliers == (0.5, 1.0, 1.5, 2.0)
    assert config.steering_vector_dir == "artifacts/steering_vectors"


def test_experiment_config_is_frozen() -> None:
    config = ExperimentConfig()

    with pytest.raises(FrozenInstanceError):
        config.seed = 1  # type: ignore[misc]


def test_shared_dataclasses_store_phase_boundaries() -> None:
    bundle = ModelBundle(model="model", processor="processor", tokenizer="tokenizer", layer_config={"decoder_block": "x.{num}"})
    splits = BenchmarkSplits(validation=[{"id": 1}], test=[{"id": 2}], fewshot_prefix="prefix")
    record = EvaluationRecord(question_id=7, gold="A", prediction="B", is_correct=False)
    result = EvaluationResult(label="baseline", correct=0, total=1, accuracy=0.0, records=[record])

    assert bundle.layer_config["decoder_block"] == "x.{num}"
    assert splits.validation == [{"id": 1}]
    assert result.records == [record]
