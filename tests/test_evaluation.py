import pytest

from physics_steering_vectors import evaluation
from physics_steering_vectors.config import ExperimentConfig
from physics_steering_vectors.logging_utils import configure_logging


def test_evaluate_generates_extracts_and_scores_rows(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    config = ExperimentConfig()
    configure_logging(config)
    rows = [
        {"question_id": "10", "question": "Q1", "options": ["a", "b"], "answer": "A"},
        {"question_id": "11", "question": "Q2", "options": ["a", "b", "c"], "answer": "B"},
    ]
    completions = iter(["Answer: A", "Answer: C"])
    calls: list[dict[str, object]] = []

    def fake_generate_completion(
        config_arg: ExperimentConfig,
        bundle: object,
        prompt: str,
        steering_vector: object | None = None,
        multiplier: float = 1.0,
        log_context: str | None = None,
    ) -> str:
        calls.append(
            {
                "config": config_arg,
                "bundle": bundle,
                "prompt": prompt,
                "steering_vector": steering_vector,
                "multiplier": multiplier,
                "log_context": log_context,
            }
        )
        return next(completions)

    monkeypatch.setattr(evaluation, "generate_completion", fake_generate_completion)
    monkeypatch.setattr(evaluation, "tqdm", lambda iterable, desc: iterable)
    bundle = object()

    result = evaluation.evaluate(
        config=config,
        bundle=bundle,
        rows=rows,
        fewshot_prefix="FEWSHOT\n\n",
        label="steered",
        steering_vector="vector",
        multiplier=2.0,
    )

    assert result.label == "steered"
    assert result.correct == 1
    assert result.total == 2
    assert result.accuracy == 0.5
    assert [record.question_id for record in result.records] == [10, 11]
    assert [record.prediction for record in result.records] == ["A", "C"]
    assert calls[0]["prompt"].startswith("FEWSHOT\n\nQuestion:\nQ1")
    assert all(call["steering_vector"] == "vector" for call in calls)
    assert all(call["multiplier"] == 2.0 for call in calls)
    assert calls[0]["log_context"] == "evaluation label=steered row_index=0 question_id=10"
    output = capsys.readouterr().out
    assert "evaluation label=steered row_index=0 question_id=10 LLM_PROMPT" in output
    assert "FEWSHOT\n\nQuestion:\nQ1" in output
    assert "evaluation label=steered row_index=0 question_id=10 LLM_COMPLETION" in output
    assert "Answer: A" in output
    assert "gold=A prediction=A is_correct=True" in output


def test_evaluate_handles_empty_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluation, "tqdm", lambda iterable, desc: iterable)

    result = evaluation.evaluate(ExperimentConfig(), object(), [], "", "empty")

    assert result.correct == 0
    assert result.total == 0
    assert result.accuracy == 0.0
    assert result.records == []
