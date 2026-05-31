import pytest

from physics_steering_vectors import activation_collection
from physics_steering_vectors.activation_collection import (
    build_training_pairs,
    classify_generated_response,
    pair_training_examples,
)
from physics_steering_vectors.config import ExperimentConfig
from physics_steering_vectors.logging_utils import configure_logging


def test_classify_generated_response_labels_correct_and_incorrect_answers() -> None:
    assert classify_generated_response("Prompt\n", "answer is (A)", "A") == (
        "positive",
        "Prompt\nanswer is (A)",
    )
    assert classify_generated_response("Prompt\n", "Answer: B", "A") == (
        "negative",
        "Prompt\nAnswer: B",
    )


def test_classify_generated_response_skips_unparsable_completion() -> None:
    assert classify_generated_response("Prompt\n", "no option letter here", "A") is None


def test_pair_training_examples_is_deterministic_balanced_and_non_mutating() -> None:
    config = ExperimentConfig(seed=123)
    positives = ["p1", "p2", "p3"]
    negatives = ["n1", "n2"]

    pairs = pair_training_examples(config, positives, negatives)

    assert pairs == pair_training_examples(config, positives, negatives)
    assert len(pairs) == 2
    assert positives == ["p1", "p2", "p3"]
    assert negatives == ["n1", "n2"]
    assert {positive for positive, _ in pairs} <= set(positives)
    assert {negative for _, negative in pairs} <= set(negatives)


def test_pair_training_examples_requires_both_classes() -> None:
    config = ExperimentConfig()

    with pytest.raises(RuntimeError, match="No correct"):
        pair_training_examples(config, [], ["negative"])
    with pytest.raises(RuntimeError, match="No incorrect"):
        pair_training_examples(config, ["positive"], [])


def test_build_training_pairs_mines_generated_positive_and_negative_responses(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    config = ExperimentConfig(train_generations_per_question=2, seed=0)
    configure_logging(config)
    rows = [
        {
            "question": "Which option is correct?",
            "options": ["alpha", "beta", "gamma", "delta"],
            "answer": "A",
        },
        {
            "question": "Which option is not correct?",
            "options": ["alpha", "beta", "gamma", "delta"],
            "answer": "D",
        },
    ]
    completions = iter(["Answer: A", "Answer: B", "Answer: C", "no answer"])
    calls: list[dict[str, object]] = []

    def fake_generate_completion(*args: object, **kwargs: object) -> str:
        calls.append(kwargs)
        return next(completions)

    monkeypatch.setattr(activation_collection, "generate_completion", fake_generate_completion)
    monkeypatch.setattr(activation_collection, "fetch_initial_prompt", lambda config: "Initial prompt.")
    monkeypatch.setattr(activation_collection, "tqdm", lambda iterable, desc: iterable)

    pairs = build_training_pairs(config, bundle=object(), rows=rows)

    assert len(pairs) == 1
    positive, negative = pairs[0]
    assert positive.endswith("Answer: A")
    assert negative.endswith(("Answer: B", "Answer: C"))
    assert len(calls) == 4
    assert all(call["do_sample"] is True for call in calls)
    assert all(call["temperature"] == config.train_temperature for call in calls)
    assert all(call["top_p"] == config.train_top_p for call in calls)
    output = capsys.readouterr().out
    assert "training_response_mining row_index=0" in output
    assert "LLM_PROMPT" in output
    assert "Question:\nWhich option is correct?" in output
    assert "LLM_COMPLETION" in output
    assert "Answer: A" in output
    assert "classification=positive" in output


def test_build_training_pairs_rejects_non_positive_generation_count() -> None:
    config = ExperimentConfig(train_generations_per_question=0)

    with pytest.raises(ValueError, match="must be positive"):
        build_training_pairs(config, bundle=object(), rows=[])
