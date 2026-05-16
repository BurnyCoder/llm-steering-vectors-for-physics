from typing import Any

import pytest

from physics_steering_vectors import data
from physics_steering_vectors.config import ExperimentConfig


def make_row(
    *,
    question_id: int = 1,
    category: str = "physics",
    question: str = "What is the force?",
    options: list[str] | None = None,
    answer: str = "B",
    cot_content: str = "A: Let's think step by step. Work through it.\nThe answer is (B).",
) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "category": category,
        "question": question,
        "options": options or ["1 N", "2 N"],
        "answer": answer,
        "cot_content": cot_content,
    }


def test_format_question_renders_lettered_options() -> None:
    row = make_row(options=["one", "two", "three"])

    assert data.format_question(row) == "Question:\nWhat is the force?\nOptions:\nA. one\nB. two\nC. three"


def test_format_cot_normalizes_answer_prefix_once() -> None:
    row = make_row(cot_content="A: Let's think step by step. First.\nA: Let's think step by step. Second.")

    assert data.format_cot(row) == "Answer: Let's think step by step. First.\nA: Let's think step by step. Second."


def test_format_solution_and_prompts_compose_question_answer_text() -> None:
    row = make_row()

    assert data.format_solution(row).startswith("Question:\nWhat is the force?")
    assert data.format_solution(row).endswith("The answer is (B).")
    assert data.build_training_prompt(row).endswith("\nAnswer: Let's think step by step.")
    assert data.build_eval_prompt(row, "PREFIX\n\n").startswith("PREFIX\n\nQuestion:")


def test_clean_row_removes_na_options_without_mutating_input() -> None:
    row = make_row(options=["real", "N/A", "also real"])

    cleaned = data._clean_row(row)

    assert cleaned["options"] == ["real", "also real"]
    assert row["options"] == ["real", "N/A", "also real"]


def test_filter_subject_keeps_and_cleans_only_matching_rows() -> None:
    rows = [
        make_row(category="physics", options=["keep", "N/A"]),
        make_row(category="chemistry", options=["drop"]),
    ]

    filtered = data._filter_subject(rows, "physics")

    assert len(filtered) == 1
    assert filtered[0]["category"] == "physics"
    assert filtered[0]["options"] == ["keep"]


def test_build_fewshot_prefix_fetches_prompt_and_appends_solutions(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        text = "These are {$} questions."

        def raise_for_status(self) -> None:
            calls["raised"] = True

    calls: dict[str, object] = {}

    def fake_get(url: str, timeout: int) -> FakeResponse:
        calls["url"] = url
        calls["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(data.requests, "get", fake_get)
    config = ExperimentConfig(initial_prompt_url="https://example.test/prompt.txt", fewshot_k=1)

    prefix = data.build_fewshot_prefix(config, [make_row(), make_row(question_id=2)])

    assert calls == {"url": "https://example.test/prompt.txt", "timeout": 30, "raised": True}
    assert prefix.startswith("These are physics questions.\n\n")
    assert prefix.count("Question:") == 1
    assert prefix.endswith("\n\n")


def test_load_physics_splits_filters_rows_caps_test_and_builds_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = {
        "validation": [
            make_row(question_id=1, category="physics", options=["v", "N/A"]),
            make_row(question_id=2, category="math"),
        ],
        "test": [
            make_row(question_id=3, category="physics"),
            make_row(question_id=4, category="physics"),
            make_row(question_id=5, category="biology"),
        ],
    }

    monkeypatch.setattr(data, "load_dataset", lambda dataset_id: dataset)
    monkeypatch.setattr(data, "build_fewshot_prefix", lambda config, validation: f"prefix for {len(validation)}")
    config = ExperimentConfig(dataset_id="dataset", max_test_examples=1)

    splits = data.load_physics_splits(config)

    assert splits.validation == [{**dataset["validation"][0], "options": ["v"]}]
    assert splits.test == [dataset["test"][0]]
    assert splits.fewshot_prefix == "prefix for 1"
