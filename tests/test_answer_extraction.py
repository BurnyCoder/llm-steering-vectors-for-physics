import pytest

from physics_steering_vectors.answer_extraction import extract_answer_letter


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("After the derivation, the answer is (C).", "C"),
        ("Answer: J\n", "J"),
        ("The final answer: A", "A"),
        ("I considered A and B, but the final choice is D", "D"),
        ("The option is 3, with no final letter.", None),
    ],
)
def test_extract_answer_letter(text: str, expected: str | None) -> None:
    assert extract_answer_letter(text) == expected


def test_extract_answer_letter_prefers_explicit_answer_pattern() -> None:
    text = "A is tempting. The answer is (B). Later text mentions C."

    assert extract_answer_letter(text) == "B"
