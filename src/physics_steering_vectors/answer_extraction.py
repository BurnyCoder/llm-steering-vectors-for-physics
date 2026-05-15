"""Answer extraction from generated text.

Sources Used:
- MMLU-Pro official evaluation script: https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_local.py

Local Function:
- Converts a generated CoT completion into one answer letter.

Global Role:
- Turns language model output into benchmark accuracy.
"""

import re  # Local: regex answer matching. Global: score generated text against gold labels.


def extract_answer_letter(text: str) -> str | None:
    """Extract final answer A-J.

    Sources Used:
    - MMLU-Pro official evaluate_from_local.py: https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_local.py

    Local Function:
    - Tries answer-specific patterns, then last standalone letter.

    Global Role:
    - Provides the prediction used for baseline-vs-steered accuracy.
    """

    patterns = [  # Source: MMLU-Pro extraction style. Local: ordered fallback patterns. Global: robust scoring.
        r"answer is \(?([A-J])\)?",  # Local: match "answer is (E)". Global: preferred official-style final answer.
        r"[Aa]nswer:\s*([A-J])",  # Local: match direct "Answer: E". Global: fallback for terse completions.
        r"\b([A-J])\b(?!.*\b[A-J]\b)",  # Local: match final standalone option letter. Global: last-resort extraction.
    ]

    for pattern in patterns:  # Local: try patterns in priority order. Global: maximize valid scored predictions.
        match = re.search(pattern, text, flags=re.DOTALL)  # Local: search completion. Global: convert free text to option letter.
        if match:  # Local: extraction succeeded. Global: avoid counting parseable answer as missing.
            return match.group(1)  # Local: return captured letter. Global: prediction for accuracy.
    return None  # Local: no parseable answer. Global: treated as wrong in evaluation.
