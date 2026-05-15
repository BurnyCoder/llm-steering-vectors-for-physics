"""MMLU-Pro data loading and prompt formatting.

Sources Used:
- HF Datasets loading docs: https://huggingface.co/docs/datasets/en/loading
- MMLU-Pro dataset card: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
- MMLU-Pro official repo: https://github.com/TIGER-AI-Lab/MMLU-Pro
- steering-vectors basic usage: https://steering-vectors.github.io/steering-vectors/basic_usage.html

Local Function:
- Loads Physics rows, builds benchmark prompts, and creates contrast pairs.

Global Role:
- Defines the data used to train steering vectors and test physics accuracy.
"""

import re  # Local: replace final answer letters. Global: create inaccurate negative solutions.
from typing import Any  # Local: type dataset rows. Global: keep external data shape flexible.

import requests  # Source: MMLU-Pro official prompt URL. Local: fetch prompt. Global: match benchmark style.
from datasets import load_dataset  # Source: HF Datasets docs. Local: load Hub dataset. Global: get MMLU-Pro rows.

from physics_steering_vectors.config import ExperimentConfig  # Local: read dataset/prompt settings. Global: central protocol.
from physics_steering_vectors.schemas import BenchmarkSplits  # Local: return typed splits. Global: phase boundary.

CHOICES = list("ABCDEFGHIJ")  # Source: MMLU-Pro 10-option format. Local: map option indices to letters. Global: score answer letters.


def load_physics_splits(config: ExperimentConfig) -> BenchmarkSplits:
    """Load and prepare MMLU-Pro Physics splits.

    Sources Used:
    - HF Datasets loading docs: https://huggingface.co/docs/datasets/en/loading
    - MMLU-Pro dataset card: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro

    Local Function:
    - Loads validation/test splits and filters category == physics.

    Global Role:
    - Keeps validation for steering-vector construction and test for final evaluation.
    """

    dataset = load_dataset(config.dataset_id)  # Source: HF Datasets docs. Local: load dataset dict. Global: fetch benchmark data.
    validation = _filter_subject(dataset["validation"], config.subject)  # Source: MMLU-Pro schema. Local: filter validation. Global: training contrast source.
    test = _filter_subject(dataset["test"], config.subject)  # Source: MMLU-Pro schema. Local: filter test. Global: held-out evaluation source.

    if config.max_test_examples is not None:  # Local: enable smoke test. Global: quick validation before full run.
        test = test[: config.max_test_examples]  # Local: truncate test rows. Global: reduce initial runtime.

    fewshot_prefix = build_fewshot_prefix(config, validation)  # Source: MMLU-Pro official prompt style. Local: build shared prompt prefix. Global: consistent eval prompting.
    return BenchmarkSplits(validation=validation, test=test, fewshot_prefix=fewshot_prefix)  # Local: package data. Global: pass to phases.


def build_training_pairs(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Build positive/negative activation pairs.

    Sources Used:
    - steering-vectors basic usage: https://steering-vectors.github.io/steering-vectors/basic_usage.html
    - MMLU-Pro dataset card: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro

    Local Function:
    - Positive: correct official CoT solution.
    - Negative: same solution text but with an incorrect final answer letter.

    Global Role:
    - Creates the activation contrast whose mean difference becomes the steering vector.
    """

    pairs: list[tuple[str, str]] = []  # Local: collect training tuples. Global: input to train_steering_vector.
    for row in rows:  # Local: iterate validation examples. Global: avoid test leakage.
        positive = format_solution(row)  # Local: correct solution string. Global: accurate activation target.
        for wrong_letter in available_wrong_letters(row):  # Local: enumerate wrong labels. Global: create inaccurate contrast examples.
            negative = format_solution(row, answer_override=wrong_letter)  # Local: inaccurate solution string. Global: negative activation target.
            pairs.append((positive, negative))  # Source: steering-vectors tuple format. Local: add pair. Global: training sample.
    return pairs  # Local: return all pairs. Global: feeds steering-vector training phase.


def build_eval_prompt(row: dict[str, Any], fewshot_prefix: str) -> str:
    """Build one evaluation prompt.

    Sources Used:
    - MMLU-Pro official evaluate_from_local.py: https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_local.py

    Local Function:
    - Appends the target question after the reusable few-shot prefix.

    Global Role:
    - Ensures baseline and steered conditions answer exactly the same prompt.
    """

    return f"{fewshot_prefix}{format_question(row)}\nAnswer: Let's think step by step."  # Local: final prompt. Global: benchmark input.


def build_fewshot_prefix(config: ExperimentConfig, validation: list[dict[str, Any]]) -> str:
    """Build official-style few-shot prompt prefix.

    Sources Used:
    - MMLU-Pro official repo: https://github.com/TIGER-AI-Lab/MMLU-Pro

    Local Function:
    - Fetches initial prompt and appends validation examples.

    Global Role:
    - Gives every test question the same task framing and demonstrations.
    """

    response = requests.get(config.initial_prompt_url, timeout=30)  # Source: MMLU-Pro repo prompt file. Local: fetch prompt text. Global: align with official eval style.
    response.raise_for_status()  # Source: requests API. Local: fail on download errors. Global: avoid silently using bad prompt.

    prefix = response.text.replace("{$}", config.subject).rstrip() + "\n\n"  # Source: MMLU-Pro prompt placeholder. Local: insert subject. Global: configure physics eval context.
    for row in validation[: config.fewshot_k]:  # Source: MMLU-Pro few-shot style. Local: select examples. Global: stable demonstrations.
        prefix += format_solution(row) + "\n\n"  # Local: add solved example. Global: condition model with answer format.
    return prefix  # Local: return reusable prefix. Global: shared by all eval rows.


def format_solution(row: dict[str, Any], answer_override: str | None = None) -> str:
    """Format a full solved example.

    Sources Used:
    - MMLU-Pro dataset card: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro

    Local Function:
    - Combines question/options with CoT answer text.

    Global Role:
    - Used both for steering examples and few-shot benchmark demonstrations.
    """

    return f"{format_question(row)}\n{format_cot(row, answer_override)}"  # Local: question plus answer. Global: full solution text.


def format_question(row: dict[str, Any]) -> str:
    """Format question and options.

    Sources Used:
    - MMLU-Pro official evaluate_from_local.py: https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_local.py

    Local Function:
    - Renders options as A., B., C., etc.

    Global Role:
    - Standardizes prompt text across training pairs and evaluation prompts.
    """

    options = "\n".join(  # Local: build option lines. Global: expose answer choices to model.
        f"{CHOICES[index]}. {option}" for index, option in enumerate(row["options"])
    )
    return f"Question:\n{row['question']}\nOptions:\n{options}"  # Local: formatted question block. Global: reusable benchmark prompt unit.


def format_cot(row: dict[str, Any], answer_override: str | None = None) -> str:
    """Format or perturb CoT answer.

    Sources Used:
    - MMLU-Pro dataset card: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro

    Local Function:
    - Normalizes the answer prefix and optionally changes the final answer.

    Global Role:
    - Supplies accurate positives and inaccurate negatives for activation subtraction.
    """

    cot = row["cot_content"].replace(  # Local: normalize prefix. Global: match evaluation completion style.
        "A: Let's think step by step.",
        "Answer: Let's think step by step.",
        1,
    )

    if answer_override is None:  # Local: no perturbation requested. Global: preserve correct official solution.
        return cot  # Local: return original CoT. Global: positive sample or few-shot example.

    replaced, count = re.subn(  # Local: change final answer. Global: create inaccurate negative sample.
        r"The answer is \([A-J]\)\.?",
        f"The answer is ({answer_override}).",
        cot,
        count=1,
    )
    if count:  # Local: replacement succeeded. Global: negative sample keeps same reasoning but wrong label.
        return replaced  # Local: return perturbed CoT. Global: negative activation text.
    return cot.rstrip() + f"\nThe answer is ({answer_override})."  # Local: fallback final answer line. Global: ensure negative has wrong answer.


def available_wrong_letters(row: dict[str, Any]) -> list[str]:
    """Return valid incorrect answer letters.

    Sources Used:
    - MMLU-Pro dataset card: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro

    Local Function:
    - Finds option letters other than the gold answer.

    Global Role:
    - Expands each validation row into several contrast pairs.
    """

    return [  # Local: list wrong letters. Global: negative labels for steering training.
        CHOICES[index]
        for index in range(len(row["options"]))
        if CHOICES[index] != row["answer"]
    ]


def _filter_subject(split: Any, subject: str) -> list[dict[str, Any]]:
    """Filter one MMLU-Pro split by subject.

    Sources Used:
    - MMLU-Pro dataset card: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro

    Local Function:
    - Keeps rows whose `category` matches `subject`.

    Global Role:
    - Makes this a physics-only experiment.
    """

    return [_clean_row(row) for row in split if row["category"] == subject]  # Local: filter and clean. Global: isolate physics rows.


def _clean_row(row: dict[str, Any]) -> dict[str, Any]:
    """Remove placeholder options.

    Sources Used:
    - MMLU-Pro official evaluate_from_local.py: https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_local.py

    Local Function:
    - Removes `N/A` options before formatting prompts.

    Global Role:
    - Keeps answer choices aligned with real options.
    """

    clean = dict(row)  # Local: copy row. Global: avoid mutating dataset object.
    clean["options"] = [option for option in clean["options"] if option != "N/A"]  # Local: drop placeholders. Global: valid answer choices only.
    return clean  # Local: return cleaned row. Global: safer downstream formatting.
