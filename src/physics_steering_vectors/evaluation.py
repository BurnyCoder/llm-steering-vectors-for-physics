"""Baseline and steered benchmark evaluation.

Sources Used:
- Qwen3.5 model card: https://huggingface.co/Qwen/Qwen3.5-0.8B
- Transformers Qwen3.5 docs: https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5
- SteeringVector API: https://steering-vectors.github.io/steering-vectors/api/steering_vector.html
- MMLU-Pro official repo: https://github.com/TIGER-AI-Lab/MMLU-Pro

Local Function:
- Generates completions, extracts answers, and scores rows.

Global Role:
- Produces the main evidence for whether steering improves physics accuracy.
"""

from typing import Any  # Local: type dataset rows and steering vector. Global: keep external objects flexible.

import torch  # Local: inference mode. Global: efficient benchmark evaluation.
from tqdm import tqdm  # Local: progress bar. Global: visible long-running eval.

from physics_steering_vectors.answer_extraction import extract_answer_letter  # Local: parse completions. Global: convert generations to scores.
from physics_steering_vectors.config import ExperimentConfig  # Local: generation settings. Global: central protocol.
from physics_steering_vectors.data import build_eval_prompt  # Local: build prompt. Global: same prompt in all conditions.
from physics_steering_vectors.generation import generate_completion  # Local: shared model completion. Global: same generation path as training mining.
from physics_steering_vectors.schemas import EvaluationRecord, EvaluationResult, ModelBundle  # Local: typed outputs. Global: phase/report contract.


@torch.inference_mode()  # Source: PyTorch inference mode. Local: disable gradients. Global: faster, lower-memory evaluation.
def evaluate(
    config: ExperimentConfig,
    bundle: ModelBundle,
    rows: list[dict[str, Any]],
    fewshot_prefix: str,
    label: str,
    steering_vector: Any | None = None,
    multiplier: float = 1.0,
) -> EvaluationResult:
    """Evaluate one condition.

    Sources Used:
    - Transformers generation docs: https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5
    - SteeringVector.apply docs: https://steering-vectors.github.io/steering-vectors/api/steering_vector.html
    - MMLU-Pro official repo: https://github.com/TIGER-AI-Lab/MMLU-Pro

    Local Function:
    - Runs each prompt, extracts one answer, and counts correct predictions.

    Global Role:
    - Measures baseline or intervention performance on held-out Physics rows.
    """

    records: list[EvaluationRecord] = []  # Local: collect row-level scores. Global: enable audit/error analysis.
    correct = 0  # Local: initialize correct count. Global: numerator for accuracy.

    for row in tqdm(rows, desc=label):  # Local: iterate benchmark rows. Global: evaluate every held-out physics item.
        prompt = build_eval_prompt(row, fewshot_prefix)  # Local: create exact prompt. Global: same input format across conditions.
        completion = generate_completion(  # Local: model output. Global: behavior under condition.
            config,
            bundle,
            prompt,
            steering_vector=steering_vector,
            multiplier=multiplier,
        )
        prediction = extract_answer_letter(completion)  # Local: parse answer. Global: benchmark prediction.
        is_correct = prediction == row["answer"]  # Local: compare to gold. Global: per-item accuracy signal.
        correct += int(is_correct)  # Local: update count. Global: aggregate metric.

        records.append(  # Local: store one scored row. Global: preserve details beyond headline accuracy.
            EvaluationRecord(
                question_id=int(row["question_id"]),  # Local: row id. Global: trace errors to dataset item.
                gold=row["answer"],  # Local: correct letter. Global: ground truth.
                prediction=prediction,  # Local: extracted model answer. Global: measured condition output.
                is_correct=is_correct,  # Local: boolean score. Global: audit metric.
            )
        )

    total = len(rows)  # Local: denominator. Global: accuracy normalization.
    return EvaluationResult(  # Local: package result. Global: input to report table.
        label=label,
        correct=correct,
        total=total,
        accuracy=correct / total if total else 0.0,
        records=records,
    )
