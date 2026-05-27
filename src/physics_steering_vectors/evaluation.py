"""Baseline and steered benchmark evaluation.

Sources Used:
- Qwen2.5 model card: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- Transformers Qwen2 docs: https://huggingface.co/docs/transformers/model_doc/qwen2
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
from physics_steering_vectors.logging_utils import get_logger, log_text_block  # Local: raw prompt/response logs. Global: auditable evaluation.
from physics_steering_vectors.schemas import EvaluationRecord, EvaluationResult, ModelBundle  # Local: typed outputs. Global: phase/report contract.


logger = get_logger(__name__)


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
    - Transformers generation docs: https://huggingface.co/docs/transformers/model_doc/qwen2
    - SteeringVector.apply docs: https://steering-vectors.github.io/steering-vectors/api/steering_vector.html
    - MMLU-Pro official repo: https://github.com/TIGER-AI-Lab/MMLU-Pro

    Local Function:
    - Runs each prompt, extracts one answer, and counts correct predictions.

    Global Role:
    - Measures baseline or intervention performance on held-out Physics rows.
    """

    logger.info(
        "Starting evaluation label=%s rows=%d steering=%s multiplier=%s",
        label,
        len(rows),
        steering_vector is not None,
        multiplier,
    )
    records: list[EvaluationRecord] = []  # Local: collect row-level scores. Global: enable audit/error analysis.
    correct = 0  # Local: initialize correct count. Global: numerator for accuracy.

    for row_index, row in enumerate(tqdm(rows, desc=label)):  # Local: iterate benchmark rows. Global: evaluate every held-out physics item.
        question_id = row["question_id"]
        prompt = build_eval_prompt(row, fewshot_prefix)  # Local: create exact prompt. Global: same input format across conditions.
        context = f"evaluation label={label} row_index={row_index} question_id={question_id}"
        log_text_block(logger, config.log_full_text, f"{context} LLM_PROMPT", prompt)
        completion = generate_completion(  # Local: model output. Global: behavior under condition.
            config,
            bundle,
            prompt,
            steering_vector=steering_vector,
            multiplier=multiplier,
            log_context=context,
        )
        log_text_block(logger, config.log_full_text, f"{context} LLM_COMPLETION", completion)
        prediction = extract_answer_letter(completion)  # Local: parse answer. Global: benchmark prediction.
        is_correct = prediction == row["answer"]  # Local: compare to gold. Global: per-item accuracy signal.
        correct += int(is_correct)  # Local: update count. Global: aggregate metric.
        logger.debug(
            "Scored evaluation row label=%s row_index=%d question_id=%s gold=%s prediction=%s is_correct=%s running_correct=%d running_total=%d",
            label,
            row_index,
            question_id,
            row["answer"],
            prediction,
            is_correct,
            correct,
            row_index + 1,
        )

        records.append(  # Local: store one scored row. Global: preserve details beyond headline accuracy.
            EvaluationRecord(
                question_id=int(question_id),  # Local: row id. Global: trace errors to dataset item.
                gold=row["answer"],  # Local: correct letter. Global: ground truth.
                prediction=prediction,  # Local: extracted model answer. Global: measured condition output.
                is_correct=is_correct,  # Local: boolean score. Global: audit metric.
            )
        )

    total = len(rows)  # Local: denominator. Global: accuracy normalization.
    result = EvaluationResult(  # Local: package result. Global: input to report table.
        label=label,
        correct=correct,
        total=total,
        accuracy=correct / total if total else 0.0,
        records=records,
    )
    logger.info("Completed evaluation label=%s accuracy=%.6f correct=%d total=%d", label, result.accuracy, result.correct, result.total)
    return result
