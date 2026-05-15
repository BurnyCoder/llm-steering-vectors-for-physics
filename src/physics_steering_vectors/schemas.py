"""Shared typed containers.

Sources Used:
- Python dataclasses: https://docs.python.org/3/library/dataclasses.html
- steering-vectors API: https://steering-vectors.github.io/steering-vectors/api/train_steering_vector.html

Local Function:
- Defines simple objects passed between phases.

Global Role:
- Keeps phase boundaries explicit: model setup returns a `ModelBundle`, data setup returns `BenchmarkSplits`, evaluation returns `EvaluationResult`.
"""

from dataclasses import dataclass  # Local: define lightweight typed records. Global: documents phase inputs/outputs.
from typing import Any  # Local: type external library objects. Global: avoids overfitting to changing HF class names.


@dataclass(frozen=True)
class ModelBundle:
    """Loaded model assets.

    Sources Used:
    - Qwen3.5 model card: https://huggingface.co/Qwen/Qwen3.5-0.8B
    - train_steering_vector API: https://steering-vectors.github.io/steering-vectors/api/train_steering_vector.html

    Local Function:
    - Groups model, processor, tokenizer, and steering hook config.

    Global Role:
    - Provides every later phase with the same loaded model runtime.
    """

    model: Any  # Local: HF model object. Global: used for activation extraction and generation.
    processor: Any  # Local: Qwen processor. Global: keeps full model preprocessing available if needed.
    tokenizer: Any  # Local: text tokenizer. Global: converts prompts for steering training and evaluation.
    layer_config: dict[str, str]  # Local: steering hook path template. Global: maps vector layers to Qwen modules.


@dataclass(frozen=True)
class BenchmarkSplits:
    """Prepared MMLU-Pro Physics data.

    Sources Used:
    - HF Datasets loading docs: https://huggingface.co/docs/datasets/en/loading
    - MMLU-Pro dataset card: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro

    Local Function:
    - Stores validation rows, test rows, and the few-shot prefix.

    Global Role:
    - Separates vector-training data from benchmark-test data.
    """

    validation: list[dict[str, Any]]  # Local: physics validation rows. Global: source for steering contrast pairs.
    test: list[dict[str, Any]]  # Local: physics test rows. Global: held-out benchmark for baseline and steered scoring.
    fewshot_prefix: str  # Local: reusable prompt prefix. Global: ensures every eval question gets the same context.


@dataclass(frozen=True)
class EvaluationRecord:
    """One scored benchmark item.

    Sources Used:
    - MMLU-Pro official repo: https://github.com/TIGER-AI-Lab/MMLU-Pro

    Local Function:
    - Stores one prediction and whether it matched the gold answer.

    Global Role:
    - Enables per-question audit of accuracy changes.
    """

    question_id: int  # Local: identify row. Global: supports later error analysis.
    gold: str  # Local: correct answer letter. Global: benchmark ground truth.
    prediction: str | None  # Local: extracted model answer. Global: measured model behavior.
    is_correct: bool  # Local: row-level score. Global: basis for total accuracy.


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate benchmark result for one condition.

    Sources Used:
    - MMLU-Pro official repo: https://github.com/TIGER-AI-Lab/MMLU-Pro

    Local Function:
    - Stores condition label, counts, accuracy, and row records.

    Global Role:
    - Lets reporting compare baseline against every steering intervention.
    """

    label: str  # Local: names baseline or steered run. Global: identifies intervention condition.
    correct: int  # Local: number of correct rows. Global: numerator for accuracy.
    total: int  # Local: number of evaluated rows. Global: denominator for accuracy.
    accuracy: float  # Local: correct / total. Global: primary outcome metric.
    records: list[EvaluationRecord]  # Local: per-row details. Global: supports deeper analysis after headline result.
