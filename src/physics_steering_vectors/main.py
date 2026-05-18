"""Top-level experiment protocol.

Sources Used:
- Qwen2.5 model card: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- MMLU-Pro dataset card: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
- steering-vectors docs: https://steering-vectors.github.io/steering-vectors/basic_usage.html

Local Function:
- Calls a few readable phase functions.

Global Role:
- Documents the complete research workflow at a glance.
"""

from physics_steering_vectors.config import ExperimentConfig  # Local: construct protocol settings. Global: fixes the experiment definition.
from physics_steering_vectors.phases import (  # Local: import named phase functions. Global: keep main readable.
    phase_1_model_setup,
    phase_2_benchmark_setup,
    phase_3_contrast_pair_setup,
    phase_4_baseline_evaluation,
    phase_5_steering_sweep,
    phase_6_report,
)


def main() -> None:
    """Run the full experiment.

    Sources Used:
    - Qwen2.5 model card: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
    - MMLU-Pro official repo: https://github.com/TIGER-AI-Lab/MMLU-Pro
    - steering-vectors basic usage: https://steering-vectors.github.io/steering-vectors/basic_usage.html

    Local Function:
    - Orchestrates setup, data prep, vector training, evaluation, and reporting.

    Global Role:
    - Executes the complete baseline-vs-steered physics performance experiment.
    """

    config = ExperimentConfig()  # Local: instantiate settings. Global: define exact model/data/sweep protocol.

    model_bundle = phase_1_model_setup(config)  # Local: load Qwen and hooks. Global: create model to steer and evaluate.
    benchmark_splits = phase_2_benchmark_setup(config)  # Local: load Physics rows. Global: define validation/test data.
    training_pairs = phase_3_contrast_pair_setup(config, model_bundle, benchmark_splits)  # Local: mine contrast pairs. Global: define steering signal.

    baseline = phase_4_baseline_evaluation(config, model_bundle, benchmark_splits)  # Local: score unsteered model. Global: control condition.
    steered_results = phase_5_steering_sweep(  # Local: train/apply vectors. Global: experimental interventions.
        config,
        model_bundle,
        benchmark_splits,
        training_pairs,
    )

    phase_6_report([baseline, *steered_results])  # Local: print comparison. Global: answer whether steering improved physics accuracy.


if __name__ == "__main__":  # Local: allow direct script execution. Global: supports simple local runs.
    main()  # Local: start orchestration. Global: run complete research pipeline.
