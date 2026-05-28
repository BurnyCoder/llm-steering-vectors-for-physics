"""Experiment configuration.

Sources Used:
- Qwen2.5 model card: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- MMLU-Pro dataset card: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
- steering-vectors basic usage: https://steering-vectors.github.io/steering-vectors/basic_usage.html

Local Function:
- Stores run settings in one dataclass.

Global Role:
- Makes the experiment protocol explicit and easy to audit before running.
"""

from dataclasses import dataclass  # Local: define immutable config objects. Global: keeps settings typed and centralized.


@dataclass(frozen=True)  # Local: make config immutable after construction. Global: prevents accidental protocol drift mid-run.
class ExperimentConfig:
    """Settings for the full steering-vector experiment.

    Sources Used:
    - Qwen2.5 model card: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
    - MMLU-Pro official repo: https://github.com/TIGER-AI-Lab/MMLU-Pro
    - train_steering_vector API: https://steering-vectors.github.io/steering-vectors/api/train_steering_vector.html

    Local Function:
    - Names the model, dataset, prompt source, generation settings, and sweep settings.

    Global Role:
    - Defines exactly what research experiment is being run.
    """

    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"  # Local: choose smallest compatible older Qwen. Global: model under intervention.
    dataset_id: str = "TIGER-Lab/MMLU-Pro"  # Local: choose benchmark dataset. Global: source of train/eval physics rows.
    subject: str = "physics"  # Local: filter MMLU-Pro category. Global: restricts experiment to physics performance.
    initial_prompt_url: str = (  # Local: official MMLU-Pro prompt source. Global: keeps evaluation prompt close to benchmark repo.
        "https://raw.githubusercontent.com/TIGER-AI-Lab/MMLU-Pro/main/"
        "cot_prompt_lib/initial_prompt.txt"
    )

    seed: int = 0  # Local: deterministic RNG seed. Global: repeatable baseline/steered comparison.
    fewshot_k: int = 5  # Local: number of validation examples in prompt. Global: matches official MMLU-Pro CoT style.
    max_new_tokens: int = 512  # Local: cap generation length. Global: limits runaway small-model CoT loops.
    max_test_examples: int | None = None  # Local: optional smoke-test cap. Global: full benchmark when None.
    train_generations_per_question: int = 4  # Local: sampled attempts per validation row. Global: mine real correct/incorrect response pools.
    train_temperature: float = 0.8  # Local: diversify training generations. Global: improve odds of mixed correct/incorrect responses.
    train_top_p: float = 0.95  # Local: nucleus sampling for training generations. Global: keep sampled mining bounded but varied.

    layer_sweep: tuple[int, ...] = (6, 12, 18)  # Local: layers to train vectors on. Global: tests intervention location.
    multipliers: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)  # Local: steering strengths. Global: tests dose response.
    train_batch_size: int = 1  # Local: conservative activation batch. Global: reduces memory risk on small GPUs.
    steering_vector_dir: str = "artifacts/steering_vectors"  # Local: runtime vector artifact directory. Global: preserves trained interventions for later analysis.
    report_dir: str = "artifacts/reports"  # Local: runtime report artifact directory. Global: preserves result tables after long runs.
    log_level: str = "DEBUG"  # Local: terminal verbosity. Global: make long experiment runs fully auditable by default.
    log_full_text: bool = True  # Local: print raw prompts/responses. Global: preserve exact LLM and steering-vector inputs in logs.
    log_file_path: str | None = "artifacts/logs/latest.log"  # Local: optional run log path. Global: preserve terminal logs after long runs.
    do_sample: bool = False  # Local: deterministic generation. Global: makes accuracy deltas easier to interpret.
