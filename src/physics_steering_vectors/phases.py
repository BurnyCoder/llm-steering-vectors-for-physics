"""Readable experiment phases.

Sources Used:
- Qwen2.5 model card: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- MMLU-Pro official repo: https://github.com/TIGER-AI-Lab/MMLU-Pro
- steering-vectors docs: https://steering-vectors.github.io/steering-vectors/basic_usage.html

Local Function:
- Groups low-level module calls into named research phases.

Global Role:
- Makes `main.py` read like the experiment protocol.
"""

from physics_steering_vectors.activation_collection import build_training_pairs  # Local: mine contrast pairs. Global: vector training input.
from physics_steering_vectors.config import ExperimentConfig  # Local: phase settings. Global: single protocol object.
from physics_steering_vectors.data import load_physics_splits  # Local: load benchmark rows. Global: benchmark setup.
from physics_steering_vectors.evaluation import evaluate  # Local: score condition. Global: baseline/intervention measurement.
from physics_steering_vectors.modeling import load_qwen_bundle  # Local: model setup. Global: shared runtime.
from physics_steering_vectors.reporting import print_result_table  # Local: output table. Global: final comparison.
from physics_steering_vectors.reproducibility import set_reproducibility  # Local: seed setup. Global: repeatability.
from physics_steering_vectors.schemas import BenchmarkSplits, EvaluationResult, ModelBundle  # Local: typed phase boundaries. Global: readable pipeline.
from physics_steering_vectors.steering import (  # Local: vector training/persistence. Global: intervention creation and preservation.
    save_steering_vector,
    steering_vector_path,
    train_vector_for_layer,
)


def phase_1_model_setup(config: ExperimentConfig) -> ModelBundle:
    """Phase 1: model setup.

    Sources Used:
    - Qwen2.5 model card: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
    - Transformers Qwen2 docs: https://huggingface.co/docs/transformers/model_doc/qwen2

    Local Function:
    - Seed RNGs, load Qwen, infer decoder hook path.

    Global Role:
    - Creates the model runtime that all later phases share.
    """

    set_reproducibility(config.seed)  # Local: seed process. Global: make comparisons repeatable.
    bundle = load_qwen_bundle(config)  # Local: load model/tokenizer/hooks. Global: prepare object to steer/evaluate.
    print(f"Decoder hook template: {bundle.layer_config['decoder_block']}")  # Local: show hook path. Global: audit intervention target.
    return bundle  # Local: return model bundle. Global: pass to downstream phases.


def phase_2_benchmark_setup(config: ExperimentConfig) -> BenchmarkSplits:
    """Phase 2: benchmark setup.

    Sources Used:
    - HF Datasets loading docs: https://huggingface.co/docs/datasets/en/loading
    - MMLU-Pro dataset card: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro

    Local Function:
    - Load Physics validation/test rows and few-shot prefix.

    Global Role:
    - Defines both steering-training data and held-out evaluation data.
    """

    splits = load_physics_splits(config)  # Local: load/filter/format data. Global: create benchmark splits.
    print(f"Validation physics rows: {len(splits.validation)}")  # Local: show validation size. Global: audit steering data volume.
    print(f"Test physics rows: {len(splits.test)}")  # Local: show test size. Global: audit evaluation data volume.
    return splits  # Local: return prepared data. Global: pass to contrast/eval phases.


def phase_3_contrast_pair_setup(
    config: ExperimentConfig,
    bundle: ModelBundle,
    splits: BenchmarkSplits,
) -> list[tuple[str, str]]:
    """Phase 3: contrast-pair setup.

    Sources Used:
    - steering-vectors basic usage: https://steering-vectors.github.io/steering-vectors/basic_usage.html

    Local Function:
    - Mine positive/negative model-generated response tuples from validation rows.

    Global Role:
    - Defines the activation direction as accurate physics solution minus inaccurate solution.
    """

    training_pairs = build_training_pairs(config, bundle, splits.validation)  # Local: mine tuples. Global: input to steering-vector training.
    print(f"Training contrast pairs: {len(training_pairs)}")  # Local: show count. Global: audit vector training signal.
    return training_pairs  # Local: return pairs. Global: pass to steering sweep.


def phase_4_baseline_evaluation(
    config: ExperimentConfig,
    bundle: ModelBundle,
    splits: BenchmarkSplits,
) -> EvaluationResult:
    """Phase 4: baseline evaluation.

    Sources Used:
    - MMLU-Pro official repo: https://github.com/TIGER-AI-Lab/MMLU-Pro

    Local Function:
    - Evaluate unsteered Qwen on Physics test rows.

    Global Role:
    - Provides the control condition for all steering deltas.
    """

    return evaluate(  # Local: run benchmark. Global: baseline score.
        config=config,
        bundle=bundle,
        rows=splits.test,
        fewshot_prefix=splits.fewshot_prefix,
        label="baseline",
    )


def phase_5_steering_sweep(
    config: ExperimentConfig,
    bundle: ModelBundle,
    splits: BenchmarkSplits,
    training_pairs: list[tuple[str, str]],
) -> list[EvaluationResult]:
    """Phase 5: steering sweep.

    Sources Used:
    - train_steering_vector API: https://steering-vectors.github.io/steering-vectors/api/train_steering_vector.html
    - SteeringVector API: https://steering-vectors.github.io/steering-vectors/api/steering_vector.html

    Local Function:
    - Train one vector per selected layer and evaluate several multipliers.

    Global Role:
    - Tests whether activation steering improves held-out physics benchmark accuracy.
    """

    results: list[EvaluationResult] = []  # Local: collect steered scores. Global: compare all interventions.

    for layer in config.layer_sweep:  # Local: iterate selected layers. Global: test where intervention works best.
        vector = train_vector_for_layer(config, bundle, training_pairs, layer)  # Local: train layer vector. Global: create intervention.
        vector_path = save_steering_vector(  # Local: persist once per trained layer. Global: preserve intervention independent of multiplier sweep.
            vector,
            steering_vector_path(config, layer),
            metadata={
                "model_id": config.model_id,
                "dataset_id": config.dataset_id,
                "subject": config.subject,
                "seed": config.seed,
                "layer": layer,
                "train_batch_size": config.train_batch_size,
                "train_generations_per_question": config.train_generations_per_question,
                "read_token_index": -1,
            },
        )
        print(f"Saved steering vector: {vector_path}")  # Local: report artifact path. Global: make experiment outputs auditable.

        for multiplier in config.multipliers:  # Local: iterate steering strengths. Global: test dose response.
            label = f"layer_{layer}_mult_{multiplier}"  # Local: name condition. Global: identify row in report.
            results.append(  # Local: store condition result. Global: build comparison set.
                evaluate(
                    config=config,
                    bundle=bundle,
                    rows=splits.test,
                    fewshot_prefix=splits.fewshot_prefix,
                    label=label,
                    steering_vector=vector,
                    multiplier=multiplier,
                )
            )

    return results  # Local: return all intervention scores. Global: feed report phase.


def phase_6_report(results: list[EvaluationResult]) -> None:
    """Phase 6: report.

    Sources Used:
    - pandas DataFrame docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html

    Local Function:
    - Print sorted accuracy table.

    Global Role:
    - Shows whether steering improved physics performance.
    """

    print_result_table(results)  # Local: render table. Global: final experiment outcome.
