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
from physics_steering_vectors.logging_utils import get_logger  # Local: phase logs. Global: terminal audit trail.
from physics_steering_vectors.modeling import load_qwen_bundle  # Local: model setup. Global: shared runtime.
from physics_steering_vectors.reporting import print_result_table, write_result_report  # Local: output table. Global: final comparison.
from physics_steering_vectors.reproducibility import set_reproducibility  # Local: seed setup. Global: repeatability.
from physics_steering_vectors.schemas import BenchmarkSplits, EvaluationResult, ModelBundle  # Local: typed phase boundaries. Global: readable pipeline.
from physics_steering_vectors.steering import (  # Local: vector training/persistence. Global: intervention creation and preservation.
    save_steering_vector,
    steering_vector_path,
    train_vector_for_layer,
)


logger = get_logger(__name__)


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

    logger.info("Phase 1 model setup starting model_id=%s seed=%s", config.model_id, config.seed)
    set_reproducibility(config.seed)  # Local: seed process. Global: make comparisons repeatable.
    bundle = load_qwen_bundle(config)  # Local: load model/tokenizer/hooks. Global: prepare object to steer/evaluate.
    logger.info("Decoder hook template: %s", bundle.layer_config["decoder_block"])  # Local: show hook path. Global: audit intervention target.
    logger.info("Phase 1 model setup complete")
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

    logger.info("Phase 2 benchmark setup starting dataset_id=%s subject=%s", config.dataset_id, config.subject)
    splits = load_physics_splits(config)  # Local: load/filter/format data. Global: create benchmark splits.
    logger.info("Validation physics rows: %d", len(splits.validation))  # Local: show validation size. Global: audit steering data volume.
    logger.info("Test physics rows: %d", len(splits.test))  # Local: show test size. Global: audit evaluation data volume.
    logger.info("Phase 2 benchmark setup complete")
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

    logger.info("Phase 3 contrast-pair setup starting validation_rows=%d", len(splits.validation))
    training_pairs = build_training_pairs(config, bundle, splits.validation)  # Local: mine tuples. Global: input to steering-vector training.
    logger.info("Training contrast pairs: %d", len(training_pairs))  # Local: show count. Global: audit vector training signal.
    logger.info("Phase 3 contrast-pair setup complete")
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

    logger.info("Phase 4 baseline evaluation starting test_rows=%d", len(splits.test))
    result = evaluate(  # Local: run benchmark. Global: baseline score.
        config=config,
        bundle=bundle,
        rows=splits.test,
        fewshot_prefix=splits.fewshot_prefix,
        label="baseline",
    )
    logger.info("Phase 4 baseline evaluation complete accuracy=%.6f correct=%d total=%d", result.accuracy, result.correct, result.total)
    return result


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

    logger.info(
        "Phase 5 steering sweep starting layers=%s multipliers=%s training_pairs=%d test_rows=%d",
        config.layer_sweep,
        config.multipliers,
        len(training_pairs),
        len(splits.test),
    )
    results: list[EvaluationResult] = []  # Local: collect steered scores. Global: compare all interventions.

    for layer in config.layer_sweep:  # Local: iterate selected layers. Global: test where intervention works best.
        logger.info("Training steering vector for layer=%d", layer)
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
        logger.info("Saved steering vector: %s", vector_path)  # Local: report artifact path. Global: make experiment outputs auditable.

        for multiplier in config.multipliers:  # Local: iterate steering strengths. Global: test dose response.
            label = f"layer_{layer}_mult_{multiplier}"  # Local: name condition. Global: identify row in report.
            logger.info("Evaluating steering condition label=%s layer=%d multiplier=%s", label, layer, multiplier)
            result = evaluate(
                config=config,
                bundle=bundle,
                rows=splits.test,
                fewshot_prefix=splits.fewshot_prefix,
                label=label,
                steering_vector=vector,
                multiplier=multiplier,
            )
            logger.info("Completed steering condition label=%s accuracy=%.6f correct=%d total=%d", label, result.accuracy, result.correct, result.total)
            results.append(result)  # Local: store condition result. Global: build comparison set.

    logger.info("Phase 5 steering sweep complete conditions=%d", len(results))
    return results  # Local: return all intervention scores. Global: feed report phase.


def phase_6_report(config: ExperimentConfig, results: list[EvaluationResult]) -> None:
    """Phase 6: report.

    Sources Used:
    - pandas DataFrame docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html

    Local Function:
    - Print and save sorted accuracy table.

    Global Role:
    - Shows and preserves whether steering improved physics performance.
    """

    logger.info("Phase 6 report starting results=%d report_dir=%s", len(results), config.report_dir)
    print_result_table(results)  # Local: render table. Global: final experiment outcome.
    report_stem = config.format_run_artifact(config.report_stem)  # Local: align report files with this run. Global: avoid overwriting prior reports.
    markdown_path, csv_path = write_result_report(results, config.report_dir, stem=report_stem)  # Local: persist table. Global: preserve run output.
    logger.info("Saved report: %s", markdown_path)  # Local: show human-readable artifact. Global: make report easy to find.
    logger.info("Saved report CSV: %s", csv_path)  # Local: show machine-readable artifact. Global: support later analysis.
    logger.info("Phase 6 report complete")
