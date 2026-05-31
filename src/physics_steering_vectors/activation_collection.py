"""Activation-pair collection from generated model responses.

Sources Used:
- MMLU-Pro dataset card: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
- steering-vectors basic usage: https://steering-vectors.github.io/steering-vectors/basic_usage.html

Local Function:
- Mines correct and incorrect model-generated responses from validation rows.

Global Role:
- Creates the activation contrast pairs used to train steering vectors.
"""

import random  # Local: seed deterministic pair assembly. Global: keep mined training tuples reproducible.
from typing import Any  # Local: type dataset rows. Global: keep external data shape flexible.

from tqdm import tqdm  # Local: progress bar for mining. Global: visible long-running training-data generation.

from physics_steering_vectors.answer_extraction import extract_answer_letter  # Local: score mined completions. Global: separate positive/negative pools.
from physics_steering_vectors.config import ExperimentConfig  # Local: read dataset/prompt settings. Global: central protocol.
from physics_steering_vectors.data import build_training_prompt, fetch_initial_prompt  # Local: build mining prompts. Global: preserve training input format.
from physics_steering_vectors.generation import generate_completion  # Local: shared model generation. Global: align training mining with evaluation generation.
from physics_steering_vectors.logging_utils import get_logger, log_text_block  # Local: raw prompt/response logs. Global: auditable mining.
from physics_steering_vectors.schemas import ModelBundle  # Local: access runtime. Global: same model/tokenizer used throughout experiment.


logger = get_logger(__name__)


def build_training_pairs(
    config: ExperimentConfig,
    bundle: ModelBundle,
    rows: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Mine positive/negative activation pairs from model responses.

    Sources Used:
    - steering-vectors basic usage: https://steering-vectors.github.io/steering-vectors/basic_usage.html
    - MMLU-Pro dataset card: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro

    Local Function:
    - Positive: unsteered model response with a correct extracted answer.
    - Negative: unsteered model response with an incorrect extracted answer.

    Global Role:
    - Creates the activation contrast whose mean difference becomes the steering vector.
    """

    if config.train_generations_per_question <= 0:  # Local: validate mining count. Global: avoid silently producing no training data.
        raise ValueError("train_generations_per_question must be positive.")

    logger.info(
        "Starting training-response mining rows=%d generations_per_question=%d temperature=%s top_p=%s",
        len(rows),
        config.train_generations_per_question,
        config.train_temperature,
        config.train_top_p,
    )

    positives: list[str] = []  # Local: collect correct generated responses. Global: positive side for vector training.
    negatives: list[str] = []  # Local: collect incorrect generated responses. Global: negative side for vector training.
    unparsable = 0  # Local: count extraction misses. Global: audit mined data quality.
    initial_prompt = fetch_initial_prompt(config)  # Source: config.initial_prompt_url. Local: share official answer-format instruction. Global: align mining with evaluation prompt source.

    for row_index, row in enumerate(tqdm(rows, desc="training_response_mining")):  # Local: iterate validation examples. Global: avoid test leakage.
        question_id = row.get("question_id", row_index)
        prompt = build_training_prompt(row, initial_prompt)  # Local: create question prompt. Global: actual model input for mined response.
        logger.debug("Mining row row_index=%d question_id=%s gold=%s prompt_chars=%d", row_index, question_id, row["answer"], len(prompt))
        for generation_index in range(config.train_generations_per_question):  # Local: sample retries. Global: obtain real correct and incorrect model outputs.
            context = f"training_response_mining row_index={row_index} question_id={question_id} generation_index={generation_index}"
            log_text_block(logger, config.log_full_text, f"{context} LLM_PROMPT", prompt)
            completion = generate_completion(  # Local: unsteered sampled response. Global: same completion code path as evaluation.
                config,
                bundle,
                prompt,
                do_sample=True,
                temperature=config.train_temperature,
                top_p=config.train_top_p,
                log_context=context,
            )
            log_text_block(logger, config.log_full_text, f"{context} LLM_COMPLETION", completion)
            classified = classify_generated_response_details(prompt, completion, row["answer"])  # Local: score completion. Global: route into activation class.
            if classified is None:  # Local: answer extraction failed. Global: do not train on unknown correctness.
                unparsable += 1
                logger.debug(
                    "Classified mining response row_index=%d question_id=%s generation_index=%d gold=%s prediction=None classification=unparsable",
                    row_index,
                    question_id,
                    generation_index,
                    row["answer"],
                )
                continue

            label, text, prediction = classified  # Local: unpack classification. Global: append actual model response only.
            logger.debug(
                "Classified mining response row_index=%d question_id=%s generation_index=%d gold=%s prediction=%s classification=%s text_chars=%d",
                row_index,
                question_id,
                generation_index,
                row["answer"],
                prediction,
                label,
                len(text),
            )
            if label == "positive":
                positives.append(text)
            else:
                negatives.append(text)

    pairs = pair_training_examples(config, positives, negatives)  # Local: adapt pools to steering-vectors tuple API. Global: training samples.
    logger.info(  # Local: show mining counts. Global: audit steering signal before vector training.
        "Mined training responses: positives=%d negatives=%d unparsable=%d pairs=%d",
        len(positives),
        len(negatives),
        unparsable,
        len(pairs),
    )
    return pairs  # Local: return paired generated responses. Global: feeds steering-vector training phase.


def classify_generated_response(
    prompt: str,
    completion: str,
    gold_answer: str,
) -> tuple[str, str] | None:
    """Classify a generated response as positive or negative.

    Local Function:
    - Parses the completion answer and joins it to the prompt actually given to the model.

    Global Role:
    - Keeps steering-vector training examples limited to real, benchmark-scored model responses.
    """

    detailed = classify_generated_response_details(prompt, completion, gold_answer)
    if detailed is None:
        return None

    label, text, _prediction = detailed
    return label, text


def classify_generated_response_details(
    prompt: str,
    completion: str,
    gold_answer: str,
) -> tuple[str, str, str] | None:
    """Classify a generated response and include the extracted prediction."""

    prediction = extract_answer_letter(completion)  # Local: parse generated final answer. Global: decide correctness class.
    if prediction is None:  # Local: extraction failed. Global: skip unknown-quality response.
        return None

    label = "positive" if prediction == gold_answer else "negative"  # Local: benchmark correctness. Global: vector contrast class.
    return label, prompt + completion, prediction  # Local: full generated response text. Global: activation input for vector training.


def pair_training_examples(
    config: ExperimentConfig,
    positives: list[str],
    negatives: list[str],
) -> list[tuple[str, str]]:
    """Adapt mined response pools to the steering-vectors pair API.

    Local Function:
    - Deterministically shuffles and zips correct/incorrect generated responses.

    Global Role:
    - Provides `train_steering_vector` with tuples while preserving generated-only data.
    """

    if not positives:  # Local: validate positive pool. Global: fail loudly before vector training.
        raise RuntimeError("No correct model-generated training responses were mined.")
    if not negatives:  # Local: validate negative pool. Global: fail loudly before vector training.
        raise RuntimeError("No incorrect model-generated training responses were mined.")

    logger.debug("Pairing training examples positives=%d negatives=%d seed=%d", len(positives), len(negatives), config.seed)
    shuffled_positives = list(positives)  # Local: avoid mutating caller list. Global: deterministic training tuple order.
    shuffled_negatives = list(negatives)  # Local: avoid mutating caller list. Global: deterministic training tuple order.
    rng = random.Random(config.seed)  # Local: seeded shuffle. Global: reproducible pair assembly.
    rng.shuffle(shuffled_positives)
    rng.shuffle(shuffled_negatives)

    pair_count = min(len(shuffled_positives), len(shuffled_negatives))  # Local: balance pools. Global: avoid synthetic fallbacks.
    pairs = list(zip(shuffled_positives[:pair_count], shuffled_negatives[:pair_count]))  # Source: steering-vectors tuple format. Local: final adapter. Global: training samples.
    logger.debug("Paired training examples pair_count=%d", pair_count)
    return pairs
