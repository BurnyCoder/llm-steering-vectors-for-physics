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

import random  # Local: seed deterministic pair assembly. Global: keep mined training tuples reproducible.
from typing import Any  # Local: type dataset rows. Global: keep external data shape flexible.

import requests  # Source: MMLU-Pro official prompt URL. Local: fetch prompt. Global: match benchmark style.
from datasets import load_dataset  # Source: HF Datasets docs. Local: load Hub dataset. Global: get MMLU-Pro rows.
from tqdm import tqdm  # Local: progress bar for mining. Global: visible long-running training-data generation.

from physics_steering_vectors.answer_extraction import extract_answer_letter  # Local: score mined completions. Global: separate positive/negative pools.
from physics_steering_vectors.config import ExperimentConfig  # Local: read dataset/prompt settings. Global: central protocol.
from physics_steering_vectors.generation import generate_completion  # Local: shared model generation. Global: align training mining with evaluation generation.
from physics_steering_vectors.schemas import BenchmarkSplits, ModelBundle  # Local: return typed splits/runtime. Global: phase boundary.

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

    positives: list[str] = []  # Local: collect correct generated responses. Global: positive side for vector training.
    negatives: list[str] = []  # Local: collect incorrect generated responses. Global: negative side for vector training.
    unparsable = 0  # Local: count extraction misses. Global: audit mined data quality.

    for row in tqdm(rows, desc="training_response_mining"):  # Local: iterate validation examples. Global: avoid test leakage.
        prompt = build_training_prompt(row)  # Local: create question prompt. Global: actual model input for mined response.
        for _ in range(config.train_generations_per_question):  # Local: sample retries. Global: obtain real correct and incorrect model outputs.
            completion = generate_completion(  # Local: unsteered sampled response. Global: same completion code path as evaluation.
                config,
                bundle,
                prompt,
                do_sample=True,
                temperature=config.train_temperature,
                top_p=config.train_top_p,
            )
            classified = classify_generated_response(prompt, completion, row["answer"])  # Local: score completion. Global: route into activation class.
            if classified is None:  # Local: answer extraction failed. Global: do not train on unknown correctness.
                unparsable += 1
                continue

            label, text = classified  # Local: unpack classification. Global: append actual model response only.
            if label == "positive":
                positives.append(text)
            else:
                negatives.append(text)

    pairs = pair_training_examples(config, positives, negatives)  # Local: adapt pools to steering-vectors tuple API. Global: training samples.
    print(  # Local: show mining counts. Global: audit steering signal before vector training.
        "Mined training responses: "
        f"positives={len(positives)} negatives={len(negatives)} "
        f"unparsable={unparsable} pairs={len(pairs)}"
    )
    return pairs  # Local: return paired generated responses. Global: feeds steering-vector training phase.


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


def build_training_prompt(row: dict[str, Any]) -> str:
    """Build one prompt used to mine a training response.

    Local Function:
    - Formats a validation question without the official answer.

    Global Role:
    - Ensures training pairs contain model-generated answers, not label-edited dataset solutions.
    """

    return f"{format_question(row)}\nAnswer: Let's think step by step."  # Local: final prompt. Global: model input for mined training response.


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

    prediction = extract_answer_letter(completion)  # Local: parse generated final answer. Global: decide correctness class.
    if prediction is None:  # Local: extraction failed. Global: skip unknown-quality response.
        return None

    label = "positive" if prediction == gold_answer else "negative"  # Local: benchmark correctness. Global: vector contrast class.
    return label, prompt + completion  # Local: full generated response text. Global: activation input for vector training.


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

    shuffled_positives = list(positives)  # Local: avoid mutating caller list. Global: deterministic training tuple order.
    shuffled_negatives = list(negatives)  # Local: avoid mutating caller list. Global: deterministic training tuple order.
    rng = random.Random(config.seed)  # Local: seeded shuffle. Global: reproducible pair assembly.
    rng.shuffle(shuffled_positives)
    rng.shuffle(shuffled_negatives)

    pair_count = min(len(shuffled_positives), len(shuffled_negatives))  # Local: balance pools. Global: avoid synthetic fallbacks.
    return list(zip(shuffled_positives[:pair_count], shuffled_negatives[:pair_count]))  # Source: steering-vectors tuple format. Local: final adapter. Global: training samples.


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


def format_solution(row: dict[str, Any]) -> str:
    """Format a full solved example.

    Sources Used:
    - MMLU-Pro dataset card: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro

    Local Function:
    - Combines question/options with CoT answer text.

    Global Role:
    - Used both for steering examples and few-shot benchmark demonstrations.
    """

    return f"{format_question(row)}\n{format_cot(row)}"  # Local: question plus answer. Global: full solution text.


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


def format_cot(row: dict[str, Any]) -> str:
    """Format official CoT answer.

    Sources Used:
    - MMLU-Pro dataset card: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro

    Local Function:
    - Normalizes the answer prefix.

    Global Role:
    - Supplies official few-shot demonstrations for benchmark prompting.
    """

    cot = row["cot_content"].replace(  # Local: normalize prefix. Global: match evaluation completion style.
        "A: Let's think step by step.",
        "Answer: Let's think step by step.",
        1,
    )

    return cot  # Local: return original CoT. Global: few-shot example only.


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
