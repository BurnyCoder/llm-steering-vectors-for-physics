# LLM Steering Vectors for Physics

This repository is a research project for testing whether activation steering can improve
small Qwen model performance on physics questions.

The experiment uses `Qwen/Qwen3.5-0.8B`, mines correct and incorrect model-generated
MMLU-Pro Physics responses from validation rows, trains steering vectors from those generated
responses, applies the vectors during generation, and compares benchmark accuracy against an
unsteered baseline.

## Research Goal

The core hypothesis is:

> A direction in activation space computed as `mean(correct physics solution activations) -
> mean(incorrect physics solution activations)` may push a language model toward more accurate
> physics reasoning during benchmark generation.

The experiment is structured as a controlled comparison:

1. Load the selected Qwen model.
2. Load MMLU-Pro Physics validation and test rows.
3. Generate sampled unsteered model responses for validation rows.
4. Classify generated responses as positive when the extracted answer is correct and negative when it is incorrect.
5. Train steering vectors from those generated-response contrast pairs.
6. Evaluate the unsteered model on held-out test rows.
7. Evaluate steered generations across a layer and multiplier sweep.
8. Report accuracy and delta from baseline.

The validation split is used to create steering vectors. The test split is reserved for evaluation.
That separation is important because it avoids directly training the steering vector on the same
examples used to measure benchmark improvement.

## Setup With `uv`

This repo is intended to use `uv` for all Python package management.

Do not use `pip`, `pip3`, or `conda` for this project.

### 1. Create a virtual environment

```bash
uv venv
```

### 2. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 3. Install/sync dependencies

```bash
uv sync
```

This will install the dependencies declared in `pyproject.toml`.

One dependency is installed from the current Hugging Face Transformers GitHub repository:

```toml
"transformers @ git+https://github.com/huggingface/transformers.git"
```

That is intentional because the selected Qwen3.5 model may require the newest Transformers support.

## Running the Experiment

After creating the venv and syncing dependencies:

```bash
uv run physics-steering
```

Equivalent module form:

```bash
uv run python -m physics_steering_vectors
```

## Smoke Test

The full run may take a while because it loads the model, trains several steering vectors, and
evaluates multiple conditions.

For a quick smoke test, temporarily set this field in
`src/physics_steering_vectors/config.py`:

```python
max_test_examples: int | None = 25
```

Then run:

```bash
uv run physics-steering
```

For the real benchmark run, set it back to:

```python
max_test_examples: int | None = None
```

## Expected Output

The script prints:

1. The inferred decoder hook template.
2. Number of validation Physics rows.
3. Number of test Physics rows.
4. A progress bar while mining generated validation responses.
5. Counts for mined positive responses, negative responses, unparsable responses, and usable training pairs.
6. Progress bars for baseline and steered evaluations.
7. A final table like:

```text
              label  correct  total  accuracy  delta_vs_baseline
 layer_12_mult_1.5       42    100      0.42               0.04
          baseline       38    100      0.38               0.00
  layer_6_mult_0.5       35    100      0.35              -0.03
```

The real numbers will depend on the model, hardware, installed library versions, and benchmark run.

## Interpreting Results

The key metric is `delta_vs_baseline`.

- Positive delta: the steering condition improved over unsteered generation.
- Zero delta: no measured improvement.
- Negative delta: steering hurt benchmark performance.

The full sweep should be reported, not only the best row. A single best row can be noisy, especially
on a small benchmark subset. The layer and multiplier sweep is meant to reveal whether improvements
are stable or isolated.


## High-Level Pipeline

```text
MMLU-Pro validation physics rows
        |
        v
sampled unsteered Qwen3.5 responses
        |
        v
answer extraction and correct/incorrect classification
        |
        v
generated positive/negative response pairs
        |
        v
steering-vectors train_steering_vector()
        |
        v
layer-specific activation steering vector
        |
        v
Qwen3.5 generation on MMLU-Pro test physics rows
        |
        v
baseline accuracy vs steered accuracy
```

## Current Design Choices

- Model: `Qwen/Qwen3.5-0.8B`
- Benchmark: `TIGER-Lab/MMLU-Pro`
- Subject/category: `physics`
- Steering library: `steering-vectors`
- Model loading: Hugging Face `transformers`
- Dataset loading: Hugging Face `datasets`
- Python package management: `uv`
- Training positives: model-generated responses whose extracted answer matches the validation gold answer.
- Training negatives: model-generated responses whose extracted answer differs from the validation gold answer.
- Training sampling: controlled by `train_generations_per_question`, `train_temperature`, and `train_top_p`.
- Activation collection: `activation_collection.py` mines and pairs generated validation responses for vector training.
- Shared generation: `generation.py` owns model completion for both activation collection and evaluation.

The model choice follows the "strict latest Qwen3.5 small model" direction. Because this checkpoint
is exposed through a multimodal image-text model class, the code uses `AutoProcessor` and
`AutoModelForImageTextToText` rather than a plain causal-LM loader.

## Repository Structure

```text
.
├── AGENTS.md
├── pyproject.toml
├── README.md
└── src
    └── physics_steering_vectors
        ├── __init__.py
        ├── __main__.py
        ├── activation_collection.py
        ├── answer_extraction.py
        ├── config.py
        ├── data.py
        ├── evaluation.py
        ├── generation.py
        ├── layers.py
        ├── main.py
        ├── modeling.py
        ├── phases.py
        ├── reporting.py
        ├── reproducibility.py
        ├── schemas.py
        └── steering.py
```

## File-by-File Context

### `pyproject.toml`

Local function:
- Declares project metadata, dependencies, build backend, and the `physics-steering` console script.

Global role:
- Defines the reproducible Python environment needed to run the experiment with `uv`.

Key dependencies:
- `torch`: model inference and activation computation.
- `transformers`: Qwen3.5 model and processor loading.
- `datasets`: MMLU-Pro loading from the Hugging Face Hub.
- `steering-vectors`: activation recording, vector training, and steering application.
- `pandas`: result table formatting.
- `requests`: fetch the official MMLU-Pro prompt template.
- `tqdm`: progress bars.
- `accelerate`: model device placement through `device_map="auto"`.
- `pytest`: unit test runner for deterministic logic.

### `src/physics_steering_vectors/config.py`

Local function:
- Stores the experiment constants in `ExperimentConfig`.

Global role:
- Makes the experimental protocol auditable from one place.

Important fields:
- `model_id`: selected Qwen checkpoint.
- `dataset_id`: selected benchmark dataset.
- `subject`: MMLU-Pro category filter.
- `layer_sweep`: decoder layers where vectors are trained.
- `multipliers`: steering strengths tested at generation time.
- `max_test_examples`: optional cap for smoke tests.
- `train_generations_per_question`: sampled unsteered completions to mine per validation row.
- `train_temperature`: sampling temperature used only while mining training responses.
- `train_top_p`: nucleus sampling value used only while mining training responses.

### `src/physics_steering_vectors/schemas.py`

Local function:
- Defines dataclasses passed between phases.

Global role:
- Keeps each phase's inputs and outputs explicit.

Main dataclasses:
- `ModelBundle`: model, processor, tokenizer, and steering hook config.
- `BenchmarkSplits`: validation rows, test rows, and few-shot prompt prefix.
- `EvaluationRecord`: one scored benchmark item.
- `EvaluationResult`: aggregate result for one baseline or steered condition.

### `src/physics_steering_vectors/reproducibility.py`

Local function:
- Sets Python and Torch random seeds.

Global role:
- Reduces avoidable randomness so baseline and steered comparisons are easier to interpret.

### `src/physics_steering_vectors/layers.py`

Local function:
- Inspects the loaded Qwen3.5 module names and infers a decoder block path template.

Global role:
- Bridges the Qwen3.5 model structure to the `steering-vectors` hook API.

Why this exists:
- Steering libraries need to know which internal module corresponds to "layer 6" or "layer 12".
- Qwen3.5 may expose the text model under a wrapper path, so the code discovers the path from the
  actual loaded model instead of assuming one fixed module name.

### `src/physics_steering_vectors/modeling.py`

Local function:
- Loads `Qwen/Qwen3.5-0.8B`, gets the tokenizer from the processor, configures padding, and builds
  the layer hook config.

Global role:
- Produces the model runtime shared by vector training and benchmark evaluation.

### `src/physics_steering_vectors/generation.py`

Local function:
- Tokenizes prompts, runs model generation, optionally applies steering hooks, and decodes only newly generated tokens.

Global role:
- Provides the single completion path used by both activation collection and benchmark evaluation.

### `src/physics_steering_vectors/activation_collection.py`

Local function:
- Mines sampled unsteered validation responses, classifies them by extracted answer correctness,
  and pairs correct/incorrect generated responses for vector training.

Global role:
- Owns the activation-training contrast data while reusing prompt formatting from `data.py` and
  model completion from `generation.py`.

Positive examples:
- Unsteered model-generated validation responses whose extracted answer matches the gold answer.

Negative examples:
- Unsteered model-generated validation responses whose extracted answer differs from the gold answer.

Unparsable responses:
- Generated responses with no extractable answer are skipped and are not used for vector training.

### `src/physics_steering_vectors/data.py`

Local function:
- Loads MMLU-Pro, filters Physics rows, formats prompts, and fetches the official prompt template.

Global role:
- Defines benchmark rows and prompt text shared by activation collection and held-out evaluation.

### `src/physics_steering_vectors/answer_extraction.py`

Local function:
- Extracts a final answer letter from generated text using regex fallbacks.

Global role:
- Converts free-form generations into benchmark predictions that can be scored.

If no answer letter is found, the prediction is `None` and the row is counted as incorrect.

### `src/physics_steering_vectors/steering.py`

Local function:
- Wraps one call to `train_steering_vector()`.

Global role:
- Produces the activation intervention that is later applied during benchmark generation.

The current implementation trains one vector per selected layer, using the final token activation
from each positive/negative prompt pair.

### `src/physics_steering_vectors/evaluation.py`

Local function:
- Builds evaluation prompts, calls the shared generation helper with or without steering, extracts answer letters, and scores each row.

Global role:
- Measures whether the steering intervention improves held-out MMLU-Pro Physics accuracy.

Baseline condition:
- Calls generation without any steering vector.

Steered condition:
- Enters `SteeringVector.apply(...)` as a context manager around the same generation call.

### `src/physics_steering_vectors/reporting.py`

Local function:
- Converts `EvaluationResult` objects into a sorted `pandas` table.

Global role:
- Shows the final result: accuracy and delta versus baseline for every layer/multiplier condition.

### `src/physics_steering_vectors/phases.py`

Local function:
- Groups lower-level helper calls into named research phases.

Global role:
- Makes the experiment readable as a protocol:

```text
phase_1_model_setup
phase_2_benchmark_setup
phase_3_contrast_pair_setup
phase_4_baseline_evaluation
phase_5_steering_sweep
phase_6_report
```

### `src/physics_steering_vectors/main.py`

Local function:
- Calls a few phase functions in order.

Global role:
- Acts as the top-level experiment script.

The main flow is intentionally small:

```python
config = ExperimentConfig()

model_bundle = phase_1_model_setup(config)
benchmark_splits = phase_2_benchmark_setup(config)
training_pairs = phase_3_contrast_pair_setup(config, model_bundle, benchmark_splits)

baseline = phase_4_baseline_evaluation(config, model_bundle, benchmark_splits)
steered_results = phase_5_steering_sweep(
    config,
    model_bundle,
    benchmark_splits,
    training_pairs,
)

phase_6_report([baseline, *steered_results])
```

## Sources Used

Model and model loading:

- Qwen3.5 model card: https://huggingface.co/Qwen/Qwen3.5-0.8B
- Transformers Qwen3.5 docs: https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5

Benchmark:

- MMLU-Pro dataset card: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
- MMLU-Pro official repo: https://github.com/TIGER-AI-Lab/MMLU-Pro

Steering:

- steering-vectors basic usage: https://steering-vectors.github.io/steering-vectors/basic_usage.html
- train_steering_vector API: https://steering-vectors.github.io/steering-vectors/api/train_steering_vector.html
- SteeringVector API: https://steering-vectors.github.io/steering-vectors/api/steering_vector.html

Python/data tooling:

- Hugging Face Datasets loading docs: https://huggingface.co/docs/datasets/en/loading
- PyTorch reproducibility notes: https://pytorch.org/docs/stable/notes/randomness.html
- pandas DataFrame docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
- Python `__main__` docs: https://docs.python.org/3/library/__main__.html

## Development Notes

Keep the experiment modular:

- Put protocol constants in `config.py`.
- Put data and prompt formatting in `data.py`.
- Put generated-response mining and positive/negative pair assembly in `activation_collection.py`.
- Put model loading in `modeling.py`.
- Put shared model completion logic in `generation.py`; do not duplicate generation code in activation collection or evaluation modules.
- Put intervention training in `steering.py`.
- Put benchmark scoring in `evaluation.py`.
- Keep `main.py` as simple phase orchestration.

When changing the experiment, prefer adding a new config field or a new phase helper over hiding
logic inside `main.py`. The goal is for a reader to understand the full research protocol quickly
without digging through implementation details first.
