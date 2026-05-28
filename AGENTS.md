# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.14 research package using a `src` layout. Runtime code lives in `src/physics_steering_vectors/`. The entry points are `main.py` for the experiment protocol and `__main__.py` for `python -m physics_steering_vectors`. Configuration is centralized in `config.py`; phase orchestration is in `phases.py`; shared model completion is in `generation.py`; data loading and prompt formatting live in `data.py`; generated-response mining and contrast-pair assembly live in `activation_collection.py`; model loading, vector training and persistence, evaluation, answer extraction, layer inference, logging, reproducibility, and reporting live in `modeling.py`, `steering.py`, `evaluation.py`, `answer_extraction.py`, `layers.py`, `logging_utils.py`, `reproducibility.py`, and `reporting.py`. Shared dataclasses live in `schemas.py`. Python and dependency state are tracked through `.python-version`, `pyproject.toml`, and `uv.lock`. Tests live under `tests/` and cover deterministic logic, model/dataset adapters with mocks, logging utilities, reproducibility, steering-vector persistence, and top-level phase orchestration. The LaTeX research write-up lives in `paper/paper.tex`, with `paper.pdf` tracked at the repository root. Generated steering-vector artifacts are written under `artifacts/steering_vectors/`, generated reports under `artifacts/reports/`, and run logs under `artifacts/logs/`; these generated artifacts are ignored by Git.

## Build, Test, and Development Commands

Use `uv` for all dependency management.

```bash
uv venv
source .venv/bin/activate
uv sync
```

Run the full experiment with:

```bash
uv run physics-steering
uv run python -m physics_steering_vectors
```

For a quick smoke run, temporarily set `max_test_examples` to a small value such as `25` and consider lowering `train_generations_per_question` in `src/physics_steering_vectors/config.py`, then run `uv run physics-steering`. Restore protocol-affecting settings before benchmark-quality runs.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, type hints, dataclasses for structured records, and small modules with explicit phase boundaries. Use `snake_case` for functions, variables, and modules; use `PascalCase` for dataclasses. Keep experiment constants in `ExperimentConfig` rather than scattering literals across modules. Keep logging setup and full prompt/completion block logging centralized in `logging_utils.py`. Keep model completion code centralized in `generation.py`; training-response mining belongs in `activation_collection.py`, and both activation collection and evaluation should call the shared generation helper rather than duplicating tokenizer/generate/decode logic. Training contrast pairs should come from actual model-generated validation responses classified by extracted answer correctness. Do not recreate the old rule-based negative path by editing only final answer letters. Keep steering-vector serialization in `steering.py`, storing versioned tensor payloads rather than pickling the external object wholesale. Avoid introducing `pip`, `conda`, or ad hoc environment files; dependencies belong in `pyproject.toml`.

## Testing Guidelines

Prefer focused unit tests for deterministic logic such as answer extraction, result aggregation, prompt formatting, generation helpers, logging utilities, reproducibility, phase orchestration, layer-name inference, and steering-vector save/load behavior. Mock Hugging Face model, tokenizer, dataset, and steering-library calls where possible so tests do not require GPU access or network downloads. Run tests with:

```bash
uv run pytest
```

For experiment changes, also run a capped smoke test before reporting results.

## Commit & Pull Request Guidelines

Use short imperative commit subjects, following the existing style, for example `Add answer extraction tests` or `Refine steering sweep reporting`. Pull requests should describe the research or code change, note any configuration changes, include unit-test, smoke-test, or benchmark commands run, and call out hardware or dataset assumptions when results depend on them.

## Security & Configuration Tips

Do not commit downloaded models, datasets, generated caches, generated steering-vector `.pt` files, generated reports, generated logs, or virtual environments. Keep local secrets and Hugging Face credentials outside the repository. Treat `max_test_examples`, `train_generations_per_question`, training sampling settings, model IDs, layer sweeps, multipliers, `steering_vector_dir`, `report_dir`, and `log_file_path` as protocol-affecting settings and document any changes in PRs.
