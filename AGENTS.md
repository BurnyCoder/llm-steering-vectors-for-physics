# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.11 research package using a `src` layout. Runtime code lives in `src/physics_steering_vectors/`. The entry points are `main.py` for the experiment protocol and `__main__.py` for `python -m physics_steering_vectors`. Configuration is centralized in `config.py`; phase orchestration is in `phases.py`; domain logic is split across `data.py`, `modeling.py`, `steering.py`, `evaluation.py`, `answer_extraction.py`, `layers.py`, and `reporting.py`. Shared dataclasses live in `schemas.py`. There is no committed test suite yet; add tests under `tests/` using names like `tests/test_answer_extraction.py`.

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

For a quick smoke run, temporarily set `max_test_examples` in `src/physics_steering_vectors/config.py` to a small value such as `25`, then run `uv run physics-steering`. Restore it to `None` before benchmark-quality runs.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, type hints, dataclasses for structured records, and small modules with explicit phase boundaries. Use `snake_case` for functions, variables, and modules; use `PascalCase` for dataclasses. Keep experiment constants in `ExperimentConfig` rather than scattering literals across modules. Avoid introducing `pip`, `conda`, or ad hoc environment files; dependencies belong in `pyproject.toml`.

## Testing Guidelines

Prefer focused unit tests for deterministic logic such as answer extraction, result aggregation, prompt formatting, and layer-name inference. Mock Hugging Face model and dataset calls where possible so tests do not require GPU access or network downloads. If `pytest` is added, run tests with:

```bash
uv run pytest
```

For experiment changes, also run a capped smoke test before reporting results.

## Commit & Pull Request Guidelines

The current history only contains `Initial project`, so no detailed commit convention is established. Use short imperative commit subjects, for example `Add answer extraction tests` or `Refine steering sweep reporting`. Pull requests should describe the research or code change, note any configuration changes, include smoke-test or benchmark commands run, and call out hardware or dataset assumptions when results depend on them.

## Security & Configuration Tips

Do not commit downloaded models, datasets, generated caches, or virtual environments. Keep local secrets and Hugging Face credentials outside the repository. Treat `max_test_examples`, model IDs, layer sweeps, and multipliers as protocol-affecting settings and document any changes in PRs.
