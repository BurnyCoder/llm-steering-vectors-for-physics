"""Steering-vector training and persistence.

Sources Used:
- steering-vectors basic usage: https://steering-vectors.github.io/steering-vectors/basic_usage.html
- train_steering_vector API: https://steering-vectors.github.io/steering-vectors/api/train_steering_vector.html
- SteeringVector API: https://steering-vectors.github.io/steering-vectors/api/steering_vector.html
- torch.save docs: https://docs.pytorch.org/docs/2.12/generated/torch.save.html

Local Function:
- Wraps the library's steering-vector trainer and saves trained vector state.

Global Role:
- Produces the activation intervention tested on physics benchmark accuracy.
"""

from pathlib import Path  # Local: build artifact paths. Global: save vectors inside the project directory.
from typing import Any  # Local: type external steering object. Global: avoid depending on concrete class internals.

import torch  # Local: serialize tensor payloads. Global: persist trained steering vectors.
from steering_vectors import SteeringVector, train_steering_vector  # Source: steering-vectors docs. Local: train/recreate direction. Global: core library reuse.

from physics_steering_vectors.config import ExperimentConfig  # Local: read batch settings. Global: central protocol.
from physics_steering_vectors.schemas import ModelBundle  # Local: access model/tokenizer/hook config. Global: shared runtime.


VECTOR_FILE_VERSION = 1  # Local: version saved vector payloads. Global: fail clearly if the format changes later.


def steering_vector_path(config: ExperimentConfig, layer: int) -> Path:
    """Return the artifact path for one trained layer vector.

    Sources Used:
    - torch.save docs: https://docs.pytorch.org/docs/2.12/generated/torch.save.html

    Local Function:
    - Builds a deterministic filename for the configured model, subject, seed, and layer.

    Global Role:
    - Gives each trained intervention a stable project-local artifact location.
    """

    model_slug = config.model_id.replace("/", "__")  # Local: make model ID path-safe. Global: keep filenames traceable to model choice.
    subject_slug = config.subject.replace("/", "__")  # Local: make subject path-safe. Global: keep artifact names benchmark-specific.
    filename = f"{model_slug}_{subject_slug}_seed_{config.seed}_layer_{layer}.pt"  # Local: one file per trained layer. Global: avoid multiplier-specific duplicates.
    return Path(config.steering_vector_dir) / filename  # Local: combine configured directory and file. Global: centralize artifact layout.


def save_steering_vector(
    vector: SteeringVector,
    path: Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist a steering vector as reconstructable tensor state.

    Sources Used:
    - SteeringVector API: https://steering-vectors.github.io/steering-vectors/api/steering_vector.html
    - torch.save docs: https://docs.pytorch.org/docs/2.12/generated/torch.save.html

    Local Function:
    - Saves layer activations, layer type, and optional run metadata.

    Global Role:
    - Preserves trained interventions without pickling the full external object.
    """

    path.parent.mkdir(parents=True, exist_ok=True)  # Local: create artifact directory lazily. Global: runs work from a clean checkout.
    payload = {  # Local: store only simple metadata and tensors. Global: keep files reconstructable across Python sessions.
        "format_version": VECTOR_FILE_VERSION,
        "layer_type": vector.layer_type,
        "layer_activations": {
            layer: activation.detach().cpu()
            for layer, activation in vector.layer_activations.items()
        },
        "metadata": metadata or {},
    }
    torch.save(payload, path)  # Local: serialize tensor payload. Global: write reusable steering-vector artifact.
    return path  # Local: return exact saved location. Global: phase output can report where the vector went.


def load_steering_vector(path: Path, map_location: Any = "cpu") -> SteeringVector:
    """Load a steering vector saved by save_steering_vector.

    Sources Used:
    - SteeringVector API: https://steering-vectors.github.io/steering-vectors/api/steering_vector.html
    - torch.load docs: https://docs.pytorch.org/docs/2.12/generated/torch.load.html

    Local Function:
    - Recreates a library SteeringVector from saved layer tensor state.

    Global Role:
    - Enables later analysis or reuse without retraining in the same process.
    """

    payload = torch.load(path, map_location=map_location, weights_only=True)  # Local: load primitive/tensor payload. Global: avoid arbitrary object pickle loading.
    if payload.get("format_version") != VECTOR_FILE_VERSION:  # Local: validate persistence schema. Global: catch incompatible artifact versions early.
        raise ValueError(f"Unsupported steering vector file format: {payload.get('format_version')}")
    return SteeringVector(  # Source: SteeringVector dataclass. Local: restore apply-compatible vector. Global: reuse saved intervention.
        layer_activations={
            int(layer): activation
            for layer, activation in payload["layer_activations"].items()
        },
        layer_type=payload["layer_type"],
    )


def train_vector_for_layer(
    config: ExperimentConfig,
    bundle: ModelBundle,
    training_pairs: list[tuple[str, str]],
    layer: int,
) -> SteeringVector:
    """Train one steering vector for one decoder layer.

    Sources Used:
    - train_steering_vector API: https://steering-vectors.github.io/steering-vectors/api/train_steering_vector.html

    Local Function:
    - Converts positive/negative text pairs into a layer-specific activation direction.

    Global Role:
    - Creates the intervention later applied during benchmark generation.
    """

    return train_steering_vector(  # Source: steering-vectors API. Local: train vector. Global: reuse existing library instead of hand-rolling hooks.
        bundle.model,  # Local: model to record activations from. Global: exact same model evaluated later.
        bundle.tokenizer,  # Local: tokenize training pairs. Global: keep activation prompts model-native.
        training_pairs,  # Local: positive/negative strings. Global: accurate-minus-inaccurate physics direction.
        layers=[layer],  # Local: train only selected layer. Global: supports layer sweep.
        layer_type="decoder_block",  # Source: steering-vectors API default layer type. Local: hook decoder blocks. Global: steer residual stream at blocks.
        layer_config=bundle.layer_config,  # Local: Qwen module path template. Global: bridge Qwen to steering-vectors.
        read_token_index=-1,  # Source: train_steering_vector API. Local: read final token activation. Global: capture solution-level representation.
        batch_size=config.train_batch_size,  # Local: memory-aware batch size. Global: keeps experiment runnable.
        show_progress=True,  # Local: show training progress. Global: long runs remain observable.
    )
