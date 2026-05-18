"""Steering-vector training.

Sources Used:
- steering-vectors basic usage: https://steering-vectors.github.io/steering-vectors/basic_usage.html
- train_steering_vector API: https://steering-vectors.github.io/steering-vectors/api/train_steering_vector.html

Local Function:
- Wraps one call to the library's steering-vector trainer.

Global Role:
- Produces the activation intervention tested on physics benchmark accuracy.
"""

from typing import Any  # Local: type external steering object. Global: avoid depending on concrete class internals.

from steering_vectors import train_steering_vector  # Source: steering-vectors docs. Local: train direction. Global: core library reuse.

from physics_steering_vectors.config import ExperimentConfig  # Local: read batch settings. Global: central protocol.
from physics_steering_vectors.schemas import ModelBundle  # Local: access model/tokenizer/hook config. Global: shared runtime.


def train_vector_for_layer(
    config: ExperimentConfig,
    bundle: ModelBundle,
    training_pairs: list[tuple[str, str]],
    layer: int,
) -> Any:
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
