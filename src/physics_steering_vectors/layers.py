"""Layer-hook discovery for Qwen models.

Sources Used:
- train_steering_vector API: https://steering-vectors.github.io/steering-vectors/api/train_steering_vector.html
- Transformers Qwen2 docs: https://huggingface.co/docs/transformers/model_doc/qwen2

Local Function:
- Finds the decoder block module path template expected by steering-vectors.

Global Role:
- Connects Qwen internals to the activation-hook library.
"""

from torch import nn  # Local: type model modules. Global: accepts any HF model exposing named_modules.

from physics_steering_vectors.logging_utils import get_logger  # Local: layer logs. Global: terminal audit trail.


logger = get_logger(__name__)


def infer_decoder_block_template(model: nn.Module) -> str:
    """Infer the decoder block path template.

    Sources Used:
    - train_steering_vector API: https://steering-vectors.github.io/steering-vectors/api/train_steering_vector.html
    - Transformers Qwen2 docs: https://huggingface.co/docs/transformers/model_doc/qwen2

    Local Function:
    - Searches likely Qwen module names for layer zero.

    Global Role:
    - Produces the `layer_config` needed to record and patch activations.
    """

    module_names = set(dict(model.named_modules()).keys())  # Local: collect module paths. Global: inspect actual loaded model, not assumptions.
    logger.debug("Inferring decoder block template module_count=%d", len(module_names))

    candidates = [  # Local: likely decoder block templates. Global: supports text-only and multimodal wrapper variants.
        "model.language_model.layers.{num}",  # Local: possible wrapped text backbone path. Global: hook target if model nests language model.
        "model.language_model.model.layers.{num}",  # Local: possible double-wrapped path. Global: robust to HF architecture wrappers.
        "language_model.layers.{num}",  # Local: possible direct language model path. Global: robust to alternate class layout.
        "language_model.model.layers.{num}",  # Local: possible nested direct path. Global: handles tokenizer/model wrappers.
        "model.layers.{num}",  # Local: common decoder-only path. Global: works for Qwen2-style causal LM modules.
    ]

    for template in candidates:  # Local: test templates in priority order. Global: choose a hook path automatically.
        logger.debug("Checking decoder block template candidate=%s", template)
        if template.format(num=0) in module_names:  # Local: check layer zero exists. Global: validates template before steering.
            logger.info("Selected decoder block template=%s", template)
            return template  # Local: return working template. Global: lets steering-vectors resolve any layer number.

    nearby = sorted(name for name in module_names if "layers.0" in name)[:25]  # Local: collect debugging clues. Global: helps update code if HF layout changes.
    logger.error("Could not infer decoder layer template nearby=%s", nearby)
    raise RuntimeError(  # Local: fail loudly. Global: avoids silently training vectors on wrong or missing layers.
        "Could not infer decoder layer template. "
        f"Nearby module names containing 'layers.0': {nearby}"
    )
