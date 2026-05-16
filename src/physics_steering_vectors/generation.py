"""Shared model completion helpers.

Sources Used:
- Qwen3.5 model card: https://huggingface.co/Qwen/Qwen3.5-0.8B
- Transformers Qwen3.5 docs: https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5
- SteeringVector API: https://steering-vectors.github.io/steering-vectors/api/steering_vector.html

Local Function:
- Tokenizes prompts, runs model generation, and decodes only new tokens.

Global Role:
- Gives evaluation and training-response mining one completion path.
"""

from contextlib import nullcontext  # Local: no-op context for unsteered generation. Global: share generation code across conditions.
from typing import Any  # Local: type steering vector object. Global: avoid binding to external library internals.

import torch  # Local: inference mode. Global: lower-memory generation calls.

from physics_steering_vectors.config import ExperimentConfig  # Local: read generation settings. Global: central protocol.
from physics_steering_vectors.modeling import model_device  # Local: place inputs. Global: avoid CPU/GPU mismatch.
from physics_steering_vectors.schemas import ModelBundle  # Local: access model/tokenizer/hooks. Global: shared runtime.


@torch.inference_mode()  # Source: PyTorch inference mode. Local: disable gradients. Global: faster, lower-memory generation.
def generate_completion(
    config: ExperimentConfig,
    bundle: ModelBundle,
    prompt: str,
    steering_vector: Any | None = None,
    multiplier: float = 1.0,
    do_sample: bool | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> str:
    """Generate one completion with optional steering and decoding overrides.

    Sources Used:
    - Qwen3.5 model card: https://huggingface.co/Qwen/Qwen3.5-0.8B
    - SteeringVector.apply docs: https://steering-vectors.github.io/steering-vectors/api/steering_vector.html

    Local Function:
    - Runs the model once and returns only generated text, excluding the prompt.

    Global Role:
    - Keeps benchmark evaluation and training-response mining behavior aligned.
    """

    inputs = bundle.tokenizer(prompt, return_tensors="pt").to(model_device(bundle.model))  # Source: HF tokenizer/generation pattern. Local: tokenize and move. Global: prepare model input.

    generation_kwargs: dict[str, Any] = {  # Local: collect shared generation arguments. Global: keep train/eval decoding differences explicit.
        **inputs,
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample if do_sample is None else do_sample,
        "pad_token_id": bundle.tokenizer.pad_token_id,
        "eos_token_id": bundle.tokenizer.eos_token_id,
    }
    if temperature is not None:  # Local: optional sampling override. Global: allow training mining to diversify completions.
        generation_kwargs["temperature"] = temperature
    if top_p is not None:  # Local: optional nucleus sampling override. Global: allow training mining to diversify completions.
        generation_kwargs["top_p"] = top_p

    if steering_vector is None:  # Local: detect unsteered condition. Global: preserve baseline/control path.
        context = nullcontext()  # Local: no-op context. Global: same generation code path as steered condition.
    else:
        context = steering_vector.apply(  # Source: SteeringVector API. Local: install temporary hooks. Global: apply activation intervention.
            bundle.model,
            layer_config=bundle.layer_config,
            multiplier=multiplier,
            min_token_index=0,
        )

    with context:  # Local: enter baseline or steering context. Global: isolate intervention to this generation.
        output_ids = bundle.model.generate(**generation_kwargs)  # Source: Transformers generation API. Local: produce answer tokens. Global: benchmark/mining model behavior.

    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]  # Local: remove prompt tokens. Global: answer extraction sees completion only.
    return bundle.tokenizer.decode(generated_ids, skip_special_tokens=True)  # Local: decode text. Global: provide extractable answer.
