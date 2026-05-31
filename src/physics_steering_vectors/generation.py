"""Shared model completion helpers.

Sources Used:
- Qwen2.5 model card: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- Transformers Qwen2 docs: https://huggingface.co/docs/transformers/model_doc/qwen2
- Qwen2.5 chat inference docs: https://qwen.readthedocs.io/en/v2.5/inference/chat.html
- Transformers chat templates docs: https://huggingface.co/docs/transformers/en/chat_templating
- Transformers text generation docs: https://huggingface.co/docs/transformers/main_classes/text_generation
- MMLU-Pro local evaluation script: https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_local.py
- SteeringVector API: https://steering-vectors.github.io/steering-vectors/api/steering_vector.html

Local Function:
- Tokenizes prompts, runs model generation, and decodes only new tokens.

Global Role:
- Gives evaluation and training-response mining one completion path.
"""

from contextlib import nullcontext  # Local: no-op context for unsteered generation. Global: share generation code across conditions.
from time import perf_counter  # Local: measure generation duration. Global: make long calls auditable.
from typing import Any  # Local: type steering vector object. Global: avoid binding to external library internals.

import torch  # Local: inference mode. Global: lower-memory generation calls.

from physics_steering_vectors.config import ExperimentConfig  # Local: read generation settings. Global: central protocol.
from physics_steering_vectors.logging_utils import get_logger  # Local: generation logs. Global: terminal audit trail.
from physics_steering_vectors.modeling import model_device  # Local: place inputs. Global: avoid CPU/GPU mismatch.
from physics_steering_vectors.schemas import ModelBundle  # Local: access model/tokenizer/hooks. Global: shared runtime.


logger = get_logger(__name__)

STOP_STRINGS = (  # Source: MMLU-Pro local eval stop=["Question:"] plus common chat spillover markers. Global: prevent run-on benchmark/dialogue completions.
    "\nQuestion:",
    "\nHuman:",
    "\nHumanity:",
    "\nAssistant:",
)
ANSWER_PREFILL = "Answer: Let's think step by step."  # Source: MMLU-Pro CoT prompt format. Local: assistant prefix for chat-template continuation.
ANSWER_PREFILL_MARKER = f"\n{ANSWER_PREFILL}"


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
    log_context: str | None = None,
) -> str:
    """Generate one completion with optional steering and decoding overrides.

    Sources Used:
    - Qwen2.5 model card: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
    - SteeringVector.apply docs: https://steering-vectors.github.io/steering-vectors/api/steering_vector.html

    Local Function:
    - Runs the model once and returns only generated text, excluding the prompt.

    Global Role:
    - Keeps benchmark evaluation and training-response mining behavior aligned.
    """

    context_label = log_context or "generation"
    device = model_device(bundle.model)
    inputs = _tokenize_prompt(bundle.tokenizer, prompt, device)  # Local: tokenize with chat template when available. Global: match instruct model input format.

    generation_kwargs: dict[str, Any] = {  # Local: collect shared generation arguments. Global: keep train/eval decoding differences explicit.
        **inputs,
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample if do_sample is None else do_sample,
        "pad_token_id": bundle.tokenizer.pad_token_id,
        "eos_token_id": _generation_eos_token_id(bundle),  # Source: Qwen generation_config preserves both <|im_end|> and <|endoftext|>. Global: stop on all model EOS IDs.
        "stop_strings": list(STOP_STRINGS),  # Source: HF generate stop_strings docs require tokenizer. Global: stop before new benchmark/chat turns.
        "tokenizer": bundle.tokenizer,
    }
    if temperature is not None:  # Local: optional sampling override. Global: allow training mining to diversify completions.
        generation_kwargs["temperature"] = temperature
    if top_p is not None:  # Local: optional nucleus sampling override. Global: allow training mining to diversify completions.
        generation_kwargs["top_p"] = top_p

    logger.debug(
        "Starting model generation context=%s prompt_chars=%d input_tokens=%d device=%s steering=%s multiplier=%s do_sample=%s max_new_tokens=%d temperature=%s top_p=%s",
        context_label,
        len(prompt),
        inputs["input_ids"].shape[-1],
        device,
        steering_vector is not None,
        multiplier,
        generation_kwargs["do_sample"],
        generation_kwargs["max_new_tokens"],
        generation_kwargs.get("temperature"),
        generation_kwargs.get("top_p"),
    )

    if steering_vector is None:  # Local: detect unsteered condition. Global: preserve baseline/control path.
        context = nullcontext()  # Local: no-op context. Global: same generation code path as steered condition.
    else:
        context = steering_vector.apply(  # Source: SteeringVector API. Local: install temporary hooks. Global: apply activation intervention.
            bundle.model,
            layer_config=bundle.layer_config,
            multiplier=multiplier,
            min_token_index=0,
        )

    started_at = perf_counter()
    with context:  # Local: enter baseline or steering context. Global: isolate intervention to this generation.
        output_ids = bundle.model.generate(**generation_kwargs)  # Source: Transformers generation API. Local: produce answer tokens. Global: benchmark/mining model behavior.
    duration_seconds = perf_counter() - started_at

    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]  # Local: remove prompt tokens. Global: answer extraction sees completion only.
    completion = bundle.tokenizer.decode(generated_ids, skip_special_tokens=True)  # Local: decode text. Global: provide extractable answer.
    completion = _strip_stop_suffix(completion)  # Local: remove generated stop marker from logs/scoring. Global: keep completions answer-only.
    logger.debug(
        "Completed model generation context=%s output_tokens=%d generated_tokens=%d completion_chars=%d duration_seconds=%.3f",
        context_label,
        output_ids.shape[-1],
        generated_ids.shape[-1],
        len(completion),
        duration_seconds,
    )
    return completion


def _tokenize_prompt(tokenizer: Any, prompt: str, device: torch.device) -> Any:
    """Tokenize one prompt, using an instruct-model chat template when available."""

    if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "apply_chat_template"):
        if prompt.endswith(ANSWER_PREFILL_MARKER):
            messages = [
                {"role": "user", "content": prompt[: -len(ANSWER_PREFILL_MARKER)]},
                {"role": "assistant", "content": ANSWER_PREFILL},
            ]
            # Source: HF chat-template docs https://huggingface.co/docs/transformers/en/chat_templating document continue_final_message for assistant prefill continuation.
            return tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

        # Source: Qwen chat docs https://qwen.readthedocs.io/en/v2.5/inference/chat.html and HF chat-template docs https://huggingface.co/docs/transformers/en/chat_templating.
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

    return tokenizer(prompt, return_tensors="pt").to(device)  # Source: HF tokenizer/generation pattern. Local: fallback for non-chat tokenizers.


def _generation_eos_token_id(bundle: ModelBundle) -> Any:
    """Return model generation EOS IDs without collapsing model-specific lists."""

    generation_config = getattr(bundle.model, "generation_config", None)
    eos_token_id = getattr(generation_config, "eos_token_id", None)
    if eos_token_id is not None:
        # Source: Qwen2.5 generation_config https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/generation_config.json lists both 151645 and 151643.
        return eos_token_id
    return bundle.tokenizer.eos_token_id


def _strip_stop_suffix(completion: str) -> str:
    """Remove a generated stop marker from the returned completion."""

    stripped = completion
    for stop_string in STOP_STRINGS:
        index = stripped.find(stop_string)
        if index != -1:
            stripped = stripped[:index]
    return stripped.rstrip()
