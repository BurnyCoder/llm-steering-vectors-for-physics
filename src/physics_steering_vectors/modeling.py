"""Model loading helpers.

Sources Used:
- Qwen3.5 model card: https://huggingface.co/Qwen/Qwen3.5-0.8B
- Transformers Qwen3.5 docs: https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5

Local Function:
- Loads Qwen3.5, tokenizer, processor, and steering hook config.

Global Role:
- Creates the shared runtime object used for activation extraction, steering, and benchmark generation.
"""

import torch  # Source: PyTorch. Local: detect CUDA/device. Global: choose practical local execution settings.
from transformers import AutoModelForImageTextToText, AutoProcessor  # Source: Qwen3.5 model card. Local: documented loaders. Global: load selected checkpoint.

from physics_steering_vectors.config import ExperimentConfig  # Local: read model id. Global: keep protocol centralized.
from physics_steering_vectors.layers import infer_decoder_block_template  # Local: find hook path. Global: enable steering-vectors.
from physics_steering_vectors.schemas import ModelBundle  # Local: return typed bundle. Global: pass model runtime between phases.


def load_qwen35_bundle(config: ExperimentConfig) -> ModelBundle:
    """Load Qwen/Qwen3.5-0.8B.

    Sources Used:
    - Qwen3.5 model card: https://huggingface.co/Qwen/Qwen3.5-0.8B
    - Transformers Qwen3.5 docs: https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5
    - train_steering_vector API: https://steering-vectors.github.io/steering-vectors/api/train_steering_vector.html

    Local Function:
    - Instantiates model, processor, tokenizer, and layer config.

    Global Role:
    - Produces the object all later phases use for both normal and steered inference.
    """

    processor = AutoProcessor.from_pretrained(config.model_id)  # Source: Qwen3.5 model card. Local: load processor assets. Global: ensures tokenizer matches checkpoint.

    model_kwargs = {"torch_dtype": "auto"}  # Source: Transformers loading convention. Local: choose checkpoint dtype automatically. Global: avoid manual dtype mismatch.
    if torch.cuda.is_available():  # Source: PyTorch CUDA API. Local: detect GPU. Global: use available acceleration.
        model_kwargs["device_map"] = "auto"  # Source: Transformers/Accelerate convention. Local: place modules automatically. Global: simplify hardware setup.

    model = AutoModelForImageTextToText.from_pretrained(  # Source: Qwen3.5 model card. Local: load model class. Global: instantiate model to steer.
        config.model_id,  # Local: selected checkpoint id. Global: strict-latest small Qwen target.
        **model_kwargs,  # Local: pass dtype/device hints. Global: make loading practical across machines.
    ).eval()  # Source: PyTorch eval mode. Local: disable train-mode behavior. Global: keep benchmark inference consistent.

    tokenizer = processor.tokenizer  # Source: Qwen processor behavior. Local: expose tokenizer. Global: steering-vectors API expects tokenizer directly.
    if tokenizer.pad_token_id is None:  # Source: HF tokenizer padding behavior. Local: check padding availability. Global: allow batched activation extraction.
        tokenizer.pad_token = tokenizer.eos_token  # Local: use EOS as pad fallback. Global: make decoder-only-style text batches possible.
    tokenizer.padding_side = "left"  # Source: HF generation convention. Local: pad prompts on left. Global: preserve final prompt tokens for generation/activation reads.

    layer_config = {"decoder_block": infer_decoder_block_template(model)}  # Source: steering-vectors layer_config docs. Local: define hook matcher. Global: align vector layers to Qwen blocks.

    return ModelBundle(  # Local: package runtime objects. Global: standard phase output.
        model=model,  # Local: loaded model. Global: used by training/eval.
        processor=processor,  # Local: loaded processor. Global: retained for Qwen compatibility.
        tokenizer=tokenizer,  # Local: tokenizer. Global: shared prompt tokenization.
        layer_config=layer_config,  # Local: hook config. Global: steering intervention wiring.
    )


def model_device(model: torch.nn.Module) -> torch.device:
    """Return the device for input tensors.

    Sources Used:
    - Transformers generation examples: https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5

    Local Function:
    - Finds where model inputs should be placed.

    Global Role:
    - Prevents CPU/GPU mismatch during generation.
    """

    return getattr(model, "device", next(model.parameters()).device)  # Local: prefer HF device attr, fallback to first param. Global: route tensors to model.
