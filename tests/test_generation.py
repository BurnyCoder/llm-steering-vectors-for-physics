from contextlib import AbstractContextManager
from types import SimpleNamespace, TracebackType
from typing import Any

import torch

from physics_steering_vectors.config import ExperimentConfig
from physics_steering_vectors.generation import generate_completion
from physics_steering_vectors.schemas import ModelBundle


class FakeBatch(dict[str, torch.Tensor]):
    def __init__(self, input_ids: list[int] | None = None) -> None:
        super().__init__({"input_ids": torch.tensor([input_ids or [101, 102]])})
        self.moved_to: torch.device | None = None

    def to(self, device: torch.device) -> "FakeBatch":
        self.moved_to = device
        return self


class FakeTokenizer:
    chat_template = None
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self, decode_text: str = "decoded completion") -> None:
        self.batch = FakeBatch()
        self.decode_text = decode_text
        self.decoded_ids: list[int] | None = None
        self.raw_prompt: str | None = None

    def __call__(self, prompt: str, return_tensors: str) -> FakeBatch:
        assert prompt == "prompt"
        assert return_tensors == "pt"
        self.raw_prompt = prompt
        return self.batch

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        self.decoded_ids = token_ids.tolist()
        return self.decode_text


class FakeChatTokenizer(FakeTokenizer):
    chat_template = "chat template"

    def __init__(self) -> None:
        super().__init__()
        self.batch = FakeBatch([301, 302, 303])
        self.chat_messages: list[dict[str, str]] | None = None
        self.chat_kwargs: dict[str, object] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        return_dict: bool = False,
        return_tensors: str = "",
    ) -> FakeBatch:
        assert tokenize is True
        assert return_dict is True
        assert return_tensors == "pt"
        self.chat_messages = messages
        self.chat_kwargs = {
            "add_generation_prompt": add_generation_prompt,
            "continue_final_message": continue_final_message,
        }
        return self.batch


class FakeModel:
    device = torch.device("cpu")

    def __init__(self, generation_config: Any | None = None) -> None:
        self.generation_kwargs: dict[str, Any] | None = None
        self.generation_config = generation_config

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.generation_kwargs = kwargs
        suffix = torch.tensor([[201, 202]])
        return torch.cat([kwargs["input_ids"], suffix], dim=1)


def make_bundle(model: FakeModel, tokenizer: FakeTokenizer) -> ModelBundle:
    return ModelBundle(
        model=model,
        processor=None,
        tokenizer=tokenizer,
        layer_config={"decoder_block": "model.layers.{num}"},
    )


def test_generate_completion_decodes_only_new_tokens() -> None:
    model = FakeModel()
    tokenizer = FakeTokenizer()
    config = ExperimentConfig(max_new_tokens=7, do_sample=False)

    text = generate_completion(config, make_bundle(model, tokenizer), "prompt")

    assert text == "decoded completion"
    assert tokenizer.batch.moved_to == torch.device("cpu")
    assert tokenizer.decoded_ids == [201, 202]
    assert model.generation_kwargs is not None
    assert model.generation_kwargs["max_new_tokens"] == 7
    assert model.generation_kwargs["do_sample"] is False
    assert model.generation_kwargs["pad_token_id"] == 0
    assert model.generation_kwargs["eos_token_id"] == 2
    assert model.generation_kwargs["stop_strings"] == ["\nQuestion:", "\nHuman:", "\nHumanity:", "\nAssistant:"]
    assert model.generation_kwargs["tokenizer"] is tokenizer


def test_generate_completion_uses_chat_template_when_available() -> None:
    model = FakeModel()
    tokenizer = FakeChatTokenizer()

    generate_completion(ExperimentConfig(), make_bundle(model, tokenizer), "prompt")

    assert tokenizer.raw_prompt is None
    assert tokenizer.chat_messages == [{"role": "user", "content": "prompt"}]
    assert tokenizer.chat_kwargs == {"add_generation_prompt": True, "continue_final_message": False}
    assert tokenizer.batch.moved_to == torch.device("cpu")
    assert model.generation_kwargs is not None
    assert model.generation_kwargs["input_ids"].tolist() == [[301, 302, 303]]


def test_generate_completion_continues_mmlu_answer_prefill_with_chat_template() -> None:
    model = FakeModel()
    tokenizer = FakeChatTokenizer()
    prompt = "Question:\nWhat is 2+2?\nAnswer: Let's think step by step."

    generate_completion(ExperimentConfig(), make_bundle(model, tokenizer), prompt)

    assert tokenizer.chat_messages == [
        {"role": "user", "content": "Question:\nWhat is 2+2?"},
        {"role": "assistant", "content": "Answer: Let's think step by step."},
    ]
    assert tokenizer.chat_kwargs == {"add_generation_prompt": False, "continue_final_message": True}


def test_generate_completion_preserves_model_generation_eos_list() -> None:
    model = FakeModel(generation_config=SimpleNamespace(eos_token_id=[151645, 151643]))
    tokenizer = FakeTokenizer()

    generate_completion(ExperimentConfig(), make_bundle(model, tokenizer), "prompt")

    assert model.generation_kwargs is not None
    assert model.generation_kwargs["eos_token_id"] == [151645, 151643]


def test_generate_completion_strips_generated_stop_suffix() -> None:
    model = FakeModel()
    tokenizer = FakeTokenizer(decode_text="The answer is (A).\nQuestion:")

    text = generate_completion(ExperimentConfig(), make_bundle(model, tokenizer), "prompt")

    assert text == "The answer is (A)."


def test_generate_completion_accepts_sampling_overrides() -> None:
    model = FakeModel()
    tokenizer = FakeTokenizer()

    generate_completion(
        ExperimentConfig(do_sample=False),
        make_bundle(model, tokenizer),
        "prompt",
        do_sample=True,
        temperature=0.4,
        top_p=0.8,
    )

    assert model.generation_kwargs is not None
    assert model.generation_kwargs["do_sample"] is True
    assert model.generation_kwargs["temperature"] == 0.4
    assert model.generation_kwargs["top_p"] == 0.8


def test_generate_completion_applies_steering_context() -> None:
    class FakeContext(AbstractContextManager[None]):
        def __init__(self) -> None:
            self.entered = False
            self.exited = False

        def __enter__(self) -> None:
            self.entered = True
            return None

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> bool:
            self.exited = True
            return False

    class FakeSteeringVector:
        def __init__(self) -> None:
            self.context = FakeContext()
            self.apply_kwargs: dict[str, object] | None = None

        def apply(self, model: object, **kwargs: object) -> FakeContext:
            self.apply_kwargs = {"model": model, **kwargs}
            return self.context

    model = FakeModel()
    tokenizer = FakeTokenizer()
    steering_vector = FakeSteeringVector()

    generate_completion(
        ExperimentConfig(),
        make_bundle(model, tokenizer),
        "prompt",
        steering_vector=steering_vector,
        multiplier=1.5,
    )

    assert steering_vector.context.entered is True
    assert steering_vector.context.exited is True
    assert steering_vector.apply_kwargs == {
        "model": model,
        "layer_config": {"decoder_block": "model.layers.{num}"},
        "multiplier": 1.5,
        "min_token_index": 0,
    }
