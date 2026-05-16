from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any

import torch

from physics_steering_vectors.config import ExperimentConfig
from physics_steering_vectors.generation import generate_completion
from physics_steering_vectors.schemas import ModelBundle


class FakeBatch(dict[str, torch.Tensor]):
    def __init__(self) -> None:
        super().__init__({"input_ids": torch.tensor([[101, 102]])})
        self.moved_to: torch.device | None = None

    def to(self, device: torch.device) -> "FakeBatch":
        self.moved_to = device
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self) -> None:
        self.batch = FakeBatch()
        self.decoded_ids: list[int] | None = None

    def __call__(self, prompt: str, return_tensors: str) -> FakeBatch:
        assert prompt == "prompt"
        assert return_tensors == "pt"
        return self.batch

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        self.decoded_ids = token_ids.tolist()
        return "decoded completion"


class FakeModel:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.generation_kwargs: dict[str, Any] | None = None

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.generation_kwargs = kwargs
        return torch.tensor([[101, 102, 201, 202]])


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
