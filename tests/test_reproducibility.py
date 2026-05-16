import random

import torch

from physics_steering_vectors import reproducibility
from physics_steering_vectors.reproducibility import set_reproducibility


def test_set_reproducibility_sets_python_and_torch_seeds(monkeypatch) -> None:
    monkeypatch.setattr(reproducibility.torch.cuda, "is_available", lambda: False)

    set_reproducibility(123)

    assert random.random() == random.Random(123).random()
    assert torch.initial_seed() == 123
