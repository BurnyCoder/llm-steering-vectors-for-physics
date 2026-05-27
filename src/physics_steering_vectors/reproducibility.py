"""Reproducibility helpers.

Sources Used:
- PyTorch reproducibility notes: https://pytorch.org/docs/stable/notes/randomness.html

Local Function:
- Sets Python and Torch seeds.

Global Role:
- Makes baseline and steered runs more comparable by reducing avoidable randomness.
"""

import random  # Local: seed Python RNG. Global: controls random choices if parse fallback or sampling is added.

import torch  # Local: seed Torch RNG. Global: controls model-related stochasticity where possible.

from physics_steering_vectors.logging_utils import get_logger  # Local: seed logs. Global: terminal audit trail.


logger = get_logger(__name__)


def set_reproducibility(seed: int) -> None:
    """Set random seeds used by the experiment.

    Sources Used:
    - PyTorch reproducibility notes: https://pytorch.org/docs/stable/notes/randomness.html

    Local Function:
    - Applies the configured seed to Python and Torch.

    Global Role:
    - Makes result differences more attributable to steering rather than incidental randomness.
    """

    logger.info("Setting reproducibility seed=%d", seed)
    random.seed(seed)  # Local: seed Python random. Global: stabilizes any Python-level randomness.
    torch.manual_seed(seed)  # Local: seed CPU Torch RNG. Global: stabilizes Torch operations where deterministic.
    if torch.cuda.is_available():  # Local: detect CUDA. Global: handle GPU runs consistently.
        torch.cuda.manual_seed_all(seed)  # Local: seed all CUDA devices. Global: stabilizes multi-GPU RNG state.
        logger.info("Seeded CUDA RNGs seed=%d", seed)
    else:
        logger.info("CUDA unavailable; skipped CUDA seeding")
