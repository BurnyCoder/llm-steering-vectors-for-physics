import pytest
from torch import nn

from physics_steering_vectors.layers import infer_decoder_block_template


def test_infer_decoder_block_template_finds_common_model_layers_path() -> None:
    class ModelWithLayers(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Linear(1, 1)])

    assert infer_decoder_block_template(ModelWithLayers()) == "model.layers.{num}"


def test_infer_decoder_block_template_uses_priority_order() -> None:
    class ModelWithMultipleCandidates(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Linear(1, 1)])
            self.model.language_model = nn.Module()
            self.model.language_model.layers = nn.ModuleList([nn.Linear(1, 1)])

    assert infer_decoder_block_template(ModelWithMultipleCandidates()) == "model.language_model.layers.{num}"


def test_infer_decoder_block_template_reports_nearby_layer_names() -> None:
    class UnknownLayerLayout(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.foo = nn.Module()
            self.foo.layers = nn.ModuleList([nn.Linear(1, 1)])

    with pytest.raises(RuntimeError, match="foo.layers.0"):
        infer_decoder_block_template(UnknownLayerLayout())
