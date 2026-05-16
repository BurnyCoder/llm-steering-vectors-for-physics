from physics_steering_vectors import __main__ as module_entrypoint
from physics_steering_vectors import main as main_module


def test_module_entrypoint_reuses_main_function() -> None:
    assert module_entrypoint.main is main_module.main
