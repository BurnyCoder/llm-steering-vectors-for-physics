from physics_steering_vectors import main as main_module
from physics_steering_vectors import phases
from physics_steering_vectors.config import ExperimentConfig
from physics_steering_vectors.schemas import BenchmarkSplits, EvaluationResult, ModelBundle


def test_phase_1_model_setup_seeds_loads_and_prints_hook(monkeypatch, capsys) -> None:
    config = ExperimentConfig(seed=99)
    bundle = ModelBundle(model=None, processor=None, tokenizer=None, layer_config={"decoder_block": "layers.{num}"})
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(phases, "set_reproducibility", lambda seed: calls.append(("seed", seed)))
    monkeypatch.setattr(phases, "load_qwen_bundle", lambda config_arg: bundle)

    assert phases.phase_1_model_setup(config) is bundle
    assert calls == [("seed", 99)]
    assert "Decoder hook template: layers.{num}" in capsys.readouterr().out


def test_phase_2_benchmark_setup_loads_and_prints_split_sizes(monkeypatch, capsys) -> None:
    splits = BenchmarkSplits(validation=[{}], test=[{}, {}], fewshot_prefix="prefix")
    monkeypatch.setattr(phases, "load_physics_splits", lambda config: splits)

    assert phases.phase_2_benchmark_setup(ExperimentConfig()) is splits
    output = capsys.readouterr().out
    assert "Validation physics rows: 1" in output
    assert "Test physics rows: 2" in output


def test_phase_3_contrast_pair_setup_mines_and_prints_pair_count(monkeypatch, capsys) -> None:
    pairs = [("positive", "negative")]
    splits = BenchmarkSplits(validation=[{"row": 1}], test=[], fewshot_prefix="")
    bundle = ModelBundle(model=None, processor=None, tokenizer=None, layer_config={})
    monkeypatch.setattr(phases, "build_training_pairs", lambda config, bundle_arg, rows: pairs)

    assert phases.phase_3_contrast_pair_setup(ExperimentConfig(), bundle, splits) == pairs
    assert "Training contrast pairs: 1" in capsys.readouterr().out


def test_phase_4_baseline_evaluation_delegates_with_baseline_label(monkeypatch) -> None:
    expected = EvaluationResult(label="baseline", correct=1, total=1, accuracy=1.0, records=[])
    splits = BenchmarkSplits(validation=[], test=[{"row": 1}], fewshot_prefix="prefix")
    bundle = ModelBundle(model=None, processor=None, tokenizer=None, layer_config={})
    calls: dict[str, object] = {}

    def fake_evaluate(**kwargs: object) -> EvaluationResult:
        calls.update(kwargs)
        return expected

    monkeypatch.setattr(phases, "evaluate", fake_evaluate)

    assert phases.phase_4_baseline_evaluation(ExperimentConfig(), bundle, splits) is expected
    assert calls["rows"] == splits.test
    assert calls["fewshot_prefix"] == "prefix"
    assert calls["label"] == "baseline"


def test_phase_5_steering_sweep_trains_each_layer_and_evaluates_each_multiplier(monkeypatch) -> None:
    config = ExperimentConfig(layer_sweep=(1, 2), multipliers=(0.5, 1.0))
    splits = BenchmarkSplits(validation=[], test=[{"row": 1}], fewshot_prefix="prefix")
    bundle = ModelBundle(model=None, processor=None, tokenizer=None, layer_config={})
    training_pairs = [("positive", "negative")]
    trained_layers: list[int] = []
    evaluate_calls: list[dict[str, object]] = []

    def fake_train_vector_for_layer(config_arg, bundle_arg, pairs_arg, layer: int) -> str:
        trained_layers.append(layer)
        return f"vector-{layer}"

    def fake_evaluate(**kwargs: object) -> EvaluationResult:
        evaluate_calls.append(kwargs)
        return EvaluationResult(label=str(kwargs["label"]), correct=0, total=1, accuracy=0.0, records=[])

    monkeypatch.setattr(phases, "train_vector_for_layer", fake_train_vector_for_layer)
    monkeypatch.setattr(phases, "evaluate", fake_evaluate)

    results = phases.phase_5_steering_sweep(config, bundle, splits, training_pairs)

    assert trained_layers == [1, 2]
    assert [result.label for result in results] == [
        "layer_1_mult_0.5",
        "layer_1_mult_1.0",
        "layer_2_mult_0.5",
        "layer_2_mult_1.0",
    ]
    assert [call["steering_vector"] for call in evaluate_calls] == ["vector-1", "vector-1", "vector-2", "vector-2"]


def test_phase_6_report_delegates_to_reporting(monkeypatch) -> None:
    results = [EvaluationResult(label="baseline", correct=0, total=1, accuracy=0.0, records=[])]
    calls: list[list[EvaluationResult]] = []
    monkeypatch.setattr(phases, "print_result_table", lambda received: calls.append(received))

    phases.phase_6_report(results)

    assert calls == [results]


def test_main_orchestrates_all_phases(monkeypatch) -> None:
    events: list[str] = []

    monkeypatch.setattr(main_module, "ExperimentConfig", lambda: "config")
    monkeypatch.setattr(main_module, "phase_1_model_setup", lambda config: events.append(f"phase1:{config}") or "bundle")
    monkeypatch.setattr(main_module, "phase_2_benchmark_setup", lambda config: events.append(f"phase2:{config}") or "splits")
    monkeypatch.setattr(
        main_module,
        "phase_3_contrast_pair_setup",
        lambda config, bundle, splits: events.append(f"phase3:{bundle}:{splits}") or "pairs",
    )
    monkeypatch.setattr(
        main_module,
        "phase_4_baseline_evaluation",
        lambda config, bundle, splits: events.append(f"phase4:{bundle}:{splits}") or "baseline",
    )
    monkeypatch.setattr(
        main_module,
        "phase_5_steering_sweep",
        lambda config, bundle, splits, pairs: events.append(f"phase5:{bundle}:{splits}:{pairs}") or ["steered"],
    )
    monkeypatch.setattr(main_module, "phase_6_report", lambda results: events.append(f"phase6:{results}"))

    main_module.main()

    assert events == [
        "phase1:config",
        "phase2:config",
        "phase3:bundle:splits",
        "phase4:bundle:splits",
        "phase5:bundle:splits:pairs",
        "phase6:['baseline', 'steered']",
    ]
