from physics_steering_vectors.reporting import print_result_table
from physics_steering_vectors.schemas import EvaluationResult


def result(label: str, correct: int, total: int) -> EvaluationResult:
    return EvaluationResult(
        label=label,
        correct=correct,
        total=total,
        accuracy=correct / total if total else 0.0,
        records=[],
    )


def test_print_result_table_sorts_by_accuracy_and_computes_delta(capsys) -> None:
    print_result_table([result("baseline", 1, 2), result("steered", 3, 4)])

    output = capsys.readouterr().out
    lines = output.splitlines()
    assert "steered" in lines[1]
    assert "baseline" in lines[2]
    assert "delta_vs_baseline" in output
    assert "0.25" in output


def test_print_result_table_handles_empty_results(capsys) -> None:
    print_result_table([])

    output = capsys.readouterr().out
    assert "Empty DataFrame" in output
    assert "accuracy" in output
