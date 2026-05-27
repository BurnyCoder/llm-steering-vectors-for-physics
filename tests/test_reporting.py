from physics_steering_vectors.reporting import build_result_frame, print_result_table, write_result_report
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


def test_build_result_frame_returns_sorted_rows() -> None:
    frame = build_result_frame([result("baseline", 1, 2), result("steered", 3, 4)])

    assert frame["label"].tolist() == ["steered", "baseline"]
    assert frame["delta_vs_baseline"].tolist() == [0.25, 0.0]


def test_write_result_report_saves_markdown_and_csv(tmp_path) -> None:
    markdown_path, csv_path = write_result_report(
        [result("baseline", 1, 2), result("steered", 3, 4)],
        tmp_path,
        stem="report",
    )

    assert markdown_path == tmp_path / "report.md"
    assert csv_path == tmp_path / "report.csv"
    assert "Physics Steering Results" in markdown_path.read_text(encoding="utf-8")
    assert "steered" in markdown_path.read_text(encoding="utf-8")
    assert "label,correct,total,accuracy,delta_vs_baseline" in csv_path.read_text(encoding="utf-8")


def test_print_result_table_handles_empty_results(capsys) -> None:
    print_result_table([])

    output = capsys.readouterr().out
    assert "Empty DataFrame" in output
    assert "accuracy" in output
