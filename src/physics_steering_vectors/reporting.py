"""Result reporting.

Sources Used:
- pandas DataFrame docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
- pathlib docs: https://docs.python.org/3/library/pathlib.html

Local Function:
- Builds, prints, and saves sorted comparison tables.

Global Role:
- Makes the experiment outcome visible and preserves it as a local artifact.
"""

from pathlib import Path  # Local: create report artifact paths. Global: persist result tables after long runs.

import pandas as pd  # Local: tabulate result rows. Global: readable final comparison.

from physics_steering_vectors.logging_utils import get_logger, log_text_block  # Local: report logs. Global: terminal audit trail.
from physics_steering_vectors.schemas import EvaluationResult  # Local: typed result input. Global: report phase contract.


logger = get_logger(__name__)


def build_result_frame(results: list[EvaluationResult]) -> pd.DataFrame:
    """Build the sorted accuracy table.

    Sources Used:
    - pandas DataFrame docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html

    Local Function:
    - Converts results into a DataFrame and sorts by accuracy.

    Global Role:
    - Gives printing and file output one shared table definition.
    """

    logger.debug("Building result frame results=%d", len(results))
    baseline_accuracy = results[0].accuracy if results else 0.0  # Local: identify control accuracy. Global: compute intervention deltas.
    columns = ["label", "correct", "total", "accuracy", "delta_vs_baseline"]
    rows = [  # Local: flatten dataclasses. Global: prepare comparison table.
        {
            "label": result.label,
            "correct": result.correct,
            "total": result.total,
            "accuracy": result.accuracy,
            "delta_vs_baseline": result.accuracy - baseline_accuracy,
        }
        for result in results
    ]
    frame = pd.DataFrame(rows, columns=columns)  # Source: pandas DataFrame docs. Local: create table. Global: stable report shape.
    if rows:
        frame = frame.sort_values("accuracy", ascending=False)  # Local: sort populated table. Global: rank interventions.
    logger.debug("Built result frame rows=%d columns=%s", len(frame.index), list(frame.columns))
    return frame  # Local: return tabular data. Global: reusable result artifact source.


def print_result_table(results: list[EvaluationResult]) -> None:
    """Print accuracy table.

    Sources Used:
    - pandas DataFrame docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html

    Local Function:
    - Prints a full text table.

    Global Role:
    - Shows whether any steering condition improved over baseline.
    """

    table_text = build_result_frame(results).to_string(index=False)
    log_text_block(logger, True, "result_table", table_text)
    print(table_text)  # Local: print full table. Global: final research readout.


def write_result_report(
    results: list[EvaluationResult],
    report_dir: str | Path,
    stem: str = "latest_results",
) -> tuple[Path, Path]:
    """Save accuracy tables to Markdown and CSV files.

    Sources Used:
    - pandas DataFrame docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
    - pathlib docs: https://docs.python.org/3/library/pathlib.html

    Local Function:
    - Writes a human-readable Markdown report and machine-readable CSV.

    Global Role:
    - Preserves experiment results after terminal output scrolls away.
    """

    logger.info("Writing result report report_dir=%s stem=%s results=%d", report_dir, stem, len(results))
    frame = build_result_frame(results)  # Local: use same table as terminal output. Global: keep printed/saved reports identical.
    output_dir = Path(report_dir)  # Local: accept config strings or caller Paths. Global: configurable artifact location.
    output_dir.mkdir(parents=True, exist_ok=True)  # Local: create artifact directory lazily. Global: runs work from a clean checkout.

    markdown_path = output_dir / f"{stem}.md"  # Local: human-readable report path. Global: durable experiment artifact.
    csv_path = output_dir / f"{stem}.csv"  # Local: tabular report path. Global: enables later analysis.
    table_text = frame.to_string(index=False)  # Local: render exact printed table. Global: keep saved Markdown dependency-free.

    markdown_path.write_text(  # Local: write a compact report. Global: preserve the final comparison.
        f"# Physics Steering Results\n\n```text\n{table_text}\n```\n",
        encoding="utf-8",
    )
    frame.to_csv(csv_path, index=False)  # Local: write machine-readable table. Global: support spreadsheet/notebook analysis.
    logger.info("Wrote result report markdown_path=%s csv_path=%s", markdown_path, csv_path)
    return markdown_path, csv_path  # Local: expose paths to caller. Global: make saved artifacts discoverable.
