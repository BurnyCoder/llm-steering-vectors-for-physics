"""Result reporting.

Sources Used:
- pandas DataFrame docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html

Local Function:
- Prints one sorted comparison table.

Global Role:
- Makes the experiment outcome visible: accuracy and delta from baseline.
"""

import pandas as pd  # Local: tabulate result rows. Global: readable final comparison.

from physics_steering_vectors.schemas import EvaluationResult  # Local: typed result input. Global: report phase contract.


def print_result_table(results: list[EvaluationResult]) -> None:
    """Print accuracy table.

    Sources Used:
    - pandas DataFrame docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html

    Local Function:
    - Converts results into a DataFrame and sorts by accuracy.

    Global Role:
    - Shows whether any steering condition improved over baseline.
    """

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
    print(frame.to_string(index=False))  # Local: print full table. Global: final research readout.
