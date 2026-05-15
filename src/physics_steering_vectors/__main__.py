"""Module entrypoint for `python -m physics_steering_vectors`.

Sources Used:
- Python package execution behavior: https://docs.python.org/3/library/__main__.html

Local Function:
- Imports and runs the same `main()` used by the console script.

Global Role:
- Gives the experiment a second clean entrypoint without duplicating orchestration logic.
"""

from physics_steering_vectors.main import main  # Local: import top-level orchestration. Global: run the same protocol from module mode.

if __name__ == "__main__":  # Local: guard direct module execution. Global: prevents accidental runs on import.
    main()  # Local: execute experiment. Global: starts all model/data/steering/eval phases.
