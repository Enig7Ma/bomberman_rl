"""Entry point for the training CLI: ``python -m training``.

::

    uv run python -m training run --curriculum docs/experiments/dqn_d5/smoke.json \\
        --out results/dqn/smoke
"""

from training.cli import main

raise SystemExit(main())
