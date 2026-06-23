"""Wrapper fino da pipeline NeuraTrade (M10).

Atalho para a CLI ``python -m src``. Útil para quem prefere chamar um script
direto; a lógica vive em ``src/pipeline.py`` (e a CLI em ``src/__main__.py``).

Exemplos:
    .venv/bin/python scripts/run_pipeline.py
    .venv/bin/python scripts/run_pipeline.py --train
    .venv/bin/python scripts/run_pipeline.py --tickers PETR4.SA VALE3.SA
"""

from __future__ import annotations

import sys

from src.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
