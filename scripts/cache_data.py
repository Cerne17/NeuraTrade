"""Refaz o cache de dados brutos da B3 via yfinance (M10).

**Único passo que acessa a rede.** Baixa OHLCV dos tickers do ``config.yaml`` e
grava em ``data/raw/<TICKER>.csv``. Os CSVs já estão versionados, então o restante
do projeto roda offline; rode este script apenas para atualizar/recriar o cache.

Uso:
    .venv/bin/python scripts/cache_data.py
    .venv/bin/python scripts/cache_data.py --tickers PETR4.SA
"""

from __future__ import annotations

import argparse

from src import data
from src.config import CONFIG


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recria o cache data/raw via yfinance.")
    parser.add_argument(
        "--tickers", nargs="+", default=None, help=f"default: {CONFIG['tickers']}"
    )
    args = parser.parse_args(argv)

    tickers = args.tickers or CONFIG["tickers"]
    print(f"Baixando {len(tickers)} ticker(s) via yfinance (requer rede)...")
    paths = data.cache_all(tickers)
    for t, p in paths.items():
        print(f"  {t} -> {p}")
    print("Cache atualizado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
