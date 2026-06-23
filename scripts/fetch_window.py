"""Baixa dados de uma janela de tempo nova para os tickers do projeto.

Útil para aplicar os modelos a um período fora do treino (ex.: Q1/2025). Grava os
CSVs em `data/inference/` (não toca no cache de treino `data/raw/`).

Uso:
    .venv/bin/python scripts/fetch_window.py --start 2025-01-01 --end 2025-03-31
    .venv/bin/python scripts/fetch_window.py --start 2025-01-01 --end 2025-03-31 \
        --tickers PETR4.SA VALE3.SA --out data/inference
"""

from __future__ import annotations

import argparse

from src.config import CONFIG
from src.inference import fetch_window


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Baixa OHLCV de uma janela nova (yfinance) para data/inference/."
    )
    parser.add_argument("--start", required=True, help="início (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="fim (YYYY-MM-DD).")
    parser.add_argument(
        "--tickers", nargs="+", default=None, help=f"default: {CONFIG['tickers']}"
    )
    parser.add_argument(
        "--out", default="data/inference", help="diretório de saída (default: data/inference)."
    )
    args = parser.parse_args(argv)

    tickers = args.tickers or CONFIG["tickers"]
    print(f"Baixando {len(tickers)} ticker(s) | {args.start} → {args.end} (requer rede)...")
    res = fetch_window(tickers, args.start, args.end, save_dir=args.out)
    for t, df in res.items():
        print(f"  {t}: {len(df)} pregões -> {args.out}/{t}.csv")
    print(f"\nPronto. Rode a inferência: .venv/bin/python scripts/run_inference.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
