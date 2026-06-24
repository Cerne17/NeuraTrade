"""Refaz o cache de dados brutos via yfinance (B3) + BCB/yfinance (macro).

**Único passo que acessa a rede.** Baixa OHLCV dos tickers (`data/raw/<TICKER>.csv`)
e, por default, os indicadores macro (`data/raw/macro.csv`: USDBRL, VIX, Selic, IPCA
— indexados por data de publicação, ADR-0012). Tudo versionado → o resto roda offline.

Uso:
    .venv/bin/python scripts/cache_data.py                  # ações + macro
    .venv/bin/python scripts/cache_data.py --tickers PETR4.SA
    .venv/bin/python scripts/cache_data.py --no-macro       # só ações
    .venv/bin/python scripts/cache_data.py --only-macro     # só macro
"""

from __future__ import annotations

import argparse

from src import data
from src.config import CONFIG


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recria o cache data/raw (ações via yfinance, macro via BCB+yfinance)."
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None, help=f"default: {CONFIG['tickers']}"
    )
    parser.add_argument("--no-macro", action="store_true", help="pula o cache da macro.")
    parser.add_argument("--only-macro", action="store_true", help="só a macro, sem ações.")
    args = parser.parse_args(argv)

    if not args.only_macro:
        tickers = args.tickers or CONFIG["tickers"]
        print(f"Baixando {len(tickers)} ticker(s) via yfinance (requer rede)...")
        for t, p in data.cache_all(tickers).items():
            print(f"  {t} -> {p}")

    if not args.no_macro:
        from src.macro import cache_macro

        print("Baixando macro (BCB/SGS: USDBRL, Selic, IPCA; yfinance: VIX)...")
        print(f"  macro -> {cache_macro()}")

    print("Cache atualizado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
