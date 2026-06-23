"""CLI da pipeline NeuraTrade (M10).

Roda a pipeline de detecção de anomalias end-to-end e imprime um resumo por ativo.

Exemplos:
    python -m src                      # carrega modelos salvos, detecta + avalia
    python -m src --train              # treina do zero (4 ativos) e detecta
    python -m src --tickers PETR4.SA   # só um ativo
    python -m src --no-evaluate        # pula a injeção sintética
"""

from __future__ import annotations

import argparse

from .config import CONFIG
from .pipeline import run_pipeline, summarize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Pipeline de detecção de anomalias (LSTM-Autoencoder) na B3.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="treina o modelo de cada ativo do zero (senão carrega models/<ticker>.keras).",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        metavar="TICKER",
        help=f"ativos a processar (default: {CONFIG['tickers']}).",
    )
    parser.add_argument(
        "--no-evaluate",
        dest="evaluate",
        action="store_false",
        help="pula a avaliação por injeção sintética (P/R/F1).",
    )
    parser.add_argument(
        "--verbose", type=int, default=0, help="verbosidade do treino Keras (0/1/2)."
    )
    args = parser.parse_args(argv)

    print(
        f"NeuraTrade · pipeline | agregação={CONFIG['detection'].get('aggregation')} "
        f"| treino={'sim' if args.train else 'carregar'}"
    )
    try:
        results = run_pipeline(
            tickers=args.tickers,
            train=args.train,
            evaluate=args.evaluate,
            verbose=args.verbose,
        )
    except FileNotFoundError as exc:
        print(f"\nERRO: {exc}\nDica: rode com --train para treinar os modelos.")
        return 1

    print("\nResumo por ativo:\n")
    print(summarize(results).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
