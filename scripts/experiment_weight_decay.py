"""Experimento: weight decay (AdamW) vs Adam puro — walk-forward (ADR-0018).

Compara, por ativo e via validação walk-forward (ADR-0010), o ``val_loss`` médio
do baseline atual (``weight_decay=0`` = Adam) contra candidatos de decay
(AdamW). O veredito segue o critério do projeto: a diferença só conta se for
**maior que o desvio inter-fold** — caso contrário é ruído (mesma régua que
rejeitou a varredura de ``latent_dim``, ADR-0010).

Uso:
    python scripts/experiment_weight_decay.py
    python scripts/experiment_weight_decay.py --tickers PETR4.SA --epochs 40

Saída: tabela por ativo + agregado + veredito no stdout; CSV em
``--out`` (default: figures/experiment_weight_decay.csv).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CONFIG, set_seeds  # noqa: E402
from src.data import load_ticker  # noqa: E402
from src.preprocessing import log_returns, temporal_split  # noqa: E402
from src.validation import cross_validate_weight_decay  # noqa: E402

CANDIDATES = [0.0, 1e-5, 1e-4, 1e-3]


def run(tickers: list[str], epochs: int | None, n_splits: int | None) -> pd.DataFrame:
    set_seeds()
    all_rows = []
    for ticker in tickers:
        print(f"\n=== {ticker} (walk-forward, {len(CANDIDATES)} candidatos) ===", flush=True)
        df = load_ticker(ticker)
        train, _ = temporal_split(log_returns(df))
        res = cross_validate_weight_decay(
            train, candidates=CANDIDATES, n_splits=n_splits, epochs=epochs
        )
        print(res.to_string(float_format=lambda x: f"{x:.6f}"), flush=True)
        res = res.reset_index()
        res.insert(0, "ticker", ticker)
        all_rows.append(res)
    return pd.concat(all_rows, ignore_index=True)


def verdict(table: pd.DataFrame) -> None:
    """Imprime o veredito por ativo: cada candidato vs baseline (wd=0)."""
    print("\n=== VEREDITO (delta de val_loss vs baseline wd=0; régua = desvio inter-fold) ===")
    for ticker, g in table.groupby("ticker"):
        base = g.loc[g["weight_decay"] == 0.0, "val_loss_mean"].iloc[0]
        base_std = g.loc[g["weight_decay"] == 0.0, "val_loss_std"].iloc[0]
        print(f"\n{ticker}: baseline val_loss = {base:.6f} (±{base_std:.6f})")
        for _, row in g[g["weight_decay"] > 0].iterrows():
            wd = row["weight_decay"]
            delta = row["val_loss_mean"] - base
            ruler = max(base_std, row["val_loss_std"])  # desvio dominante
            if abs(delta) <= ruler:
                tag = "INDIFERENTE (dentro do ruído inter-fold)"
            elif delta < 0:
                tag = "MELHOR (val_loss menor além do ruído)"
            else:
                tag = "PIOR (val_loss maior além do ruído)"
            print(f"  wd={wd:<8g} delta={delta:+.6f}  régua=±{ruler:.6f}  -> {tag}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", nargs="+", default=CONFIG["tickers"])
    ap.add_argument("--epochs", type=int, default=None, help="teto de épocas/fold")
    ap.add_argument("--n-splits", type=int, default=None, help="folds walk-forward")
    ap.add_argument(
        "--out", default="figures/experiment_weight_decay.csv", help="CSV de saída"
    )
    args = ap.parse_args()

    table = run(args.tickers, args.epochs, args.n_splits)
    verdict(table)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\nCSV salvo em {out}")


if __name__ == "__main__":
    main()
