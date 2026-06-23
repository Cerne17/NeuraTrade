"""Gera as figuras finais do relatório em report/figures/ (issue #28).

Utilitário versionado (exceção no .gitignore): regenera, de forma reprodutível,
as figuras embutidas no relatório a partir dos modelos treinados. Usa os mesmos
plots de src/viz.py que os notebooks, garantindo consistência. As figuras ficam
em report/figures/ (auto-contido para compilar o PDF).

Rode: .venv/bin/python scripts/build_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import CONFIG, set_seeds
from src import data, preprocessing as pp, train as T, detect as D, events as E
from src.viz import (
    plot_error_distribution,
    plot_error_timeseries,
    plot_events_overlay,
)

OUT = Path(__file__).resolve().parent.parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SETORES = {
    "PETR4.SA": "Energia/commodities",
    "VALE3.SA": "Mineração",
    "AMER3.SA": "Varejo",
    "ITUB4.SA": "Financeiro",
}


def _save(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"escrito: {path}")


def main():
    set_seeds()
    tickers = CONFIG["tickers"]
    W = CONFIG["preprocessing"]["window_size"]

    pre = {t: pp.preprocess_ticker(df) for t, df in data.load_all().items()}
    models = {t: T.load_model(t) for t in tickers}

    err_tr = {t: D.reconstruction_error(models[t], pre[t]["X_train"]) for t in tickers}
    err_te = {t: D.reconstruction_error(models[t], pre[t]["X_test"]) for t in tickers}
    thr_s = {t: D.static_threshold(err_tr[t]) for t in tickers}
    thr_d = {t: D.dynamic_threshold(err_te[t]) for t in tickers}
    dates = {t: pre[t]["test_index"][W - 1:] for t in tickers}
    flags_s = {t: D.flag_anomalies(err_te[t], thr_s[t]) for t in tickers}

    # --- Fig M4: erro de teste com limiar estático e dinâmico (4 ativos) ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    for ax, t in zip(axes.ravel(), tickers):
        ax.plot(dates[t], err_te[t], lw=0.6, label="erro", color="steelblue")
        ax.axhline(thr_s[t], color="red", ls="--", lw=1, label="estático p95")
        ax.plot(dates[t], thr_d[t], color="green", lw=1, label="dinâmico")
        ax.set_title(t)
        ax.legend(fontsize=8)
    fig.suptitle("Erro de reconstrução (teste 2020–2024) e limiares")
    fig.tight_layout()
    _save(fig, "m4_erro_limiares")

    # --- Fig M5: distribuição do erro treino vs teste com p95 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, t in zip(axes.ravel(), tickers):
        plot_error_distribution(err_tr[t], err_te[t], thr_s[t], t, ax=ax)
    fig.suptitle("Distribuição do erro de reconstrução (MAE) — treino vs. teste")
    fig.tight_layout()
    _save(fig, "m5_distribuicao_erro")

    # --- Fig M6a: AMER3 detalhe (fraude 2023) com eventos ---
    t = "AMER3.SA"
    d = dates[t]
    mask = d >= "2022-01-01"
    fig, ax = plt.subplots(figsize=(14, 4))
    plot_error_timeseries(err_te[t][mask], d[mask], thr_s[t], flags_s[t][mask], t, ax=ax)
    plot_events_overlay(ax, E.events_in_range("2022-01-01", d.max(), ticker=t))
    fig.suptitle("AMER3 — erro, anomalias e eventos (2022–2024)")
    fig.tight_layout()
    _save(fig, "m6_amer3_detalhe")

    # --- Fig M6b: contágio COVID (4 ativos) ---
    cs, ce = "2020-02-01", "2020-05-31"
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    for ax, t in zip(axes.ravel(), tickers):
        d = dates[t]
        m = (d >= cs) & (d <= ce)
        plot_error_timeseries(err_te[t][m], d[m], thr_s[t], flags_s[t][m], t, ax=ax)
        plot_events_overlay(ax, E.events_in_range(cs, ce))
        ax.set_title(f"{t} ({SETORES[t]}) — {int(flags_s[t][m].sum())} anomalias")
    fig.suptitle("Crash COVID (fev–mai/2020): contágio entre setores")
    fig.tight_layout()
    _save(fig, "m6_covid_contagio")


if __name__ == "__main__":
    main()
