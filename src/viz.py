"""Plots padronizados do projeto (issue #24).

Todas as funções aceitam um ``ax`` externo — o notebook monta o grid de
subplots e passa cada eixo, mantendo o estilo uniforme entre ativos.
Se ``ax=None``, a função cria sua própria figura (útil para plots isolados).

Estilo: matplotlib puro, sem seaborn, coerente com 04_detection_thresholds.ipynb.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import CONFIG, PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / CONFIG["figures_dir"]


def save_fig(fig: plt.Figure, name: str) -> Path:
    """Salva ``fig`` em ``figures/<name>.png`` e retorna o caminho."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_error_distribution(
    errors_train: np.ndarray,
    errors_test: np.ndarray,
    threshold: float,
    ticker: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Histograma sobreposto do erro de treino e teste com linha do threshold (issue #24).

    O threshold (calculado sobre o treino) separa visualmente a cauda direita
    que o detector marca como anômala. Erros de teste além do threshold são os
    candidatos a anomalia.

    Args:
        errors_train: erros de reconstrução do período de treino.
        errors_test:  erros de reconstrução do período de teste.
        threshold:    limiar estático (saída de ``static_threshold``).
        ticker:       nome do ativo (título do plot).
        ax:           eixo externo; cria figura própria se ``None``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    bins = 60
    ax.hist(errors_train, bins=bins, alpha=0.5, label="treino", color="steelblue")
    ax.hist(errors_test, bins=bins, alpha=0.5, label="teste", color="orange")
    ax.axvline(threshold, color="red", ls="--", lw=1, label=f"threshold p95 ({threshold:.4f})")
    ax.set_title(ticker)
    ax.set_xlabel("erro de reconstrução (MAE)")
    ax.set_ylabel("frequência")
    ax.legend(fontsize=8)
    return ax


def plot_events_overlay(
    ax: plt.Axes,
    events: "pd.DataFrame",
    color: str = "gray",
) -> None:
    """Sobrepõe linhas verticais de eventos em um eixo existente (issue #26).

    Cada evento é desenhado como uma linha vertical tracejada; o rótulo é
    anotado no topo do eixo em texto rotacionado. Projetado para ser chamado
    após ``plot_error_timeseries``, reutilizando o mesmo ``ax``.

    Args:
        ax:     eixo já populado com a série de erros.
        events: DataFrame de ``events_in_range`` (colunas date/label/tickers).
        color:  cor das linhas e anotações.
    """
    import pandas as pd

    if events.empty:
        return

    ymin, ymax = ax.get_ylim()
    for _, row in events.iterrows():
        ax.axvline(row["date"], color=color, ls=":", lw=0.8, alpha=0.7)
        ax.text(
            row["date"],
            ymax,
            f"  {row['label']}",
            rotation=90,
            va="top",
            ha="left",
            fontsize=5,
            color=color,
            alpha=0.9,
        )


def plot_error_timeseries(
    errors: np.ndarray,
    dates,
    threshold: float | np.ndarray,
    flags: np.ndarray,
    ticker: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Série temporal do erro com threshold e marcação das anomalias (issue #24).

    Anomalias (``flags=True``) são destacadas como pontos vermelhos sobre a
    série — facilita a inspeção visual de quais períodos foram sinalizados.
    ``threshold`` pode ser escalar (estático) ou vetor (dinâmico).

    Args:
        errors:    vetor de erros de reconstrução por janela.
        dates:     índice de datas alinhado às janelas (``test_index[W-1:]``).
        threshold: limiar estático (float) ou dinâmico (array).
        flags:     booleano por janela (saída de ``flag_anomalies``).
        ticker:    nome do ativo (título do plot).
        ax:        eixo externo; cria figura própria se ``None``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    ax.plot(dates, errors, lw=0.6, label="erro", color="steelblue")

    if np.ndim(threshold) == 0:
        ax.axhline(float(threshold), color="red", ls="--", lw=1, label="threshold")
    else:
        ax.plot(dates, threshold, color="red", lw=1, ls="--", label="threshold")

    anomaly_dates = dates[flags]
    anomaly_errors = errors[flags]
    if len(anomaly_dates):
        ax.scatter(anomaly_dates, anomaly_errors, color="red", s=10, zorder=3, label="anomalia")

    ax.set_title(ticker)
    ax.set_ylabel("erro de reconstrução (MAE)")
    ax.legend(fontsize=8)
    return ax
