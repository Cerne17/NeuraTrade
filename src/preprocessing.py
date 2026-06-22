"""Pré-processamento: log-retornos, split temporal, normalização e janelas.

Ordem metodológica (ADR-0001):
1. ``log_returns`` — feature principal.
2. ``temporal_split`` — separa treino/teste ANTES de normalizar.
3. ``fit_scaler`` (só no treino) + ``apply_scaler`` — evita vazamento.
4. ``make_windows`` — janelas deslizantes, geradas dentro de cada partição
   (nunca cruzando a fronteira treino/teste).

``preprocess_ticker`` encadeia tudo e devolve os tensores prontos para o modelo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import CONFIG


def log_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.Series:
    """Retorno logarítmico diário ``r_t = ln(P_t / P_{t-1})`` (issue #9).

    O primeiro valor (NaN) é descartado. Invariante a escala multiplicativa,
    logo imune a fatores de split/grupamento já aplicados em ``Close``.
    """
    r = np.log(df[price_col] / df[price_col].shift(1))
    return r.dropna().rename("log_return")


def temporal_split(
    series: pd.Series, train_end: str | None = None
) -> tuple[pd.Series, pd.Series]:
    """Split temporal treino/teste (issue #10).

    Treino = até ``train_end`` (inclusive); teste = estritamente depois.
    Não embaralha: preserva a ordem cronológica.
    """
    train_end = train_end or CONFIG["data"]["train_end"]
    cut = pd.Timestamp(train_end)
    train = series.loc[series.index <= cut]
    test = series.loc[series.index > cut]
    return train, test


def fit_scaler(train_series: pd.Series) -> MinMaxScaler:
    """Ajusta um ``MinMaxScaler`` APENAS no treino (issue #11).

    Ajustar sobre treino+teste vazaria estatísticas (min/max) do futuro.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_series.to_numpy().reshape(-1, 1))
    return scaler


def apply_scaler(scaler: MinMaxScaler, series: pd.Series) -> np.ndarray:
    """Aplica um scaler já ajustado. Valores do teste podem sair de [0,1]
    (extremos pós-2020 fora do range de treino) — esperado (ADR-0001/0003)."""
    return scaler.transform(series.to_numpy().reshape(-1, 1)).ravel()


def make_windows(
    values: np.ndarray, window_size: int | None = None, step: int | None = None
) -> np.ndarray:
    """Janelas deslizantes (issue #12).

    Args:
        values: vetor 1D (uma partição já normalizada).
        window_size/step: usam ``CONFIG["preprocessing"]`` se ``None``.

    Returns:
        Tensor ``(n_janelas, window_size, 1)`` pronto para o LSTM. Vazio se a
        partição for menor que ``window_size``.
    """
    window_size = window_size or CONFIG["preprocessing"]["window_size"]
    step = step or CONFIG["preprocessing"]["step"]

    values = np.asarray(values, dtype="float32").ravel()
    n = len(values)
    if n < window_size:
        return np.empty((0, window_size, 1), dtype="float32")

    idx = range(0, n - window_size + 1, step)
    windows = np.stack([values[i : i + window_size] for i in idx])
    return windows[..., np.newaxis]


def preprocess_ticker(
    df: pd.DataFrame,
    price_col: str = "Close",
    train_end: str | None = None,
    window_size: int | None = None,
    step: int | None = None,
) -> dict:
    """Pipeline completo de um ativo: retornos → split → scaler → janelas.

    Returns:
        dict com ``X_train``/``X_test`` (tensores de janelas), o ``scaler``
        ajustado e os índices de datas de cada partição (para alinhar
        detecções com o calendário).
    """
    r = log_returns(df, price_col=price_col)
    r_train, r_test = temporal_split(r, train_end=train_end)

    scaler = fit_scaler(r_train)
    train_scaled = apply_scaler(scaler, r_train)
    test_scaled = apply_scaler(scaler, r_test)

    return {
        "X_train": make_windows(train_scaled, window_size, step),
        "X_test": make_windows(test_scaled, window_size, step),
        "scaler": scaler,
        "train_index": r_train.index,
        "test_index": r_test.index,
    }
