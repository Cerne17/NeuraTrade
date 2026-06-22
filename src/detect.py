"""Detecção de anomalias por erro de reconstrução (ADR-0005).

Erro por janela → limiar estático (percentil do erro de treino) e/ou limiar
dinâmico (percentil em janela móvel **causal**, só passado). Uma janela é
anômala quando o erro supera o limiar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONFIG


def reconstruction_error(model, X: np.ndarray, metric: str = "mae") -> np.ndarray:
    """Erro de reconstrução por janela.

    Args:
        metric: ``"mae"`` (padrão, robusto a outliers) ou ``"mse"`` (issue #18).

    Returns:
        Vetor de tamanho ``len(X)``, ordenado no tempo (janela i → passo i).
    """
    recon = model.predict(X, verbose=0)
    diff = recon - X
    if metric == "mse":
        return np.mean(diff**2, axis=(1, 2))
    if metric == "mae":
        return np.mean(np.abs(diff), axis=(1, 2))
    raise ValueError(f"metric desconhecida: {metric!r} (use 'mae' ou 'mse').")


def static_threshold(
    train_errors: np.ndarray, percentile: float | None = None
) -> float:
    """Limiar estático: percentil do erro de reconstrução do TREINO (issue #19).

    Calculado sobre o treino (normalidade), não sobre o teste — evita vazamento.
    """
    percentile = percentile or CONFIG["detection"]["threshold_percentile"]
    return float(np.percentile(train_errors, percentile))


def dynamic_threshold(
    errors: np.ndarray,
    window: int | None = None,
    percentile: float | None = None,
    causal: bool = True,
) -> np.ndarray:
    """Limiar dinâmico: percentil em janela móvel (issue #20).

    Para cada ponto, o limiar é o percentil dos erros de uma janela de tamanho
    ``window``. Com ``causal=True`` a janela usa **apenas o passado** (exclui o
    próprio ponto, via ``shift(1)``), evitando que a anomalia infle seu próprio
    limiar. Os primeiros pontos (janela incompleta) retornam ``NaN``.

    Returns:
        Vetor de limiares alinhado a ``errors``.
    """
    window = window or CONFIG["detection"]["dynamic_window"]
    percentile = percentile or CONFIG["detection"]["threshold_percentile"]
    q = percentile / 100.0

    s = pd.Series(errors)
    roll = s.shift(1) if causal else s
    thr = roll.rolling(window, min_periods=window).quantile(q)
    return thr.to_numpy()


def flag_anomalies(
    errors: np.ndarray, threshold: float | np.ndarray
) -> np.ndarray:
    """Marca janelas anômalas: ``erro > limiar``.

    ``threshold`` pode ser escalar (estático) ou vetor (dinâmico). Posições com
    limiar ``NaN`` (janela inicial do dinâmico) são ``False``.
    """
    thr = np.asarray(threshold, dtype="float64")
    flags = errors > thr
    if thr.ndim > 0:
        flags = np.where(np.isnan(thr), False, flags)
    return flags.astype(bool)
