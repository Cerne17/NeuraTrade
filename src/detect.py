"""Detecção de anomalias por erro de reconstrução (ADR-0005).

Erro por janela → limiar estático (percentil do erro de treino) e/ou limiar
dinâmico (percentil em janela móvel **causal**, só passado). Uma janela é
anômala quando o erro supera o limiar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONFIG


def _per_step_error(diff: np.ndarray, metric: str) -> np.ndarray:
    """Erro por passo de tempo: reduz o eixo de *features* (ADR-0009).

    ``diff`` tem forma ``(n_janelas, window, n_features)``. Reduz primeiro as
    features (``axis=-1``) para obter um erro escalar por passo, devolvendo
    ``(n_janelas, window)``. Esta é a primeira das duas etapas; a agregação
    temporal vem depois (não trocar a ordem dos eixos — ADR-0009).
    """
    if metric == "mse":
        return np.mean(diff**2, axis=-1)
    if metric == "mae":
        return np.mean(np.abs(diff), axis=-1)
    raise ValueError(f"metric desconhecida: {metric!r} (use 'mae' ou 'mse').")


def aggregate_window_error(
    per_step: np.ndarray,
    aggregation: str | None = None,
    percentile: float | None = None,
) -> np.ndarray:
    """Agrega o erro por passo em um escore por janela (ADR-0009).

    Segunda etapa da redução: opera sobre o eixo **temporal** (``axis=1``) de um
    tensor ``(n_janelas, window)`` produzido por :func:`_per_step_error`.

    Args:
        aggregation: ``"mean"`` (padrão, atual), ``"max"`` (pior passo da janela,
            preserva choque de 1 dia → Recall) ou ``"percentile"`` (meio-termo
            robusto a um único ponto ruidoso). Usa ``CONFIG["detection"]`` se ``None``.
        percentile: percentil usado quando ``aggregation == "percentile"``. Usa
            ``CONFIG["detection"]["aggregation_percentile"]`` se ``None``.

    Returns:
        Vetor de escores por janela, tamanho ``len(per_step)``.
    """
    dcfg = CONFIG["detection"]
    aggregation = aggregation or dcfg.get("aggregation", "mean")

    if aggregation == "mean":
        return np.mean(per_step, axis=1)
    if aggregation == "max":
        return np.max(per_step, axis=1)
    if aggregation == "percentile":
        p = percentile if percentile is not None else dcfg.get(
            "aggregation_percentile", 90
        )
        return np.percentile(per_step, p, axis=1)
    raise ValueError(
        f"aggregation desconhecida: {aggregation!r} "
        "(use 'mean', 'max' ou 'percentile')."
    )


def reconstruction_error(
    model,
    X: np.ndarray,
    metric: str = "mae",
    aggregation: str | None = None,
    percentile: float | None = None,
) -> np.ndarray:
    """Erro de reconstrução por janela (ADR-0005, agregação ADR-0009).

    Redução em **duas etapas, nesta ordem**: (1) erro por passo, reduzindo o
    eixo de features (:func:`_per_step_error`); (2) agregação temporal
    (:func:`aggregate_window_error`). A média sobre o tempo (``aggregation="mean"``)
    reproduz o comportamento histórico; ``max``/``percentile`` evitam que um choque
    de um único dia seja diluído pelos demais passos da janela.

    Args:
        metric: ``"mae"`` (padrão, robusto a outliers) ou ``"mse"`` (issue #18).
        aggregation: agregação temporal (ver :func:`aggregate_window_error`).
            Usa ``CONFIG["detection"]["aggregation"]`` se ``None``.
        percentile: percentil para ``aggregation="percentile"``.

    Returns:
        Vetor de tamanho ``len(X)``, ordenado no tempo (janela i → passo i).
    """
    recon = model.predict(X, verbose=0)
    diff = np.asarray(recon) - np.asarray(X)
    per_step = _per_step_error(diff, metric)
    return aggregate_window_error(per_step, aggregation, percentile)


def reconstruction_error_per_channel(
    model,
    X: np.ndarray,
    metric: str = "mae",
    aggregation: str | None = None,
    percentile: float | None = None,
) -> np.ndarray:
    """Erro de reconstrução por janela **e por canal** (ADR-0011, issue #53).

    No caso multivariado (OHLCV), decompõe o erro por feature em vez de colapsá-lo
    num escalar. Permite **atribuir** a anomalia (ex.: *volume-driven* vs
    *price-driven*) e diagnosticar canais que a reconstrução está ignorando (MSE
    desbalanceado entre canais).

    A agregação temporal (``aggregation``) é a mesma da detecção escalar, aplicada
    canal a canal.

    Returns:
        Matriz ``(n_janelas, n_features)``: erro agregado no tempo, por canal.
    """
    recon = model.predict(X, verbose=0)
    diff = np.asarray(recon) - np.asarray(X)  # (n, window, n_features)

    if metric == "mse":
        per_step = diff**2
    elif metric == "mae":
        per_step = np.abs(diff)
    else:
        raise ValueError(f"metric desconhecida: {metric!r} (use 'mae' ou 'mse').")

    # Agrega o eixo temporal (axis=1), preservando o eixo de canais (axis=-1).
    dcfg = CONFIG["detection"]
    aggregation = aggregation or dcfg.get("aggregation", "mean")
    if aggregation == "mean":
        return np.mean(per_step, axis=1)
    if aggregation == "max":
        return np.max(per_step, axis=1)
    if aggregation == "percentile":
        p = percentile if percentile is not None else dcfg.get(
            "aggregation_percentile", 90
        )
        return np.percentile(per_step, p, axis=1)
    raise ValueError(
        f"aggregation desconhecida: {aggregation!r} (use 'mean', 'max' ou 'percentile')."
    )


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
