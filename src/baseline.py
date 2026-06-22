"""Baselines simples de detecção (issue #14).

Chão de comparação para o LSTM-Autoencoder. Métodos clássicos, sem aprendizado,
que atribuem um score de anomalia a cada observação de log-retorno. Servem para
demonstrar que o modelo profundo agrega valor sobre heurísticas triviais.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONFIG


def rolling_zscore(returns: pd.Series, window: int | None = None) -> pd.Series:
    """Score de anomalia por z-score em janela móvel causal.

    $z_t = |r_t - \\mu_{t-w:t}| / \\sigma_{t-w:t}$, usando apenas o passado.
    Valores altos indicam desvio atípico do regime recente.
    """
    window = window or CONFIG["detection"]["dynamic_window"]
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std()
    z = (returns - mean) / std
    return z.abs().rename("zscore")


def moving_average_residual(
    returns: pd.Series, window: int | None = None
) -> pd.Series:
    """Score de anomalia pelo resíduo absoluto vs. média móvel causal.

    $|r_t - \\mu_{t-w:t}|$. Mais simples que o z-score (não normaliza pela
    volatilidade local).
    """
    window = window or CONFIG["detection"]["dynamic_window"]
    mean = returns.rolling(window).mean()
    return (returns - mean).abs().rename("ma_residual")
