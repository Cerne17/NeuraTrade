"""Testes da validação Walk-Forward (ADR-0010, issue #49).

Foco no invariante anti-vazamento: o scaler de cada fold enxerga **apenas** o
passado daquele fold, nunca o futuro.
"""

import numpy as np
import pandas as pd

from src.validation import walk_forward_splits


def _series(n=120):
    # Série estritamente crescente: se o scaler vazar o futuro, seu max delata.
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    return pd.Series(np.arange(n, dtype=float), index=idx, name="r")


def test_folds_expandem_e_geram_janelas():
    s = _series(120)
    folds = list(walk_forward_splits(s, n_splits=4, window_size=5, step=1))
    assert len(folds) == 4

    train_sizes = [len(f["train_index"]) for f in folds]
    assert train_sizes == sorted(train_sizes)            # treino expansível
    assert all(np.diff(train_sizes) > 0)                 # estritamente crescente

    for f in folds:
        n_train = len(f["train_index"])
        assert f["X_train"].shape == (n_train - 5 + 1, 5, 1)
        assert f["X_val"].shape[0] > 0


def test_scaler_nao_ve_o_futuro():
    s = _series(120)
    full_max = s.max()
    for f in walk_forward_splits(s, n_splits=4, window_size=5):
        scaler = f["scaler"]
        train_max = s.loc[f["train_index"]].max()
        # O scaler foi ajustado só no treino do fold.
        assert scaler.data_max_[0] == train_max
        # E como a série cresce, esse max é menor que o máximo global (sem vazamento).
        assert scaler.data_max_[0] < full_max


def test_janela_nao_cruza_fronteira_train_val():
    # Treino e validação não compartilham datas (TimeSeriesSplit é disjunto).
    s = _series(90)
    for f in walk_forward_splits(s, n_splits=3, window_size=4):
        inter = f["train_index"].intersection(f["val_index"])
        assert len(inter) == 0
