"""Testes da agregação de erro por janela (ADR-0009, issue #46)."""

import numpy as np
import pytest

from src.detect import (
    aggregate_window_error,
    reconstruction_error,
    reconstruction_error_per_channel,
)


class _ZeroModel:
    """Stub: reconstrói tudo como zero → erro = |X| (mae) determinístico, sem TF."""

    def predict(self, X, verbose=0):
        return np.zeros_like(X)


def test_aggregate_mean_vs_max_isola_choque():
    # Janela com um único passo de choque: max deve preservar; mean deve diluir.
    per_step = np.array([[0.0, 0.0, 9.0, 0.0, 0.0]])  # (1 janela, 5 passos)
    mean = aggregate_window_error(per_step, aggregation="mean")
    mx = aggregate_window_error(per_step, aggregation="max")
    assert mean[0] == pytest.approx(9.0 / 5)
    assert mx[0] == pytest.approx(9.0)
    assert mx[0] > mean[0]


def test_aggregate_percentile_entre_mean_e_max():
    per_step = np.array([[0.0, 0.0, 9.0, 0.0, 0.0]])
    p90 = aggregate_window_error(per_step, aggregation="percentile", percentile=90)
    mean = aggregate_window_error(per_step, aggregation="mean")
    mx = aggregate_window_error(per_step, aggregation="max")
    assert mean[0] <= p90[0] <= mx[0]


def test_aggregate_rejeita_modo_invalido():
    with pytest.raises(ValueError):
        aggregate_window_error(np.zeros((1, 3)), aggregation="median")


def test_reconstruction_error_shape_e_max_pega_pico():
    # 2 janelas, 4 passos, 1 feature. Segunda janela tem um pico.
    X = np.zeros((2, 4, 1), dtype="float32")
    X[1, 2, 0] = 5.0
    err_mean = reconstruction_error(_ZeroModel(), X, aggregation="mean")
    err_max = reconstruction_error(_ZeroModel(), X, aggregation="max")
    assert err_mean.shape == (2,)
    assert err_max.shape == (2,)
    assert err_max[1] == pytest.approx(5.0)        # max preserva o pico
    assert err_mean[1] == pytest.approx(5.0 / 4)   # mean dilui


def test_per_channel_preserva_eixo_de_canais():
    # 3 janelas, 4 passos, 2 canais (ex.: Close, Volume).
    X = np.zeros((3, 4, 2), dtype="float32")
    X[0, 1, 1] = 7.0  # anomalia só no canal 1 (volume)
    per_ch = reconstruction_error_per_channel(_ZeroModel(), X, aggregation="max")
    assert per_ch.shape == (3, 2)
    assert per_ch[0, 1] == pytest.approx(7.0)  # canal de volume disparou
    assert per_ch[0, 0] == pytest.approx(0.0)  # canal de preço, não


def test_metric_invalida_levanta():
    X = np.zeros((1, 3, 1), dtype="float32")
    with pytest.raises(ValueError):
        reconstruction_error(_ZeroModel(), X, metric="rmse")
