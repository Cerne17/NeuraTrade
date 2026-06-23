"""Testes do pipeline multivariado OHLCV (ADR-0011, issue #52)."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    build_features,
    make_windows,
    preprocess_ticker_multivariate,
)


def _ohlcv(n=600):
    """OHLCV sintético: treino (<=2019) calmo, teste (2020+) com volume maior."""
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    base = 10 + np.linspace(0, 1, n)
    vol = np.full(n, 1_000_000.0)
    # Pós-2020 com volume muito maior — testa saturação / fit só no treino.
    vol[idx > pd.Timestamp("2019-12-31")] = 50_000_000.0
    return pd.DataFrame(
        {
            "Open": base,
            "High": base * 1.01,
            "Low": base * 0.99,
            "Close": base,
            "Volume": vol,
        },
        index=idx,
    )


def test_make_windows_multivariado_preserva_canais():
    arr = np.arange(20, dtype="float32").reshape(10, 2)  # (T=10, 2 features)
    w = make_windows(arr, window_size=4, step=1)
    assert w.shape == (7, 4, 2)


def test_make_windows_1d_retrocompativel():
    w = make_windows(np.arange(10), window_size=4, step=1)
    assert w.shape == (7, 4, 1)  # 1D continua virando 1 canal


def test_build_features_volume_log1p_e_retorno_preco():
    df = _ohlcv()
    feats = build_features(df, features=["Close", "Volume"], volume_log1p=True)
    assert list(feats.columns) == ["Close", "Volume"]
    assert not feats.isna().any().any()  # log-retorno dropa o 1º NaN
    # Volume virou log1p do bruto (alinhado após o dropna do log-retorno).
    expected = np.log1p(df["Volume"]).loc[feats.index]
    np.testing.assert_allclose(feats["Volume"].to_numpy(), expected.to_numpy())


def test_feature_invalida_levanta():
    with pytest.raises(ValueError):
        build_features(_ohlcv(), features=["Close", "Spread"])


def test_pipeline_multivariado_shape_e_scaler_por_coluna():
    df = _ohlcv()
    out = preprocess_ticker_multivariate(
        df, features=["Close", "Volume"], window_size=10, step=1
    )
    assert out["X_train"].shape[1:] == (10, 2)
    assert out["X_test"].shape[1:] == (10, 2)
    assert out["features"] == ["Close", "Volume"]
    # Scaler por coluna: um min/max por feature.
    assert out["scaler"].data_min_.shape == (2,)
    assert out["scaler"].data_max_.shape == (2,)


def test_scaler_ajustado_so_no_treino():
    # Volume de teste (5e7) >> treino (1e6): após escalar, o teste deve estourar
    # [0,1] no canal de volume — prova de que o scaler não viu o futuro (ADR-0001).
    df = _ohlcv()
    out = preprocess_ticker_multivariate(
        df, features=["Close", "Volume"], window_size=10
    )
    assert out["X_test"][..., 1].max() > 1.0
