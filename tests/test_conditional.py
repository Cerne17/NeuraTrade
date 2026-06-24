"""Testes do Conditional AE / contexto macro (ADR-0012, proposto)."""

import numpy as np
import pandas as pd
import pytest

from src.conditional import (
    block_scores,
    block_thresholds,
    classify_regime,
    conditional_features,
)
from src.macro import align_macro, stationarize_macro


def _asset(n=300):
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    base = 10 + np.linspace(0, 2, n)
    return pd.DataFrame(
        {"Close": base, "Volume": np.full(n, 1e6), "Open": base, "High": base, "Low": base},
        index=idx,
    )


# --- macro: alinhamento causal -------------------------------------------- #

def test_align_macro_ffill_causal_e_sem_bfill():
    asset = _asset(20)
    # macro publicada só em 2 datas dentro da janela
    pub = pd.DataFrame(
        {"VIX": [15.0, 22.0]},
        index=[asset.index[5], asset.index[12]],
    )
    out = align_macro(asset, pub, macro_cols=["VIX"])
    # antes da 1ª publicação: linhas descartadas (sem bfill puxando o futuro)
    assert out.index.min() == asset.index[5]
    # entre as publicações: vale a 1ª (ffill), não a 2ª (futuro)
    assert out.loc[asset.index[8], "VIX"] == 15.0
    # a partir da 2ª publicação: vale a 2ª
    assert out.loc[asset.index[15], "VIX"] == 22.0


def test_stationarize_macro_metodos():
    idx = pd.date_range("2019-01-01", periods=5, freq="B")
    df = pd.DataFrame(
        {"USDBRL": [4.0, 4.0, 4.4, 4.4, 4.84], "Selic": [10.0, 10.0, 11.0, 11.0, 11.0],
         "VIX": [15.0, 16.0, 17.0, 18.0, 19.0]},
        index=idx,
    )
    out = stationarize_macro(df, methods={"USDBRL": "logret", "Selic": "delta", "VIX": "level"})
    # logret: ln(4.4/4.0) ~ 0.0953
    assert out["USDBRL"].iloc[1] == pytest.approx(np.log(4.4 / 4.0))
    # delta: 11-10 = 1 no passo da mudança, 0 quando estável
    assert out["Selic"].iloc[1] == pytest.approx(1.0)
    assert out["Selic"].iloc[2] == pytest.approx(0.0)
    # level: inalterado
    assert out["VIX"].iloc[0] == 16.0  # 1ª linha cai pelo dropna do diff/logret


# --- features condicionais ------------------------------------------------- #

def test_conditional_features_pv_primeiro():
    asset = _asset(200)
    idx = asset.index
    macro = pd.DataFrame(
        {"USDBRL": np.linspace(4, 5, 200), "VIX": np.linspace(15, 20, 200)}, index=idx
    )
    frame, n_pv = conditional_features(
        asset, macro, pv_cols=["Close", "Volume"], macro_cols=["USDBRL", "VIX"]
    )
    assert n_pv == 2
    assert list(frame.columns[:2]) == ["Close", "Volume"]      # pv primeiro
    assert set(frame.columns[2:]) == {"USDBRL", "VIX"}          # macro depois
    assert not frame.isna().any().any()


# --- scores por bloco + classificação ------------------------------------- #

class _ZeroPV:
    """Stub: reconstrói o bloco pv como zero → pv_error = |X_pv| agregado."""

    def __init__(self, n_pv):
        self.n_pv = n_pv

    def predict(self, X, verbose=0):
        return np.zeros((len(X), X.shape[1], self.n_pv), dtype="float32")


def test_block_scores_pv_error_e_macro_stress():
    n, W, n_pv, n_macro = 3, 5, 2, 2
    X = np.zeros((n, W, n_pv + n_macro), dtype="float32")
    X[0, 2, 0] = 4.0                 # pico no canal de preço, janela 0
    X[1, :, n_pv] = np.array([0, 0, 0.9, 0, 0])  # macro mexe na janela 1
    s = block_scores(_ZeroPV(n_pv), X, n_price_volume=n_pv, aggregation="max")
    assert s["pv_error"].shape == (n,)
    assert s["pv_error"][0] == pytest.approx(4.0 / n_pv)   # |4|/2 features, max no tempo
    assert s["pv_error"][2] == pytest.approx(0.0)          # janela sem pico pv
    # macro_stress: amplitude intra-janela; janela 1 mexeu, 0 e 2 estáveis
    assert s["macro_stress"][1] > 0
    assert s["macro_stress"][0] == pytest.approx(0.0)


def test_classify_regime_matriz_2x2():
    pv = np.array([0.1, 0.9, 0.9, 0.2])
    macro = np.array([0.0, 0.0, 0.5, 0.5])
    labels = classify_regime(pv, macro, pv_threshold=0.5, macro_threshold=0.3)
    assert list(labels) == ["normal", "idiossincratico", "sistemico", "normal"]


def test_block_thresholds_percentil():
    train = {"pv_error": np.arange(100.0), "macro_stress": np.arange(100.0) * 2}
    pv_thr, macro_thr = block_thresholds(train, percentile=95)
    assert pv_thr == pytest.approx(np.percentile(np.arange(100.0), 95))
    assert macro_thr == pytest.approx(np.percentile(np.arange(100.0) * 2, 95))
