"""Conditional Autoencoder com contexto macro — pipeline e decisão (ADR-0012, proposto).

Monta o tensor ``[preço/volume | macro]``, treina o Conditional AE (que reconstrói
**só** o bloco preço/volume), e extrai **scores por bloco** para a lógica de regime:

| erro Preço/Volume | estresse Macro | regime |
| ----------------- | -------------- | ------ |
| alto              | baixo          | **idiossincrático** (estresse de liquidez do ativo) |
| alto              | alto           | **sistêmico** (choque macro global) |
| baixo             | —              | normal |

A macro entra na **entrada** (condiciona o encoder), nunca no alvo da perda
([model.build_conditional_autoencoder][]). O "estresse macro" é medido direto do
bloco macro (amplitude intra-janela), não por reconstrução — eventos macro
agendados não devem inflar o erro do ativo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import CONFIG
from .macro import align_macro, stationarize_macro
from .preprocessing import build_features, make_windows


def conditional_features(
    df_asset: pd.DataFrame,
    df_macro: pd.DataFrame,
    pv_cols: list[str] | None = None,
    macro_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Monta o frame ``[preço/volume | macro]`` estacionarizado e alinhado.

    Bloco preço/volume via :func:`src.preprocessing.build_features` (Close→log-retorno,
    Volume→log1p); bloco macro via :func:`src.macro.align_macro` (ffill causal) +
    :func:`src.macro.stationarize_macro`. As colunas pv vêm **primeiro** (índices
    ``0..n_pv-1``), as macro depois — convenção usada por todo o módulo.

    Returns:
        ``(frame, n_price_volume)`` com índice alinhado (interseção das datas).
    """
    pv_cols = pv_cols or CONFIG["preprocessing"].get("features", ["Close", "Volume"])
    pv = build_features(df_asset, features=pv_cols)

    macro_aligned = align_macro(df_asset, df_macro, macro_cols=macro_cols)
    macro = stationarize_macro(macro_aligned)

    frame = pv.join(macro, how="inner").dropna()
    return frame, len(pv_cols)


def prepare_conditional(
    frame: pd.DataFrame,
    n_price_volume: int,
    train_end: str | None = None,
    window_size: int | None = None,
    step: int | None = None,
) -> dict:
    """Split temporal + scaler por coluna (só no treino) + janelas (ADR-0001/0012).

    Returns:
        dict com ``X_train``/``X_test`` ``(n, window, n_pv+n_macro)``, ``scaler``,
        ``n_price_volume`` e os índices de datas.
    """
    train_end = train_end or CONFIG["data"]["train_end"]
    cut = pd.Timestamp(train_end)
    train = frame.loc[frame.index <= cut]
    test = frame.loc[frame.index > cut]

    scaler = MinMaxScaler().fit(train.to_numpy())  # por coluna, só no treino
    return {
        "X_train": make_windows(scaler.transform(train.to_numpy()), window_size, step),
        "X_test": make_windows(scaler.transform(test.to_numpy()), window_size, step),
        "scaler": scaler,
        "n_price_volume": n_price_volume,
        "train_index": train.index,
        "test_index": test.index,
    }


def train_conditional(
    X_train: np.ndarray,
    n_price_volume: int,
    latent_dim: int | None = None,
    verbose: int = 0,
):
    """Treina o Conditional AE: entrada = pv+macro, **alvo = só o bloco pv**.

    Returns:
        ``(model, history)``.
    """
    from tensorflow import keras

    from .model import build_conditional_autoencoder

    tcfg = CONFIG["train"]
    n_macro = X_train.shape[-1] - n_price_volume
    model = build_conditional_autoencoder(
        n_price_volume=n_price_volume, n_macro=n_macro, latent_dim=latent_dim
    )
    es = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=tcfg["early_stopping_patience"],
        restore_best_weights=True,
    )
    y_train = X_train[..., :n_price_volume]  # alvo: só preço/volume
    history = model.fit(
        X_train,
        y_train,
        epochs=tcfg["epochs"],
        batch_size=tcfg["batch_size"],
        validation_split=tcfg["validation_split"],
        shuffle=tcfg["shuffle"],
        callbacks=[es],
        verbose=verbose,
    )
    return model, history


def block_scores(
    model,
    X_full: np.ndarray,
    n_price_volume: int,
    aggregation: str | None = None,
    percentile: float | None = None,
) -> dict[str, np.ndarray]:
    """Scores por bloco para a decisão de regime (ADR-0012).

    - **pv_error:** erro de reconstrução do bloco preço/volume (a anomalia do
      ativo), agregado no tempo pela mesma política do detector (``max`` default,
      ADR-0009).
    - **macro_stress:** amplitude intra-janela do bloco macro (max−min por feature,
      média entre features). Constante (estável) → ≈0; choque macro → alto.
      Medido **direto da entrada**, sem reconstrução — eventos macro não inflam o
      erro do ativo.

    Returns:
        ``{"pv_error": (n,), "macro_stress": (n,)}``.
    """
    dcfg = CONFIG["detection"]
    aggregation = aggregation or dcfg.get("aggregation", "max")
    p = percentile if percentile is not None else dcfg.get("aggregation_percentile", 90)

    recon_pv = np.asarray(model.predict(X_full, verbose=0))
    X_pv = X_full[..., :n_price_volume]
    pv_step = np.mean(np.abs(recon_pv - X_pv), axis=-1)  # (n, window)

    if aggregation == "mean":
        pv_error = pv_step.mean(axis=1)
    elif aggregation == "max":
        pv_error = pv_step.max(axis=1)
    elif aggregation == "percentile":
        pv_error = np.percentile(pv_step, p, axis=1)
    else:
        raise ValueError(f"aggregation desconhecida: {aggregation!r}")

    X_macro = X_full[..., n_price_volume:]  # (n, window, n_macro)
    # amplitude intra-janela por feature, média entre features (offset-invariante)
    macro_stress = (X_macro.max(axis=1) - X_macro.min(axis=1)).mean(axis=-1)

    return {"pv_error": pv_error, "macro_stress": macro_stress}


def block_thresholds(
    scores_train: dict[str, np.ndarray], percentile: float | None = None
) -> tuple[float, float]:
    """Limiares (pv, macro) = percentil do **treino** de cada score (normalidade)."""
    percentile = percentile or CONFIG["detection"]["threshold_percentile"]
    return (
        float(np.percentile(scores_train["pv_error"], percentile)),
        float(np.percentile(scores_train["macro_stress"], percentile)),
    )


def classify_regime(
    pv_error: np.ndarray,
    macro_stress: np.ndarray,
    pv_threshold: float,
    macro_threshold: float,
) -> np.ndarray:
    """Classifica cada janela em ``normal`` / ``idiossincratico`` / ``sistemico``.

    Anomalia = ``pv_error > pv_threshold`` (o ativo está sob estresse). Entre as
    anômalas, ``macro_stress > macro_threshold`` separa choque sistêmico (macro
    também se mexeu) de idiossincrático (macro estável). Janelas não-anômalas são
    ``normal`` independentemente da macro --- macro mexendo sozinha, sem estresse
    no ativo, não é anomalia do ativo.
    """
    pv_error = np.asarray(pv_error)
    macro_active = np.asarray(macro_stress) > macro_threshold
    anomaly = pv_error > pv_threshold

    labels = np.full(len(pv_error), "normal", dtype=object)
    labels[anomaly & ~macro_active] = "idiossincratico"
    labels[anomaly & macro_active] = "sistemico"
    return labels
