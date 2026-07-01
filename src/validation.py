"""Validação Walk-Forward para séries temporais (ADR-0010).

Substitui o holdout único por validação cruzada *walk-forward*
(``sklearn.model_selection.TimeSeriesSplit``): fatias **expansíveis** do passado,
avaliação no bloco futuro adjacente. Serve à seleção rigorosa de hiperparâmetros
(ex.: ``latent_dim``), com estimativa de menor variância que um único corte.

Invariante anti-vazamento (crítico, ADR-0001 **por fold**):
- o ``MinMaxScaler`` é **reajustado dentro de cada fold**, só no treino daquele
  fold; o bloco de validação é apenas transformado (nunca refit);
- as janelas deslizantes são geradas **dentro** de cada partição, sem cruzar a
  fronteira treino/validação.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from sklearn.preprocessing import MinMaxScaler

from .config import CONFIG
from .preprocessing import apply_scaler, build_features, fit_scaler, make_windows


def walk_forward_splits(
    series: pd.Series,
    n_splits: int | None = None,
    window_size: int | None = None,
    step: int | None = None,
) -> Iterator[dict]:
    """Gera folds *walk-forward* prontos para o autoencoder (ADR-0010).

    Para cada fold do ``TimeSeriesSplit``: ajusta um ``MinMaxScaler`` **apenas**
    no recorte de treino, transforma treino e validação com esse scaler e gera as
    janelas dentro de cada recorte (nunca cruzando o corte).

    Args:
        series: série 1D (tipicamente log-retornos do período de **treino**,
            i.e. a "normalidade"). Mantém a ordem cronológica.
        n_splits: número de folds. Usa ``CONFIG["validation"]["n_splits"]`` se ``None``.
        window_size/step: usam ``CONFIG["preprocessing"]`` se ``None``.

    Yields:
        Dict por fold com ``fold`` (índice), ``X_train``/``X_val`` (tensores de
        janelas), ``scaler`` (ajustado só no treino) e ``train_index``/``val_index``
        (datas de cada recorte, para diagnóstico).
    """
    n_splits = n_splits if n_splits is not None else CONFIG["validation"]["n_splits"]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    values_index = series.index

    for fold, (train_idx, val_idx) in enumerate(tscv.split(series.to_numpy())):
        s_train = series.iloc[train_idx]
        s_val = series.iloc[val_idx]

        # Scaler ajustado SÓ no treino do fold (anti-vazamento por fold).
        scaler = fit_scaler(s_train)
        train_scaled = apply_scaler(scaler, s_train)
        val_scaled = apply_scaler(scaler, s_val)

        # Janelas geradas DENTRO de cada recorte — nenhuma cruza a fronteira.
        yield {
            "fold": fold,
            "X_train": make_windows(train_scaled, window_size, step),
            "X_val": make_windows(val_scaled, window_size, step),
            "scaler": scaler,
            "train_index": values_index[train_idx],
            "val_index": values_index[val_idx],
        }


def walk_forward_splits_multivariate(
    feats: pd.DataFrame,
    n_splits: int | None = None,
    window_size: int | None = None,
    step: int | None = None,
) -> Iterator[dict]:
    """Versão multivariada de :func:`walk_forward_splits` (ADR-0010 + ADR-0011).

    Recebe um frame de features já construído (saída de
    :func:`src.preprocessing.build_features`, p.ex. Close+Volume) e, em cada fold,
    ajusta um ``MinMaxScaler`` **por coluna** apenas no recorte de treino — mesma
    garantia anti-vazamento da versão univariada, agora canal a canal.

    Yields:
        Dict por fold com ``X_train``/``X_val`` de forma ``(n, window, n_features)``.
    """
    n_splits = n_splits if n_splits is not None else CONFIG["validation"]["n_splits"]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    idx = feats.index

    for fold, (train_idx, val_idx) in enumerate(tscv.split(feats.to_numpy())):
        tr, va = feats.iloc[train_idx], feats.iloc[val_idx]

        scaler = MinMaxScaler().fit(tr.to_numpy())  # por coluna, só no treino do fold
        tr_scaled = scaler.transform(tr.to_numpy())
        va_scaled = scaler.transform(va.to_numpy())

        yield {
            "fold": fold,
            "X_train": make_windows(tr_scaled, window_size, step),
            "X_val": make_windows(va_scaled, window_size, step),
            "scaler": scaler,
            "train_index": idx[train_idx],
            "val_index": idx[val_idx],
        }


def cross_validate_latent_dim(
    series: pd.Series,
    candidates: list[int],
    n_splits: int | None = None,
    epochs: int | None = None,
    folds: list[dict] | None = None,
) -> pd.DataFrame:
    """Seleciona ``latent_dim`` por validação *walk-forward* (ADR-0010, issue #50).

    Para cada candidato de ``latent_dim``, treina o autoencoder em cada fold e
    coleta o ``val_loss`` (melhor época, via ``EarlyStopping``). Retorna a média e
    o desvio por candidato — base honesta para a escolha, com variância estimada.

    Importa TensorFlow de forma preguiçosa (etapas sem treino não pagam o custo).

    Args:
        candidates: valores de ``latent_dim`` a comparar (ex.: ``[8, 16, 32]``).
        epochs: teto de épocas por fold (reduzido no CV para conter custo). Usa
            ``CONFIG["train"]["epochs"]`` se ``None``.
        folds: folds pré-computados (ex.: de
            :func:`walk_forward_splits_multivariate` para o caso OHLCV). Se ``None``,
            usa a malha univariada sobre ``series``. O ``n_features`` é inferido das
            janelas, então o mesmo loop serve uni e multivariado.

    Returns:
        ``DataFrame`` indexado por ``latent_dim`` com colunas ``val_loss_mean``,
        ``val_loss_std`` e ``n_folds``.
    """
    from tensorflow import keras

    from .model import build_lstm_autoencoder

    tcfg = CONFIG["train"]
    epochs = epochs if epochs is not None else tcfg["epochs"]
    if folds is None:
        folds = list(walk_forward_splits(series, n_splits=n_splits))

    rows = []
    for latent_dim in candidates:
        fold_losses = []
        for f in folds:
            X_tr, X_val = f["X_train"], f["X_val"]
            if len(X_tr) == 0 or len(X_val) == 0:
                continue  # fold curto demais para gerar janelas
            n_features = X_tr.shape[-1]
            model = build_lstm_autoencoder(latent_dim=latent_dim, n_features=n_features)
            es = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=tcfg["early_stopping_patience"],
                restore_best_weights=True,
            )
            hist = model.fit(
                X_tr,
                X_tr,
                validation_data=(X_val, X_val),
                epochs=epochs,
                batch_size=tcfg["batch_size"],
                shuffle=tcfg["shuffle"],
                callbacks=[es],
                verbose=0,
            )
            fold_losses.append(min(hist.history["val_loss"]))
            keras.backend.clear_session()

        rows.append(
            {
                "latent_dim": latent_dim,
                "val_loss_mean": float(np.mean(fold_losses)) if fold_losses else np.nan,
                "val_loss_std": float(np.std(fold_losses)) if fold_losses else np.nan,
                "n_folds": len(fold_losses),
            }
        )

    return pd.DataFrame(rows).set_index("latent_dim")


def cross_validate_weight_decay(
    series: pd.Series,
    candidates: list[float],
    n_splits: int | None = None,
    epochs: int | None = None,
    folds: list[dict] | None = None,
) -> pd.DataFrame:
    """Compara valores de ``weight_decay`` por walk-forward (ADR-0018).

    Mesmo protocolo de :func:`cross_validate_latent_dim`, variando a penalidade
    L2 desacoplada (AdamW) em vez do ``latent_dim``. ``weight_decay=0.0`` é o
    baseline atual (Adam puro). Útil para responder, com a variância inter-fold à
    vista, se a regularização por decay ajuda, atrapalha ou é indiferente num
    dataset pequeno cujo gargalo já é insensível.

    Args:
        candidates: valores de ``weight_decay`` a comparar (ex.: ``[0.0, 1e-5, 1e-4, 1e-3]``).
        epochs: teto de épocas por fold. Usa ``CONFIG["train"]["epochs"]`` se ``None``.
        folds: folds pré-computados; se ``None``, usa a malha univariada sobre ``series``.

    Returns:
        ``DataFrame`` indexado por ``weight_decay`` com ``val_loss_mean``,
        ``val_loss_std`` e ``n_folds``.
    """
    from tensorflow import keras

    from .model import build_lstm_autoencoder

    tcfg = CONFIG["train"]
    epochs = epochs if epochs is not None else tcfg["epochs"]
    if folds is None:
        folds = list(walk_forward_splits(series, n_splits=n_splits))

    rows = []
    for wd in candidates:
        fold_losses = []
        for f in folds:
            X_tr, X_val = f["X_train"], f["X_val"]
            if len(X_tr) == 0 or len(X_val) == 0:
                continue
            n_features = X_tr.shape[-1]
            model = build_lstm_autoencoder(n_features=n_features, weight_decay=wd)
            es = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=tcfg["early_stopping_patience"],
                restore_best_weights=True,
            )
            hist = model.fit(
                X_tr,
                X_tr,
                validation_data=(X_val, X_val),
                epochs=epochs,
                batch_size=tcfg["batch_size"],
                shuffle=tcfg["shuffle"],
                callbacks=[es],
                verbose=0,
            )
            fold_losses.append(min(hist.history["val_loss"]))
            keras.backend.clear_session()

        rows.append(
            {
                "weight_decay": wd,
                "val_loss_mean": float(np.mean(fold_losses)) if fold_losses else np.nan,
                "val_loss_std": float(np.std(fold_losses)) if fold_losses else np.nan,
                "n_folds": len(fold_losses),
            }
        )

    return pd.DataFrame(rows).set_index("weight_decay")


def cross_validate_latent_dim_conditional(
    folds: list[dict],
    n_price_volume: int,
    candidates: list[int],
    epochs: int | None = None,
) -> pd.DataFrame:
    """Seleciona ``latent_dim`` do **Conditional AE** por walk-forward (ADR-0010/0012).

    Como :func:`cross_validate_latent_dim`, mas treina o modelo condicional
    (`build_conditional_autoencoder`): a entrada é o tensor completo pv+macro e o
    **alvo é só o bloco preço/volume** (`X[..., :n_price_volume]`). Os ``folds`` vêm
    de :func:`walk_forward_splits_multivariate` sobre o frame `[pv | macro]`.

    Returns:
        ``DataFrame`` por ``latent_dim`` com ``val_loss_mean``/``val_loss_std``/``n_folds``.
    """
    from tensorflow import keras

    from .model import build_conditional_autoencoder

    tcfg = CONFIG["train"]
    epochs = epochs if epochs is not None else tcfg["epochs"]

    rows = []
    for latent_dim in candidates:
        fold_losses = []
        for f in folds:
            X_tr, X_val = f["X_train"], f["X_val"]
            if len(X_tr) == 0 or len(X_val) == 0:
                continue
            n_macro = X_tr.shape[-1] - n_price_volume
            model = build_conditional_autoencoder(
                n_price_volume=n_price_volume, n_macro=n_macro, latent_dim=latent_dim
            )
            es = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=tcfg["early_stopping_patience"],
                restore_best_weights=True,
            )
            hist = model.fit(
                X_tr,
                X_tr[..., :n_price_volume],   # alvo: só preço/volume
                validation_data=(X_val, X_val[..., :n_price_volume]),
                epochs=epochs,
                batch_size=tcfg["batch_size"],
                shuffle=tcfg["shuffle"],
                callbacks=[es],
                verbose=0,
            )
            fold_losses.append(min(hist.history["val_loss"]))
            keras.backend.clear_session()

        rows.append(
            {
                "latent_dim": latent_dim,
                "val_loss_mean": float(np.mean(fold_losses)) if fold_losses else np.nan,
                "val_loss_std": float(np.std(fold_losses)) if fold_losses else np.nan,
                "n_folds": len(fold_losses),
            }
        )

    return pd.DataFrame(rows).set_index("latent_dim")
