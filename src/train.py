"""Treino do LSTM-Autoencoder (ADR-0004).

Um modelo por ativo. Validação por bloco cronológico (``shuffle=False``),
``EarlyStopping`` com ``restore_best_weights`` e persistência em
``models/<ticker>.keras``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import CONFIG, PROJECT_ROOT
from .model import build_lstm_autoencoder

MODELS_DIR = PROJECT_ROOT / CONFIG["train"]["models_dir"]


def _model_path(ticker: str, suffix: str = "") -> Path:
    return MODELS_DIR / f"{ticker}{suffix}.keras"


def train_model(
    X_train: np.ndarray,
    ticker: str | None = None,
    save: bool = True,
    verbose: int = 0,
    n_features: int | None = None,
    suffix: str = "",
):
    """Treina o autoencoder sobre janelas de treino (reconstrução: X→X).

    Usa os hiperparâmetros de ``CONFIG["train"]``. Se ``ticker`` for dado e
    ``save=True``, salva os pesos em ``models/<ticker><suffix>.keras``.

    Args:
        n_features: nº de canais da entrada. Inferido de ``X_train`` se ``None``;
            >1 ativa o caminho multivariado (OHLCV, ADR-0011).
        suffix: sufixo do arquivo de modelo (ex.: ``"_multi"``) para não
            sobrescrever o modelo univariado do mesmo ativo.

    Returns:
        ``(model, history)``.
    """
    from tensorflow import keras

    tcfg = CONFIG["train"]
    if n_features is None:
        n_features = X_train.shape[-1]
    model = build_lstm_autoencoder(n_features=n_features)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=tcfg["early_stopping_patience"],
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        X_train,
        X_train,
        epochs=tcfg["epochs"],
        batch_size=tcfg["batch_size"],
        validation_split=tcfg["validation_split"],
        shuffle=tcfg["shuffle"],  # False: preserva ordem temporal (anti-vazamento)
        callbacks=callbacks,
        verbose=verbose,
    )

    if save and ticker is not None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save(_model_path(ticker, suffix))

    return model, history


def load_model(ticker: str, suffix: str = ""):
    """Carrega um modelo salvo de ``models/<ticker><suffix>.keras``."""
    from tensorflow import keras

    path = _model_path(ticker, suffix)
    if not path.exists():
        raise FileNotFoundError(f"Modelo ausente: {path}. Treine antes (train_model).")
    return keras.models.load_model(path)


def reconstruction_error(model, X: np.ndarray) -> np.ndarray:
    """Erro de reconstrução por janela: MAE entre entrada e reconstrução.

    Base para a detecção (M4). Retorna um vetor de tamanho ``len(X)``.
    """
    recon = model.predict(X, verbose=0)
    return np.mean(np.abs(recon - X), axis=(1, 2))
