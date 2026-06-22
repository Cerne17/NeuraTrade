"""Arquitetura do LSTM-Autoencoder (ADR-0003).

Encoder LSTM → gargalo denso (latent) → RepeatVector → Decoder LSTM →
TimeDistributed(Dense). Perda MSE, otimizador Adam. Rede pequena, coerente com
o tamanho do dataset (~2450 janelas/ativo). Um modelo por ativo (comparação
setorial).
"""

from __future__ import annotations

from .config import CONFIG


def build_lstm_autoencoder(
    window_size: int | None = None,
    n_features: int = 1,
    lstm_units: int | None = None,
    latent_dim: int | None = None,
    dropout: float | None = None,
    learning_rate: float | None = None,
    loss: str | None = None,
):
    """Constrói e compila o LSTM-Autoencoder.

    Hiperparâmetros omitidos usam ``CONFIG`` (fonte única). Retorna um
    ``keras.Model`` já compilado.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    mcfg = CONFIG["model"]
    tcfg = CONFIG["train"]
    window_size = window_size or CONFIG["preprocessing"]["window_size"]
    lstm_units = lstm_units or mcfg["lstm_units"]
    latent_dim = latent_dim or mcfg["latent_dim"]
    dropout = mcfg["dropout"] if dropout is None else dropout
    learning_rate = learning_rate or tcfg["learning_rate"]
    loss = loss or mcfg["loss"]

    inputs = keras.Input(shape=(window_size, n_features), name="janela")

    # Encoder: comprime a janela em um vetor latente.
    x = layers.LSTM(lstm_units, name="encoder_lstm")(inputs)
    x = layers.Dropout(dropout, name="encoder_dropout")(x)
    latent = layers.Dense(latent_dim, activation="relu", name="bottleneck")(x)

    # Decoder: reconstrói a sequência a partir do latente.
    x = layers.RepeatVector(window_size, name="repeat")(latent)
    x = layers.LSTM(lstm_units, return_sequences=True, name="decoder_lstm")(x)
    x = layers.Dropout(dropout, name="decoder_dropout")(x)
    outputs = layers.TimeDistributed(
        layers.Dense(n_features), name="reconstrucao"
    )(x)

    model = keras.Model(inputs, outputs, name="lstm_autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=loss)
    return model
