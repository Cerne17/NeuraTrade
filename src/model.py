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


def build_conditional_autoencoder(
    window_size: int | None = None,
    n_price_volume: int = 2,
    n_macro: int = 4,
    lstm_units: int | None = None,
    latent_dim: int | None = None,
    dropout: float | None = None,
    learning_rate: float | None = None,
    bottleneck_activation: str = "tanh",
):
    """Conditional LSTM-Autoencoder com contexto macro (ADR-0012, proposto).

    O encoder enxerga o tensor completo ``(window, n_price_volume + n_macro)`` ---
    a macro **condiciona** a codificação. O decoder reconstrói **apenas** o bloco
    de preço/volume (``n_price_volume`` canais): a perda MSE recai só sobre ele.
    Assim a macro dá contexto sistêmico sem poluir a loss nem ser "esquecida" por
    ser quase-constante (ver ADR-0012, e o problema do MSE dominado por canais de
    alta variância).

    Treino: ``model.fit(X_full, X_full[..., :n_price_volume])`` --- alvo é só o
    bloco pv; a macro entra na entrada, nunca no alvo.

    Args:
        n_price_volume: nº de canais reconstruídos (ex.: Close+Volume → 2).
        n_macro: nº de canais macro de contexto (não reconstruídos).
        bottleneck_activation: ativação do gargalo. Default ``"tanh"`` (preserva
            sinal no espaço macro mais rico; evita dead-units do ``relu`` --- ver
            review do ADR-0012). Use ``"relu"`` para casar com o modelo base.

    Returns:
        ``keras.Model`` compilado (entrada pv+macro, saída pv).
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
    n_features = n_price_volume + n_macro

    inputs = keras.Input(shape=(window_size, n_features), name="janela_pv_macro")

    # Encoder: vê preço/volume + macro (condicionamento).
    x = layers.LSTM(lstm_units, name="encoder_lstm")(inputs)
    x = layers.Dropout(dropout, name="encoder_dropout")(x)
    latent = layers.Dense(
        latent_dim, activation=bottleneck_activation, name="bottleneck"
    )(x)

    # Decoder: reconstrói SÓ o bloco de preço/volume.
    x = layers.RepeatVector(window_size, name="repeat")(latent)
    x = layers.LSTM(lstm_units, return_sequences=True, name="decoder_lstm")(x)
    x = layers.Dropout(dropout, name="decoder_dropout")(x)
    outputs = layers.TimeDistributed(
        layers.Dense(n_price_volume), name="reconstrucao_pv"
    )(x)

    model = keras.Model(inputs, outputs, name="conditional_lstm_ae")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mse")
    return model
