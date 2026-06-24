"""Barra de progresso de treino — callback Keras, sem dependência externa.

Mostra uma barra ao vivo (atualizada em uma única linha via ``\\r``) por época, para
o usuário ver que o treino está andando. Funciona no terminal e em notebooks.
``EarlyStopping`` pode encerrar antes de atingir o total de épocas.
"""

from __future__ import annotations


def keras_progress(label: str = "", width: int = 24):
    """Retorna um callback Keras que imprime uma barra de progresso por época.

    Args:
        label: prefixo da linha (ex.: ``"PETR4.SA "``).
        width: largura da barra em caracteres.
    """
    from tensorflow import keras

    class _EpochProgress(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.total = self.params.get("epochs", 0) or 0

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            e = epoch + 1
            frac = e / self.total if self.total else 0.0
            full = int(width * frac)
            bar = "█" * full + "░" * (width - full)
            msg = f"\r  {label}[{bar}] {e}/{self.total}"
            if "loss" in logs:
                msg += f"  loss={logs['loss']:.4f}"
            if "val_loss" in logs:
                msg += f"  val={logs['val_loss']:.4f}"
            print(msg, end="", flush=True)

        def on_train_end(self, logs=None):
            print()  # quebra a linha ao terminar

    return _EpochProgress()
