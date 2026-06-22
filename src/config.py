"""Carregamento de configuração e controle de reprodutibilidade.

Fonte única de hiperparâmetros: ``config.yaml`` (raiz do projeto).
Importar ``CONFIG`` para ler hiperparâmetros; chamar ``set_seeds()`` no
início de qualquer notebook/script para fixar as seeds globais.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml

# Raiz do projeto = diretório que contém este pacote (src/) -> sobe um nível.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Lê o ``config.yaml`` e retorna um dicionário.

    Args:
        path: caminho alternativo para o YAML. Usa ``CONFIG_PATH`` se ``None``.
    """
    cfg_path = Path(path) if path is not None else CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# Configuração global, carregada uma vez na importação do módulo.
CONFIG: dict[str, Any] = load_config()


def set_seeds(seed: int | None = None) -> int:
    """Fixa as seeds de ``random``, ``numpy`` e ``tensorflow``.

    Garante reprodutibilidade entre execuções. Lê ``CONFIG["seed"]`` por
    padrão; aceita override via argumento.

    Returns:
        A seed efetivamente aplicada.
    """
    if seed is None:
        seed = int(CONFIG.get("seed", 42))

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        # TensorFlow é opcional em etapas que não treinam (ex.: EDA).
        pass

    return seed
