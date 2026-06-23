"""Avaliação por injeção sintética de anomalias (ADR-0006).

Estratégia: inserir perturbações conhecidas (price shocks) em posições
controladas da série de teste, produzindo ground-truth binário. O detector é
então avaliado contra esse ground-truth via Precision / Recall / F1 (issue #23).

Ordem de uso:
1. ``inject_price_shocks``  — perturba a série e devolve labels.
2. ``labels_to_window_labels`` — converte labels por passo em labels por janela.
3. ``compute_metrics``       — Precision / Recall / F1 (issue #23).
"""

from __future__ import annotations

import numpy as np

from .config import CONFIG


def inject_price_shocks(
    test_scaled: np.ndarray,
    n_injections: int | None = None,
    shock_magnitude: float | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Injeta price shocks em posições aleatórias da série de teste (issue #22).

    Opera sobre log-retornos já escalados (saída de ``apply_scaler``), que é o
    espaço em que o modelo foi treinado. Injetar no preço bruto produziria um
    choque de magnitude variável dependendo do nível de preço do ativo.

    Args:
        test_scaled: vetor 1D de log-retornos escalados do período de teste.
        n_injections: número de posições a perturbar. Usa
            ``CONFIG["evaluation"]["n_injections"]`` se ``None``.
        shock_magnitude: delta somado ao log-retorno na posição injetada. Usa
            ``CONFIG["evaluation"]["price_shock"]`` se ``None``. **PROVISÓRIO**
            (ADR-0006): calibrar em k desvios-padrão do retorno real do ativo.
        seed: seed do gerador. Usa ``CONFIG["seed"]`` se ``None``.

    Returns:
        Tupla ``(perturbed, labels, positions)`` onde:
        - ``perturbed``: cópia de ``test_scaled`` com choques aplicados.
        - ``labels``: vetor inteiro de 0/1, ``labels[i] = 1`` se o passo ``i``
          foi injetado.
        - ``positions``: índices das posições injetadas (para diagnóstico e plot).
    """
    cfg = CONFIG["evaluation"]
    n = n_injections if n_injections is not None else cfg["n_injections"]
    mag = shock_magnitude if shock_magnitude is not None else cfg["price_shock"]
    seed = seed if seed is not None else CONFIG["seed"]

    values = np.asarray(test_scaled, dtype="float32").ravel()

    if n > len(values):
        raise ValueError(
            f"n_injections={n} maior que o comprimento da série ({len(values)})."
        )

    rng = np.random.default_rng(seed)
    positions = rng.choice(len(values), size=n, replace=False)

    perturbed = values.copy()
    perturbed[positions] += mag

    labels = np.zeros(len(values), dtype=int)
    labels[positions] = 1

    return perturbed, labels, positions


def labels_to_window_labels(
    labels: np.ndarray,
    window_size: int | None = None,
    step: int | None = None,
) -> np.ndarray:
    """Converte labels por passo em labels por janela (issue #23).

    Uma janela é positiva (label=1) se **qualquer** passo injetado cair dentro
    dela. Isso reflete que o modelo processa a janela inteira — um choque em
    qualquer posição da janela eleva o erro de reconstrução de toda ela.

    O número de janelas geradas segue a mesma lógica de ``make_windows``:
    ``range(0, len(labels) - window_size + 1, step)``, mantendo alinhamento
    exato com os tensores ``X_test`` produzidos pelo pré-processamento.

    Args:
        labels: vetor 1D de labels por passo (saída de ``inject_price_shocks``).
        window_size/step: usam ``CONFIG["preprocessing"]`` se ``None``.

    Returns:
        Vetor 1D de labels por janela, de tamanho igual ao número de janelas.
    """
    pcfg = CONFIG["preprocessing"]
    window_size = window_size if window_size is not None else pcfg["window_size"]
    step = step if step is not None else pcfg["step"]

    n = len(labels)
    starts = range(0, n - window_size + 1, step)
    return np.array(
        [int(labels[i : i + window_size].any()) for i in starts],
        dtype=int,
    )
