"""Avaliação por injeção sintética de anomalias (ADR-0006).

Estratégia: inserir perturbações conhecidas (price shocks) em posições
controladas da série de teste, produzindo ground-truth binário. O detector é
então avaliado contra esse ground-truth via Precision / Recall / F1 (issue #23).

Ordem de uso:
1. ``inject_price_shocks``    — perturba a série e devolve labels.
2. ``labels_to_window_labels`` — converte labels por passo em labels por janela.
3. ``compute_metrics``         — Precision / Recall / F1 (issue #23).
"""

from __future__ import annotations

import numpy as np

from .config import CONFIG


def inject_price_shocks(
    test_scaled: np.ndarray,
    n_injections: int | None = None,
    shock_magnitude: float | None = None,
    k_sigma: float | None = None,
    sigma: float | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Injeta price shocks em posições aleatórias da série de teste (issue #22).

    Opera sobre log-retornos já escalados (saída de ``apply_scaler``), que é o
    espaço em que o modelo foi treinado. Injetar no preço bruto produziria um
    choque de magnitude variável dependendo do nível de preço do ativo.

    Magnitude do choque (ADR-0006): por padrão é **relativa** à dispersão do
    ativo, ``mag = k_sigma * sigma`` --- assim um "choque de k desvios-padrão"
    tem o mesmo significado estatístico em todos os ativos, em vez de um valor
    absoluto cuja dificuldade varia por ativo. Passe ``shock_magnitude`` para
    forçar um valor absoluto (modo legado).

    Args:
        test_scaled: vetor 1D de log-retornos escalados do período de teste.
        n_injections: número de posições a perturbar. Usa
            ``CONFIG["evaluation"]["n_injections"]`` se ``None``.
        shock_magnitude: delta **absoluto** somado ao log-retorno escalado. Se
            informado, tem precedência sobre ``k_sigma`` (modo legado, ADR-0006).
        k_sigma: múltiplo de desvios-padrão do choque relativo. Usa
            ``CONFIG["evaluation"]["shock_k_sigma"]`` se ``None``.
        sigma: dispersão de referência (idealmente o desvio-padrão da
            **normalidade**, i.e. dos retornos escalados de treino). Se ``None``,
            usa ``np.std(test_scaled)``. Ignorado quando ``shock_magnitude`` é dado.
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
    seed = seed if seed is not None else CONFIG["seed"]

    values = np.asarray(test_scaled, dtype="float32").ravel()

    if shock_magnitude is not None:
        mag = float(shock_magnitude)
    else:
        k = k_sigma if k_sigma is not None else cfg["shock_k_sigma"]
        s = sigma if sigma is not None else float(np.std(values))
        mag = float(k) * s

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


def compute_metrics(
    window_flags: np.ndarray,
    window_labels: np.ndarray,
) -> dict[str, float]:
    """Precision, Recall e F1 do detector contra o ground-truth sintético (issue #23).

    ``window_flags`` é a saída de ``flag_anomalies`` (detect.py); ``window_labels``
    é a saída de ``labels_to_window_labels``. Os dois vetores devem estar alinhados:
    mesma janela no mesmo índice.

    ``zero_division=0`` evita erro quando o detector não dispara nenhuma flag
    (Precision indefinida); nesse caso retorna 0.0 para todas as métricas.

    Returns:
        Dict com chaves ``"precision"``, ``"recall"`` e ``"f1"``, valores em [0, 1].
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    y_true = np.asarray(window_labels, dtype=int)
    y_pred = np.asarray(window_flags, dtype=int)

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def score_curves(
    window_scores: np.ndarray,
    window_labels: np.ndarray,
) -> dict:
    """Métricas **independentes de limiar** para classe rara (ADR-0015).

    Sob forte desbalanceamento (anomalias ≈ poucos %), a **ROC-AUC engana**: o
    excesso de verdadeiros-negativos infla a pontuação. A **PR-AUC** (área sob a
    curva Precision–Recall, = *average precision*) foca na classe positiva e é a
    métrica honesta. Reporta ambas — e a *baseline* da PR-AUC (a prevalência), que
    é o que um classificador aleatório atinge.

    Args:
        window_scores: escore contínuo por janela (erro de reconstrução agregado),
            **não** as flags binárias — PR/ROC precisam de escore para varrer limiares.
        window_labels: ground-truth 0/1 por janela.

    Returns:
        Dict com ``pr_auc``, ``roc_auc``, ``prevalence`` (baseline da PR) e
        ``pr_lift`` (pr_auc / prevalence). ``roc_auc`` é ``nan`` se só houver uma classe.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(window_labels, dtype=int)
    s = np.asarray(window_scores, dtype=float)
    prevalence = float(y.mean())

    pr_auc = float(average_precision_score(y, s)) if 0 < prevalence < 1 else float("nan")
    roc_auc = float(roc_auc_score(y, s)) if 0 < prevalence < 1 else float("nan")
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "prevalence": prevalence,
        "pr_lift": pr_auc / prevalence if prevalence > 0 else float("nan"),
    }


def expected_cost(
    window_flags: np.ndarray,
    window_labels: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
) -> dict[str, float]:
    """Custo assimétrico de Falsos Positivos vs Falsos Negativos (ADR-0015/0016).

    Em detecção de anomalias o erro não é simétrico: deixar passar um choque (FN)
    raramente custa o mesmo que um alarme falso (FP). Esta é a tradução mínima e
    honesta para "custo" — sem inventar uma estratégia de portfólio (ver ADR-0016).

    Args:
        cost_fp/cost_fn: custos relativos. Ex.: ``cost_fn=5`` = perder uma anomalia
            é 5× pior que um alarme falso.

    Returns:
        Dict com contagens ``fp``/``fn``/``tp``/``tn`` e ``cost`` total ponderado.
    """
    y = np.asarray(window_labels, dtype=int)
    p = np.asarray(window_flags, dtype=int)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "cost": float(cost_fp * fp + cost_fn * fn),
    }
