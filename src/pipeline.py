"""Pipeline funcional end-to-end (M10).

Encadeia o projeto inteiro com a **configuração atual** (`config.yaml`, agregação
``max`` por default desde M9): para cada ativo, carrega dados → pré-processa →
treina/carrega o modelo → calcula o erro de reconstrução → aplica limiares
estático e dinâmico → (opcional) avalia por injeção sintética.

Esta é a via de **produção/reprodução**, distinta dos notebooks (que são estudos).
Uso típico:

    from src.pipeline import run_pipeline, summarize
    res = run_pipeline()           # carrega modelos existentes
    print(summarize(res))

ou pela CLI: ``python -m src --train`` (ver ``src/__main__.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data, detect
from . import preprocessing as pp
from . import train as train_mod
from .config import CONFIG, set_seeds


def run_pipeline(
    tickers: list[str] | None = None,
    train: bool = False,
    evaluate: bool = True,
    verbose: int = 0,
) -> dict[str, dict]:
    """Roda a pipeline completa para cada ativo.

    Args:
        tickers: lista de ativos. Usa ``CONFIG["tickers"]`` se ``None``.
        train: se ``True``, treina o modelo de cada ativo do zero (e salva). Se
            ``False`` (default), carrega ``models/<ticker>.keras`` e falha com
            mensagem clara se ausente.
        evaluate: se ``True``, injeta choques sintéticos (k·σ) e calcula P/R/F1.
        verbose: verbosidade do treino Keras.

    Returns:
        Dict ``{ticker: resumo}`` com fração de janelas marcadas (estático e
        dinâmico), limiar estático, nº de janelas de teste e, se ``evaluate``,
        precision/recall/f1. A agregação do erro é a de ``config.yaml``.
    """
    set_seeds()
    tickers = tickers or CONFIG["tickers"]
    agg = CONFIG["detection"].get("aggregation", "mean")
    data_map = data.load_all(tickers)

    results: dict[str, dict] = {}
    for t in tickers:
        pre = pp.preprocess_ticker(data_map[t])

        if train:
            model, _ = train_mod.train_model(pre["X_train"], ticker=t, verbose=verbose)
        else:
            model = train_mod.load_model(t)  # FileNotFoundError com dica se ausente

        err_tr = detect.reconstruction_error(model, pre["X_train"])
        err_te = detect.reconstruction_error(model, pre["X_test"])
        thr_s = detect.static_threshold(err_tr)
        thr_d = detect.dynamic_threshold(err_te)

        flags_s = detect.flag_anomalies(err_te, thr_s)
        flags_d = detect.flag_anomalies(err_te, thr_d)

        res = {
            "agregacao": agg,
            "n_test_windows": int(len(err_te)),
            "limiar_estatico": round(float(thr_s), 5),
            "frac_estatico": round(float(np.mean(flags_s)), 4),
            "frac_dinamico": round(float(np.mean(flags_d)), 4),
        }

        if evaluate:
            res.update(_evaluate_ticker(model, data_map[t]))

        results[t] = res

    return results


def _evaluate_ticker(model, df) -> dict:
    """Injeção sintética (k·σ) + P/R/F1 para um ativo, usando a config atual."""
    from .evaluate import (
        compute_metrics,
        inject_price_shocks,
        labels_to_window_labels,
    )

    r = pp.log_returns(df)
    r_train, r_test = pp.temporal_split(r)
    scaler = pp.fit_scaler(r_train)
    sigma_norm = float(np.std(pp.apply_scaler(scaler, r_train)))
    test_scaled = pp.apply_scaler(scaler, r_test)

    perturbed, labels, _ = inject_price_shocks(test_scaled, sigma=sigma_norm)
    err_pert = detect.reconstruction_error(model, pp.make_windows(perturbed))
    wl = labels_to_window_labels(labels)
    err_tr = detect.reconstruction_error(model, pp.make_windows(pp.apply_scaler(scaler, r_train)))
    m = compute_metrics(detect.flag_anomalies(err_pert, detect.static_threshold(err_tr)), wl)
    return {k: round(v, 3) for k, v in m.items()}


def summarize(results: dict[str, dict]) -> pd.DataFrame:
    """Converte a saída de :func:`run_pipeline` em um ``DataFrame`` indexado por ativo."""
    return pd.DataFrame(results).T.rename_axis("ticker")
