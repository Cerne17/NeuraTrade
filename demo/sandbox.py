"""NeuraTrade — Sandbox interativo (demonstração ao vivo).

Carrega os modelos LSTM-Autoencoder já treinados (2010–2019) e deixa variar, em
tempo real, os parâmetros de **detecção** e do **protocolo de avaliação**:
agregação do erro, tipo/percentil de limiar e injeção sintética. O erro é
recalculado uma vez por ativo (cache) e a resposta às mudanças de limiar/injeção
é instantânea — sem retreinar.

Rodar (a partir da raiz do repo):
    streamlit run demo/sandbox.py

Requisitos: modelos em ``models/<ticker>.keras`` e dados em ``data/raw/`` (offline).
Se faltarem modelos, treine antes: ``python -m src --train``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data, detect, events  # noqa: E402
from src import preprocessing as pp  # noqa: E402
from src.config import CONFIG, set_seeds  # noqa: E402
from src.evaluate import (  # noqa: E402
    event_metrics,
    inject_price_shocks,
    labels_to_window_labels,
    score_curves,
)
from src.train import load_model  # noqa: E402

st.set_page_config(page_title="NeuraTrade — Sandbox", layout="wide")
W = CONFIG["preprocessing"]["window_size"]


# ----------------------------------------------------------------------------
# Cache pesado: modelo + erro por janela (recalcula só ao trocar de ativo/período)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Carregando modelo…")
def _model(ticker: str):
    return load_model(ticker)


@st.cache_data(show_spinner="Pré-processando série…")
def _prep(ticker: str):
    """scaler + tensores de treino/teste do ticker (split 2010–2019 / 2020–2024)."""
    pre = pp.preprocess_ticker(data.load_ticker(ticker))
    return pre


@st.cache_data(show_spinner="Reconstruindo (erro por passo)…")
def _per_step_error(ticker: str, split: str) -> np.ndarray:
    """Erro POR PASSO (n_janelas, window) — base para reagregar sem reprever."""
    pre = _prep(ticker)
    X = pre["X_train"] if split == "treino" else pre["X_test"]
    model = _model(ticker)
    recon = model.predict(X, verbose=0)
    diff = np.asarray(recon) - np.asarray(X)
    return detect._per_step_error(diff, "mae")  # (n, window)


def _dates(ticker: str, split: str, n: int) -> pd.DatetimeIndex:
    pre = _prep(ticker)
    idx = pre["train_index"] if split == "treino" else pre["test_index"]
    return pd.DatetimeIndex(idx[W - 1 : W - 1 + n])


# ----------------------------------------------------------------------------
# Sidebar — controles
# ----------------------------------------------------------------------------
set_seeds()
st.sidebar.title("⚙️ Parâmetros")

ticker = st.sidebar.selectbox("Ativo", CONFIG["tickers"])
split = st.sidebar.radio(
    "Período", ["teste", "treino"],
    help="treino = normalidade 2010–2019 · teste = 2020–2024",
)

st.sidebar.subheader("Agregação do erro (ADR-0009)")
aggregation = st.sidebar.selectbox("aggregation", ["max", "mean", "percentile"])
agg_pct = st.sidebar.slider("percentil (se percentile)", 50, 99, 90, disabled=aggregation != "percentile")

st.sidebar.subheader("Limiar (ADR-0005)")
thr_kind = st.sidebar.radio("tipo", ["estático", "dinâmico"])
thr_pct = st.sidebar.slider("percentil do limiar", 80, 99, int(CONFIG["detection"]["threshold_percentile"]))
dyn_win = st.sidebar.slider(
    "janela do dinâmico (pregões)", 60, 504, int(CONFIG["detection"]["dynamic_window"]),
    step=21, disabled=thr_kind != "dinâmico",
)

st.sidebar.subheader("Injeção sintética (ADR-0006/0015)")
inject = st.sidebar.checkbox("injetar choques + medir P/R/F1", value=(split == "teste"))
n_inj = st.sidebar.slider("n_injections", 2, 80, 10, disabled=not inject)
k_sigma = st.sidebar.slider("choque (k·σ)", 2.0, 8.0, float(CONFIG["evaluation"]["shock_k_sigma"]), 0.5, disabled=not inject)


# ----------------------------------------------------------------------------
# Cálculo (rápido: reagrega o erro por passo já em cache)
# ----------------------------------------------------------------------------
per_step = _per_step_error(ticker, split).copy()  # (n, window)

# Injeção: perturba o erro por passo nas janelas que contêm o choque.
labels_win = None
if inject:
    # Choque no espaço do erro: soma k·σ ao pior passo das janelas sorteadas.
    n_windows = per_step.shape[0]
    sigma = float(per_step.std())
    _, step_labels, _ = inject_price_shocks(
        np.zeros(n_windows + W - 1), n_injections=n_inj, k_sigma=k_sigma, sigma=1.0
    )
    labels_win = labels_to_window_labels(step_labels)[:n_windows]
    bump = k_sigma * sigma
    per_step[labels_win.astype(bool)] += bump

# Agrega em escore por janela conforme o controle.
scores = detect.aggregate_window_error(per_step, aggregation, agg_pct)

# Limiar. Estático: percentil do TREINO (anti-vazamento). Dinâmico: janela causal.
train_ps = _per_step_error(ticker, "treino")
train_scores = detect.aggregate_window_error(train_ps, aggregation, agg_pct)
if thr_kind == "estático":
    threshold = detect.static_threshold(train_scores, thr_pct)
    thr_series = np.full(len(scores), threshold)
else:
    thr_series = detect.dynamic_threshold(scores, dyn_win, thr_pct)

flags = detect.flag_anomalies(scores, thr_series)
dates = _dates(ticker, split, len(scores))


# ----------------------------------------------------------------------------
# Painel principal
# ----------------------------------------------------------------------------
st.title("🔍 NeuraTrade — Sandbox de Detecção")
st.caption(
    f"{ticker} · período {split} · agregação **{aggregation}** · "
    f"limiar **{thr_kind} p{thr_pct}**"
    + (f" · injeção n={n_inj}, {k_sigma:g}σ" if inject else "")
)

# Métricas de topo
c1, c2, c3, c4 = st.columns(4)
c1.metric("Janelas", len(scores))
c2.metric("Anomalias marcadas", int(flags.sum()), f"{100*flags.mean():.1f}%")
if inject and labels_win is not None:
    em = event_metrics(flags, labels_win)
    sc = score_curves(scores, labels_win)
    c3.metric("Recall / Precision (evento)", f"{em['recall']:.2f} / {em['precision']:.2f}")
    c4.metric("PR-AUC (janela)", f"{sc['pr_auc']:.3f}", f"prev {sc['prevalence']:.2f}")
else:
    c3.metric("Limiar (mediano)", f"{np.nanmedian(thr_series):.4f}")
    c4.metric("Erro médio", f"{scores.mean():.4f}")

# Gráfico: erro + limiar + anomalias
chart = pd.DataFrame({"erro": scores, "limiar": thr_series}, index=dates)
st.line_chart(chart, height=320)

anom_dates = dates[flags]
if len(anom_dates):
    st.markdown("**Janelas anômalas (datas):** " + ", ".join(
        d.strftime("%Y-%m-%d") for d in anom_dates[:25]
    ) + (" …" if len(anom_dates) > 25 else ""))

# Eventos conhecidos no período (correlação — ADR-0008)
if len(dates):
    ev = events.events_in_range(dates.min().date().isoformat(), dates.max().date().isoformat(), ticker=ticker)
    if len(ev):
        st.subheader("📅 Eventos documentados no período (ADR-0008)")
        st.dataframe(ev[["date", "label", "tickers"]], hide_index=True, width="stretch")

with st.expander("ℹ️ O que é ao vivo e o que já foi decidido"):
    st.markdown(
        "- **Ao vivo** (recalcula sem retreinar): agregação, limiar (tipo/percentil/janela), injeção.\n"
        "- **Já decidido por experimento** (fixo aqui): arquitetura, `latent_dim=16`, `weight_decay=0` "
        "(ADR-0018), Close+Volume, macro USDBRL/VIX. Mudá-los exige retreino (minutos) — ver ADRs.\n"
        "- **Anti-vazamento:** o limiar estático usa o percentil do **treino** (normalidade), nunca do período avaliado."
    )
