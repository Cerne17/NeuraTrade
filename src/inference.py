"""Inferência em janelas de tempo novas, fora do período de treino.

Aplica os modelos já treinados (em 2010–2019) a um intervalo arbitrário (ex.: Q1
de 2025) e marca janelas anômalas pelo erro de reconstrução.

**Invariante metodológica (ADR-0001):** a normalização e o limiar usados na nova
janela são os do **treino** (a "normalidade" 2010–2019) — o `MinMaxScaler` e o
limiar estático **não** são reajustados sobre o período novo. Reajustar redefiniria
"normal" em função do próprio intervalo avaliado, anulando a detecção.

Fluxo:
1. ``fetch_window`` — baixa OHLCV da janela nova (rede).
2. ``infer_window`` — log-retornos → escala com o scaler do treino → janelas →
   erro de reconstrução (agregação de ``config.yaml``) → marca anomalias vs. o
   limiar estático do treino.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import data, detect
from . import preprocessing as pp
from .config import CONFIG
from .train import load_model


def fetch_window(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    save_dir: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Baixa OHLCV das tickers para ``[start, end]`` (rede, via yfinance).

    Não escreve em ``data/raw/`` (cache de treino) — se ``save_dir`` for dado,
    grava os CSVs lá (ex.: ``data/inference/``).

    Args:
        tickers: ativos. Usa ``CONFIG["tickers"]`` se ``None``.
        start/end: datas ISO da janela nova.
        save_dir: diretório para persistir os CSVs (opcional).

    Returns:
        ``{ticker: DataFrame OHLCV}``.
    """
    tickers = tickers or CONFIG["tickers"]
    out: dict[str, pd.DataFrame] = {}
    dest = Path(save_dir) if save_dir else None
    if dest:
        dest.mkdir(parents=True, exist_ok=True)

    for t in tickers:
        df = data.download_ticker(t, start=start, end=end)
        out[t] = df
        if dest:
            df.to_csv(dest / f"{t}.csv")
    return out


def _training_reference(ticker: str, model) -> tuple:
    """Devolve ``(scaler, limiar_estatico)`` do **treino** (normalidade) do ticker.

    Carrega o histórico de ``data/raw`` e refaz o pré-processamento (cujo split de
    treino é 2010–2019), de onde saem o scaler e o erro de treino para o limiar.
    """
    pre = pp.preprocess_ticker(data.load_ticker(ticker))
    err_train = detect.reconstruction_error(model, pre["X_train"])
    return pre["scaler"], detect.static_threshold(err_train)


def infer_window(
    ticker: str,
    df_window: pd.DataFrame,
    model=None,
) -> pd.DataFrame:
    """Roda o modelo treinado sobre uma janela nova e marca anomalias.

    Args:
        ticker: ativo (usado para carregar modelo, scaler e limiar do treino).
        df_window: OHLCV da janela nova (saída de :func:`fetch_window`).
        model: modelo já carregado (opcional; senão carrega ``models/<ticker>.keras``).

    Returns:
        ``DataFrame`` indexado por data (fim de cada janela deslizante) com colunas
        ``erro``, ``limiar`` e ``anomalia`` (bool).

    Raises:
        ValueError: se a janela tiver menos pregões que ``window_size + 1``.
    """
    model = model or load_model(ticker)
    scaler, threshold = _training_reference(ticker, model)

    r = pp.log_returns(df_window)
    W = CONFIG["preprocessing"]["window_size"]
    if len(r) < W:
        raise ValueError(
            f"janela curta para {ticker}: {len(r)} retornos < window_size={W}. "
            "Use um intervalo maior (≳ 2 meses de pregão)."
        )

    scaled = pp.apply_scaler(scaler, r)
    X = pp.make_windows(scaled)
    err = detect.reconstruction_error(model, X)
    flags = detect.flag_anomalies(err, threshold)

    dates = r.index[W - 1 : W - 1 + len(err)]
    return pd.DataFrame(
        {"erro": err, "limiar": threshold, "anomalia": flags}, index=dates
    )


def infer_all(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    window_data: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Conveniência: busca (ou recebe) a janela e roda a inferência em cada ticker.

    Args:
        window_data: dados já baixados (de :func:`fetch_window`); se ``None``,
            baixa via ``fetch_window``.
    """
    tickers = tickers or CONFIG["tickers"]
    if window_data is None:
        window_data = fetch_window(tickers, start, end)
    return {t: infer_window(t, window_data[t]) for t in tickers}
