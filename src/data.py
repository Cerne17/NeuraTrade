"""Coleta, cache e carregamento dos dados de preço (B3 via yfinance).

Separação deliberada (issue #6):
- ``cache_ticker`` / ``cache_all`` — ÚNICO ponto que acessa a rede (yfinance) e
  persiste em ``data/raw/<TICKER>.csv``.
- ``load_ticker`` / ``load_all`` — usados pelo pipeline; leem **somente** o cache,
  nunca a API. Assim os notebooks rodam offline e de forma reprodutível.

Decisões em docs/adr/0007-coleta-e-tratamento-amer3.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CONFIG, PROJECT_ROOT

RAW_DIR = PROJECT_ROOT / CONFIG["data"]["raw_dir"]

# Colunas OHLCV mantidas. Com auto_adjust=True, "Close" já vem ajustado por
# splits/dividendos (ver ADR-0007).
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _raw_path(ticker: str) -> Path:
    return RAW_DIR / f"{ticker}.csv"


def download_ticker(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Baixa OHLCV de um ticker via yfinance. **Acessa a rede.**

    Achata o ``MultiIndex`` de colunas que o yfinance retorna para um único
    ticker e mantém apenas as colunas OHLCV.

    Args:
        ticker: símbolo B3 (ex.: ``"PETR4.SA"``).
        start/end: datas (ISO). Usam ``CONFIG["data"]`` se ``None``.
        auto_adjust: ajusta preços por splits/dividendos (recomendado).

    Raises:
        ValueError: se o yfinance não retornar dados.
    """
    import yfinance as yf

    start = start or CONFIG["data"]["start"]
    end = end or CONFIG["data"]["end"]

    df = yf.download(
        ticker, start=start, end=end, auto_adjust=auto_adjust, progress=False
    )
    if df.empty:
        raise ValueError(f"yfinance retornou vazio para {ticker} ({start}..{end}).")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel("Ticker")
    df.index.name = "Date"
    return df[OHLCV]


def cache_ticker(ticker: str, **kwargs) -> Path:
    """Baixa e persiste ``data/raw/<TICKER>.csv``. Retorna o caminho."""
    df = download_ticker(ticker, **kwargs)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = _raw_path(ticker)
    df.to_csv(path)
    return path


def cache_all(tickers: list[str] | None = None, **kwargs) -> dict[str, Path]:
    """Baixa e persiste todos os tickers do ``CONFIG`` (ou os informados)."""
    tickers = tickers or CONFIG["tickers"]
    return {t: cache_ticker(t, **kwargs) for t in tickers}


def load_ticker(ticker: str) -> pd.DataFrame:
    """Carrega um ticker do cache local. **Nunca acessa a rede.**

    Raises:
        FileNotFoundError: se o cache não existir (rode ``cache_all`` antes).
    """
    path = _raw_path(ticker)
    if not path.exists():
        raise FileNotFoundError(
            f"Cache ausente: {path}. Rode data.cache_all() uma vez (requer rede)."
        )
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def load_all(tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Carrega todos os tickers do cache em um dict ``{ticker: DataFrame}``."""
    tickers = tickers or CONFIG["tickers"]
    return {t: load_ticker(t) for t in tickers}


def integrity_report(df: pd.DataFrame) -> dict:
    """Resumo de integridade de uma série OHLCV (para EDA / issue #7).

    Reporta linhas, intervalo de datas, NaNs, dias com volume zero (halt/
    iliquidez), preços de fechamento não positivos, índices duplicados e a
    maior queda/alta de log-retorno diário (sinaliza crashes e possíveis
    artefatos de grupamento não ajustado).
    """
    import numpy as np

    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    return {
        "n_rows": len(df),
        "date_min": df.index.min(),
        "date_max": df.index.max(),
        "nans": int(df.isna().sum().sum()),
        "zero_volume_days": int((df["Volume"] == 0).sum()),
        "nonpositive_close": int((df["Close"] <= 0).sum()),
        "duplicated_index": int(df.index.duplicated().sum()),
        "max_drop_logret": float(log_ret.min()),
        "max_drop_date": log_ret.idxmin(),
        "max_jump_logret": float(log_ret.max()),
        "max_jump_date": log_ret.idxmax(),
    }
