"""Engenharia de features macroeconômicas para o Conditional AE (ADR-0012, proposto).

Alinha indicadores macro (Selic, IPCA, USDBRL, VIX, ...) ao calendário de pregão da
B3 sem vazar o futuro e os estaciona antes de alimentar a rede.

**Contrato anti-leakage (CRÍTICO).** O `DataFrame` macro **deve estar indexado pela
data de PUBLICAÇÃO** do dado, não pela data de referência. O IPCA de março é
divulgado em meados de abril; se vier indexado por março, o `ffill` espalharia ~6
semanas de futuro. O mesmo vale para decisões do COPOM (data da decisão vs vigência).
Este módulo assume o contrato e só faz preenchimento **causal** (`ffill`); não há
`bfill` (que puxaria o futuro para o passado).

O conector de dados (BCB/SGS, FRED, etc.) é externo a este módulo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG, PROJECT_ROOT

# --- Conector de dados macro (BCB/SGS + yfinance) -------------------------- #

# Séries do Banco Central (SGS). Diárias (dias úteis) → dadas pela data em que
# valem (causais, conhecidas no próprio dia). IPCA é MENSAL e exige lag de publicação.
# Selic = 1178 (taxa anualizada base 252, dias úteis); a meta (432) inclui fins de
# semana e estoura o limite de pontos do SGS em janelas longas.
BCB_SERIES = {"USDBRL": 1, "Selic": 1178, "IPCA": 433}

# IPCA (ref. mês) é divulgado ~10 dias após o fim do mês de referência. A SGS data
# pelo 1º dia do mês de referência → desloca-se para a data aproximada de
# PUBLICAÇÃO (mês seguinte + ~9 dias) para não vazar o futuro (ADR-0012). É uma
# aproximação conservadora (publicação um pouco tarde é seguro; cedo vazaria).
PUBLICATION_LAG = {"IPCA": pd.DateOffset(months=1, days=9)}

MACRO_CACHE = PROJECT_ROOT / "data" / "raw" / "macro.csv"


def _bcb_request(code: int, d0: pd.Timestamp, d1: pd.Timestamp) -> list:
    import json
    import urllib.request

    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
        f"?formato=json&dataInicial={d0.strftime('%d/%m/%Y')}"
        f"&dataFinal={d1.strftime('%d/%m/%Y')}"
    )
    req = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "NeuraTrade/1.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
    if not body.lstrip().startswith(b"["):  # SGS devolve HTML/XML quando excede o limite
        raise ValueError(
            f"SGS série {code}: resposta não-JSON ({d0.date()}..{d1.date()}); "
            "reduza a janela do chunk."
        )
    return json.loads(body)


def fetch_bcb_series(code: int, start: str, end: str) -> pd.Series:
    """Baixa uma série do SGS/BCB (JSON). Índice = data; valores = float. **Rede.**

    O SGS limita o nº de pontos por request (séries diárias com fim de semana
    estouram em poucos anos) → busca em blocos de 4 anos e concatena.
    """
    import time

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    rows: list = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + pd.DateOffset(years=4) - pd.Timedelta(days=1), end)
        rows.extend(_bcb_request(code, cursor, chunk_end))
        cursor = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.4)  # cortesia / evita throttle do SGS

    s = pd.Series(
        {pd.to_datetime(r["data"], dayfirst=True): float(r["valor"]) for r in rows}
    )
    return s[~s.index.duplicated()].sort_index()


def fetch_macro(
    start: str | None = None,
    end: str | None = None,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Baixa os indicadores macro indexados por **data de publicação** (ADR-0012). **Rede.**

    BCB/SGS: USDBRL (1), Selic meta (432), IPCA (433); VIX via yfinance (``^VIX``).
    O IPCA (mensal) recebe o lag de publicação (``PUBLICATION_LAG``) — as demais são
    conhecidas no próprio dia. Colunas na ordem de ``features``.
    """
    start = start or CONFIG["data"]["start"]
    end = end or CONFIG["data"]["end"]
    features = features or CONFIG.get("macro", {}).get(
        "features", ["USDBRL", "VIX", "Selic", "IPCA"]
    )

    cols = {}
    for feat in features:
        if feat == "VIX":
            import yfinance as yf

            vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
            s = vix["Close"]
            s = s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
            cols[feat] = s
        elif feat in BCB_SERIES:
            s = fetch_bcb_series(BCB_SERIES[feat], start, end)
            if feat in PUBLICATION_LAG:  # desloca p/ a data de publicação (IPCA)
                s.index = s.index + PUBLICATION_LAG[feat]
            cols[feat] = s
        else:
            raise ValueError(f"feature macro desconhecida: {feat!r}")

    df = pd.DataFrame(cols)[features]
    df.index.name = "Date"
    return df


def cache_macro(
    start: str | None = None, end: str | None = None, path: str | Path | None = None
) -> Path:
    """Baixa e persiste a macro em ``data/raw/macro.csv``. Retorna o caminho."""
    path = Path(path) if path is not None else MACRO_CACHE
    df = fetch_macro(start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return path


def load_macro(path: str | Path | None = None) -> pd.DataFrame:
    """Carrega a macro do cache (``data/raw/macro.csv``). **Nunca acessa a rede.**"""
    path = Path(path) if path is not None else MACRO_CACHE
    if not path.exists():
        raise FileNotFoundError(
            f"Cache macro ausente: {path}. Rode scripts/cache_data.py --macro (requer rede)."
        )
    return pd.read_csv(path, index_col="Date", parse_dates=True)

# Política de estacionarização por feature (ADR-0012). Níveis não-estacionários
# viram inovações; séries já estacionárias/limitadas ficam em nível.
DEFAULT_STATIONARIZE = {
    "USDBRL": "logret",  # nível tendência → log-retorno
    "Selic": "delta",    # taxa em escada → variação (≠0 só em mudança)
    "IPCA": "level",     # já é uma taxa (%); mantém nível
    "VIX": "level",      # mean-reverting limitado; mantém nível (ou z-score externo)
}


def align_macro(
    df_asset: pd.DataFrame,
    df_macro: pd.DataFrame,
    macro_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Alinha macro ao calendário do ativo, com preenchimento **causal** (ADR-0012).

    Junta `df_macro` (indexado por **data de publicação** — ver contrato no módulo)
    ao índice de pregão de `df_asset` e propaga para frente (`ffill`). **Não** faz
    `bfill`: as linhas iniciais sem macro são descartadas (perder o começo é o preço
    de não inventar passado).

    Args:
        df_asset: dados do ativo (índice = pregões da B3).
        df_macro: indicadores macro, índice = data de publicação.
        macro_cols: colunas a usar. Usa todas de `df_macro` se ``None``.

    Returns:
        `DataFrame` no índice de `df_asset` contendo **apenas as colunas macro**
        (não as do ativo), preenchidas causalmente; linhas iniciais sem cobertura
        macro são removidas.
    """
    macro_cols = macro_cols or list(df_macro.columns)
    # Reindexa a macro ao calendário do ativo (só as colunas macro, sem as do ativo).
    aligned = df_macro[macro_cols].reindex(df_asset.index.union(df_macro.index))
    # Preenchimento SÓ para frente (causal): valor vale do dia da publicação em diante.
    aligned = aligned.ffill().reindex(df_asset.index)
    # Sem bfill — descarta o cabeçalho ainda sem macro (evita lookahead em D0).
    return aligned.dropna(subset=macro_cols)


def stationarize_macro(
    df: pd.DataFrame,
    methods: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Estaciona as colunas macro conforme a política (ADR-0012).

    Métodos por coluna: ``"logret"`` (log-retorno, p/ níveis com tendência),
    ``"delta"`` (primeira diferença, p/ taxas em escada) ou ``"level"`` (mantém).
    Colunas fora de `methods` ficam em nível. A primeira linha (NaN do diff/logret)
    é descartada, mantendo todas as colunas alinhadas.

    Returns:
        `DataFrame` com as mesmas colunas, estacionarizadas.
    """
    methods = methods or DEFAULT_STATIONARIZE
    out = {}
    for col in df.columns:
        m = methods.get(col, "level")
        if m == "logret":
            out[col] = np.log(df[col] / df[col].shift(1))
        elif m == "delta":
            out[col] = df[col].diff()
        elif m == "level":
            out[col] = df[col].astype(float)
        else:
            raise ValueError(f"método de estacionarização desconhecido: {m!r}")
    return pd.DataFrame(out, index=df.index).dropna()
