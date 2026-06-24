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

import numpy as np
import pandas as pd

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
