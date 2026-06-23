"""Linha do tempo de eventos econômicos e políticos brasileiros (issue #25).

Fonte de verdade narrativa para M6: cada anomalia detectada é cruzada com
eventos desta lista para validação qualitativa (ADR-0006, via narrativa).

``tickers=None`` indica evento sistêmico (afeta todos os ativos analisados).
Lista ordenada cronologicamente; cobre o período do dataset (2010–2024).
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Linha do tempo
# ---------------------------------------------------------------------------

_EVENTS: list[dict] = [
    # 2010–2013 — recuperação pós-crise e boom de commodities
    {"date": "2011-08-05", "label": "EUA perde rating AAA; crise da dívida europeia", "tickers": None},
    {"date": "2012-06-01", "label": "Fitch rebaixa perspectiva do rating Brasil", "tickers": None},
    {"date": "2013-06-17", "label": "Protestos de junho: manifestações em massa no Brasil", "tickers": None},

    # 2014–2016 — recessão, Lava Jato e impeachment
    {"date": "2014-10-26", "label": "Eleições: Dilma Rousseff reeleita (2º turno)", "tickers": None},
    {"date": "2015-01-15", "label": "Petrobras: divulgação bilhão de perdas Lava Jato", "tickers": ["PETR4.SA"]},
    {"date": "2015-09-09", "label": "S&P rebaixa Brasil para grau especulativo (junk)", "tickers": None},
    {"date": "2015-11-05", "label": "Samarco/Mariana: rompimento da barragem de Fundão", "tickers": ["VALE3.SA"]},
    {"date": "2016-05-12", "label": "Dilma afastada; Temer assume interinamente", "tickers": None},
    {"date": "2016-08-31", "label": "Impeachment de Dilma Rousseff efetivado pelo Senado", "tickers": None},

    # 2017–2018 — Joesley Day, greve e eleições
    {"date": "2017-05-17", "label": "Joesley Day: gravação de Temer vaza; Ibovespa cai 9% em 1 dia", "tickers": None},
    {"date": "2018-05-21", "label": "Greve dos caminhoneiros: paralisação nacional ~10 dias", "tickers": ["PETR4.SA"]},
    {"date": "2018-10-28", "label": "Eleições: Jair Bolsonaro eleito presidente (2º turno)", "tickers": None},

    # 2019 — Brumadinho
    {"date": "2019-01-25", "label": "Brumadinho: rompimento da barragem Córrego do Feijão (Vale)", "tickers": ["VALE3.SA"]},

    # 2020 — COVID
    {"date": "2020-01-30", "label": "OMS declara emergência de saúde pública internacional — COVID-19", "tickers": None},
    {"date": "2020-03-11", "label": "OMS declara pandemia de COVID-19", "tickers": None},
    {"date": "2020-03-23", "label": "Pico do crash COVID na B3: Ibovespa acumula ~30% de queda no mês", "tickers": None},

    # 2021 — intervenção Petrobras
    {"date": "2021-02-19", "label": "Bolsonaro demite CEO da Petrobras (Castello Branco); interferência política", "tickers": ["PETR4.SA"]},
    {"date": "2021-04-13", "label": "Petrobras: novo CEO (Silva e Luna) confirmado pelo conselho", "tickers": ["PETR4.SA"]},

    # 2022 — eleições e tensão fiscal
    {"date": "2022-10-30", "label": "Eleições: Luiz Inácio Lula da Silva eleito presidente (2º turno)", "tickers": None},
    {"date": "2022-12-22", "label": "PEC da Transição aprovada; expansão fiscal gera pressão no câmbio", "tickers": None},

    # 2023 — caso Americanas e outros
    {"date": "2023-01-08", "label": "Atos golpistas: invasão do Congresso, STF e Palácio do Planalto em Brasília", "tickers": None},
    {"date": "2023-01-11", "label": "Americanas: rombo contábil de R$ 20 bi divulgado; ação cai ~78%", "tickers": ["AMER3.SA"]},
    {"date": "2023-01-19", "label": "Americanas: pedido de recuperação judicial protocolado", "tickers": ["AMER3.SA"]},
    {"date": "2023-08-30", "label": "Arcabouço fiscal aprovado pelo Congresso", "tickers": None},

    # 2024 — AMER3 recuperação judicial + commodities
    {"date": "2024-08-15", "label": "AMER3: salto especulativo durante negociação da recuperação judicial", "tickers": ["AMER3.SA"]},
    {"date": "2024-11-14", "label": "AMER3: novo salto; volume 3× a média (especulação em recuperação judicial)", "tickers": ["AMER3.SA"]},
]


# ---------------------------------------------------------------------------
# Funções de acesso
# ---------------------------------------------------------------------------


def events_dataframe() -> pd.DataFrame:
    """Retorna todos os eventos como DataFrame com colunas date/label/tickers.

    ``date`` é ``pd.Timestamp``; ``tickers`` é lista de strings ou ``None``
    (sistêmico). Ordenado cronologicamente.
    """
    df = pd.DataFrame(_EVENTS)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def events_in_range(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Filtra eventos dentro de um intervalo de datas [start, end].

    Args:
        start/end: limites inclusive do intervalo.
        ticker: se informado, retorna apenas eventos sistêmicos (``tickers=None``)
            **ou** que incluam explicitamente este ticker. Omitir retorna tudo.

    Returns:
        DataFrame filtrado, mesma estrutura de ``events_dataframe()``.
    """
    df = events_dataframe()
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    df = df[mask]

    if ticker is not None:
        def _matches(tickers):
            return tickers is None or ticker in tickers

        df = df[df["tickers"].apply(_matches)]

    return df.reset_index(drop=True)
