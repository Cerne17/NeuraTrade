# Demo — Sandbox interativo NeuraTrade

Demonstração ao vivo da detecção de anomalias: varia parâmetros de **detecção** e
do **protocolo de avaliação** em tempo real, sobre os modelos já treinados
(2010–2019). Reaproveita `src/detect.py` e `src/evaluate.py`; o erro por passo é
calculado uma vez por ativo (cache) e reagregado instantaneamente.

## Rodar

```bash
pip install streamlit          # dependência só do demo
streamlit run demo/sandbox.py  # a partir da raiz do repo
```

Abre em `http://localhost:8501`. Requer modelos em `models/<ticker>.keras` e dados
em `data/raw/` (ambos versionados → roda **offline**). Se faltarem modelos:
`python -m src --train`.

## O que dá para mexer ao vivo (sem retreinar)

| Controle | Efeito | ADR |
| --- | --- | --- |
| Ativo / Período | troca ativo (PETR4/VALE3/AMER3/ITUB4) e treino↔teste | — |
| Agregação | `max` / `mean` / `percentile` do erro na janela | 0009 |
| Limiar | estático (percentil do treino) vs dinâmico (janela causal) | 0005 |
| Injeção sintética | `n_injections`, choque `k·σ` → P/R/F1 por evento + PR-AUC ao vivo | 0006/0015/0017 |

O gráfico mostra erro por janela + linha de limiar + anomalias marcadas; abaixo,
os eventos brasileiros documentados no período (correlação, ADR-0008).

## O que **não** muda ao vivo (decidido por experimento)

Arquitetura, `latent_dim=16`, `weight_decay=0` (ADR-0018), features Close+Volume,
macro USDBRL/VIX. Mudá-los exige retreino (minutos) — a decisão e a evidência estão
nos ADRs. É de propósito: o sandbox mostra que o *detector* é robusto a como se
lê o erro, não a quanta capacidade se empilha.
