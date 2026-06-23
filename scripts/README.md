# scripts/ — utilitários

Scripts auxiliares versionados (M10). A **lógica** do projeto vive em `src/`; estes
scripts são atalhos finos e ferramentas operacionais. Rode-os a partir da raiz do
repositório, com o ambiente ativado (`.venv/bin/python ...` ou `python ...`).

| Script | O que faz | Rede? |
| ------ | --------- | ----- |
| `run_pipeline.py` | Wrapper da CLI `python -m src`: roda a pipeline de detecção end-to-end e imprime o resumo por ativo. | não |
| `fetch_window.py` | Baixa OHLCV de uma **janela nova** (ex.: Q1/2025) para `data/inference/`, sem tocar no cache de treino. | **sim** |
| `run_inference.py` | **Inferência interativa**: pergunta intervalo + ticker, baixa, roda os modelos e lista as janelas anômalas. Fácil de usar. | **sim** |
| `cache_data.py` | Refaz o cache `data/raw/*.csv` via yfinance (período de treino). | **sim** |
| `build_figures.py` | Regenera as figuras do relatório em `report/figures/` a partir dos modelos treinados, usando os plots de `src/viz.py`. | não |

## Exemplos

```bash
# pipeline completa (carrega modelos salvos)
.venv/bin/python scripts/run_pipeline.py
# ...ou treinando do zero
.venv/bin/python scripts/run_pipeline.py --train

# aplicar os modelos a uma janela NOVA (fora do treino)
.venv/bin/python scripts/fetch_window.py --start 2025-01-01 --end 2025-03-31
.venv/bin/python scripts/run_inference.py        # interativo: digita intervalo + ticker

# recriar o cache de treino / regenerar figuras
.venv/bin/python scripts/cache_data.py
.venv/bin/python scripts/build_figures.py
```

> **Inferência fora do treino (`src/inference.py`):** a normalização e o limiar usados na janela
> nova são os do **treino** (normalidade 2010–2019) — não são reajustados sobre o período novo
> (ADR-0001). Assim "anomalia" continua significando "desvio da normalidade aprendida", não
> "desvio do próprio intervalo".

> A pipeline equivalente também roda direto por `python -m src` (ver `src/__main__.py`).
> Os **notebooks** (`notebooks/`) são os **estudos** do projeto, não a pipeline de produção.
