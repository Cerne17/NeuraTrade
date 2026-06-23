# scripts/ — utilitários

Scripts auxiliares versionados (M10). A **lógica** do projeto vive em `src/`; estes
scripts são atalhos finos e ferramentas operacionais. Rode-os a partir da raiz do
repositório, com o ambiente ativado (`.venv/bin/python ...` ou `python ...`).

| Script | O que faz | Rede? |
| ------ | --------- | ----- |
| `run_pipeline.py` | Wrapper da CLI `python -m src`: roda a pipeline de detecção end-to-end e imprime o resumo por ativo. | não |
| `cache_data.py` | Refaz o cache `data/raw/*.csv` via yfinance (único passo de rede). | **sim** |
| `build_figures.py` | Regenera as figuras do relatório em `report/figures/` a partir dos modelos treinados, usando os plots de `src/viz.py`. | não |

## Exemplos

```bash
# pipeline completa (carrega modelos salvos)
.venv/bin/python scripts/run_pipeline.py
# ...ou treinando do zero
.venv/bin/python scripts/run_pipeline.py --train

# recriar o cache de dados (precisa de internet)
.venv/bin/python scripts/cache_data.py

# regenerar as figuras do relatório
.venv/bin/python scripts/build_figures.py
```

> A pipeline equivalente também roda direto por `python -m src` (ver `src/__main__.py`).
> Os **notebooks** (`notebooks/`) são os **estudos** do projeto, não a pipeline de produção.
