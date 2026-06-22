## Estrutura do Projeto
```
NeuraTrade/
├── README.md
├── requirements.txt          
├── pyproject.toml            
├── .gitignore
├── config.yaml               # fonte única de hiperparâmetros
├── data/
│   ├── raw/                  # cache do yfinance — COMMITADO (pequeno)
│   │   ├── PETR4.csv
│   │   ├── VALE3.csv
│   │   ├── AMER3.csv
│   │   └── ITUB4.csv
│   └── processed/            # gerado → no .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py             # carrega o config.yaml + seeds globais
│   ├── data.py               # download, cache, load
│   ├── preprocessing.py      # log-returns, split, scaler, janelas
│   ├── model.py              # arquitetura do LSTM-AE (build/compile)
│   ├── train.py              # loop de treino + callbacks
│   ├── detect.py             # erro de reconstrução, threshold estático/dinâmico
│   ├── evaluate.py           # injeção sintética, precision/recall/f1
│   ├── events.py             # timeline de eventos BR (dict datado)
│   └── viz.py                # funções de plot padronizadas
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_train.ipynb
│   ├── 04_detection_thresholds.ipynb
│   ├── 05_evaluation_synthetic.ipynb
│   └── 06_events_correlation.ipynb
├── models/                   # pesos salvos por ativo → .gitignore
├── figures/                  # saídas pro relatório
└── report/
    ├── relatorio.tex
    └── refs.bib
```