# Detecção de Anomalias em Séries Temporais Financeiras Brasileiras com LSTM-Autoencoder

> Trabalho Final — **Redes Neurais** (2026.1)
> Autores: **Ana Beatriz** e **Miguel Cerne**

Detecção não supervisionada de anomalias em ações da **B3** usando uma arquitetura **LSTM-Autoencoder**. O modelo é treinado apenas em períodos de comportamento "normal" e aprende a reconstruir padrões típicos; quando confrontado com movimentos atípicos, o **erro de reconstrução** dispara e sinaliza a anomalia. As anomalias detectadas são então cruzadas com uma linha do tempo de eventos econômicos e políticos brasileiros documentados.

---

## Sobre o projeto

A maior parte da literatura de detecção de anomalias com deep learning se concentra no mercado americano e em criptomoedas. Este trabalho aplica a abordagem a **ativos brasileiros** e adiciona quatro contribuições combinadas:

- Aplicação em ativos nacionais (B3), contexto sub-representado na literatura.
- Correlação sistemática e documentada entre anomalias detectadas e eventos históricos brasileiros.
- Comparação entre **threshold estático** (percentil fixo) e **threshold dinâmico** (percentil em janela móvel).
- Avaliação de generalização entre setores econômicos distintos.

### Ativos analisados

| Ticker     | Empresa         | Setor               |
| ---------- | --------------- | ------------------- |
| `PETR4.SA` | Petrobras PN    | Energia/commodities |
| `VALE3.SA` | Vale S.A.       | Mineração           |
| `AMER3.SA` | Americanas S.A. | Varejo              |
| `ITUB4.SA` | Itaú Unibanco   | Financeiro          |

Período de coleta: **2010–2024**. Treino em 2010–2019 ("normalidade"), teste em 2020–2024.

---

## Quick start

### Google Colab (recomendado)

No topo de qualquer notebook:

```python
!git clone https://github.com/<usuario>/anomaly-detection-b3.git
%cd anomaly-detection-b3
!pip install -e .

from src.config import set_seeds, CONFIG
set_seeds()
```

### Local

```bash
git clone https://github.com/<usuario>/anomaly-detection-b3.git
cd anomaly-detection-b3
python -m venv .venv && source .venv/bin/activate
pip install -e .            # ou: pip install -r requirements.txt
```

Os dados brutos já estão versionados em `data/raw/`, então o pipeline roda offline — não é necessário baixar nada do yfinance para reproduzir os resultados.

---

## Estrutura do repositório

```
anomaly-detection-b3/
├── config.yaml               # fonte única de hiperparâmetros
├── data/
│   ├── raw/                  # cache do yfinance (versionado)
│   └── processed/            # gerado (não versionado)
├── src/
│   ├── config.py             # carrega config.yaml + seeds globais
│   ├── data.py               # download, cache, load
│   ├── preprocessing.py      # log-returns, split, scaler, janelas
│   ├── model.py              # arquitetura do LSTM-Autoencoder
│   ├── train.py              # treino + callbacks
│   ├── detect.py             # erro de reconstrução, thresholds
│   ├── evaluate.py           # injeção sintética, precision/recall/f1
│   ├── events.py             # linha do tempo de eventos BR
│   └── viz.py                # plots padronizados
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_train.ipynb
│   ├── 04_detection_thresholds.ipynb
│   ├── 05_evaluation_synthetic.ipynb
│   └── 06_events_correlation.ipynb
├── models/                   # pesos treinados (não versionado)
├── figures/                  # saídas para o relatório
└── report/                   # relatório LaTeX
```

A lógica vive em `src/`; os notebooks apenas orquestram e visualizam. Isso permite reaproveitar o mesmo pipeline nos quatro ativos e reduz conflitos de merge no `.ipynb`.

---

## Como rodar

Os notebooks são numerados na ordem de execução e mapeiam as fases do projeto:

| Notebook                       | Etapa                                                              |
| ------------------------------ | ----------------------------------------------------------------- |
| `01_eda`                       | Inspeção das séries, gaps, integridade dos dados                  |
| `02_preprocessing`             | Log-returns, split temporal, normalização, janelas deslizantes    |
| `03_train`                     | Treino do LSTM-Autoencoder (um modelo por ativo)                  |
| `04_detection_thresholds`      | Erro de reconstrução; threshold estático vs. dinâmico             |
| `05_evaluation_synthetic`      | Injeção de anomalias artificiais; Precision, Recall, F1           |
| `06_events_correlation`        | Sobreposição das anomalias com eventos econômicos/políticos       |

---

## Metodologia

1. **Pré-processamento** — retorno logarítmico diário como feature principal; *split temporal antes da normalização* (o `MinMaxScaler` é ajustado apenas no treino, para evitar vazamento de informação do futuro); janelas deslizantes.
2. **Modelo** — Encoder LSTM → bottleneck → Decoder LSTM, otimizando o MSE de reconstrução, com `EarlyStopping`. Um modelo por ativo, para permitir a comparação setorial.
3. **Detecção** — uma janela é marcada como anômala quando seu erro de reconstrução ultrapassa o limiar (percentil 95 do erro de treino, no caso estático).
4. **Threshold dinâmico** — percentil calculado em janela temporal móvel, comparado ao estático.

### Avaliação

Como o problema é não supervisionado, a avaliação combina duas vias:

- **Narrativa** — verificação de que as anomalias detectadas correspondem a eventos reais conhecidos (crash de março/2020, caso Americanas em janeiro/2023, etc.).
- **Quantitativa** — injeção controlada de perturbações artificiais (*price shocks* e *volume spikes*) em posições conhecidas, permitindo calcular Precision, Recall e F1.

> **Nota metodológica:** o período de treino (2010–2019) contém eventos de forte volatilidade (recessão de 2014–2016, Lava Jato, impeachment de 2016). A definição de "normalidade" é, portanto, relativa e está discutida no relatório.

---

## Configuração

Todos os hiperparâmetros ficam em `config.yaml` (tickers, datas de split, `window_size`, percentil do threshold, tamanho da janela móvel, seed). É a fonte única de verdade — alterar um experimento significa alterar esse arquivo, não os notebooks.

---

## Stack

`Python` · `TensorFlow/Keras` · `yfinance` · `pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn`

---

## Referências

- Li, S. (2020). *Time Series of Price Anomaly Detection with LSTM.*
- Petrovic, D. *Anomaly Detection in Stock Price with LSTM Autoencoder.* (GitHub)
- *Anomaly Detection on Bitcoin Values.* IEEE (2021).
- Liu et al. (2025). *Robust Anomaly Detection in Financial Markets Using LSTM Autoencoders and GANs.*
- *LSTM e DBSCAN para Bitcoin.* Springer Nature (2025).

Repositórios públicos foram usados apenas como guia arquitetural. Toda a implementação, os dados e as análises são originais.

---

## Licença

Projeto acadêmico desenvolvido para a disciplina de Redes Neurais (2026.1). Uso educacional.