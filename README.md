# NeuraTrade — Detecção de Anomalias em Séries Financeiras da B3 com LSTM-Autoencoder

> Trabalho Final — **Redes Neurais** (2026.1)
> Autores: **Ana Beatriz** e **Miguel Cerne**

Detecção **não supervisionada** de anomalias em ações da **B3** com uma arquitetura
**LSTM-Autoencoder**. O modelo é treinado apenas em períodos de comportamento "normal"
(2010–2019) e aprende a reconstruir padrões típicos; diante de movimentos atípicos, o **erro de
reconstrução** dispara e sinaliza a anomalia. As anomalias detectadas são cruzadas com uma linha
do tempo de eventos econômicos e políticos brasileiros documentados.

---

## Sobre o projeto

A maior parte da literatura de detecção de anomalias com deep learning foca o mercado americano e
criptomoedas. Este trabalho aplica a abordagem a **ativos brasileiros** e reúne:

- Aplicação em ativos nacionais (B3), contexto sub-representado na literatura.
- Correlação documentada entre anomalias detectadas e eventos históricos brasileiros.
- Comparação entre **limiar estático** (percentil fixo) e **limiar dinâmico** (percentil em janela
  móvel causal).
- Avaliação de generalização entre setores econômicos distintos.

A modelagem evoluiu em duas fases, com cada decisão registrada como ADR e validada por
experimento (ver [Resultados](#resultados) e [Documentação](#documentação)):

- **Fase 1 (M0–M7)** — pipeline base: coleta, pré-processamento sem vazamento, treino,
  detecção (estático vs. dinâmico), avaliação sintética e correlação com eventos.
- **Fase 2 (M8–M9)** — evolução da modelagem: **agregação `max`** do erro por janela (Recall),
  **validação walk-forward** do `latent_dim` e **entrada multivariada Close+Volume** (anomalias
  de volume).
- **M10** — organização do repositório: pipeline executável (`python -m src`), `scripts/`
  versionado e índices.

### Ativos analisados

| Ticker     | Empresa         | Setor               |
| ---------- | --------------- | ------------------- |
| `PETR4.SA` | Petrobras PN    | Energia/commodities |
| `VALE3.SA` | Vale S.A.       | Mineração           |
| `AMER3.SA` | Americanas S.A. | Varejo              |
| `ITUB4.SA` | Itaú Unibanco   | Financeiro          |

Período de coleta: **2010–2024**. Treino em 2010–2019 ("normalidade"), teste em 2020–2024. Um
modelo por ativo, para permitir a comparação setorial.

---

## Quickstart

```bash
git clone https://github.com/Cerne17/NeuraTrade.git
cd NeuraTrade
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # ou: pip install -r requirements.txt

# pipeline de detecção end-to-end
python -m src --train            # 1ª vez: treina os 4 modelos (models/ não é versionado)
python -m src                    # execuções seguintes: carrega os modelos salvos
```

Os dados brutos já estão versionados em `data/raw/`, então tudo roda **offline** — o `yfinance`
só é necessário para recriar o cache (`python scripts/cache_data.py`).

A pipeline usa a configuração atual de `config.yaml` (agregação do erro **`max`** por default,
adotada em M9) e imprime um resumo por ativo:

```
$ python -m src --no-evaluate
         agregacao n_test_windows limiar_estatico frac_estatico frac_dinamico
PETR4.SA       max           1215         0.40772        0.0584        0.0568
VALE3.SA       max           1215          0.2631         0.0280        0.0856
AMER3.SA       max           1215          0.2844         0.2700        0.1251
ITUB4.SA       max           1215         0.37819         0.0560        0.0543
```

> **Rodando no Google Colab?** Veja o guia abaixo.

### Aplicar os modelos a uma janela nova (fora do treino)

Para rodar os modelos treinados em um período recente (ex.: Q1/2025) sem retreinar:

```bash
# 1) baixa a janela nova (não toca no cache de treino data/raw/)
python scripts/fetch_window.py --start 2025-01-01 --end 2025-03-31

# 2) inferência interativa: digita intervalo + ticker, lista as janelas anômalas
python scripts/run_inference.py
```

A normalização e o limiar continuam sendo os do **treino** (normalidade 2010–2019), não
reajustados sobre o período novo — "anomalia" segue significando *desvio da normalidade
aprendida* (ADR-0001). Em código: `from src.inference import infer_all`.

---

## Google Colab

Roda inteiramente no Colab — útil para ter GPU grátis no treino e não instalar nada localmente.

### 1. Setup (primeira célula)

Crie um notebook novo em [colab.research.google.com](https://colab.research.google.com) e cole:

```python
# clona o repositório e instala o pacote (data/raw já vem versionado → roda offline)
!git clone https://github.com/Cerne17/NeuraTrade.git
%cd NeuraTrade
!pip install -q -e .

from src.config import set_seeds, CONFIG
set_seeds()                      # reprodutibilidade (seed=42)
print("tickers:", CONFIG["tickers"], "| agregação:", CONFIG["detection"]["aggregation"])
```

> O Colab já traz TensorFlow; o `pip install -e .` apenas registra o pacote `src` e garante as
> demais dependências. Se o Colab pedir, clique em **"Restart runtime"** após a instalação e
> rode a célula de setup de novo (sem reclonar: `%cd /content/NeuraTrade`).

### 2. GPU (opcional, acelera o treino)

**Runtime → Change runtime type → Hardware accelerator: T4 GPU.** Sem GPU o treino dos quatro
modelos roda em CPU em poucos minutos; a detecção (sem `--train`) não precisa de GPU.

### 3. Rodar a pipeline

```python
# treina os 4 modelos e imprime o resumo por ativo (models/ não é versionado)
!python -m src --train

# execuções seguintes (modelos já treinados na sessão):
!python -m src
```

Ou chamando a função diretamente, para inspecionar o resultado como `DataFrame`:

```python
from src.pipeline import run_pipeline, summarize
res = run_pipeline(train=True)   # 1ª vez; depois train=False
summarize(res)
```

### 4. Rodar os estudos (notebooks)

Após o clone, abra um notebook pelo painel de arquivos do Colab
(`notebooks/03_train.ipynb`, etc.) **na mesma sessão** — ele já enxerga o `src/` instalado.
Para abrir um notebook direto do GitHub (**File → Open notebook → GitHub →**
`Cerne17/NeuraTrade`), adicione a célula de setup do passo 1 no topo antes de rodar as demais.

### 5. Atualizar os dados (opcional, requer rede)

Os CSVs já estão versionados; só rode isto para recriar o cache do zero:

```python
!python scripts/cache_data.py
```

---

## Estrutura do repositório

```
NeuraTrade/
├── config.yaml               # fonte única de hiperparâmetros (cada chave referencia um ADR)
├── data/
│   ├── raw/                  # cache do yfinance (versionado → roda offline)
│   └── processed/            # gerado (não versionado)
├── src/                      # PIPELINE (lógica do projeto)
│   ├── __main__.py           # CLI: `python -m src` roda a pipeline end-to-end
│   ├── pipeline.py           # run_pipeline(): data→preproc→train/load→detect→avaliar
│   ├── config.py             # carrega config.yaml + seeds globais
│   ├── data.py               # download, cache, load (yfinance)
│   ├── preprocessing.py      # log-returns, split, scaler, janelas (uni e multivariado)
│   ├── model.py              # arquitetura do LSTM-Autoencoder (n_features ≥ 1)
│   ├── train.py              # treino + EarlyStopping (univariado e multivariado)
│   ├── detect.py             # erro de reconstrução (mean/max/percentil), limiares
│   ├── validation.py         # validação Walk-Forward (TimeSeriesSplit), uni e multivariado
│   ├── evaluate.py           # injeção sintética, precision/recall/f1
│   ├── inference.py          # aplica modelos a janelas novas (fora do treino)
│   ├── events.py             # linha do tempo de eventos BR
│   └── viz.py                # plots padronizados
├── scripts/                  # UTILITÁRIOS (atalhos operacionais)
│   ├── run_pipeline.py       # wrapper da CLI
│   ├── fetch_window.py       # baixa uma janela nova (ex.: Q1/2025) → data/inference/
│   ├── run_inference.py      # inferência interativa em janelas à escolha
│   ├── cache_data.py         # refaz data/raw via yfinance (passo de rede)
│   ├── build_figures.py      # regenera report/figures/
│   └── README.md
├── notebooks/                # ESTUDOS (orquestram src/, versionados com saídas)
│   ├── README.md             # índice dos estudos por fase
│   └── 01_eda … 11_ohlcv_full.ipynb
├── docs/adr/                 # Architecture Decision Records (0001–0011)
├── report/                   # relatório LaTeX (preliminar)
├── teoria/                   # guia teórico autocontido (LaTeX)
├── debate/                   # confrontações teóricas / defesa das decisões (LaTeX)
├── models/                   # pesos treinados (não versionado)
└── figures/                  # saídas dos estudos
```

**Três papéis, três pastas:**

- **`src/`** — a **pipeline** (lógica do projeto). Ponto de entrada: `python -m src`.
- **`scripts/`** — **utilitários** operacionais (atalhos finos sobre `src/`). Ver [`scripts/README.md`](scripts/README.md).
- **`notebooks/`** — os **estudos** (exploração/medição, com saídas). Ver [`notebooks/README.md`](notebooks/README.md).

A lógica vive em `src/`; notebooks e scripts apenas a orquestram. Isso reaproveita o mesmo
pipeline nos quatro ativos e reduz conflitos de merge no `.ipynb`.

---

## Estudos (notebooks)

Numerados na ordem de execução (índice completo em [`notebooks/README.md`](notebooks/README.md)):

| Notebook                       | Estudo |
| ------------------------------ | ------ |
| `01_eda`                       | Inspeção das séries, gaps, integridade dos dados (M1). |
| `02_preprocessing`             | Log-retornos, split temporal, normalização, janelas (M2). |
| `03_train`                     | Treino do LSTM-Autoencoder, um modelo por ativo (M3). |
| `04_detection_thresholds`      | Erro de reconstrução; limiar estático vs. dinâmico (M4). |
| `05_evaluation_synthetic`      | Injeção sintética; Precision/Recall/F1 (M5). |
| `06_events_correlation`        | Anomalias × eventos econômicos/políticos (M6). |
| `07_aggregation_recalibration` | Agregação `max`/`percentil` + recalibração do limiar (M8, ADR-0009). |
| `08_walkforward`               | Seleção de `latent_dim` por walk-forward (M8, ADR-0010). |
| `09_multivariate_ohlcv`        | Close+Volume; atribuição de anomalia de volume (M8, ADR-0011). |
| `10_max_default_decision`      | `max` no teste real 2020–2024; adoção como default (M9, ADR-0009). |
| `11_ohlcv_full`                | OHLCV `(30,5)` vs Close+Volume; `latent_dim` multivariado (M9, ADR-0011). |

---

## Metodologia

1. **Pré-processamento** — retorno logarítmico diário como feature; *split temporal antes da
   normalização* (`MinMaxScaler` ajustado **apenas no treino**, evitando vazamento do futuro);
   janelas deslizantes de 30 passos. A via multivariada (Close+Volume) escala **por coluna** e
   aplica `log1p` ao volume.
2. **Modelo** — Encoder LSTM → bottleneck (`latent_dim=16`, validado por walk-forward) → Decoder
   LSTM, perda MSE, `EarlyStopping`. Um modelo por ativo.
3. **Erro por janela** — reduzido em duas etapas (features → tempo). A agregação temporal é
   **`max`** por default (desde M9): preserva choques de um único dia que a média diluía.
4. **Detecção** — uma janela é anômala quando o erro supera o limiar: **estático** (percentil 95
   do erro de treino) ou **dinâmico** (percentil em janela móvel **causal** de 252 pregões). O
   limiar é recalibrado sobre o mesmo escore de agregação.
5. **Validação de hiperparâmetro** — walk-forward (`TimeSeriesSplit`), com scaler reajustado
   **por fold** (anti-vazamento por fold).

### Avaliação

Problema não supervisionado → avaliação em duas vias:

- **Narrativa** — as anomalias detectadas correspondem a eventos reais conhecidos (crash de
  março/2020, caso Americanas em janeiro/2023, etc.).
- **Quantitativa** — injeção controlada de perturbações em posições conhecidas (*price shocks*; e
  *volume spikes* na via multivariada), permitindo Precision/Recall/F1.

> **Nota metodológica:** o período de treino (2010–2019) contém eventos de forte volatilidade
> (recessão de 2014–2016, Lava Jato, impeachment de 2016). "Normalidade" é, portanto, relativa —
> discutido no relatório.

---

## Resultados

Principais achados da Fase 2, todos medidos (notebooks 07–11) e registrados nos ADRs:

- **Agregação `max` (ADR-0009).** Na injeção sintética, trocar a média pelo `max` **dobrou o
  Recall** (0,16 → 0,35) e ainda **elevou a Precision** (0,55 → 0,84) — o choque de um dia, antes
  diluído, separa-se no pior passo. No teste real 2020–2024 a fração de janelas marcadas ficou
  ~igual à da média (≈0,10), sem inflar falsos positivos → **`max` adotado como default**.
- **`latent_dim` (ADR-0010).** Walk-forward confirmou `latent_dim=16`; o modelo é **insensível** ao
  tamanho do gargalo em [8, 32] (diferenças ≪ desvio entre folds) — evidência de robustez.
- **Multivariado Close+Volume (ADR-0011).** Um pico injetado só no canal de volume eleva o erro
  **daquele** canal (+0,05 a +0,36) e deixa o canal de preço em ≈0 — a anomalia de volume, antes
  invisível ao modelo univariado, passa a ser detectada e **atribuída**. O **OHLCV completo
  `(30,5)`** foi testado e **rejeitado** (piora o `val_loss` em 2/4 ativos sem ganho).

---

## Documentação

- **[`docs/adr/`](docs/adr/)** — Architecture Decision Records (0001–0011): cada decisão de
  metodologia/hiperparâmetro, com proveniência e evidência. `config.yaml` referencia o ADR de cada
  chave.
- **[`report/`](report/)** — relatório LaTeX (preliminar), uma seção por milestone.
- **[`teoria/`](teoria/)** — guia teórico autocontido (finanças, estatística, redes neurais).
- **[`debate/`](debate/)** — confrontações teóricas: defesa crítica das decisões e limitações
  assumidas.

---

## Configuração

Todos os hiperparâmetros vivem em `config.yaml` — **fonte única de verdade**; alterar um
experimento significa editar esse arquivo, não os notebooks. Chaves principais (cada uma referencia
seu ADR):

| Chave | Valor | ADR |
| ----- | ----- | --- |
| `preprocessing.window_size` | 30 | 0002 |
| `model.latent_dim` | 16 | 0003 / 0010 |
| `detection.threshold_percentile` | 95 | 0005 |
| `detection.dynamic_window` | 252 | 0005 |
| `detection.aggregation` | **max** | 0009 |
| `validation.n_splits` | 10 | 0010 |
| `preprocessing.features` | `[Close, Volume]` | 0011 |

---

## Stack

`Python` · `TensorFlow/Keras` · `yfinance` · `pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn` · `pytest`

---

## Referências

- Li, S. (2020). *Time Series of Price Anomaly Detection with LSTM.*
- Valkov, V. *Time Series Anomaly Detection with LSTM Autoencoders using Keras.* (curiousily.com)
- Petrovic, D. *Anomaly Detection in Stock Price with LSTM Autoencoder.* (GitHub)
- *Anomaly Detection on Bitcoin Values.* IEEE (2021).
- Liu et al. (2025). *Robust Anomaly Detection in Financial Markets Using LSTM Autoencoders and GANs.*
- Kohavi, R. (1995). *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection.* IJCAI. — base para `validation.n_splits = 10`.

Repositórios públicos foram usados apenas como guia arquitetural. Toda a implementação, os dados e
as análises são originais.

---

## Licença

Projeto acadêmico desenvolvido para a disciplina de Redes Neurais (2026.1). Uso educacional.
