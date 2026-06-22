# ADR-0007: Coleta de dados, ajuste de preços e tratamento do caso AMER3

- **Status:** aceito
- **Data:** 2026-06-22
- **Proveniência:** FUNDAMENTADO (README) + DECISÃO DE PROJETO
- **Milestone:** M1 (Coleta & EDA)
- **Chaves em `config.yaml`:** `data.raw_dir`, `data.start`, `data.end`
- **Issues:** #6 (download+cache), #7 (AMER3)

## Contexto

O pipeline precisa de dados OHLCV diários dos 4 ativos (2010–2024), de forma
reprodutível e offline. Além disso, a série da Americanas (`AMER3.SA`) tem
particularidades societárias e eventos extremos que podem ser sinal (anomalia
real) ou ruído (artefato de evento corporativo).

## Decisão

### Coleta e cache (#6)
1. **Fonte única de rede:** `data.cache_ticker` / `cache_all` baixam via yfinance e
   persistem `data/raw/<TICKER>.csv` (CSV, versionado — ver `.gitignore`).
2. **Pipeline offline:** `data.load_ticker` / `load_all` leem **somente** o cache e
   nunca acessam a API. Os notebooks dependem só de `load_*`.
3. **`auto_adjust=True`:** o `Close` já vem ajustado por splits e dividendos.

### Tratamento AMER3 (#7)
4. **Não limpar anomalias reais.** A queda de ~78% em **12/01/2023** (log-ret ≈ -1.48),
   após a divulgação da fraude contábil, é o evento-alvo do projeto — permanece na série.
5. **Sinalizar artefato de grupamento.** O salto de log-ret ≈ +1.03 em **14/11/2024**
   é compatível com um grupamento (reverse split) não totalmente ajustado pelo yfinance.
   Documentado e marcado na EDA; decisão de correção fica para M2 (pré-processamento).
6. **Dias de volume zero** (halts/iliquidez) são contados em `integrity_report` e
   reportados, não removidos nesta fase.

## Justificativa

- yfinance back-fillou a série de AMER3 desde 2010 via continuidade de ticker
  (herdada da Lojas Americanas), então **há histórico de treino 2010–2019** — a
  preocupação inicial de "delisting / sem histórico" não se confirmou na EDA.
- A integridade verificada (3724 pregões, 0 NaN em todos) torna desnecessário
  imputar valores; o foco do tratamento é distinguir **sinal** (fraude) de
  **artefato** (grupamento).
- CSV em `data/raw/` cumpre o requisito do README de pipeline reprodutível offline
  (~1.3 MB no total, aceitável para versionar).

## Consequências

- O artefato de 14/11/2024 em AMER3, se não corrigido, vira uma anomalia espúria nas
  métricas de M5. **Decisão (M2): corrigir o fator de grupamento** (reverse split), **não
  recortar** o período — preserva-se a continuidade da série e o caso AMER3 inteiro.
  Implementação no pré-processamento (M2).
- `auto_adjust=True` muda valores históricos quando há novos proventos/splits → re-baixar
  o cache pode alterar resultados. O cache versionado **congela** os dados do experimento;
  re-rodar `cache_all` é uma decisão consciente, não automática.
- A taxa de "feriados/halts" (business days − pregões) não distingue feriado de halt; é
  panorama, não validação rigorosa de calendário.

## Alternativas consideradas

- **`auto_adjust=False` + usar `Adj Close`:** equivalente; `auto_adjust=True` é mais direto.
- **Remover dias de volume zero:** rejeitado — quebra a continuidade temporal das janelas.
- **Excluir AMER3:** rejeitado — varejo é um dos setores-alvo e o caso Americanas é a
  anomalia narrativa mais forte do trabalho.
