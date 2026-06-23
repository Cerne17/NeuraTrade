# ADR-0005: Threshold estático (p95) e dinâmico (janela causal)

- **Status:** aceito (`dynamic_window=252` confirmado em M7)
- **Data:** 2026-06-22
- **Proveniência:** misto
- **Milestone:** M4 (Detecção & Thresholds)
- **Chaves em `config.yaml`:** `detection.threshold_percentile`, `detection.dynamic_window`

## Contexto

Uma janela é anômala quando o erro de reconstrução ultrapassa um limiar. O projeto compara
duas estratégias de limiar — é uma das contribuições declaradas no README.

## Decisão

1. **Estático:** percentil **95** do erro de reconstrução **calculado sobre o treino**.
   Limiar fixo aplicado a todo o teste.
2. **Dinâmico:** percentil 95 sobre uma **janela móvel causal** de `dynamic_window = 252`
   observações (≈ 1 ano útil de pregão), usando **somente o passado** de cada ponto.

## Justificativa

- **p95 (FUNDAMENTADO):** o README fixa "percentil 95 do erro de treino". É mais robusto que
  o **máximo do treino** usado por Li (2020) / Valkov, que é refém de um único outlier do treino.
- **Threshold sobre o treino, não o teste:** usar a distribuição do teste para definir o limiar
  vazaria informação do período avaliado.
- **Causalidade do dinâmico (crítico):** a janela móvel deve incluir apenas observações
  **anteriores** ao ponto avaliado. Usar uma janela centrada ou que inclua o futuro vaza
  informação e invalida a comparação. `252` ≈ pregões/ano, capturando regime recente sem
  fixar um limiar global.

## Consequências

- `dynamic_window = 252` é **PROVISÓRIO**: não há fonte para o número; calibrar em M4
  (testar p.ex. 126 / 252 / 504) e reportar sensibilidade.
- p95 implica ~5% de janelas marcadas como anômalas no treino por construção — é a taxa-base
  esperada de falsos positivos; considerar na leitura das métricas ([ADR-0006](0006-avaliacao-injecao-e-setorial.md)).
- Janelas sobrepostas ([ADR-0002](0002-window-size-30.md)) fazem uma anomalia gerar rajada de
  detecções; agrupar detecções contíguas em "eventos" antes de comparar com eventos reais.

> **Atualização (M4, 2026-06-22) — implementação e evidência.**
> - **Causalidade:** o limiar dinâmico usa `rolling(window).quantile(p)` com `shift(1)`,
>   isto é, janela de tamanho `dynamic_window` que **exclui o próprio ponto** (só passado).
>   Os primeiros `window` pontos ficam `NaN` (não marcados).
> - **Erro de detecção:** MAE por janela (`metric="mae"`, padrão), mais robusto a outlier
>   que o MSE; MSE disponível via parâmetro (issue #18).
> - **Evidência empírica (teste 2020–2024):** sob mudança de regime, o estático **satura** —
>   AMER3 marca ~27% das janelas (todo o pós-fraude), ITUB4 ~11%. O dinâmico recalibra ao
>   regime local e reduz para ~14% (AMER3) e ~5% (ITUB4), sem usar o futuro. Em ativos
>   estáveis (PETR4, VALE3) os dois quase coincidem (~3%). Confirma o valor do esquema dinâmico.
> - `dynamic_window=252` segue **provisório**; estudo de sensibilidade (126/252/504) fica para M5.

> **Atualização (M7, 2026-06-23) — estudo de sensibilidade `dynamic_window`.**
> Fração de janelas de teste marcadas (limiar dinâmico) por tamanho de janela:
>
> | Ticker | 126 | 252 | 504 |
> | --- | --- | --- | --- |
> | PETR4.SA | 8,5% | 2,9% | 0,0% |
> | VALE3.SA | 8,2% | 3,4% | 0,3% |
> | AMER3.SA | 15,2% | 14,3% | 5,7% |
> | ITUB4.SA | 6,3% | 4,9% | 0,0% |
>
> Comportamento monotônico: `126` é reativo (≈8% — segue o ruído local, infla falsos
> positivos), `504` super-suaviza (≈0% nos ativos estáveis — aproxima-se de um limiar
> global e perde a capacidade de marcar qualquer coisa). **`252` é confirmado**: produz
> taxas próximas da base-rate alvo (~5% do p95) sem colapsar nem oscilar. Status de
> PROVISÓRIO removido.

> **Atualização (M9, 2026-06-23) — agregação default = `max`.**
> A partir de M9 (issue #58, [ADR-0009](0009-agregacao-erro-janela.md)), o erro por janela usa
> agregação **`max`** (pior passo) em vez da média, por default. O esquema de limiares (estático
> p95 do treino + dinâmico causal 252) é **inalterado**: muda apenas o escore sobre o qual o
> limiar é calculado --- e, por isso, o limiar é recalibrado sobre o erro `max` do treino. No
> teste real 2020–2024 a fração marcada permaneceu próxima da de `mean` (≈0,10 estático), de modo
> que a base-rate esperada do p95 segue válida.

## Alternativas consideradas

- **Máximo do erro de treino (Li/Valkov):** rejeitado — frágil a outlier único.
- **Threshold por desvio-padrão (μ + kσ):** assume normalidade do erro, improvável; percentil é não-paramétrico.
