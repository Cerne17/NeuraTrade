# ADR-0005: Threshold estático (p95) e dinâmico (janela causal)

- **Status:** aceito (janela dinâmica PROVISÓRIA)
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

## Alternativas consideradas

- **Máximo do erro de treino (Li/Valkov):** rejeitado — frágil a outlier único.
- **Threshold por desvio-padrão (μ + kσ):** assume normalidade do erro, improvável; percentil é não-paramétrico.
