# ADR-0002: Tamanho da janela deslizante = 30

- **Status:** aceito
- **Data:** 2026-06-22
- **Proveniência:** FUNDAMENTADO
- **Milestone:** M2 (Pré-processamento)
- **Chaves em `config.yaml`:** `preprocessing.window_size`, `preprocessing.step`

## Contexto

O autoencoder reconstrói sequências de comprimento fixo. A janela define quanto contexto
temporal o modelo vê por amostra: curta demais não captura padrão; longa demais aumenta
overfitting e reduz o nº de amostras.

## Decisão

`window_size = 30` dias úteis (≈ 6 semanas de pregão), `step = 1` (janelas deslizantes
com sobreposição máxima).

## Justificativa

- **Li (2020):** `TIME_STEPS = 30` para retorno de preço diário.
- **Valkov (curiousily):** `TIME_STEPS = 30` na implementação Keras de referência.

Convergência das duas fontes-âncora do projeto sobre o mesmo valor. `step = 1` maximiza
o número de amostras de treino, importante dado o histórico relativamente curto por ativo.

## Consequências

- Janelas vizinhas são altamente correlacionadas (sobreposição de 29/30). Isso interage
  com `validation_split` — ver [ADR-0004](0004-configuracao-de-treino.md): com `shuffle=False`
  a validação é o bloco final, mas as janelas na fronteira treino/validação compartilham dias.
- Uma anomalia pontual afeta até 30 janelas consecutivas; a lógica de detecção/avaliação
  precisa considerar isso ao contar acertos (ver [ADR-0006](0006-avaliacao-injecao-e-setorial.md)).

## Alternativas consideradas

- Janelas de 60/90 dias: descartadas por ora; reduzem amostras e fogem da âncora da literatura.
  Podem entrar como estudo de sensibilidade em M3 se houver tempo.
