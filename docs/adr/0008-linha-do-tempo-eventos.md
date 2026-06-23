# ADR-0008: Linha do tempo de eventos e tolerância de matching narrativo

- **Status:** aceito
- **Data:** 2026-06-23
- **Proveniência:** DECISÃO DE PROJETO
- **Milestone:** M6 (Eventos & Comparação Setorial)
- **Chaves em `config.yaml`:** — (sem chave; lista curada em `src/events.py`, tolerância derivada de `preprocessing.window_size`)

## Contexto

A validação narrativa do projeto cruza as anomalias detectadas com eventos econômicos e
políticos brasileiros reais (crash COVID, fraude Americanas, Brumadinho, etc.). Para que esse
cruzamento seja reprodutível e auditável, é preciso (a) fixar uma lista curada de eventos como
fonte de verdade e (b) definir a regra que decide quando uma anomalia "corresponde" a um evento.
O ADR-0006 deixou ambos como pendência ("definir regra de matching e documentá-la").

## Decisão

1. **Curadoria (`src/events.py`):** lista ordenada cronologicamente cobrindo 2010–2024. Cada
   evento tem `date`, `label` e `tickers`:
   - `tickers=None` → evento **sistêmico** (afeta todos os ativos analisados).
   - `tickers=[...]` → evento **específico** dos ativos listados (ex.: Brumadinho → `VALE3.SA`;
     fraude Americanas → `AMER3.SA`; intervenção/troca de CEO da Petrobras → `PETR4.SA`).
   Critério de inclusão: eventos de impacto macroeconômico amplo (rebaixamentos de rating,
   eleições, impeachment, COVID) ou choques idiossincráticos documentados nos ativos do estudo.
2. **Tolerância de matching narrativo:** um evento é considerado "coberto" se há ao menos uma
   anomalia detectada dentro de **±`window_size` (30 dias)** da data do evento.

## Justificativa

- **Sistêmico vs. específico (DECISÃO DE PROJETO):** a distinção permite testar contágio
  (um evento sistêmico deve aparecer em todos os ativos) e poder discriminativo (um evento
  setorial deve isolar-se no ativo afetado) — os dois eixos da comparação setorial da M6.
- **Tolerância = `window_size`:** a unidade de detecção é a janela de 30 passos; uma anomalia
  é, por construção, imprecisa em até ~30 pregões. Amarrar a tolerância ao próprio
  `window_size` evita um número mágico independente e mantém coerência com o ADR-0002.

## Consequências

- A cobertura narrativa é sensível à tolerância: ±30 dias é permissivo; janelas menores
  reduziriam a cobertura agregada. Reportar a tolerância junto da métrica.
- A lista é curada manualmente: completude não é garantida; serve à validação qualitativa,
  não a uma métrica de recall sobre eventos reais (esse papel cabe à injeção sintética, ADR-0006).
- Datas de eventos são âncoras aproximadas (anúncio vs. reação do mercado podem diferir em
  dias) — outra razão para a tolerância temporal.

## Alternativas consideradas

- **Tolerância fixa em dias absolutos (ex.: ±7):** descartada — desacoplada da granularidade
  real da detecção (a janela).
- **Extração automática de eventos (notícias/NLP):** fora de escopo; introduz ruído e
  dependência de fonte externa não reprodutível offline.
