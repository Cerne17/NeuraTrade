# CONTEXTO — Brief para geração dos slides (NeuraTrade)

> **Para a IA que vai montar os slides:** você recebe **três arquivos**:
> `SLIDES.md` (o conteúdo slide a slide), `ROTEIRO.md` (a fala de cada slide) e
> este `CONTEXTO.md` (o pano de fundo do projeto e as regras de design). Use
> `SLIDES.md` como **estrutura canônica** — não invente slides nem reordene sem
> motivo. Use `ROTEIRO.md` só para calibrar ênfase; **não** cole a fala nos slides.
> Use este arquivo para acertar tom, termos, números e visual. **Não invente
> números nem resultados** que não estejam aqui ou em `SLIDES.md`.

---

## 1. O que é o projeto (resumo de 30 segundos)

NeuraTrade é um projeto acadêmico (Redes Neurais 2026.1) de **detecção não
supervisionada de anomalias** em ações da bolsa brasileira (B3) usando um
**LSTM-Autoencoder**. O modelo treina só em período "normal" (2010–2019), aprende
a **reconstruir** o padrão típico, e trata janelas com **erro de reconstrução**
alto como anomalias. As anomalias são cruzadas com uma linha do tempo de eventos
econômicos/políticos brasileiros. Quatro ativos, um modelo cada, para comparação
setorial: **PETR4** (energia), **VALE3** (mineração), **AMER3** (varejo), **ITUB4**
(financeiro).

**Público da apresentação:** banca acadêmica + colegas. **Duração:** 10 minutos.
**Idioma:** português do Brasil. **Tom:** direto, técnico, honesto — o diferencial
do projeto é o **rigor metodológico**, não sofisticação de arquitetura.

## 2. Fatos e números canônicos (fonte da verdade — use estes)

- Janela de entrada: **30 pregões** de log-retornos. Features: **Close + Volume**.
- Arquitetura: Encoder LSTM (64 unidades) → gargalo denso **`latent_dim=16`** →
  Decoder LSTM → reconstrução. Perda **MAE**. ~2450 janelas de treino por ativo.
- Split temporal: treino **2010–2019** (normalidade), teste **2020–2024**.
- Validação: **walk-forward** (`TimeSeriesSplit`, **k=10**).
- Limiar: percentil do erro de **treino** (padrão **p95**), estático ou dinâmico causal.
- Agregação do erro na janela: **`max`** (default). Ganho medido: **Recall 0,16 → 0,35**
  e **Precision 0,55 → 0,84** ao trocar `mean` por `max`.
- Conditional Autoencoder com macro (**USD/BRL, VIX**): separou **COVID/2020 =
  sistêmico** de **Americanas/2023 = idiossincrático**. Contagens: PETR4 na COVID =
  **30 janelas sistêmicas**; AMER3 = **49 janelas idiossincráticas**.
- Métrica de classe rara: **PR-AUC** no lugar de ROC-AUC. No regime raro (prevalência
  **4,9%**): **ROC-AUC = 0,84 mascara** o que **PR-AUC = 0,15 expõe**.
- Protocolo sintético corrigido: janelas sobrepostas inflavam a prevalência a ~70% →
  **avaliação por evento** (agrupa janelas contíguas).
- Tema central (repetir como fio condutor): **"mais capacidade ≠ mais sinal"**.
  Rejeitados **com evidência**: aumentar `latent_dim` (insensível em [8,32]), Atenção/
  Transformers, Optuna, **weight decay** (ADR-0018) — todos dentro do ruído inter-fold.
- Reprodutibilidade: roda **offline**, **32 testes** automatizados, **18 ADRs**.

> Se algum número faltar para um slide, **omita** — não preencha com estimativa.

## 3. Os 4 desafios (é o núcleo — dê peso visual a eles)

1. **Vazamento temporal** → split antes de normalizar; scaler só no treino (por fold).
2. **Choque de 1 dia diluído** pela média → agregação **`max`** (dobra o Recall).
3. **Idiossincrático vs sistêmico** → Conditional AE vê macro, perde só em preço/volume.
4. **Métrica que engana** (ROC em classe rara) → **PR-AUC** + custo assimétrico FP×FN;
   de brinde, correção do protocolo sintético (avaliação por evento).

Cada desafio segue o padrão **Problema → Solução → Evidência (número)**. Mantenha
esse tríptico visível em cada slide de desafio.

## 4. Recursos visuais disponíveis (caminhos reais no repositório)

Use estas figuras onde `SLIDES.md` pedir; **não** invente gráficos que não existem.

| Arquivo | O que mostra | Slide sugerido |
| --- | --- | --- |
| `report/figures/m4_erro_limiares.png` | erro de reconstrução vs limiares | 7 (agregação/limiar) |
| `report/figures/m5_distribuicao_erro.png` | distribuição do erro / detecção | 11 (resultados) |
| `report/figures/m6_covid_contagio.png` | contágio sistêmico na COVID | 8 (sistêmico) |
| `report/figures/m6_amer3_detalhe.png` | detalhe da anomalia da AMER3 | 8 (idiossincrático) |
| `figures/experiment_weight_decay.csv` | tabela do experimento de weight decay | 10 / apêndice A4 |

Diagramas a **gerar** (não existem como imagem; descreva/desenhe): arquitetura
Encoder→gargalo→Decoder (slide 4); barra temporal treino|teste + folds walk-forward
(slide 5); barras "delta vs desvio inter-fold" (slide 10). Podem ser esquemáticos.

## 5. Diretrizes de design

- **1 ideia por slide.** Máx. ~5 bullets. Bullets curtos (fragmentos, não frases longas).
- **Números em destaque** (cor/negrito): `max` dobra Recall, PR-AUC 0,15 vs ROC 0,84,
  COVID 30 / Americanas 49. São os pontos que a banca lembra.
- Paleta sóbria (tema financeiro/acadêmico): um azul/verde-petróleo + cinza + 1 cor de
  destaque para "anomalia" (vermelho/laranja). Fundo claro, alto contraste.
- Fonte sem serifa, legível a 6 m. Código/monoespaçado só quando citar nome de função.
- Rodapé discreto com "NeuraTrade · Redes Neurais 2026.1" + número do slide.
- Ícones simples nos 4 desafios (cadeado=vazamento, pico=choque, dois-mundos=idio/sist,
  gráfico-enganoso=métrica). Sem excesso de clipart.
- Slide 12 é a **demonstração ao vivo** (sandbox Streamlit) — deixe um slide-moldura
  simples (título + 1 print/placeholder), pois a tela real assume.

## 6. Formato de saída desejado

Gere os slides em **Marp** (Markdown → slides; `marp-cli` exporta PDF/PPTX/HTML).
Requisitos:
- Front-matter Marp (`marp: true`, `theme:`, `paginate: true`, `size: 16:9`).
- Um `---` separando cada slide, na **mesma ordem** de `SLIDES.md` (13 slides + apêndice).
- Título de cada slide como `##`; bullets como lista. Referencie as imagens pelos
  caminhos da tabela acima (`![](../report/figures/....png)`), assumindo que o `.md` de
  saída fica em `presentation/`.
- Speaker notes: coloque a fala correspondente de `ROTEIRO.md` como comentário Marp
  (`<!-- ... -->`) no fim de cada slide — assim o roteiro viaja junto sem poluir o slide.
- Não exceda o conteúdo de `SLIDES.md`; se um slide ficar cheio, **quebre em dois**
  em vez de encolher a fonte.

Se preferir outro formato (Reveal.js, PPTX nativo, Google Slides), mantenha as mesmas
regras: ordem de `SLIDES.md`, números canônicos desta seção 2, speaker notes do roteiro.

## 7. O que NÃO fazer

- Não inventar resultados, benchmarks, acurácias ou datas fora deste arquivo/`SLIDES.md`.
- Não transformar o projeto em "robô de trading" — é **detecção**, não estratégia de
  compra/venda (o backtest financeiro é explicitamente trabalho **futuro**).
- Não sugerir que atenção/Transformers/Optuna/weight decay "melhorariam" — foram
  testados e **rejeitados com evidência**. O argumento é parcimônia, não moda.
- Não colar parágrafos longos do roteiro dentro do slide. Roteiro = speaker notes.
