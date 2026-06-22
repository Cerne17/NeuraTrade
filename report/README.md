# Relatório (LaTeX)

Relatório preliminar **vivo** do projeto NeuraTrade — atualizado a cada fechamento de
milestone, base para o artigo final.

## Estrutura

- `main.tex` — documento principal (classe `article`, `babel` pt-BR, `natbib`).
- `references.bib` — bibliografia.
- `sections/` — uma seção por arquivo; milestones entram como `NN_mX_*.tex` e são
  incluídos via `\input` em `main.tex`.

## Compilar

```bash
tectonic main.tex          # recomendado (binário único, resolve pacotes/bibtex)
# ou, com TeX Live:
latexmk -pdf main.tex
```

Gera `main.pdf`.

## Convenção por milestone

Ao fechar um milestone:
1. Criar `sections/NN_mX_nome.tex` documentando decisões, fontes, observações e conclusões.
2. Adicionar `\input{sections/NN_mX_nome.tex}` em `main.tex`.
3. Manter coerência com os ADRs em `../docs/adr/` (o relatório consolida; os ADRs detalham).
