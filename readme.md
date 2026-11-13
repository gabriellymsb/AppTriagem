---
language: pt
license: mit
base_model: neuralmind/bert-base-portuguese-cased
pipeline_tag: text-classification
tags:
  - text-classification
  - portuguese
  - customer-reviews
  - weak-supervision
  - llm-labeling
metrics:
  - f1
  - accuracy
---

# BERTimbau para triagem de avaliações de clientes

Classifica a avaliação de um cliente, em português, pelo problema principal do pedido. É um [BERTimbau base](https://huggingface.co/neuralmind/bert-base-portuguese-cased) ajustado (fine-tuning) com rótulos gerados por um LLM (Gemini) e avaliado contra rótulos humanos.

Projeto completo, com código, guia de rotulagem e análise de erros: [github.com/gabriellymsb/AppTriagem](https://github.com/gabriellymsb/AppTriagem)

## Classes

| Rótulo | Quando usar |
| --- | --- |
| `nao_recebido` | O produto ainda não chegou ou não foi entregue no endereço |
| `atraso` | Chegou depois do prazo, ou o cliente reclama de demora |
| `incompleto` | Chegou parte do pedido |
| `produto_errado` | Chegou um item diferente do pedido ou do anunciado (modelo, cor, marca, falsificado) |
| `defeito_qualidade` | Chegou o item certo, mas com defeito ou de qualidade ruim |
| `atendimento` | Troca, devolução, reembolso, cancelamento, cobrança ou falta de resposta |
| `elogio` | Avaliação positiva, sem problema relevante |
| `outros` | Sem informação suficiente ou reclamação genérica |

Quando uma avaliação tem mais de um problema, vale o principal, nesta ordem de prioridade: `nao_recebido` > `incompleto` > `produto_errado` > `defeito_qualidade` > `atraso` > `atendimento` > `elogio` > `outros`.

## Como usar

```python
from transformers import pipeline

triagem = pipeline("text-classification", model="gabriellymsb/bertimbau-triagem-avaliacoes")

triagem([
    "Comprei 2 colchas e só veio 1.",
    "Ainda não recebi o produto e ninguém responde.",
    "Chegou antes do prazo, recomendo!",
])
```

## Dados de treino

Avaliações públicas do [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-e-commerce) (e-commerce, usado como aproximação de um app de delivery).

| Conjunto | Avaliações | Rótulo |
| --- | --- | --- |
| Treino | 877 | gerado por LLM (Gemini), seguindo um guia de rotulagem |
| Validação | 300 | gerado por LLM |
| Teste | 301 | **humano**, sem ver o rótulo do LLM |

A amostra privilegia notas baixas (o dataset original é dominado por notas 5), e as classes são desbalanceadas.

## Treinamento

- Modelo base: `neuralmind/bert-base-portuguese-cased`
- 8 épocas, lotes de 16, até 64 tokens por texto
- AdamW, taxa de aprendizado 3e-5, aquecimento de 10% e queda linear, corte de gradientes em 1,0
- Entropia cruzada sem peso por classe (o peso por classe empatou na validação, em 3 sementes)
- Seleção da melhor época pelo F1 macro na validação (época 7); semente 42

## Resultados

No teste com rótulos humanos (301 avaliações):

| Métrica | Valor |
| --- | --- |
| F1 macro | 0,754 |
| Acurácia | 81,7% |
| Latência | ~3,4 ms por avaliação (GPU) |

Comparação nas 213 avaliações do teste que também foram rotuladas pelo LLM:

| Modelo | F1 macro | Acurácia |
| --- | --- | --- |
| TF-IDF + regressão logística | 0,644 | 70,9% |
| **Este modelo** | **0,774** | **83,6%** |
| LLM professor (Gemini 3.8 Flash) | 0,841 | 89,7% |

F1 por classe no teste: `incompleto` 0,92 · `elogio` 0,90 · `nao_recebido` 0,86 · `defeito_qualidade` 0,86 · `produto_errado` 0,81 · `atraso` 0,65 · `atendimento` 0,61 · `outros` 0,43.

## Limitações

- **Treino pequeno** (877 exemplos) e com ruído: os rótulos do LLM concordam com os humanos em cerca de 90% dos casos.
- **Herda vícios do professor:** tende a usar `outros` em excesso, como o LLM que gerou os rótulos.
- **Texto em maiúsculas** pode ser mal classificado (o modelo diferencia maiúsculas de minúsculas).
- **Contrastes** como "recomendo a loja, mas não o produto" podem ser lidos como elogio.
- **Erros de digitação e falta de acento** ("produto de ma qualidade") prejudicam a classificação.
- **Um rótulo por avaliação:** textos com dois problemas recebem só o principal.
- **Domínio:** treinado com avaliações de e-commerce brasileiro; o desempenho em outros domínios não foi medido.
- Classes pequenas no teste (13 a 20 exemplos) deixam o F1 de `outros`, `atraso` e `atendimento` instável.
