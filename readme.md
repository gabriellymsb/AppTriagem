# AppTriagem

Classificador que lê avaliações de clientes de um app de delivery e identifica o problema do pedido (atraso, produto errado, pedido não recebido, item faltando etc.). O modelo é pequeno e aprende com rótulos gerados por um LLM, o que evita rotulagem manual e mantém o custo de inferência baixo.

O projeto cobre o ciclo completo: treinar, ajustar, avaliar e publicar modelos no Hugging Face, com LLMs no meio do processo (rotulagem).

## Problema

Em um app de delivery, milhares de avaliações chegam todos os dias em texto livre ("a comida chegou fria e 1h depois do combinado", "veio outro lanche no meu pedido"). Ler e triar tudo manualmente não escala, e problemas operacionais acabam enterrados em comentários genéricos.

## Por que importa

Classificar por causa permite rotear casos para o time certo (suporte, logística, restaurante), priorizar reembolsos e identificar padrões, como um restaurante que erra pedidos com frequência ou uma região com atrasos recorrentes. Um modelo pequeno roda rápido e barato em volume alto, o que seria inviável chamando um LLM para cada avaliação.

## Estrutura do projeto

```
triagem-avaliacoes/
├── notebooks/   # exploração e análise de erros
├── src/         # rotulagem, treino, avaliação
├── api/         # FastAPI + Dockerfile
├── results/     # tabela de métricas e gráficos
└── README.md
```
