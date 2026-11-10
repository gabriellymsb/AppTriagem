# Guia de rotulagem — avaliações de pedidos

Versão 1.1

## Objetivo

Classificar cada avaliação de cliente em **exatamente um** rótulo, que descreve o **problema principal** do pedido. Se não houver problema, o rótulo descreve o tom da avaliação (elogio ou outros).

## Regras gerais

1. **Classifique pelo texto, nunca pela nota.** O modelo final só vai ver o texto. Se a nota é 5 mas o texto reclama de atraso, o rótulo é `atraso`.
2. **Um único rótulo por avaliação.** Quando houver mais de um problema, use a ordem de prioridade abaixo.
3. **Problema vence elogio.** "Chegou rápido, mas veio quebrado" é `defeito_qualidade`.
4. **Na dúvida entre um rótulo específico e `outros`, prefira o específico** se o texto der evidência clara dele.
5. Ignore nomes de lojas: eles foram anonimizados (lannister, baratheon, targaryen etc.).

## Rótulos

### `nao_recebido`
O produto **ainda não chegou** no momento em que a avaliação foi escrita, ou o cliente diz que nunca recebeu.
- Inclui: "ainda não recebi", "estou esperando até hoje", produto extraviado ou roubado no transporte, rastreio parado.
- Não inclui: chegou parte do pedido (→ `incompleto`); chegou, mas tarde (→ `atraso`); pedido cancelado com pedido de reembolso (→ `atendimento`).
- Exemplos: "Eu ainda não recebi o produto." · "Estou esperando até hoje." · "Meu produto foi roubado, ninguém resolveu."

### `atraso`
O produto **chegou**, mas depois do prazo prometido, ou o cliente reclama de demora/falta de pontualidade sem dizer que não recebeu.
- Inclui: "chegou um mês depois", "demorou muito", "sem pontualidade".
- Não inclui: ainda não chegou (→ `nao_recebido`).
- Exemplos: "O prazo era de 14 dias e chegou quase um mês depois." · "Sem pontualidade."

### `incompleto`
Chegou **parte** do pedido: faltou um item, uma peça ou uma quantidade.
- Inclui: "só veio 1 de 2", "faltou o cartucho colorido", "o resto ainda não chegou" (quando parte já chegou).
- Exemplos: "Só foi entregue o cartucho da tinta preta. Falta o colorido." · "Comprei 2 colchas e só veio 1."

### `produto_errado`
Chegou um item **diferente** do pedido ou do anunciado em característica **objetiva**: outro produto, modelo, tamanho, cor, voltagem, capacidade.
- Inclui: "me entregaram outra coisa", "veio copo de 400ml em vez de 470ml", "cor diferente da escolhida", marca diferente da anunciada, produto "compatível", "similar" ou falsificado no lugar do original.
- Não inclui: mesmo produto, mas de qualidade ruim ou "diferente da foto" em aspecto subjetivo (→ `defeito_qualidade`).
- Exemplos: "Comprei uma coisa e me entregaram outra." · "Veio o copo errado: anunciado 470ml, veio 400ml."

### `defeito_qualidade`
Chegou o item certo, mas **com defeito, quebrado, danificado ou de qualidade ruim**.
- Inclui: não funciona, quebrado, avariado, material frágil, "não é como na foto" em qualidade/acabamento.
- Não inclui: falsificado ou de outra marca que não a anunciada (→ `produto_errado`).
- Exemplos: "Produto defeituoso, parece que não foi testado." · "Material muito fraquinho, pano muito fino."

### `atendimento`
O problema principal é **com a loja ou o pós-venda**: troca, devolução, reembolso, cancelamento, cobrança indevida, falta de resposta ou de contato.
- Inclui: "não consigo falar com a central", "devolução sem resposta", "pedido cancelado e não fui ressarcida", "cobraram em dobro".
- Não inclui: reclamação de atendimento junto com problema do produto (o problema do produto tem prioridade).
- Exemplos: "Informaram 48 horas para confirmar a devolução e até agora nada." · "O pedido foi cancelado e não fui ressarcida."

### `elogio`
Avaliação positiva **sem problema relevante**: satisfação com o produto, a entrega ou a loja.
- Inclui: "recomendo", "chegou antes do prazo", "ótimo produto"; elogio com sugestão leve que não descreve problema ("gostei, a embalagem podia ser melhor").
- Exemplos: "Parabéns, adorei comprar pela loja." · "Super recomendo, entrega antes do prazo."

### `outros`
Texto sem informação suficiente para os rótulos acima, ou reclamação vaga/genérica.
- Inclui: "obrigado", "ok", "empresa deixando a desejar", reclamação de logística que não é atraso, falta nem não entrega ("deveriam entregar tudo de uma vez"), perguntas sem problema descrito.
- Exemplos: "Obrigado." · "Empresa está deixando a desejar."

## Ordem de prioridade (quando houver mais de um problema)

1. `nao_recebido`
2. `incompleto`
3. `produto_errado`
4. `defeito_qualidade`
5. `atraso`
6. `atendimento`
7. `elogio`
8. `outros`

Exceção: **pedido cancelado** com reclamação de reembolso é `atendimento`, mesmo que o produto não tenha chegado. O produto não vai chegar; o problema é o dinheiro.

## Casos ambíguos já decididos

| Texto (resumo) | Rótulo | Por quê |
| --- | --- | --- |
| Comprei uma coisa, veio outra, e não consigo falar com a central | `produto_errado` | Problema do produto vence atendimento |
| Qualidade frágil e chegou quase um mês depois | `defeito_qualidade` | Qualidade vem antes de atraso na prioridade |
| Só faturaram o mosquiteiro; a garrafa ainda não chegou | `incompleto` | Parte do pedido chegou |
| Não recebi e estou pedindo reembolso (sem cancelamento) | `nao_recebido` | O problema central ainda é a não entrega |
| Pedido cancelado e não fui ressarcida | `atendimento` | Exceção do cancelamento |
| Produto muito diferente do que mostra, material fraquinho | `defeito_qualidade` | A diferença descrita é de qualidade, não de especificação |
| Faltou um cartucho e cobraram frete em dobro | `incompleto` | Falta de item vence cobrança indevida |
| As mercadorias deveriam ser entregues de uma vez só | `outros` | Reclamação de logística sem atraso, falta ou não entrega |
| Nota 5, texto: "obrigado" | `outros` | Sem avaliação explícita; a nota não conta |
| Onde consigo informações? Ainda não recebi | `nao_recebido` | A pergunta é consequência da não entrega |
| Anúncio dizia Canon, veio um "compatível"/falsificado | `produto_errado` | Marca e originalidade são especificação objetiva do anúncio |

<!-- FIM DO PROMPT: o que vem abaixo é só para revisão humana -->

## Revisão humana (conjunto de teste)

1. **Rotule sem ver o rótulo do LLM.** A coluna do LLM fica escondida até você terminar, para não se deixar influenciar.
2. Leia o texto inteiro, aplique as regras gerais e a ordem de prioridade.
3. Se ficar em dúvida, escreva o motivo na coluna `obs`. Não pule a avaliação.
4. Se aparecer um caso que o guia não cobre, decida, adicione à tabela de casos ambíguos e suba a versão do guia (1.1, 1.2...).
5. As avaliações revisadas à mão **nunca** entram no treino: elas são o conjunto de teste.

## Histórico

- 1.0 — versão inicial, a partir de 30 amostras de notas 1 e 2.
- 1.1 — produto falsificado ou de outra marca passa a ser `produto_errado` (contradição encontrada no teste com 40 avaliações).
