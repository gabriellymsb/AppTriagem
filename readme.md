# AppTriagem: triagem automática de avaliações de clientes

Classificador que lê a avaliação de um cliente e diz qual é o problema do pedido: não recebido, atraso, item faltando, produto errado, defeito, atendimento. Um LLM (Gemini) rotula os dados de treino, e um modelo de linguagem pequeno (BERTimbau) aprende com esses rótulos. O resultado é um classificador rápido e barato o suficiente para rodar em todas as avaliações, algo inviável chamando um LLM para cada uma.

**Resultado principal:** treinado com apenas 877 exemplos rotulados por LLM, o BERTimbau chega a **92% do F1 macro do próprio LLM** (0,774 contra 0,841, nas mesmas avaliações), respondendo em **cerca de 3 ms por avaliação**.

| | |
| --- | --- |
| **Demo** (roda no navegador) | [huggingface.co/spaces/gabriellymsb/triagem-avaliacoes](https://huggingface.co/spaces/gabriellymsb/triagem-avaliacoes) |
| **Modelo** | [huggingface.co/gabriellymsb/bertimbau-triagem-avaliacoes](https://huggingface.co/gabriellymsb/bertimbau-triagem-avaliacoes) |

## Sumário

- [O problema](#o-problema)
- [Objetivo](#objetivo)
- [O que o projeto testa](#o-que-o-projeto-testa)
- [Dados](#dados)
- [Como funciona](#como-funciona)
- [Divisão dos dados](#divisão-dos-dados)
- [Por que um teste com rótulo humano](#por-que-um-teste-com-rótulo-humano)
- [Técnicas](#técnicas)
- [Resultados](#resultados)
- [Análise de erros](#análise-de-erros)
- [Publicação e demo](#publicação-e-demo)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Os scripts](#os-scripts)
- [Passo a passo](#passo-a-passo)
- [Decisões e aprendizados](#decisões-e-aprendizados)
- [Limitações](#limitações)
- [Como rodar](#como-rodar)
- [Próximos passos](#próximos-passos)

## O problema

Em um app de delivery ou marketplace, milhares de avaliações chegam todos os dias em texto livre. Ler e triar tudo à mão não escala, e problemas operacionais ficam enterrados em comentários genéricos. Exemplos reais do dataset:

| Avaliação | Classe |
| --- | --- |
| "Eu ainda não recebi o produto." | `nao_recebido` |
| "O prazo era de 14 dias e chegou quase um mês depois." | `atraso` |
| "Só foi entregue o cartucho da tinta preta. Falta o outro cartucho da tinta colorida." | `incompleto` |
| "Veio o copo errado. Vendeu como copo de 470ml e veio copos de 400ml." | `produto_errado` |
| "Parece que o produto não foi testado antes de sair da loja, produto defeituoso." | `defeito_qualidade` |
| "Informaram que seria 48 horas pra confirmar a devolução e até o momento nada." | `atendimento` |
| "Parabéns lojas lannister adorei comprar pela Internet." | `elogio` |
| "Empresa está deixando a desejar." | `outros` |

Classificar por causa permite mandar cada caso para o time certo (logística, suporte, loja parceira), priorizar reembolsos e enxergar padrões, como um parceiro que erra pedidos com frequência.

## Objetivo

Responder com números a uma pergunta de engenharia:

> **Um modelo pequeno, treinado com rótulos gerados por um LLM, chega perto da qualidade do próprio LLM, sendo muito mais rápido e barato?**

Entregas:

1. Comparação entre LLM, TF-IDF e BERTimbau em qualidade e latência, medida contra rótulos humanos.
2. Análise de erros: o que cada modelo erra e por quê.
3. Modelo publicado no Hugging Face e demo em Gradio.

## O que o projeto testa

| Pergunta | Como é medida |
| --- | --- |
| Os rótulos do LLM são confiáveis? | Concordância e F1 do LLM contra rótulos humanos |
| Um modelo simples já resolve? | Baseline TF-IDF + regressão logística |
| Entender contexto faz diferença? | BERTimbau ajustado × baseline, classe por classe |
| Vale o custo? | Latência por avaliação e escala possível |
| Onde os modelos erram, e por quê? | Tipos de erro, matriz de confusão e leitura dos casos |

## Dados

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-e-commerce), arquivo `olist_order_reviews_dataset.csv`.

- 99.224 avaliações, das quais **40.977 (41%) têm comentário em texto**.
- **Notas muito desbalanceadas:** entre as que têm texto, 20.554 são nota 5 e só 2.145 são nota 2.
- **Textos curtos:** mediana de 9 palavras, máximo de 45.
- Os nomes das lojas foram anonimizados pela Olist com casas de Game of Thrones (lannister, stark, baratheon).

São dados de e-commerce, usados como aproximação de um app de delivery: os problemas (atraso, não recebido, item faltando, produto errado) são os mesmos. Os dados não são versionados no repositório; baixe pelo link acima.

## Como funciona

```
 dataset Olist (40.977 com texto)
        │  limpeza + sorteio estratificado por nota
        ▼
 amostra de 2.000 avaliações
        │
        ├──► 301 rotuladas à mão ─────────────────────────► TESTE
        │      (213 delas também rotuladas pelo LLM, para comparação)
        │
        └──► rotuladas pelo LLM (Gemini), com docs/guia_rotulagem.md como prompt
                 ├──► 300 ────────────────────────────────► VALIDAÇÃO
                 └──► 877 ────────────────────────────────► TREINO

 TREINO ──► TF-IDF + regressão logística ┐
        └─► fine-tuning do BERTimbau      ├─► escolha pela VALIDAÇÃO ─► nota final no TESTE
                                          ┘
```

O LLM **não é treinado** neste projeto. Ele faz o papel de "professor": rotula os exemplos de treino seguindo o guia. Quem é treinado é o "aluno", um modelo menor que aprende a imitar esses rótulos e depois roda sozinho, sem chamar o LLM. É uma forma de **supervisão fraca** (os rótulos de treino têm ruído), próxima da **destilação de conhecimento**.

## Divisão dos dados

| Conjunto | Planejado | Efetivo | Rótulo | Papel |
| --- | --- | --- | --- | --- |
| **Treino** | ~1.400 (70%) | 877 (59%) | do LLM | O modelo aprende os padrões e ajusta seus pesos |
| **Validação** | 300 (15%) | 300 (20%) | do LLM | O "simulado": comparar configurações, escolher a melhor época, detectar overfitting |
| **Teste** | 300 (15%) | 301 (20%) | **humano** | A "prova final", usada uma única vez por modelo |

A divisão planejada era 70/15/15. O treino ficou menor porque a rotulagem com LLM parou em 1.390 avaliações, pelos limites do plano gratuito do Gemini (veja [Limitações](#limitações)). Validação e teste têm o tamanho planejado; só o treino diminuiu.

Regras seguidas:

- **Toda escolha é feita na validação.** O teste nunca é usado para decidir nada. Se fosse, ele viraria um segundo simulado e deixaria de medir o desempenho real.
- A divisão entre treino e validação é **estratificada**: cada classe aparece na mesma proporção nos dois conjuntos.
- Sementes aleatórias fixas: a divisão é a mesma em qualquer computador.

## Por que um teste com rótulo humano

Se o teste também fosse rotulado pelo LLM, o projeto mediria **o quanto o modelo imita o LLM**, e não o quanto ele acerta. Por isso, as 301 avaliações do teste foram rotuladas à mão, **sem ver o rótulo do LLM**, seguindo o mesmo guia.

Isso tem dois usos:

1. **Nota real dos modelos:** o F1 no teste é medido contra o julgamento humano.
2. **Qualidade do próprio LLM:** comparando os rótulos dele com os humanos nas mesmas avaliações, dá para medir quanto ruído entra no treino.

| Momento | Concordância LLM × humano |
| --- | --- |
| Primeira medição | 79,8% |
| Após a primeira revisão dos rótulos humanos | 86,4% |
| **Final** | **89,7%** (F1 macro 0,841, em 213 avaliações) |

As revisões corrigiram inconsistências **nos rótulos humanos**, aplicando o guia a **classes inteiras** (e não só às avaliações em que o LLM discordava), para não viciar o teste a favor do LLM. Detalhes em [Decisões e aprendizados](#decisões-e-aprendizados).

## Técnicas

### Baseline: TF-IDF + regressão logística

**TF-IDF** transforma cada avaliação em um vetor de números, um por termo do vocabulário. O peso de cada termo cresce quanto mais ele aparece *naquele* texto (TF) e quanto mais raro ele é *no conjunto todo* (IDF). Assim, "faltou" pesa muito e "produto" pesa pouco.

- Palavras soltas e pares de palavras (`ngram_range=(1, 2)`): "não recebi" diz mais que "não" e "recebi" separados.
- Opcionalmente, pedaços de 2 a 5 letras (`char_wb`): torna o modelo tolerante a erros de digitação.
- Texto em minúsculas e sem acentos, `sublinear_tf` e `min_df=2`.

A **regressão logística** aprende um peso por termo para cada classe; a classe com a maior soma de pesos vence. Hiperparâmetros:

- `C`: força da regularização (o "freio" contra decorar o treino).
- `class_weight="balanced"`: erros em classes raras custam mais.

A busca testa 20 combinações (5 valores de C × com/sem peso × com/sem n-gramas de letras) e escolhe pela validação.

### BERTimbau ajustado (fine-tuning)

[BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased) é um modelo BERT pré-treinado em bilhões de palavras em português. O fine-tuning ensina a ele as 8 classes deste problema.

Diferente do TF-IDF, ele lê a frase inteira com **atenção**, considerando cada palavra no contexto das outras. É isso que separa "veio outra marca" (`produto_errado`) de "veio quebrado" (`defeito_qualidade`).

Como o treino funciona (`src/finetune.py`):

1. **Tokenização:** cada texto vira uma sequência de até 64 *tokens* (pedaços de palavra).
2. **Modelo:** o BERTimbau pré-treinado recebe uma **cabeça de classificação nova**, com 8 saídas e pesos aleatórios. As cabeças do pré-treino são descartadas; por isso o log mostra `UNEXPECTED` (descartadas) e `MISSING` (a nova).
3. **Função de perda:** entropia cruzada. Com `--peso-classes`, erros em classes raras pesam mais.
4. **Otimização:** AdamW com taxa de aprendizado de 3e-5, aquecimento nos primeiros 10% dos passos e queda linear depois, e corte de gradientes em 1,0.
5. **Épocas:** o modelo percorre o treino inteiro várias vezes, em lotes de 16, e mede o F1 macro na **validação** ao fim de cada época.
6. **Melhor época:** só a época com maior F1 de validação é salva, descartando épocas em que o modelo começa a decorar o treino.
7. **Prova final:** a melhor época é avaliada uma única vez no **teste**.

### Métrica: F1 macro

Para cada classe, o F1 equilibra **precisão** (dos que o modelo chamou de X, quantos eram X?) e **revocação** (dos X reais, quantos ele achou?). O **macro** tira a média simples entre as 8 classes, então `outros` (13 exemplos no teste) pesa tanto quanto `elogio` (64). A acurácia esconderia o desempenho nas classes raras.

## Resultados

### Comparação geral

Os modelos foram avaliados nas 301 avaliações do teste; o LLM, só nas 213 que ele rotulou. Para comparar os três de forma justa, a coluna do meio usa **as mesmas 213** para todos.

| Modelo | F1 macro (301) | F1 macro (as mesmas 213) | Acurácia (213) | Latência por avaliação |
| --- | --- | --- | --- | --- |
| TF-IDF + regressão logística | 0,628 | 0,644 | 70,9% | 0,36 ms (CPU) |
| **BERTimbau, 8 épocas** | **0,754** | **0,774** | **83,6%** | **3,35 ms** (GPU) |
| LLM (Gemini 3.8 Flash) | — | 0,841 | 89,7% | dezenas de segundos por chamada de 125 avaliações (não medido com precisão) |

Leituras:

- **O BERTimbau chega a 92% do F1 do professor** (0,774 / 0,841) aprendendo com só 877 exemplos rotulados por ele.
- **O contexto vale muito:** do baseline para o BERTimbau, o F1 sobe 0,13 e a acurácia, 13 pontos.
- **Escala:** no plano gratuito do Gemini (20 chamadas por dia, até 125 avaliações por chamada), rotular 1 milhão de avaliações levaria **mais de um ano**. O BERTimbau classifica 1 milhão em menos de 1 hora em uma GPU, sem custo por chamada.
- **Ressalva:** os rótulos humanos foram revisados depois de comparados com os do LLM. Mesmo revisando classes inteiras, isso pode ter aproximado os rótulos humanos dos do LLM; o F1 do LLM pode estar um pouco otimista.

### Por classe

| Classe | Exemplos (teste) | Baseline | BERTimbau | LLM¹ |
| --- | --- | --- | --- | --- |
| `nao_recebido` | 58 | 0,77 | 0,86 | 0,92 |
| `atraso` | 18 | 0,70 | 0,65 | 0,87 |
| `incompleto` | 38 | 0,86 | 0,92 | 0,96 |
| `produto_errado` | 26 | 0,60 | 0,81 | 0,90 |
| `defeito_qualidade` | 64 | 0,67 | 0,86 | 0,90 |
| `atendimento` | 20 | 0,43 | 0,61 | 0,70 |
| `elogio` | 64 | 0,75 | 0,90 | 0,97 |
| `outros` | 13 | 0,24 | 0,43 | 0,50 |

¹ LLM medido nas 213 avaliações que ele rotulou. Com 13 a 20 exemplos nas classes pequenas, um único acerto muda o F1 delas em cerca de 5 pontos.

- **Onde o contexto mais ajudou:** `produto_errado` (+0,21) e `defeito_qualidade` (+0,19), a fronteira mais difícil entre as classes.
- **As classes mais difíceis para todos:** `outros` e `atendimento`, as mais vagas e as menos frequentes.

### Baseline: busca de hiperparâmetros

Melhor configuração: `C=0.3`, `class_weight="balanced"`, com n-gramas de letras (F1 de validação 0,703).

- `class_weight="balanced"` foi o que mais ajudou: de 0,44 para 0,66 de F1 de validação, com o mesmo C.
- Os n-gramas de letras somaram cerca de 0,04 em todas as configurações.
- O F1 de validação (0,70) é maior que o de teste (0,63) porque a validação usa rótulos do LLM, e o teste, humanos: a diferença mede, em parte, o ruído dos rótulos.

### BERTimbau: épocas

| Época | Perda de treino | F1 validação |
| --- | --- | --- |
| 1 | 1,93 | 0,344 |
| 2 | 1,06 | 0,540 |
| 3 | 0,51 | 0,687 |
| 4 | 0,24 | 0,734 |
| 5 | 0,11 | 0,777 |
| 6 | 0,06 | 0,764 |
| **7** | **0,03** | **0,785** ← melhor época, salva |
| 8 | 0,02 | 0,782 |

Com 4 épocas, o modelo ainda estava aprendendo (F1 de teste 0,712). A partir da época 5, a perda de treino continua caindo, mas o F1 de validação estabiliza perto de 0,78: é o começo do overfitting, e a seleção pela validação evita usar as épocas piores.

### BERTimbau: peso por classe e sementes

O peso por classe (`--peso-classes`) e a versão sem peso empataram na validação (0,7849 × 0,7847). Para desempatar com honestidade, cada configuração foi treinada com 3 sementes aleatórias, e a decisão olhou **só o F1 de validação**, com a regra definida antes: maior média; em caso de empate, a configuração mais simples.

| Configuração | Semente 42 | Semente 1 | Semente 2 | Média |
| --- | --- | --- | --- | --- |
| **Sem peso** | 0,785 | 0,771 | 0,806 | **0,787** |
| Com peso | 0,785 | 0,778 | 0,796 | 0,786 |

- **A variação entre sementes (3,5 pontos) foi maior que a diferença entre as configurações (0,1 ponto).** Comparar uma rodada de cada levaria a uma conclusão baseada em ruído.
- O peso por classe não melhorou a média, mas deixou o treino mais estável entre sementes e mais rápido para convergir.
- Modelo final: **sem peso, 8 épocas, semente 42**.

## Análise de erros

O BERTimbau erra 55 das 301 avaliações do teste. Usando o rótulo do LLM como pista, cada erro foi classificado (`src/analise_erros.py`):

| Tipo | Erros | O que significa |
| --- | --- | --- |
| Erro próprio | 21 | O LLM acertou, e o modelo errou |
| Sem rótulo do LLM | 20 | Avaliação que o LLM não rotulou |
| **Herdado do professor** | **10** | O modelo previu o mesmo que o LLM, e o LLM errou |
| Ambos erram diferente | 4 | Os dois erraram, cada um de um jeito |

**29% dos erros com rótulo do LLM foram herdados do professor:** o modelo aprendeu exatamente o que lhe ensinaram.

**Comparação com o baseline:** o BERTimbau corrigiu 56 erros do baseline e introduziu 16 novos (saldo de +40 acertos). Mas **39 dos 55 erros do BERTimbau (71%) também são erros do baseline**: dois modelos muito diferentes erram nas mesmas avaliações, o que indica casos difíceis por natureza.

**O padrão mais forte: excesso de `outros`.** 14 dos 55 erros (25%) são o modelo dizendo `outros` para avaliações com um problema concreto. O LLM usa `outros` 3,5 vezes mais do que o rotulador humano (8,2% contra 2,3%), e o modelo herdou esse viés.

### Leitura de 24 erros

Os três grupos de confusão mais frequentes foram lidos um a um (motivos em `results/analise_erros.csv`):

| Motivo | Casos | Exemplo |
| --- | --- | --- |
| Ambiguidade real | 6 | "Poderia entregar mais rápido": sugestão ou reclamação de atraso? |
| Regra nova do guia que o professor não conhecia | 5 | "Tive que retirar no correio, não entregaram em casa": a regra entrou no guia v1.3, depois de o LLM ter rotulado o treino |
| Limitação do modelo | 5 | "Recomendo a loja, mas não o produto" previsto como `elogio`; texto TODO EM MAIÚSCULAS sobre produto sujo e riscado previsto como `elogio` |
| Rótulo humano discutível pelo guia | 4 | "Estou avaliando a entrega, ainda não instalei" rotulado como `nao_recebido`, mas o produto chegou |
| Erro de digitação, sem acento ou texto curto | 4 | "nao ecebi prodoutos"; "produto de ma qualidade" (sem o acento de "má") |

**Só cerca de 1 em cada 5 desses erros é uma limitação clara do modelo.** A maioria vem dos dados (rótulos do professor, regras do guia, rótulos humanos) ou de ambiguidade no próprio texto. Melhorar os dados deve render mais do que trocar de modelo.

Os rótulos humanos discutíveis **não foram corrigidos**: alterar o teste depois de ver os erros do modelo, e só onde ele errou, viciaria a avaliação a favor do modelo.

## Publicação e demo

**Modelo no Hugging Face:** o BERTimbau final (8 épocas, semente 42) foi treinado de novo no Colab, reproduzindo exatamente as métricas do experimento, e publicado com um *model card* (classes, uso, dados, treino, resultados e limitações):

```python
from transformers import pipeline

triagem = pipeline("text-classification", model="gabriellymsb/bertimbau-triagem-avaliacoes")
triagem("Comprei 2 colchas e só veio 1.")   # [{'label': 'incompleto', 'score': 0.99}]
```

**Demo no navegador, sem servidor:** Spaces com Gradio passaram a exigir plano pago no Hugging Face, então a demo foi feita como **página estática com [Transformers.js](https://huggingface.co/docs/transformers.js)**. O modelo roda no navegador de quem visita:

1. **Exportação para ONNX** com `torch.onnx.export`, conferida contra o PyTorch (diferença máxima da ordem de 10⁻⁹ nos logits).
2. **Quantização dinâmica para 8 bits** com o ONNX Runtime: de cerca de 420 MB para cerca de 110 MB, com as mesmas previsões nos exemplos testados.
3. **Página estática** (`demo/index.html`) que baixa a versão quantizada uma vez e classifica em milissegundos, sem enviar o texto a nenhum servidor.

Os arquivos ONNX ficam na pasta `onnx/` do repositório do modelo.

## Estrutura do repositório

```
AppTriagem/
├── demo/
│   └── index.html                 demo estática (Transformers.js), publicada no Hugging Face Spaces
├── docs/
│   └── guia_rotulagem.md          guia das 8 classes (v1.3); também é o prompt do LLM
├── notebooks/
│   └── 01_exploracao.ipynb        exploração inicial do dataset
├── rotulos/
│   └── teste_humano.csv           as 301 avaliações rotuladas à mão (conjunto de teste)
├── results/
│   ├── baseline877.json           métricas finais do baseline
│   ├── baseline_busca877.csv      as 20 configurações testadas na validação
│   ├── baseline_confusao877.csv   matriz de confusão do baseline no teste
│   ├── baseline_erros877.csv      erros do baseline no teste
│   ├── bertimbau_8ep.json         métricas finais do BERTimbau (modelo final)
│   ├── bertimbau_8ep_erros.csv    erros do BERTimbau no teste
│   └── analise_erros.csv          erros do BERTimbau por tipo, com motivos
├── src/
│   ├── rotular.py                 rotulagem com LLM
│   ├── normalizar_humano.py       padroniza o CSV humano depois de editar na planilha
│   ├── dividir.py                 divisão treino/validação/teste e concordância LLM × humano
│   ├── discordancias.py           análise das discordâncias LLM × humano
│   ├── baseline.py                TF-IDF + regressão logística
│   ├── finetune.py                fine-tuning do BERTimbau
│   └── analise_erros.py           tipos de erro e comparação entre modelos
├── .gitignore
└── readme.md

Fora do Git (.gitignore):
├── .env                           chave da API do LLM
├── .venv/                         ambiente virtual Python
├── data/                          dataset, amostra, rótulos do LLM e os conjuntos gerados
└── modelos/                       modelos treinados (o BERTimbau vai para o Hugging Face)
```

O sufixo `877` indica resultados com 877 exemplos de treino; `8ep`, o BERTimbau com 8 épocas.

## Os scripts

### `src/rotular.py`: rotulagem com LLM

- Limpa o dataset (textos vazios, quebras de linha, duplicados) e sorteia **2.000 avaliações estratificadas por nota**: 700 de nota 1, 400 de nota 2, 400 de nota 3, 250 de nota 4 e 250 de nota 5. Um sorteio uniforme viria quase todo de elogios. A amostra é salva uma vez, para ser reprodutível.
- Monta o prompt com o **guia de rotulagem** e uma instrução de saída em JSON.
- Envia lotes a qualquer API **compatível com a da OpenAI** (configurada no `.env`). Ajusta os parâmetros ao provedor: a camada de compatibilidade da API do Claude, por exemplo, não aceita `response_format` do tipo `json_object`.
- Valida cada rótulo; os inválidos ficam pendentes.
- **Retoma de onde parou:** cada lote é gravado ao terminar. Erros 429 e 503 são tentados de novo com espera crescente a partir de 30 segundos, e as tentativas automáticas da biblioteca ficam desligadas, para que cada tentativa custe exatamente uma chamada da cota.
- `--dry-run` mostra o prompt exato, sem chamar a API.

### `src/normalizar_humano.py`: padronização dos rótulos humanos

Planilhas mudam o arquivo ao salvar (codificação, separador, cabeçalho). Este script aceita qualquer variação, padroniza as colunas para `id,texto,rotulo_humano,obs`, recupera o texto original da amostra (desfazendo acentos quebrados), corrige variações de rótulo (`Não_recebido` → `nao_recebido`) e aponta o que continuar inválido. `--definir id=rotulo` grava uma decisão sem abrir a planilha.

### `src/dividir.py`: divisão dos dados

- **Teste:** as avaliações rotuladas à mão, com o rótulo humano; o rótulo do LLM fica ao lado, só para comparação.
- **Validação e treino:** as demais, com o rótulo do LLM, em divisão estratificada.
- Mostra a distribuição de classes por conjunto e a concordância LLM × humano.

### `src/discordancias.py`: discordâncias LLM × humano

Tabela cruzada entre rótulo humano e rótulo do LLM, com os textos dos pares que mais se confundem. Usado para encontrar lacunas no guia e inconsistências nos rótulos humanos.

### `src/baseline.py`: TF-IDF + regressão logística

Treina as 20 configurações no treino, escolhe a melhor pela validação, avalia uma única vez no teste e salva métricas, matriz de confusão, erros e o modelo.

### `src/finetune.py`: BERTimbau

Fine-tuning com PyTorch e Hugging Face Transformers, como descrito em [Técnicas](#bertimbau-ajustado-fine-tuning). Roda em GPU (Colab ou Kaggle) ou CPU. Opções: `--epocas`, `--lr`, `--batch`, `--max-len`, `--semente`, `--peso-classes`.

### `src/analise_erros.py`: análise de erros

Classifica os erros do modelo em herdados do professor, próprios, ambíguos ou sem rótulo do LLM; lista as confusões mais comuns; compara os erros de dois modelos; e gera um CSV com uma coluna `motivo` para leitura humana.

## Passo a passo

1. **Exploração** (`notebooks/01_exploracao.ipynb`): tamanho, proporção com texto, distribuição de notas, tamanho dos textos e leitura de avaliações de notas baixas para definir as classes.
2. **Guia de rotulagem v1.0:** 8 classes definidas a partir de 30 avaliações reais, com ordem de prioridade para avaliações com mais de um problema.
3. **Script de rotulagem** genérico para APIs compatíveis com a da OpenAI.
4. **Escolha do LLM:** o Ollama Cloud exigia créditos (erro 402); o Gemini, no plano gratuito, foi o escolhido.
5. **Teste com 40 avaliações**, que revelou uma contradição no guia sobre produtos falsificados → **guia v1.1**.
6. **Rotulagem em escala**, limitada pelo plano gratuito (5 requisições por minuto, 20 por dia) e por instabilidade do servidor.
7. **Rotulagem humana** de 301 avaliações, sem ver o rótulo do LLM. Um caso novo (cliente que comprou errado e pediu troca) virou o **guia v1.2**.
8. **Divisão** com o teste formado pelas avaliações humanas.
9. **Análise de discordâncias LLM × humano:** uma lacuna no guia (retirada nos Correios) e três padrões de inconsistência nos rótulos humanos → **guia v1.3** e revisão de classes inteiras.
10. **Baseline:** F1 macro de 0,628 no teste.
11. **BERTimbau, 4 épocas:** 0,712. **8 épocas:** 0,754.
12. **Peso por classe e 3 sementes:** empate; vence a configuração sem peso.
13. **Rotulagem congelada** em 1.390 avaliações, por instabilidade do Gemini.
14. **Análise de erros** do BERTimbau e comparação com o baseline e com o LLM.
15. **Publicação:** modelo no Hugging Face e demo estática com o modelo em ONNX quantizado rodando no navegador.

## Decisões e aprendizados

- **Um rótulo por avaliação.** Muitas avaliações têm mais de um problema. O guia define uma ordem de prioridade (`nao_recebido` > `incompleto` > `produto_errado` > `defeito_qualidade` > `atraso` > `atendimento` > `elogio` > `outros`), e o rótulo descreve o problema principal.
- **O rótulo descreve o que o caso precisa, não de quem é a culpa.** Um cliente que comprou errado e pediu troca é `atendimento`: o caso precisa do pós-venda, mesmo sem falha da loja.
- **Testar pequeno antes de rodar tudo.** O teste com 40 avaliações revelou que "produto falsificado" aparecia como exemplo em duas classes; o guia foi corrigido antes de gastar a cota.
- **Verificar que a mudança chegou ao sistema.** Na primeira correção do guia, o rótulo não mudou; o `--dry-run` mostrou que a edição não tinha sido salva.
- **Medir a qualidade dos próprios rótulos humanos.** A análise de discordâncias revelou três padrões de inconsistência nos rótulos humanos (usar `atendimento` quando o problema principal era outro; `atraso` quando o produto nem tinha chegado; `defeito_qualidade` para críticas leves em avaliações positivas). Foram revisadas classes inteiras pela regra do guia, e não só as discordâncias, para não viciar o teste a favor do LLM.
- **Decidir pela validação, e com mais de uma semente.** Entre duas configurações empatadas, a variação entre sementes foi maior que a diferença entre elas. A regra de decisão foi definida antes de ver o resultado, e o F1 de teste nunca entrou na escolha.
- **O aluno aprende os vícios do professor.** O excesso de `outros` do LLM e uma regra do guia criada depois da rotulagem reaparecem nos erros do modelo. Melhorar os rótulos deve render mais do que trocar de modelo.
- **Com cota pequena, o tamanho do lote é a variável que mais importa.** O guia (cerca de 3 mil tokens) vai em toda chamada; lotes maiores aproveitam melhor um limite de 20 chamadas por dia.
- **Bibliotecas podem gastar cota em silêncio.** A biblioteca `openai` tenta de novo, por padrão, até 2 vezes antes de devolver um erro. Com 20 chamadas por dia, cada erro visível chegava a custar 3. As tentativas passaram a ser controladas só pelo script.
- **Dados públicos no plano gratuito.** O plano gratuito do Gemini pode usar os dados enviados para melhorar os produtos do Google. Não há problema com um dataset público; com dados reais de clientes, a escolha seria um plano pago ou um modelo local.

## Limitações

- **Treino menor que o planejado:** 877 exemplos em vez de cerca de 1.400. A rotulagem parou em 1.390 avaliações pelo limite diário do plano gratuito do Gemini e por instabilidade do servidor (erros 503 por dias seguidos). Os experimentos usam esse retrato de 1.390 rótulos.
- **Rótulos do LLM de versões diferentes do guia:** o treino foi rotulado com as versões 1.1 e 1.2; o teste humano segue a 1.3.
- **LLM comparado em 213 das 301 avaliações do teste**, e com a ressalva de que os rótulos humanos foram revisados depois de comparados com os dele.
- **Teste pequeno nas classes raras** (13 a 20 exemplos), o que deixa o F1 dessas classes instável.
- **Um único rotulador humano**: sem medida de concordância entre pessoas.
- **Dados de e-commerce** usados como aproximação de delivery.

## Como rodar

```bash
git clone https://github.com/gabriellymsb/AppTriagem.git
cd AppTriagem
python3 -m venv .venv && source .venv/bin/activate
pip install pandas scikit-learn joblib openai python-dotenv
pip install torch --index-url https://download.pytorch.org/whl/cpu   # ou com CUDA, se houver GPU
pip install transformers
```

1. Baixe `olist_order_reviews_dataset.csv` do [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-e-commerce) e coloque em `data/`.
2. Crie um `.env` na raiz:
   ```
   LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
   LLM_API_KEY=sua_chave
   LLM_MODEL=nome_do_modelo
   ```
3. Rode:
   ```bash
   python src/rotular.py --dry-run               # confere o prompt
   python src/rotular.py --lote 125 --pausa 20   # rotulagem (retoma se parar)
   python src/normalizar_humano.py               # padroniza os rótulos humanos
   python src/dividir.py                         # divisão + concordância LLM × humano
   python src/discordancias.py                   # onde LLM e humano discordam
   python src/baseline.py                        # TF-IDF + regressão logística
   python src/finetune.py --epocas 8             # BERTimbau (de preferência em GPU)
   python src/analise_erros.py                   # tipos de erro e comparação
   ```

## Próximos passos

- [x] Exploração dos dados
- [x] Guia de rotulagem (v1.3)
- [x] Rotulagem com LLM (1.390 avaliações) e humana (301)
- [x] Baseline TF-IDF + regressão logística
- [x] Fine-tuning do BERTimbau, com escolha de épocas, peso por classe e sementes
- [x] Análise de erros
- [x] Modelo no Hugging Face, com model card
- [x] Demo no navegador (ONNX quantizado + Transformers.js) no Hugging Face Spaces
- [ ] API com FastAPI + Docker
- [ ] Terminar a rotulagem e rotular o treino de novo com o guia v1.3
- [ ] Texto em minúsculas antes do modelo (casos em caixa alta) ou um modelo *uncased*
- [ ] Classificação multi-rótulo, para avaliações com mais de um problema
- [ ] Comparar professores: Gemini × Claude × um modelo aberto, contra os rótulos humanos
