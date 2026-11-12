# AppTriagem: triagem automática de avaliações de clientes

Classificador que lê a avaliação de um cliente e diz qual é o problema do pedido: não recebido, atraso, item faltando, produto errado, defeito, atendimento. Um LLM rotula os dados de treino, e um modelo de linguagem pequeno (BERTimbau) aprende com esses rótulos. O resultado é um classificador rápido e barato o suficiente para rodar em todas as avaliações, algo inviável chamando um LLM para cada uma.

> **Status (11/11/2026):** rotulagem com LLM em andamento (1.390 de 2.000). Os resultados abaixo são **preliminares**, com 877 exemplos de treino. A versão final terá cerca de 1.400.

## Sumário

- [O problema](#o-problema)
- [Objetivo final](#objetivo-final)
- [O que o projeto testa](#o-que-o-projeto-testa)
- [Dados](#dados)
- [Como funciona](#como-funciona)
- [Divisão dos dados: 70/15/15](#divisão-dos-dados-701515)
- [Por que um teste com rótulo humano](#por-que-um-teste-com-rótulo-humano)
- [Técnicas](#técnicas)
- [Resultados preliminares](#resultados-preliminares)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Os scripts](#os-scripts)
- [Passo a passo até aqui](#passo-a-passo-até-aqui)
- [Decisões e aprendizados](#decisões-e-aprendizados)
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

## Objetivo final

Responder com números a uma pergunta de engenharia:

> **Um modelo pequeno, treinado com rótulos gerados por um LLM, chega perto da qualidade do próprio LLM, sendo muito mais rápido e barato?**

Entregas previstas:

1. Tabela comparativa (LLM × TF-IDF × BERTimbau) em qualidade, latência e custo.
2. Modelo publicado no Hugging Face, com model card.
3. Demo em Gradio, para qualquer pessoa testar.
4. API com FastAPI empacotada em Docker.

## O que o projeto testa

| Pergunta | Como é medida |
| --- | --- |
| Os rótulos do LLM são confiáveis? | Concordância e F1 do LLM contra 301 rótulos humanos |
| Um modelo simples já resolve? | Baseline TF-IDF + regressão logística |
| Entender contexto faz diferença? | BERTimbau ajustado × baseline, classe por classe |
| Vale o custo? | Latência por avaliação e custo estimado por 1 milhão de avaliações |
| Onde cada modelo erra, e por quê? | Matriz de confusão e leitura dos erros |

## Dados

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-e-commerce), arquivo `olist_order_reviews_dataset.csv`.

- 99.224 avaliações, das quais **40.977 (41%) têm comentário em texto**.
- **Notas muito desbalanceadas:** entre as que têm texto, 20.554 são nota 5 e só 2.145 são nota 2.
- **Textos curtos:** mediana de 9 palavras, máximo de 45.
- Os nomes das lojas foram anonimizados pela Olist com casas de Game of Thrones (lannister, stark, baratheon).

São dados de e-commerce, usados aqui como aproximação de um app de delivery: os problemas (atraso, não recebido, item faltando, produto errado) são os mesmos. Os dados não são versionados no repositório; baixe pelo link acima.

## Como funciona

```
 dataset Olist (40.977 com texto)
        │  limpeza + sorteio por nota
        ▼
 amostra de 2.000 avaliações
        │
        ├──► 301 rotuladas à mão ──────────────────────► TESTE (15%)
        │
        └──► 1.700 rotuladas pelo LLM (Gemini)
                 │  usando docs/guia_rotulagem.md como prompt
                 ├──► 300 ─────────────────────────────► VALIDAÇÃO (15%)
                 └──► ~1.400 ──────────────────────────► TREINO (70%)

 TREINO ──► TF-IDF + regressão logística ┐
        └─► fine-tuning do BERTimbau      ├─► escolha pela VALIDAÇÃO ─► nota final no TESTE
                                          ┘
```

O LLM **não é treinado** neste projeto. Ele faz o papel de "professor": rotula os exemplos de treino seguindo o guia. Quem é treinado é o "aluno", um modelo menor (TF-IDF ou BERTimbau) que aprende a imitar esses rótulos e depois roda sozinho, sem chamar o LLM. É uma forma de **supervisão fraca** (os rótulos de treino têm ruído) parecida com **destilação de conhecimento**.

## Divisão dos dados: 70/15/15

| Conjunto | Tamanho | Rótulo | Papel |
| --- | --- | --- | --- |
| **Treino** | ~1.400 (70%) | do LLM | O modelo aprende os padrões e ajusta os pesos internos, vendo esses exemplos repetidas vezes |
| **Validação** | 300 (15%) | do LLM | O "simulado": comparar configurações (hiperparâmetros), escolher a melhor época e detectar overfitting |
| **Teste** | 301 (15%) | **humano** | A "prova final", guardada até o modelo estar pronto e usada uma única vez |

Regras seguidas:

- **Toda escolha é feita na validação.** O teste nunca é usado para decidir nada. Se fosse, ele viraria um segundo simulado e deixaria de medir o desempenho real.
- A divisão entre treino e validação é **estratificada**: cada classe aparece na mesma proporção nos dois conjuntos.
- Sementes aleatórias fixas: a divisão é sempre a mesma em qualquer computador.

## Por que um teste com rótulo humano

Se o teste também fosse rotulado pelo LLM, o projeto mediria **o quanto o modelo imita o LLM**, e não o quanto ele acerta. Por isso, as 301 avaliações do teste foram rotuladas à mão, **sem ver o rótulo do LLM**, seguindo o mesmo guia.

Isso tem dois usos:

1. **Nota real dos modelos:** o F1 no teste é medido contra o julgamento humano.
2. **Qualidade do próprio LLM:** comparando os rótulos dele com os humanos nas mesmas avaliações, dá para medir quanto ruído entra no treino.

| Medida | LLM × humano (213 avaliações com os dois rótulos) |
| --- | --- |
| Concordância antes da revisão | 79,8% |
| Concordância depois da revisão | 86,4% |

A revisão corrigiu inconsistências **nos rótulos humanos**, não nos do LLM. Para não viciar o teste a favor do LLM, foram revisadas **classes inteiras** pela regra do guia, e não só as avaliações em que o LLM discordava (detalhes em [Decisões e aprendizados](#decisões-e-aprendizados)).

## Técnicas

### Baseline: TF-IDF + regressão logística

**TF-IDF** transforma cada avaliação em um vetor de números, um por termo do vocabulário. O peso de cada termo cresce quanto mais ele aparece *naquele* texto (TF) e quanto mais raro ele é *no conjunto todo* (IDF). Assim, "faltou" pesa muito e "produto" pesa pouco.

- Palavras soltas e pares de palavras (`ngram_range=(1, 2)`): "não recebi" diz mais que "não" e "recebi" separados.
- Opcionalmente, pedaços de 2 a 5 letras (`char_wb`): torna o modelo tolerante a erros de digitação ("nao", "recebí").
- Texto em minúsculas e sem acentos, `sublinear_tf` e `min_df=2`.

A **regressão logística** aprende um peso por termo para cada classe. Para prever, soma os pesos dos termos presentes no texto, e a classe com a maior soma vence. Dois hiperparâmetros importam:

- `C`: a força da regularização (o "freio" contra decorar o treino).
- `class_weight="balanced"`: erros em classes raras custam mais, compensando o desbalanceamento.

A busca testa 20 combinações (5 valores de C × com/sem peso × com/sem n-gramas de letras) e escolhe pela validação.

### BERTimbau ajustado (fine-tuning)

[BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased) é um modelo BERT pré-treinado em bilhões de palavras em português. Ele já "entende" a língua; o fine-tuning só ensina a ele as 8 classes deste problema.

Diferente do TF-IDF, ele lê a frase inteira com **atenção**, considerando cada palavra no contexto das outras. É isso que permite separar "veio outra marca" (`produto_errado`) de "veio quebrado" (`defeito_qualidade`), ou "chegou antes do prazo" (`elogio`) de "não chegou no prazo" (`atraso`).

Como o treino funciona (`src/finetune.py`):

1. **Tokenização:** cada texto vira uma sequência de até 64 pedaços de palavra (*tokens*). Os textos têm no máximo cerca de 45 palavras.
2. **Modelo:** o BERTimbau pré-treinado recebe uma **cabeça de classificação nova**, com 8 saídas e pesos aleatórios. As cabeças do pré-treino são descartadas; por isso o log mostra `UNEXPECTED` (descartadas) e `MISSING` (a nova).
3. **Função de perda:** entropia cruzada entre a previsão e o rótulo. Com `--peso-classes`, os erros em classes raras pesam mais, como no `balanced` do baseline.
4. **Otimização:** AdamW com taxa de aprendizado de 3e-5, aquecimento nos primeiros 10% dos passos e queda linear depois, e corte de gradientes em 1,0 para estabilidade.
5. **Épocas:** o modelo percorre o treino inteiro várias vezes, em lotes de 16. Ao fim de cada época, mede o F1 macro na **validação**.
6. **Melhor época:** só é salva a época com o maior F1 de validação. Se o modelo começar a decorar o treino (overfitting), as épocas piores são descartadas.
7. **Prova final:** a melhor época é avaliada uma única vez no **teste**.

### Métrica: F1 macro

Para cada classe, o F1 equilibra **precisão** (dos que o modelo chamou de X, quantos eram X?) e **revocação** (dos X reais, quantos ele achou?). O **macro** tira a média simples entre as 8 classes, então `outros` (13 exemplos no teste) pesa tanto quanto `elogio` (64). A acurácia esconderia o desempenho nas classes raras: um modelo que só chutasse as classes grandes teria acurácia razoável e seria inútil para `atendimento`.

## Resultados preliminares

> 877 exemplos de treino. Os números finais saem com ~1.400.

| Modelo | F1 macro (validação) | F1 macro (teste) | Acurácia (teste) | Latência |
| --- | --- | --- | --- | --- |
| TF-IDF + regressão logística | 0,703 | 0,628 | 68% | 0,36 ms (CPU) |
| BERTimbau, 4 épocas | 0,725 | **0,712** | **80%** | 2,98 ms (GPU) |
| LLM (Gemini) | — | aguardando rótulos completos | — | segundos por lote |

Melhor configuração do baseline: `C=0.3`, `class_weight="balanced"`, com n-gramas de letras.

**F1 por classe no teste:**

| Classe | Exemplos no teste | Baseline | BERTimbau |
| --- | --- | --- | --- |
| `nao_recebido` | 58 | 0,77 | **0,82** |
| `atraso` | 18 | **0,70** | 0,67 |
| `incompleto` | 38 | 0,86 | **0,92** |
| `produto_errado` | 26 | 0,60 | **0,80** |
| `defeito_qualidade` | 64 | 0,67 | **0,88** |
| `atendimento` | 20 | **0,43** | 0,36 |
| `elogio` | 64 | 0,75 | **0,90** |
| `outros` | 13 | 0,24 | **0,34** |

Leituras:

- **O contexto ajuda onde as palavras se parecem:** `produto_errado` e `defeito_qualidade` subiram cerca de 0,20 cada com o BERTimbau.
- **`atendimento` piorou:** revocação de 0,25. O LLM rotula pouco essa classe (cerca de 5% do treino), e este BERTimbau treinou sem peso por classe. A opção `--peso-classes` foi criada para isso e será testada na rodada final.
- **O F1 de validação subiu em todas as 4 épocas**, então mais épocas devem ajudar.
- **A diferença entre validação e teste** no baseline (0,70 → 0,63) reflete em boa parte o ruído dos rótulos: a validação é rotulada pelo LLM, e o teste, por uma pessoa.
- **Cautela com as classes pequenas:** com 13 a 20 exemplos, um único acerto muda o F1 dessas classes em cerca de 5 pontos.

## Estrutura do repositório

```
AppTriagem/
├── docs/
│   └── guia_rotulagem.md       guia das 8 classes (v1.3); também é o prompt do LLM
├── notebooks/
│   └── 01_exploracao.ipynb     exploração inicial do dataset
├── rotulos/
│   └── teste_humano.csv        as 301 avaliações rotuladas à mão (conjunto de teste)
├── results/
│   ├── baseline.json           métricas finais do baseline
│   ├── baseline_busca.csv      as 20 configurações testadas na validação
│   ├── baseline_confusao.csv   matriz de confusão no teste
│   ├── baseline_erros.csv      erros do baseline no teste
│   ├── bertimbau.json          métricas finais do BERTimbau
│   └── bertimbau_erros.csv     erros do BERTimbau no teste
├── src/
│   ├── rotular.py              rotulagem com LLM
│   ├── normalizar_humano.py    padroniza o CSV humano depois de editar na planilha
│   ├── dividir.py              divisão 70/15/15 e concordância LLM × humano
│   ├── discordancias.py        análise das discordâncias LLM × humano
│   ├── baseline.py             TF-IDF + regressão logística
│   └── finetune.py             fine-tuning do BERTimbau
├── .gitignore
└── readme.md

Fora do Git (.gitignore):
├── .env                        chave da API do LLM
├── .venv/                      ambiente virtual Python
├── data/
│   ├── olist_order_reviews_dataset.csv   dataset original
│   ├── amostra.csv             as 2.000 avaliações sorteadas
│   ├── rotulos_llm.jsonl       rótulos do LLM, gravados a cada lote
│   ├── rotulados_llm.csv       amostra + rótulos do LLM
│   ├── discordancias.csv       avaliações em que o LLM e o humano discordam
│   ├── treino/treino.csv       ┐
│   ├── validacao/validacao.csv ├ gerados por dividir.py
│   └── teste/teste.csv         ┘
└── modelos/
    ├── baseline.joblib         baseline treinado
    └── bertimbau-triagem/      BERTimbau ajustado (~420 MB, vai para o Hugging Face)
```

## Os scripts

### `src/rotular.py`: rotulagem com LLM

- Limpa o dataset (remove textos vazios, quebras de linha e duplicados) e sorteia **2.000 avaliações estratificadas por nota**: 700 de nota 1, 400 de nota 2, 400 de nota 3, 250 de nota 4 e 250 de nota 5. Um sorteio uniforme viria quase todo de elogios. A amostra é salva uma vez só, para ser reprodutível.
- Monta o prompt com o **guia de rotulagem** (a parte de revisão humana é cortada) e uma instrução de saída em JSON.
- Envia lotes de avaliações a qualquer API **compatível com a da OpenAI** (configurada no `.env`), com `temperature=0`.
- Valida cada rótulo recebido; os inválidos ficam pendentes.
- **Retoma de onde parou:** cada lote é gravado ao terminar, e erros de rede ou de limite de uso (429, 503) são tentados de novo com espera crescente.
- `--dry-run` mostra o prompt exato sem chamar a API.

### `src/normalizar_humano.py`: padronização dos rótulos humanos

Planilhas mudam o arquivo ao salvar (codificação, separador, cabeçalho). Este script lê qualquer variação, padroniza as colunas para `id,texto,rotulo_humano,obs`, recupera o texto original da amostra (desfazendo acentos quebrados), corrige variações de rótulo (`Não_recebido` → `nao_recebido`) e aponta o que continuar inválido. `--definir id=rotulo` grava uma decisão sem abrir a planilha.

### `src/dividir.py`: divisão 70/15/15

- **Teste:** as avaliações rotuladas à mão, com o rótulo humano. O rótulo do LLM fica ao lado, só para comparação.
- **Validação e treino:** as demais, com o rótulo do LLM, divididas de forma estratificada (300 para validação, o resto para treino).
- Mostra a distribuição de classes por conjunto e a **concordância LLM × humano**.

### `src/discordancias.py`: análise das discordâncias

Tabela cruzada entre rótulo humano e rótulo do LLM, com os textos dos pares que mais se confundem. Foi usado para encontrar lacunas no guia e inconsistências nos rótulos humanos.

### `src/baseline.py`: TF-IDF + regressão logística

Treina as 20 configurações no treino, escolhe a melhor pelo F1 macro na validação, avalia uma única vez no teste e salva métricas, matriz de confusão, erros e o modelo.

### `src/finetune.py`: BERTimbau

Fine-tuning com PyTorch e Hugging Face Transformers, conforme descrito em [Técnicas](#bertimbau-ajustado-fine-tuning). Roda em GPU (Colab ou Kaggle) ou CPU. Opções: `--epocas`, `--lr`, `--batch`, `--max-len`, `--peso-classes`.

## Passo a passo até aqui

1. **Exploração** (`notebooks/01_exploracao.ipynb`): tamanho do dataset, proporção com texto, distribuição de notas, tamanho dos textos e leitura de amostras de notas baixas para definir as classes.
2. **Guia de rotulagem v1.0:** 8 classes, definidas a partir de 30 avaliações reais de notas 1 e 2, com ordem de prioridade para avaliações com mais de um problema.
3. **Script de rotulagem** genérico para APIs compatíveis com a OpenAI.
4. **Escolha do LLM:** o Ollama Cloud exigia créditos pagos (erro 402); o Gemini, no plano gratuito, foi o escolhido.
5. **Teste com 40 avaliações**, que revelou uma contradição no guia sobre produtos falsificados → **guia v1.1**.
6. **Rotulagem em escala**, limitada pelo plano gratuito (5 requisições por minuto, 20 por dia). Ajuste: lotes de até 125 avaliações por chamada.
7. **Rotulagem humana** de 301 avaliações, sem ver o rótulo do LLM. Um caso novo (cliente que comprou errado e pediu troca) virou o **guia v1.2**.
8. **Divisão 70/15/15** com o teste formado pelas avaliações humanas.
9. **Análise de discordâncias LLM × humano:** uma lacuna no guia (retirada nos Correios) e três padrões de inconsistência nos rótulos humanos → **guia v1.3** e revisão de classes inteiras. Concordância de 79,8% para 86,4%.
10. **Baseline:** F1 macro de 0,628 no teste.
11. **BERTimbau** no Colab com GPU: F1 macro de 0,712 no teste.

## Decisões e aprendizados

- **Um rótulo por avaliação.** Muitas avaliações têm mais de um problema ("veio outro produto e não consigo falar com a central"). O guia define uma ordem de prioridade, e o rótulo descreve o problema principal: `nao_recebido` > `incompleto` > `produto_errado` > `defeito_qualidade` > `atraso` > `atendimento` > `elogio` > `outros`. Classificação multi-rótulo fica como próximo passo.
- **O rótulo descreve o que o caso precisa, não de quem é a culpa.** Um cliente que comprou errado e pediu troca é `atendimento`: o caso precisa do pós-venda, mesmo sem falha da loja.
- **Testar pequeno antes de rodar tudo.** O teste com 40 avaliações revelou que "produto falsificado" aparecia como exemplo em duas classes. O guia foi corrigido antes de gastar a cota com 2.000 avaliações.
- **Verificar que a mudança chegou ao sistema.** Na primeira correção do guia, o rótulo não mudou; o `--dry-run` mostrou que a edição não tinha sido salva.
- **Medir a qualidade dos próprios rótulos humanos.** A análise de discordâncias mostrou que os rótulos humanos tinham três padrões de inconsistência (usar `atendimento` quando o problema principal era outro; `atraso` quando o produto nem tinha chegado; `defeito_qualidade` para críticas leves em avaliações positivas). Corrigir só onde o LLM discordava viciaria o teste a favor dele; por isso, foram revisadas todas as avaliações dessas classes pela regra do guia.
- **Com cota pequena, o tamanho do lote é a variável que mais importa.** O guia (cerca de 3 mil tokens) vai em toda chamada, então lotes maiores aproveitam melhor um limite de 20 requisições por dia.
- **Desbalanceamento importa na prática.** No baseline, `class_weight="balanced"` levou o F1 de validação de 0,44 para 0,66.
- **Dados públicos no plano gratuito.** O plano gratuito do Gemini pode usar os dados enviados para melhorar os produtos do Google. Não há problema com um dataset público; com dados reais de clientes, a escolha seria um plano pago ou um modelo local.

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
   python src/dividir.py                         # 70/15/15 + concordância LLM × humano
   python src/discordancias.py                   # onde LLM e humano discordam
   python src/baseline.py                        # TF-IDF + regressão logística
   python src/finetune.py --epocas 8             # BERTimbau (de preferência em GPU)
   ```

## Próximos passos

- [x] Exploração dos dados
- [x] Guia de rotulagem (v1.3)
- [x] Script de rotulagem com LLM
- [x] Rotulagem humana do conjunto de teste (301)
- [x] Divisão 70/15/15
- [x] Baseline TF-IDF + regressão logística (preliminar)
- [x] Fine-tuning do BERTimbau (preliminar)
- [ ] Concluir a rotulagem com LLM (faltam 610)
- [ ] Rodada final: baseline e BERTimbau com ~1.400 exemplos, mais épocas e teste de `--peso-classes`
- [ ] F1 do LLM no teste completo e custo estimado por 1 milhão de avaliações
- [ ] Análise de erros dos três modelos
- [ ] Publicação no Hugging Face, demo em Gradio e API com FastAPI + Docker
