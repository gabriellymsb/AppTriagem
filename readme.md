# AppTriagem

Classificador que lê avaliações de clientes e identifica o problema do pedido: **atraso**, **não recebido**, **item faltando**, **produto errado**, **defeito**, **atendimento**. Um LLM rotula os dados de treino, e um modelo pequeno aprende com esses rótulos. O resultado é um classificador rápido e barato o bastante para rodar em todas as avaliações, algo inviável chamando um LLM para cada uma.

O projeto cobre o ciclo completo: rotular, treinar, avaliar e publicar modelos no Hugging Face, com LLMs no meio do processo.

---

## Problema

Em um app de delivery ou marketplace, milhares de avaliações chegam todos os dias em texto livre. Ler e triar tudo à mão não escala, e problemas operacionais acabam enterrados em comentários genéricos. Exemplos reais do dataset:

* *"Eu ainda não recebi o produto."* → `nao_recebido`
* *"Só foi entregue o cartucho da tinta preta. Falta o outro cartucho da tinta colorida."* → `incompleto`
* *"Veio o copo errado. Vendeu como copo de 470ml e veio copos de 400ml"* → `produto_errado`
* *"O prazo era de 14 dias e chegou quase um mês depois"* → `atraso`

---

## Por que importa

Classificar por causa permite rotear cada caso para o time certo (suporte, logística, loja parceira), priorizar reembolsos e enxergar padrões, como um parceiro que erra pedidos com frequência ou uma região com atrasos recorrentes.

---

## Dados

**Brazilian E-Commerce Public Dataset by Olist**: cerca de 99 mil avaliações reais de clientes brasileiros, das quais 41% têm comentário em texto. Os nomes das lojas foram anonimizados pela própria Olist.

São dados de e-commerce, usados aqui como aproximação de um aplicativo de delivery: os problemas (*atraso, não recebido, item faltando, produto errado*) são os mesmos. Os dados não são versionados neste repositório; baixe pelo link oficial.

Duas características guiaram as decisões do projeto:
- **Notas muito desbalanceadas:** metade das avaliações com texto tem nota 5. A amostra de treino privilegia notas baixas, e a métrica principal é *F1 macro*, não acurácia.
- **Textos curtos:** mediana de 9 palavras, máximo de 45.

---

## Como funciona

1. **Guia de rotulagem (`docs/guia_rotulagem.md`):** 8 classes com definições, exemplos, ordem de prioridade e casos ambíguos. O mesmo arquivo serve de prompt para o LLM e de referência para a revisão humana.
2. **Rotulagem com LLM (`src/rotular.py`):** 2.000 avaliações sorteadas por nota, enviadas em lotes de 20 para um LLM compatível com a API da OpenAI (Gemini, no plano gratuito). O script retoma de onde parou se a rede ou a API falharem.
3. **Revisão humana:** 300 avaliações rotuladas à mão, sem ver o rótulo do LLM. Elas formam o conjunto de teste e nunca entram no treino. Isso reduz a rotulagem manual, em vez de eliminá-la, e permite medir a qualidade dos próprios rótulos do LLM.
4. **Treino:** baseline com TF-IDF + regressão logística e fine-tuning do BERTimbau.
5. **Avaliação:** comparação entre baseline, BERTimbau e LLM sem treino, em qualidade, latência e custo.
6. **Publicação:** modelo no Hugging Face, demo em Gradio e API com FastAPI + Docker.

---

## Decisões e aprendizados

- **Um rótulo por avaliação:** Muitas avaliações têm mais de um problema (*"veio outro produto e não consigo falar com a central"*). Para caber no prazo, o guia define uma ordem de prioridade e o rótulo descreve o problema principal. Classificação multi-rótulo fica como próximo passo.
- **Testar pequeno antes de rodar tudo:** Um teste com 40 avaliações revelou uma contradição no guia: *"produto falsificado"* estava listado como defeito, mas marca diferente da anunciada era produto errado. O guia foi corrigido e versionado (`v1.1`) antes de rotular as 2.000.
- **Dados públicos no plano gratuito:** O plano gratuito do Gemini pode usar os dados enviados para melhorar os produtos do Google. Não há problema com um dataset público; com dados reais de clientes, a escolha seria um plano pago ou um modelo local.

---

## Estrutura do projeto

```text
AppTriagem/
├── docs/        # Guia de rotulagem
├── notebooks/   # Exploração e análise de erros
├── src/         # Rotulagem, treino, avaliação
├── api/         # FastAPI + Dockerfile (em breve)
├── results/     # Tabela de métricas e gráficos (em breve)
└── README.md
```

## Como rodar
### 1. Criar e ativar o ambiente virtual
```bash
python3 -m venv .venv && source .venv/bin/activate
```

### 2. Instalar dependências
```bash
pip install pandas openai python-dotenv
```

### 3. Configuração dos dados e ambiente:

#### 1. Coloque 'olist_order_reviews_dataset.csv' no diretório data/
#### 2. Crie um arquivo .env com: LLM_BASE_URL, LLM_API_KEY e LLM_MODEL

### 4. Execução do script de rotulagem
```bash
# Mostra o prompt, sem chamar a API
python src/rotular.py --dry-run
```
``` bash
# Teste rápido com 40 amostras
python src/rotular.py --limite 40
```
```bash
# Amostra completa respeitando rate limit
python src/rotular.py --pausa 7
```
