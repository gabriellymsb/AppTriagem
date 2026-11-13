"""Rotula avaliações da Olist com um LLM, usando o guia de rotulagem como prompt.

Funciona com qualquer provedor compatível com a API da OpenAI
(Groq, Gemini, OpenAI, Ollama local). Configure no arquivo .env ou no terminal:

    LLM_BASE_URL  endereço da API do provedor
    LLM_API_KEY   sua chave (no Ollama, qualquer texto)
    LLM_MODEL     nome do modelo, conforme a lista do provedor

Uso (a partir da raiz do projeto):
    python src/rotular.py --dry-run     # mostra o prompt do 1º lote, sem chamar a API
    python src/rotular.py --limite 40   # teste rápido com 40 avaliações
    python src/rotular.py               # amostra completa; retoma de onde parou
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "data" / "olist_order_reviews_dataset.csv"
GUIA = RAIZ / "docs" / "guia_rotulagem.md"
AMOSTRA = RAIZ / "data" / "amostra.csv"
SAIDA_JSONL = RAIZ / "data" / "rotulos_llm.jsonl"
SAIDA_CSV = RAIZ / "data" / "rotulados_llm.csv"

ROTULOS = [
    "nao_recebido", "atraso", "incompleto", "produto_errado",
    "defeito_qualidade", "atendimento", "elogio", "outros",
]

# Quantas avaliações sortear por nota. Privilegia notas baixas,
# porque um sorteio uniforme viria quase todo de elogios (nota 5).
POR_NOTA = {1: 700, 2: 400, 3: 400, 4: 250, 5: 250}
SEMENTE = 42

INSTRUCAO_SAIDA = f"""
Você vai receber uma lista JSON de avaliações, cada uma com "id" e "texto".
Classifique cada avaliação em exatamente um rótulo, seguindo o guia acima.
Rótulos válidos: {", ".join(ROTULOS)}.
Responda somente com um objeto JSON no formato:
{{"rotulos": {{"<id>": "<rotulo>", ...}}}}
incluindo todos os ids recebidos, sem nenhum texto fora do JSON.
"""


def carregar_e_limpar() -> pd.DataFrame:
    df = pd.read_csv(DADOS)
    df = df.dropna(subset=["review_comment_message"]).copy()
    df["texto"] = (df["review_comment_message"]
                   .str.replace(r"\s+", " ", regex=True)
                   .str.strip())
    df = df[df["texto"].str.len() > 2]
    return df.drop_duplicates(subset="texto")


def montar_amostra() -> pd.DataFrame:
    """Sorteia a amostra uma única vez e salva, para ser reprodutível."""
    if AMOSTRA.exists():
        return pd.read_csv(AMOSTRA, dtype={"id": str})
    df = carregar_e_limpar()
    partes = [df[df["review_score"] == nota].sample(n, random_state=SEMENTE)
              for nota, n in POR_NOTA.items()]
    amostra = (pd.concat(partes)
               .sample(frac=1, random_state=SEMENTE)
               .reset_index(drop=True))
    amostra["id"] = [f"a{i:04d}" for i in range(len(amostra))]
    amostra = amostra[["id", "review_id", "review_score", "texto"]]
    amostra.to_csv(AMOSTRA, index=False)
    print(f"Amostra criada: {len(amostra)} avaliações -> {AMOSTRA}")
    return amostra


def montar_prompt_sistema() -> str:
    guia = GUIA.read_text(encoding="utf-8")
    guia = guia.split("<!-- FIM DO PROMPT")[0]  # remove a parte só para humanos
    return guia.strip() + "\n\n" + INSTRUCAO_SAIDA.strip()


def ids_ja_rotulados() -> set:
    if not SAIDA_JSONL.exists():
        return set()
    with SAIDA_JSONL.open(encoding="utf-8") as f:
        return {json.loads(linha)["id"] for linha in f if linha.strip()}


def extrair_json(conteudo: str) -> dict:
    """Aceita a resposta mesmo se o modelo envolver o JSON em ```json ... ```."""
    inicio, fim = conteudo.find("{"), conteudo.rfind("}")
    return json.loads(conteudo[inicio:fim + 1])


def opcoes_do_provedor(base_url: str) -> dict:
    """Parâmetros extras aceitos por cada provedor.

    A camada de compatibilidade da API do Claude recusa response_format
    "json_object" e, nos modelos mais novos, temperature abaixo de 1.
    O script já extrai o JSON da resposta mesmo sem esses parâmetros.
    """
    if "anthropic.com" in base_url:
        return {"max_tokens": 16000}
    return {"temperature": 0, "response_format": {"type": "json_object"}}


def classificar_lote(cliente, modelo, sistema, lote, tentativas=5) -> dict:
    mensagem = json.dumps(lote, ensure_ascii=False)
    extras = opcoes_do_provedor(str(cliente.base_url))
    for t in range(tentativas):
        try:
            resp = cliente.chat.completions.create(
                model=modelo,
                messages=[
                    {"role": "system", "content": sistema},
                    {"role": "user", "content": mensagem},
                ],
                **extras,
            )
            return extrair_json(resp.choices[0].message.content)["rotulos"]
        except Exception as erro:  # rede, limite de uso, JSON inválido
            espera = 30 * 2 ** t
            print(f"  erro ({type(erro).__name__}: {erro}); "
                  f"nova tentativa em {espera}s", file=sys.stderr)
            time.sleep(espera)
    return {}


def consolidar(amostra: pd.DataFrame) -> None:
    with SAIDA_JSONL.open(encoding="utf-8") as f:
        rotulos = pd.DataFrame([json.loads(l) for l in f if l.strip()])
    final = amostra.merge(rotulos.drop_duplicates("id"), on="id", how="inner")
    final.to_csv(SAIDA_CSV, index=False)
    print(f"\n{len(final)} de {len(amostra)} rotuladas -> {SAIDA_CSV}")
    print(final["rotulo_llm"].value_counts())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limite", type=int, help="rotula só N avaliações pendentes")
    p.add_argument("--lote", type=int, default=20, help="avaliações por chamada")
    p.add_argument("--pausa", type=float, default=2.0,
                   help="segundos entre chamadas (evita limite de uso)")
    p.add_argument("--dry-run", action="store_true",
                   help="mostra o prompt do primeiro lote e sai")
    args = p.parse_args()

    amostra = montar_amostra()
    sistema = montar_prompt_sistema()
    feitos = ids_ja_rotulados()
    pendentes = amostra[~amostra["id"].isin(feitos)]
    if args.limite:
        pendentes = pendentes.head(args.limite)
    print(f"Já rotuladas: {len(feitos)} | pendentes nesta execução: {len(pendentes)}")

    if args.dry_run:
        lote = pendentes.head(args.lote)[["id", "texto"]].to_dict("records")
        print("\n===== PROMPT DE SISTEMA =====\n" + sistema)
        print("\n===== MENSAGEM DO 1º LOTE =====\n"
              + json.dumps(lote, ensure_ascii=False, indent=1))
        return

    try:
        from dotenv import load_dotenv
        load_dotenv(RAIZ / ".env")
    except ImportError:
        pass
    faltando = [v for v in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")
                if not os.getenv(v)]
    if faltando:
        sys.exit(f"Configure as variáveis: {', '.join(faltando)}")

    from openai import OpenAI
    cliente = OpenAI(base_url=os.environ["LLM_BASE_URL"],
                     api_key=os.environ["LLM_API_KEY"],
                     max_retries=0,   # quem decide as novas tentativas é o script
                     timeout=300)     # desiste de uma chamada após 5 minutos
    modelo = os.environ["LLM_MODEL"]

    invalidos = 0
    total_lotes = (len(pendentes) + args.lote - 1) // args.lote
    for n, i in enumerate(range(0, len(pendentes), args.lote), start=1):
        lote = pendentes.iloc[i:i + args.lote][["id", "texto"]].to_dict("records")
        rotulos = classificar_lote(cliente, modelo, sistema, lote)
        with SAIDA_JSONL.open("a", encoding="utf-8") as f:
            for item in lote:
                rotulo = rotulos.get(item["id"])
                if rotulo not in ROTULOS:
                    invalidos += 1  # fica pendente para a próxima execução
                    continue
                f.write(json.dumps({"id": item["id"], "rotulo_llm": rotulo},
                                   ensure_ascii=False) + "\n")
        print(f"Lote {n}/{total_lotes} ok")
        time.sleep(args.pausa)

    if invalidos:
        print(f"\n{invalidos} sem rótulo válido. Rode o script de novo "
              "para tentar só essas.")
    consolidar(amostra)


if __name__ == "__main__":
    main()
