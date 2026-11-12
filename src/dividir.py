"""Divide as 2.000 avaliações em treino (70%), validação (15%) e teste (15%).

- teste: as 300 avaliações revisadas à mão (rotulos/teste_humano.csv).
  O rótulo é o humano; o do LLM fica ao lado só para medir a qualidade dele.
- validação e treino: as demais, com rótulo do LLM, divididas mantendo
  a proporção de cada classe.

Uso (a partir da raiz do projeto):
    python src/dividir.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parent.parent
AMOSTRA = RAIZ / "data" / "amostra.csv"
ROTULOS_LLM = RAIZ / "data" / "rotulos_llm.jsonl"
HUMANO = RAIZ / "rotulos" / "teste_humano.csv"
SAIDA = RAIZ / "data"

ROTULOS = [
    "nao_recebido", "atraso", "incompleto", "produto_errado",
    "defeito_qualidade", "atendimento", "elogio", "outros",
]
SEMENTE = 42


def carregar_llm() -> pd.DataFrame:
    with ROTULOS_LLM.open(encoding="utf-8") as f:
        r = pd.DataFrame([json.loads(l) for l in f if l.strip()])
    return r.drop_duplicates("id", keep="last")  # vale o rótulo mais recente


def ler_planilha(caminho) -> pd.DataFrame:
    """Lê o CSV salvo por LibreOffice/Excel: aceita UTF-8 ou Windows-1252, e , ou ;."""
    for codificacao in ("utf-8-sig", "cp1252"):
        try:
            df = pd.read_csv(caminho, dtype=str, sep=None, engine="python",
                             encoding=codificacao)
            break
        except UnicodeDecodeError:
            continue
    df.columns = [c.strip().lower() for c in df.columns]
    faltando = {"id", "rotulo_humano"} - set(df.columns)
    if faltando:
        sys.exit(f"Colunas não encontradas em {caminho.name}: {', '.join(faltando)}. "
                 f"Colunas lidas: {', '.join(df.columns)}")
    if "obs" not in df.columns:
        df["obs"] = ""
    return df


def carregar_humano() -> pd.DataFrame:
    h = ler_planilha(HUMANO).fillna("")
    h["id"] = h["id"].str.strip()
    h["rotulo_humano"] = h["rotulo_humano"].str.strip().str.lower()
    vazios = h[h["rotulo_humano"] == ""]
    invalidos = h[(h["rotulo_humano"] != "") & ~h["rotulo_humano"].isin(ROTULOS)]
    if len(vazios) or len(invalidos):
        if len(vazios):
            print(f"{len(vazios)} sem rótulo humano: {', '.join(vazios['id'].head(10))}")
        for _, r in invalidos.iterrows():
            print(f"rótulo inválido em {r['id']}: '{r['rotulo_humano']}'")
        sys.exit("Corrija rotulos/teste_humano.csv e rode de novo.")
    return h


def main():
    amostra = pd.read_csv(AMOSTRA, dtype={"id": str})
    llm = carregar_llm()
    humano = carregar_humano()

    # Teste: as revisadas à mão
    teste = (humano[["id", "rotulo_humano", "obs"]]
             .merge(amostra[["id", "texto"]], on="id")
             .merge(llm, on="id", how="left")
             .rename(columns={"rotulo_humano": "rotulo"}))

    # Treino + validação: o resto, com rótulo do LLM
    resto = amostra[~amostra["id"].isin(teste["id"])].merge(llm, on="id", how="left")
    sem_rotulo = resto["rotulo_llm"].isna().sum()
    if sem_rotulo:
        print(f"Aviso: {sem_rotulo} avaliações ainda sem rótulo do LLM ficaram de fora. "
              "Rode src/rotular.py até zerar as pendentes.")
    resto = resto.dropna(subset=["rotulo_llm"]).rename(columns={"rotulo_llm": "rotulo"})

    n_val = round(0.15 * len(amostra))  # 15% do total, igual ao teste
    treino, validacao = train_test_split(
        resto[["id", "texto", "rotulo"]], test_size=n_val,
        stratify=resto["rotulo"], random_state=SEMENTE)

    partes = {"treino": treino,
              "validacao": validacao,
              "teste": teste[["id", "texto", "rotulo", "rotulo_llm", "obs"]]}
    for nome, df in partes.items():
        pasta = SAIDA / nome
        pasta.mkdir(parents=True, exist_ok=True)
        df.to_csv(pasta / f"{nome}.csv", index=False)

    total = sum(len(d) for d in partes.values())
    print("\nDivisão:")
    for nome, df in partes.items():
        print(f"  {nome:<10} {len(df):>5}  ({len(df) / total:.0%})")

    tabela = pd.DataFrame({n: d["rotulo"].value_counts() for n, d in partes.items()})
    print("\nRótulos por conjunto:")
    print(tabela.reindex(ROTULOS).fillna(0).astype(int))

    # Qualidade dos rótulos do LLM, medida contra os seus
    t = teste.dropna(subset=["rotulo_llm"])
    if len(t):
        print(f"\nLLM × humano no teste ({len(t)} avaliações):")
        print(f"  concordância: {accuracy_score(t['rotulo'], t['rotulo_llm']):.1%}")
        print(f"  F1 macro:     {f1_score(t['rotulo'], t['rotulo_llm'], average='macro', labels=ROTULOS, zero_division=0):.3f}")
        print("  Os erros do LLM viram ruído no treino: guarde esse número para o README.")


if __name__ == "__main__":
    main()
