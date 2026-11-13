"""Análise de erros dos modelos no conjunto de teste.

Separa cada erro em um tipo, usando o rótulo do LLM (o "professor") como pista:

- herdado do professor: o modelo previu o mesmo que o LLM, e o LLM errou.
  O modelo aprendeu certo; o ruído estava nos rótulos de treino.
- erro próprio: o LLM acertou, e o modelo errou. Limitação do modelo.
- ambos erram diferente: o modelo e o LLM erraram, cada um de um jeito.
  Costuma indicar avaliação ambígua ou difícil.
- sem rótulo do LLM: avaliação do teste que o LLM não chegou a rotular.

Também compara os erros de dois modelos (quais são comuns e quais são de cada um)
e gera um CSV para leitura, com uma coluna "motivo" para você preencher.

Uso (a partir da raiz do projeto):
    python src/analise_erros.py
    python src/analise_erros.py --principal results/bertimbau_8ep_erros.csv \
                                --comparar results/baseline_erros.csv
"""

import argparse
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
TESTE = RAIZ / "data" / "teste" / "teste.csv"
SAIDA = RAIZ / "results" / "analise_erros.csv"


def classificar(linha) -> str:
    if pd.isna(linha["rotulo_llm"]):
        return "sem rótulo do LLM"
    if linha["previsto"] == linha["rotulo_llm"]:
        return "herdado do professor"
    if linha["rotulo_llm"] == linha["rotulo"]:
        return "erro próprio"
    return "ambos erram diferente"


def carregar(caminho, teste):
    erros = pd.read_csv(caminho)[["id", "previsto"]]
    return erros.merge(teste, on="id", how="left")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--principal", default="results/bertimbau_8ep_erros.csv")
    p.add_argument("--comparar", default="results/baseline_erros.csv")
    args = p.parse_args()

    pd.set_option("display.width", 200)
    teste = pd.read_csv(TESTE)[["id", "texto", "rotulo", "rotulo_llm"]]
    principal = carregar(RAIZ / args.principal, teste)
    principal["tipo"] = principal.apply(classificar, axis=1)

    print(f"Modelo principal: {args.principal}")
    print(f"{len(principal)} erros em {len(teste)} avaliações "
          f"(acurácia {1 - len(principal) / len(teste):.1%})\n")

    print("Tipos de erro:")
    tipos = principal["tipo"].value_counts()
    for tipo, n in tipos.items():
        print(f"  {tipo:<24} {n:>3}  ({n / len(principal):.0%})")

    com_llm = principal[principal["tipo"] != "sem rótulo do LLM"]
    if len(com_llm):
        herdados = (com_llm["tipo"] == "herdado do professor").mean()
        print(f"\nEntre os erros com rótulo do LLM, {herdados:.0%} foram herdados do professor.")

    print("\nConfusões mais comuns (humano -> previsto):")
    pares = principal.groupby(["rotulo", "previsto"]).size().sort_values(ascending=False)
    for (humano, previsto), n in pares.head(8).items():
        print(f"  {humano:<18} -> {previsto:<18} {n}")

    if args.comparar and (RAIZ / args.comparar).exists():
        outro = carregar(RAIZ / args.comparar, teste)
        a, b = set(principal["id"]), set(outro["id"])
        print(f"\nComparação com {args.comparar}:")
        print(f"  erros em comum:              {len(a & b)}")
        print(f"  só do modelo principal:      {len(a - b)}")
        print(f"  só do modelo comparado:      {len(b - a)}  (o principal acertou esses)")

    leitura = principal[["id", "tipo", "rotulo", "previsto", "rotulo_llm", "texto"]].copy()
    leitura["motivo"] = ""
    leitura.sort_values(["tipo", "rotulo", "previsto"]).to_csv(SAIDA, index=False, encoding="utf-8")
    print(f"\nPara leitura: {SAIDA}")
    print("Preencha a coluna 'motivo' em uns 15-20 erros (ex.: rótulo humano duvidoso, "
          "ambíguo, sarcasmo, dois problemas, texto curto demais).")


if __name__ == "__main__":
    main()
