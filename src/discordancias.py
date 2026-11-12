"""Compara os rótulos do LLM com os humanos no conjunto de teste.

Mostra a tabela cruzada completa e exemplos dos pares de classes
em que mais há discordância. Salva tudo em data/discordancias.csv.

Uso (a partir da raiz do projeto, depois de src/dividir.py):
    python src/discordancias.py
    python src/discordancias.py --pares 8 --exemplos 6
"""

import argparse
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
TESTE = RAIZ / "data" / "teste" / "teste.csv"
SAIDA = RAIZ / "data" / "discordancias.csv"

CURTO = {
    "nao_recebido": "nao_receb", "atraso": "atraso", "incompleto": "incompl",
    "produto_errado": "prod_err", "defeito_qualidade": "defeito",
    "atendimento": "atend", "elogio": "elogio", "outros": "outros",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pares", type=int, default=5, help="quantos pares mostrar")
    p.add_argument("--exemplos", type=int, default=4, help="exemplos por par")
    args = p.parse_args()

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)

    t = pd.read_csv(TESTE).dropna(subset=["rotulo_llm"])
    ordem = list(CURTO.values())
    tabela = pd.crosstab(t["rotulo"].map(CURTO), t["rotulo_llm"].map(CURTO),
                         rownames=["você"], colnames=["LLM"])
    tabela = tabela.reindex(index=ordem, columns=ordem, fill_value=0)
    print("Linhas = seu rótulo | colunas = rótulo do LLM | diagonal = concordância\n")
    print(tabela)

    d = t[t["rotulo"] != t["rotulo_llm"]]
    d[["id", "rotulo", "rotulo_llm", "texto", "obs"]].sort_values(
        ["rotulo", "rotulo_llm"]).to_csv(SAIDA, index=False, encoding="utf-8")
    print(f"\n{len(d)} discordâncias em {len(t)} avaliações -> {SAIDA}")

    pares = d.groupby(["rotulo", "rotulo_llm"]).size().sort_values(ascending=False)
    for (voce, llm), n in pares.head(args.pares).items():
        print(f"\n=== você: {voce} | LLM: {llm} ({n}) ===")
        grupo = d[(d["rotulo"] == voce) & (d["rotulo_llm"] == llm)]
        for _, r in grupo.head(args.exemplos).iterrows():
            print(f"- {r['id']}: {r['texto']}")


if __name__ == "__main__":
    main()
