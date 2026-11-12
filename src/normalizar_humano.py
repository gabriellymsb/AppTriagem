"""Deixa rotulos/teste_humano.csv no formato padrão depois de editar na planilha.

Rode sempre depois de salvar no LibreOffice/Excel. Ele:
- aceita UTF-8 ou Windows-1252, separador , ou ;, cabeçalho em qualquer caixa;
- padroniza as colunas para id, texto, rotulo_humano, obs;
- recupera o texto original de data/amostra.csv (desfaz acentos quebrados);
- corrige variações comuns de rótulo (acento, maiúscula, espaço);
- avisa o que continuar inválido.

Uso (a partir da raiz do projeto):
    python src/normalizar_humano.py
    python src/normalizar_humano.py --definir a1809=atendimento
"""

import argparse
import sys
import unicodedata
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
HUMANO = RAIZ / "rotulos" / "teste_humano.csv"
AMOSTRA = RAIZ / "data" / "amostra.csv"

ROTULOS = [
    "nao_recebido", "atraso", "incompleto", "produto_errado",
    "defeito_qualidade", "atendimento", "elogio", "outros",
]


def sem_acento(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def ler(caminho) -> pd.DataFrame:
    for codificacao in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(caminho, dtype=str, sep=None, engine="python",
                               encoding=codificacao).fillna("")
        except UnicodeDecodeError:
            continue
    sys.exit("Não consegui ler o arquivo.")


def achar_coluna(colunas, *nomes):
    for c in colunas:
        if sem_acento(c).strip().lower() in nomes:
            return c
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--definir", nargs="*", default=[],
                   help="força rótulos, ex.: a1809=atendimento")
    args = p.parse_args()

    h = ler(HUMANO)
    print("Colunas lidas:", list(h.columns))
    c_id = achar_coluna(h.columns, "id")
    c_rot = achar_coluna(h.columns, "rotulo_humano", "rotulo", "label")
    c_obs = achar_coluna(h.columns, "obs", "observacao", "observacoes")
    if not c_id or not c_rot:
        sys.exit("Não achei as colunas de id e de rótulo. Me mande a linha 'Colunas lidas'.")

    h = pd.DataFrame({
        "id": h[c_id].str.strip(),
        "rotulo_humano": h[c_rot],
        "obs": h[c_obs] if c_obs else "",
    })
    h["rotulo_humano"] = (h["rotulo_humano"].map(sem_acento).str.strip().str.lower()
                          .str.replace(" ", "_").str.rstrip("?"))

    for item in args.definir:
        rid, rotulo = item.split("=")
        h.loc[h["id"] == rid, "rotulo_humano"] = rotulo
        print(f"definido: {rid} = {rotulo}")

    texto = pd.read_csv(AMOSTRA, dtype=str)[["id", "texto"]]
    h = h.merge(texto, on="id", how="left")[["id", "texto", "rotulo_humano", "obs"]]
    h.to_csv(HUMANO, index=False, encoding="utf-8")

    print(f"\n{len(h)} linhas salvas em UTF-8 | {h['texto'].isna().sum()} ids não encontrados")
    print(h["rotulo_humano"].value_counts().to_string())
    vazios = h[h["rotulo_humano"] == ""]["id"].tolist()
    invalidos = h[(h["rotulo_humano"] != "") & ~h["rotulo_humano"].isin(ROTULOS)]
    if vazios:
        print(f"\nSem rótulo: {', '.join(vazios)}")
    for _, r in invalidos.iterrows():
        print(f"Inválido: {r['id']} = '{r['rotulo_humano']}'")
    if not vazios and invalidos.empty:
        print("\nTudo certo. Pode rodar src/dividir.py")


if __name__ == "__main__":
    main()
