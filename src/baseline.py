"""Baseline: TF-IDF + regressão logística.

1. Treina várias configurações no treino.
2. Escolhe a melhor pelo F1 macro na validação (o "simulado").
3. Só então avalia uma única vez no teste (a "prova final").

Uso (a partir da raiz do projeto):
    python src/baseline.py
"""

import json
import time
from itertools import product
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "data"
RESULTADOS = RAIZ / "results"
MODELOS = RAIZ / "modelos"

ROTULOS = [
    "nao_recebido", "atraso", "incompleto", "produto_errado",
    "defeito_qualidade", "atendimento", "elogio", "outros",
]


def ler(nome):
    return pd.read_csv(DADOS / nome / f"{nome}.csv")


def montar(C, peso, usar_char):
    palavras = TfidfVectorizer(lowercase=True, strip_accents="unicode",
                               ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    if usar_char:
        # n-gramas de caracteres ajudam com erros de digitação ("nao", "recebi")
        letras = TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                 analyzer="char_wb", ngram_range=(2, 5),
                                 min_df=2, sublinear_tf=True)
        vetor = FeatureUnion([("palavras", palavras), ("letras", letras)])
    else:
        vetor = palavras
    clf = LogisticRegression(C=C, class_weight=peso, max_iter=2000)
    return Pipeline([("tfidf", vetor), ("clf", clf)])


def f1(y, p):
    return f1_score(y, p, average="macro", labels=ROTULOS, zero_division=0)


def main():
    treino, validacao, teste = ler("treino"), ler("validacao"), ler("teste")

    # 1-2. Busca de hiperparâmetros usando só treino e validação
    grade = list(product([0.3, 1, 3, 10, 30], [None, "balanced"], [False, True]))
    tentativas = []
    for C, peso, usar_char in grade:
        modelo = montar(C, peso, usar_char).fit(treino["texto"], treino["rotulo"])
        nota = f1(validacao["rotulo"], modelo.predict(validacao["texto"]))
        tentativas.append({"C": C, "peso": peso, "char": usar_char, "f1_val": nota})
        print(f"C={C:<4} peso={str(peso):<8} char={str(usar_char):<5}  F1 val={nota:.3f}")

    melhor = max(tentativas, key=lambda t: t["f1_val"])
    print(f"\nMelhor na validação: {melhor}")

    # Retreina com a melhor configuração e congela
    modelo = montar(melhor["C"], melhor["peso"], melhor["char"])
    modelo.fit(treino["texto"], treino["rotulo"])

    # 3. Prova final: uma única vez no teste
    inicio = time.perf_counter()
    pred = modelo.predict(teste["texto"])
    ms_por_avaliacao = (time.perf_counter() - inicio) / len(teste) * 1000

    print("\n===== TESTE (rótulos humanos) =====")
    print(classification_report(teste["rotulo"], pred, labels=ROTULOS, zero_division=0))
    print(f"Latência: {ms_por_avaliacao:.3f} ms por avaliação (CPU)")

    resultado = {
        "modelo": "tfidf+regressao_logistica",
        "config": melhor,
        "f1_macro_validacao": melhor["f1_val"],
        "f1_macro_teste": f1(teste["rotulo"], pred),
        "f1_por_classe_teste": dict(zip(ROTULOS, f1_score(
            teste["rotulo"], pred, average=None, labels=ROTULOS, zero_division=0).round(3))),
        "latencia_ms": round(ms_por_avaliacao, 4),
    }

    # Referência: o próprio LLM no mesmo teste
    if "rotulo_llm" in teste and teste["rotulo_llm"].notna().all():
        resultado["f1_macro_teste_llm"] = f1(teste["rotulo"], teste["rotulo_llm"])
        print(f"F1 macro do LLM no mesmo teste: {resultado['f1_macro_teste_llm']:.3f}")

    RESULTADOS.mkdir(exist_ok=True)
    MODELOS.mkdir(exist_ok=True)
    (RESULTADOS / "baseline.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    pd.DataFrame(tentativas).to_csv(RESULTADOS / "baseline_busca.csv", index=False)
    pd.DataFrame(confusion_matrix(teste["rotulo"], pred, labels=ROTULOS),
                 index=ROTULOS, columns=ROTULOS).to_csv(RESULTADOS / "baseline_confusao.csv")
    erros = teste.assign(previsto=pred)[lambda d: d["rotulo"] != d["previsto"]]
    erros[["id", "texto", "rotulo", "previsto"]].to_csv(
        RESULTADOS / "baseline_erros.csv", index=False)
    joblib.dump(modelo, MODELOS / "baseline.joblib")

    print(f"\nF1 macro no teste: {resultado['f1_macro_teste']:.3f}")
    print("Resultados em results/, modelo em modelos/baseline.joblib")


if __name__ == "__main__":
    main()
