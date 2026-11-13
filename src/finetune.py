"""Fine-tuning do BERTimbau para classificar as avaliações.

- Treina no treino; ao fim de cada época, mede o F1 macro na validação.
- Guarda a época com melhor F1 na validação (evita overfitting).
- Só no fim avalia uma única vez no teste.

Roda em GPU (Colab/Kaggle) ou CPU. Na CPU, com ~1.400 textos curtos,
leva em torno de 10 a 30 minutos, dependendo do computador.

Uso (a partir da raiz do projeto):
    python src/finetune.py                  # configuração padrão
    python src/finetune.py --epocas 5 --lr 3e-5
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "data"
RESULTADOS = RAIZ / "results"
SAIDA_MODELO = RAIZ / "modelos" / "bertimbau-triagem"

ROTULOS = [
    "nao_recebido", "atraso", "incompleto", "produto_errado",
    "defeito_qualidade", "atendimento", "elogio", "outros",
]
ID = {r: i for i, r in enumerate(ROTULOS)}


def fixar_semente(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def ler(nome):
    return pd.read_csv(DADOS / nome / f"{nome}.csv")


def carregador(df, tok, max_len, batch, embaralhar):
    enc = tok(df["texto"].tolist(), truncation=True, max_length=max_len,
              padding="max_length", return_tensors="pt")
    y = torch.tensor([ID[r] for r in df["rotulo"]])
    itens = [{"input_ids": enc["input_ids"][i],
              "attention_mask": enc["attention_mask"][i],
              "labels": y[i]} for i in range(len(df))]
    return DataLoader(itens, batch_size=batch, shuffle=embaralhar)


@torch.no_grad()
def prever(modelo, dl, disp):
    modelo.eval()
    saida = []
    for b in dl:
        logits = modelo(input_ids=b["input_ids"].to(disp),
                        attention_mask=b["attention_mask"].to(disp)).logits
        saida.extend(logits.argmax(-1).cpu().tolist())
    return [ROTULOS[i] for i in saida]


def f1(y, p):
    return f1_score(y, p, average="macro", labels=ROTULOS, zero_division=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", default="neuralmind/bert-base-portuguese-cased")
    p.add_argument("--epocas", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-len", type=int, default=64)  # textos têm no máx. ~45 palavras
    p.add_argument("--semente", type=int, default=42)
    args = p.parse_args()

    fixar_semente(args.semente)
    disp = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {disp}")

    treino, validacao, teste = ler("treino"), ler("validacao"), ler("teste")
    tok = AutoTokenizer.from_pretrained(args.modelo)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        args.modelo, num_labels=len(ROTULOS),
        id2label=dict(enumerate(ROTULOS)), label2id=ID,
        ignore_mismatched_sizes=True).to(disp)  # cabeça de classificação nova, 8 classes

    dl_treino = carregador(treino, tok, args.max_len, args.batch, True)
    dl_val = carregador(validacao, tok, args.max_len, 64, False)
    dl_teste = carregador(teste, tok, args.max_len, 64, False)

    otim = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=0.01)
    passos = len(dl_treino) * args.epocas
    agenda = get_linear_schedule_with_warmup(otim, int(0.1 * passos), passos)

    melhor_f1, historico = -1.0, []
    for epoca in range(1, args.epocas + 1):
        modelo.train()
        inicio, perda_total = time.time(), 0.0
        for b in dl_treino:
            b = {k: v.to(disp) for k, v in b.items()}
            perda = modelo(**b).loss
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otim.step(); agenda.step(); otim.zero_grad()
            perda_total += perda.item()

        f1_val = f1(validacao["rotulo"], prever(modelo, dl_val, disp))
        perda_media = perda_total / len(dl_treino)
        historico.append({"epoca": epoca, "perda_treino": perda_media, "f1_val": f1_val})
        print(f"Época {epoca}: perda treino={perda_media:.4f}  F1 val={f1_val:.3f}  "
              f"({time.time() - inicio:.0f}s)")

        if f1_val > melhor_f1:  # guarda só a melhor época
            melhor_f1 = f1_val
            SAIDA_MODELO.mkdir(parents=True, exist_ok=True)
            modelo.save_pretrained(SAIDA_MODELO)
            tok.save_pretrained(SAIDA_MODELO)

    # Prova final com a melhor época
    modelo = AutoModelForSequenceClassification.from_pretrained(SAIDA_MODELO).to(disp)
    inicio = time.perf_counter()
    pred = prever(modelo, dl_teste, disp)
    ms = (time.perf_counter() - inicio) / len(teste) * 1000

    print("\n===== TESTE (rótulos humanos) =====")
    print(classification_report(teste["rotulo"], pred, labels=ROTULOS, zero_division=0))
    print(f"F1 macro no teste: {f1(teste['rotulo'], pred):.3f}")
    print(f"Latência: {ms:.2f} ms por avaliação ({disp})")

    RESULTADOS.mkdir(exist_ok=True)
    resultado = {
        "modelo": args.modelo, "config": vars(args), "dispositivo": disp,
        "historico": historico, "f1_macro_validacao": melhor_f1,
        "f1_macro_teste": f1(teste["rotulo"], pred),
        "f1_por_classe_teste": dict(zip(ROTULOS, f1_score(
            teste["rotulo"], pred, average=None, labels=ROTULOS, zero_division=0).round(3))),
        "latencia_ms": round(ms, 3),
    }
    (RESULTADOS / "bertimbau.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    teste.assign(previsto=pred)[lambda d: d["rotulo"] != d["previsto"]][
        ["id", "texto", "rotulo", "previsto"]].to_csv(RESULTADOS / "bertimbau_erros.csv", index=False)
    print(f"Modelo salvo em {SAIDA_MODELO}")


if __name__ == "__main__":
    main()
