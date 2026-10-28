"""Metricas do TCC de PDI: compara uma rodada do executor com a anotacao do video.

Anotacao (JSON, uma por video, fora do Git como o video):
    {
      "video": "mao_oculta_r1.avi",
      "pessoas": [
        {"id": "p1", "nome": "Nome do cadastro", "rosto": [[0.0, 30.0]]},
        {"id": "p2", "nome": null, "rosto": [[4.0, 12.0]]}
      ],
      "gestos": [{"pessoa": "p1", "classe": "Mao Oculta", "inicio": 2.0, "fim": 14.0}]
    }

- "nome" e o nome do cadastro; null e pessoa nao cadastrada (classe de rejeicao).
- "rosto" sao os trechos, em segundos de video, em que o rosto aparece e deveria
  ser reconhecido. Fora deles, a pessoa nao conta naquele quadro.
- "gestos" sao os trechos de cada gesto; a classe e um dos cinco alertas.

Rosto, por quadro amostrado: cada nome esperado presente na saida e TP, ausente e
FN; nome reconhecido sem par esperado e FP (aceitacao falsa, nome trocado ou
deteccao espuria); nao cadastrado com um "NAO ALUNO" para ele e rejeicao correta
(TN). Rosto pendente nao confirma nome: conta como "NAO ALUNO".

Gesto, por classe: um trecho anotado com o alerta em algum quadro dentro dele
(ou ate --tolerancia s depois do fim) e TP; sem o alerta, FN. Cada sequencia de
quadros seguidos com o alerta fora de todo trecho da classe e um FP. A media
usa so as classes com trecho anotado; os alarmes das outras saem em
"alarmes_sem_trecho". O trecho de
"Mao Fechada + Braco Estendido" tambem vale para "Braco Estendido" e "Mao
Fechada", que o detector emite junto.
"""
import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

UNKNOWN = "NAO ALUNO"
CLASSES = ("Braco Estendido", "Rendicao", "Mao Fechada", "Mao Fechada + Braco Estendido",
           "Mao Oculta")
IMPLIED = {"Mao Fechada + Braco Estendido": ("Braco Estendido", "Mao Fechada")}


def normalize(label):
    """Sem acento e sem diferenca de caixa: 'Mão Oculta' e 'Mao Oculta' sao iguais."""
    text = unicodedata.normalize("NFKD", label)
    return " ".join("".join(char for char in text if not unicodedata.combining(char)).split()).lower()


CANONICAL = {normalize(name): name for name in CLASSES}


def canonical_class(label):
    try:
        return CANONICAL[normalize(label)]
    except KeyError:
        raise ValueError(f"Classe de gesto desconhecida: {label}") from None


def load_run(path):
    lines = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    header = next(line for line in lines if line["tipo"] == "cabecalho")
    frames = [line for line in lines if line["tipo"] == "quadro"]
    return header, frames


def inside(instant, spans):
    # Folga de 1 us: 10,2 + 1,7 da 11,8999... em ponto flutuante.
    return any(start - 1e-6 <= instant <= end + 1e-6 for start, end in spans)


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = None
    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precisao": precision, "revocacao": recall, "f1": f1}


def face_metrics(frames, annotation):
    tp = fp = fn = tn = 0
    for frame in frames:
        expected = [person.get("nome") for person in annotation.get("pessoas", [])
                    if inside(frame["t"], person.get("rosto", []))]
        predicted = Counter(UNKNOWN if face.get("pendente") else face.get("nome")
                            for face in frame.get("rostos", []))
        for name in [name for name in expected if name is not None]:
            if predicted[name] > 0:
                predicted[name] -= 1
                tp += 1
            else:
                fn += 1
        for _ in [name for name in expected if name is None]:
            if predicted[UNKNOWN] > 0:
                predicted[UNKNOWN] -= 1
                tn += 1
        # Nome reconhecido que sobrou nao tinha par esperado.
        fp += sum(count for name, count in predicted.items() if name != UNKNOWN and count > 0)
    total = tp + tn + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "acuracia": (tp + tn) / total if total else None, **prf(tp, fp, fn)}


def gesture_spans(annotation):
    spans = {name: [] for name in CLASSES}
    for gesture in annotation.get("gestos", []):
        name = canonical_class(gesture["classe"])
        for target in (name, *IMPLIED.get(name, ())):
            spans[target].append((gesture["inicio"], gesture["fim"]))
    return spans


def gesture_metrics(frames, annotation, tolerance):
    spans = gesture_spans(annotation)
    per_class = {}
    for name in CLASSES:
        alert_times = [frame["t"] for frame in frames
                       if any(canonical_class(alert) == name
                              for gesture in frame.get("gestos", []) for alert in gesture["alertas"])]
        widened = [(start, end + tolerance) for start, end in spans[name]]
        tp = sum(any(inside(instant, [span]) for instant in alert_times) for span in widened)
        fn = len(widened) - tp
        # Sequencias de quadros seguidos com o alerta fora de qualquer trecho.
        fp, previous_outside = 0, False
        times = [frame["t"] for frame in frames]
        alerted = set(alert_times)
        for instant in times:
            outside = instant in alerted and not inside(instant, widened)
            if outside and not previous_outside:
                fp += 1
            previous_outside = outside
        per_class[name] = {"trechos": len(widened), "tp": tp, "fp": fp, "fn": fn, **prf(tp, fp, fn)}
    # Media so das classes com trecho anotado: sem trecho, revocacao e F1 nao
    # existem, e a classe so entraria pela precisao. Os alarmes delas saem a parte.
    evaluated = [values for values in per_class.values() if values["trechos"]]

    def mean(key):
        values = [item[key] for item in evaluated if item[key] is not None]
        return sum(values) / len(values) if values else None

    return {"por_classe": per_class,
            "media": {"precisao": mean("precisao"), "revocacao": mean("revocacao"),
                      "f1": mean("f1"), "classes": len(evaluated)},
            "alarmes_sem_trecho": sum(values["fp"] for values in per_class.values()
                                      if not values["trechos"])}


def evaluate(run_path, annotation_path, tolerance):
    header, frames = load_run(run_path)
    annotation = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    if annotation.get("video") and annotation["video"] != header["video"]["arquivo"]:
        raise ValueError(f"Anotacao de {annotation['video']}, rodada de {header['video']['arquivo']}")
    return {"rodada": header["rodada"], "pdi": header["pdi"], "video": header["video"]["arquivo"],
            "quadros": len(frames), "tolerancia_s": tolerance,
            "rosto": face_metrics(frames, annotation),
            "gesto": gesture_metrics(frames, annotation, tolerance)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rodada", type=Path, help="JSONL do tools/executor_pdi.py")
    parser.add_argument("anotacao", type=Path, help="JSON da anotacao do mesmo video")
    parser.add_argument("--tolerancia", type=float, default=1.7,
                        help="Segundos depois do fim do trecho em que o alerta ainda conta")
    parser.add_argument("--saida", type=Path, help="Grava o resultado em JSON")
    args = parser.parse_args(argv)
    result = evaluate(args.rodada, args.anotacao, args.tolerancia)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.saida:
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        args.saida.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
