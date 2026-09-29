"""Banco de replay das regras de gesto, sem modelo, camera nem Raspberry.

Cada sequencia traz, por observacao, o que o GestureAnalyzer recebe no stream:
keypoints, caixa e contexto das maos. O replay entrega as mesmas observacoes
com o intervalo escolhido e mostra em qual delas cada alerta dispara, antes e
depois do criterio duplo (acoes 1 e 2 de docs/PLANO_GESTOS.md).

As sequencias sinteticas cobrem as cinco regras e a cena neutra. Gravacoes
reais, feitas com tools/record_gesture_sequence.py, usam o mesmo formato:

    {"esperados": ["Rendicao"],
     "frames": [{"keypoints": [[x, y, conf], ...17], "box": [x1, y1, x2, y2],
                 "hand_context": {"left_visible": true, "right_closed": false}}]}

Um frame com "keypoints": null e um frame sem ninguem detectado.

Uso:
    python tools/gesture_replay.py
    python tools/gesture_replay.py gravacao.json --intervalos 0.033 6.2
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from App.GestureRecon.detector import GestureAnalyzer, MIN_OBSERVATIONS

# 30 FPS, 1 e 3 s, o frame de duas pessoas no Pi hoje e o da linha de base.
INTERVALS = (1 / 30, 1.0, 3.0, 6.2, 13.2)
# Uma observacao por regra equivale ao criterio anterior, so de duracao.
PREVIOUS_MIN_OBSERVATIONS = {rule: 1 for rule in MIN_OBSERVATIONS}
CODES = {"Mao Fechada": "F", "Mao Fechada + Braco Estendido": "A", "Rendicao": "R",
         "Mao Oculta": "O", "Braco Estendido": "B"}
# Condicao do detector por tras de cada alerta, para medir uma gravacao.
RULE_KEYS = {"Mao Fechada": ("fist_frames",),
             "Mao Fechada + Braco Estendido": ("threat_frames",),
             "Rendicao": ("surrender_frames",),
             "Mao Oculta": ("left_hidden_frames", "right_hidden_frames"),
             "Braco Estendido": ("aiming_frames",)}

# Pessoa num quadro de 320x240. De frente: ombros a 50 px, quadril a 70 px
# abaixo. As posicoes dos bracos cumprem a geometria de cada regra do detector.
FRONT = {"shoulders": ((140, 80), (190, 80)), "hips": ((145, 150), (185, 150))}
SIDE = {"shoulders": ((160, 80), (170, 80)), "hips": ((160, 150), (170, 150))}
ARMS_DOWN = ((135, 115), (133, 148), (195, 115), (197, 148))
ARMS_DOWN_WIDE = ((120, 115), (116, 150), (210, 115), (214, 150))
HANDS_UP = ((110, 80), (110, 40), (220, 80), (220, 40))
RIGHT_ARM_EXTENDED = ((135, 115), (133, 148), (225, 80), (260, 80))
# Mao atras das costas, na cintura, com o cotovelo dobrado a cerca de 105 graus.
RIGHT_HAND_BEHIND_BACK = ((135, 115), (133, 148), (205, 115), (180, 135))
SIDE_ARMS_DOWN = ((162, 115), (164, 148), (168, 115), (166, 148))
OPEN = {"left_visible": True, "right_visible": True}
RIGHT_FIST = {"left_visible": True, "right_visible": True, "right_closed": True}
RIGHT_HIDDEN = {"left_visible": True, "right_visible": False}
NO_HANDS = {"left_visible": False, "right_visible": False}
GESTURES = {
    "mao_fechada": (ARMS_DOWN, RIGHT_FIST, {"Mao Fechada"}),
    "ameaca": (RIGHT_ARM_EXTENDED, RIGHT_FIST,
               {"Mao Fechada", "Mao Fechada + Braco Estendido", "Braco Estendido"}),
    "rendicao": (HANDS_UP, OPEN, {"Rendicao"}),
    "mao_oculta": (RIGHT_HAND_BEHIND_BACK, RIGHT_HIDDEN, {"Mao Oculta"}),
    "braco_estendido": (RIGHT_ARM_EXTENDED, OPEN, {"Braco Estendido"}),
}
# Situacoes do video neutro de 29/09/2026, mais a mao do lado que a camera nao ve.
POSTURES = {
    "neutro": (ARMS_DOWN, OPEN, {}, set()),
    "neutro_bracos_afastados": (ARMS_DOWN_WIDE, OPEN, {}, set()),
    "neutro_mao_perdida": (ARMS_DOWN, RIGHT_HIDDEN, {}, set()),
    "neutro_de_costas": (ARMS_DOWN, NO_HANDS, {"face": 0.1}, set()),
    "de_lado_mao_do_outro_lado": (SIDE_ARMS_DOWN, RIGHT_HIDDEN, {"body": SIDE}, {"Mao Oculta"}),
}


def _frame(arms, hands, body=FRONT, face=0.9):
    """Um quadro sintetico; com a face em confianca baixa, a pessoa esta de costas."""
    left_elbow, left_wrist, right_elbow, right_wrist = arms
    head = [((165, 50), face), ((160, 45), face), ((170, 45), face), ((155, 48), 0.9),
            ((175, 48), 0.9)]
    body_points = [*body["shoulders"], left_elbow, right_elbow, left_wrist, right_wrist,
                   *body["hips"], (145, 190), (185, 190), (145, 230), (185, 230)]
    points = head + [(point, 0.9) for point in body_points]
    return {"keypoints": [[float(x), float(y), conf] for (x, y), conf in points],
            "box": [120.0, 30.0, 270.0, 235.0], "hand_context": dict(hands)}


def synthetic_sequences(sustained=20):
    """Posturas e gestos mantidos, e cada gesto breve (duas observacoes)."""
    sequences = {}
    for name, (arms, hands, options, expected) in POSTURES.items():
        sequences[name] = {"frames": [_frame(arms, hands, **options)] * sustained,
                           "esperados": expected}
    neutral = _frame(ARMS_DOWN, OPEN)
    for name, (arms, hands, expected) in GESTURES.items():
        gesture = _frame(arms, hands)
        sequences[f"{name}_mantido"] = {"frames": [gesture] * sustained, "esperados": expected}
        sequences[f"{name}_breve"] = {"frames": [neutral, gesture, gesture] + [neutral] * 3,
                                      "esperados": expected}
    return sequences


def _observe(frames, interval, min_observations=None):
    """Entrega cada observacao ao analisador: indice, alertas e condicoes ativas."""
    analyzer = GestureAnalyzer(min_observations=min_observations)
    for index, frame in enumerate(frames):
        if frame.get("keypoints") is None:
            # Ninguem detectado neste frame: o servico esquece o track.
            analyzer.clean_old_tracks([])
            yield index, [], {}
            continue
        result = analyzer.analyze(1, frame["keypoints"], frame.get("box"),
                                  hand_context=frame.get("hand_context"),
                                  observed_at=index * interval)
        yield index, result["alerts"], analyzer.active_states


def replay(frames, interval, min_observations=None):
    """Observacao, contada a partir de 1, em que cada alerta disparou primeiro."""
    first = {}
    for index, alerts, _states in _observe(frames, interval, min_observations):
        for alert in alerts:
            first.setdefault(alert, index + 1)
    return first


def activity(frames):
    """Por alerta: fracao das observacoes com a condicao ativa e a maior sequencia."""
    active = {alert: [] for alert in RULE_KEYS}
    for _index, _alerts, states in _observe(frames, 1.0):
        for alert, keys in RULE_KEYS.items():
            active[alert].append(any(states.get((1, key), False) for key in keys))
    summary = {}
    for alert, values in active.items():
        longest = current = 0
        for value in values:
            current = current + 1 if value else 0
            longest = max(longest, current)
        summary[alert] = (sum(values) / max(len(values), 1), longest)
    return summary


def table(sequences, intervals, min_observations):
    header = ["sequencia".ljust(24)] + [f"{interval:.3g} s".rjust(12) for interval in intervals]
    lines = ["".join(header)]
    for name, sequence in sequences.items():
        cells = []
        for interval in intervals:
            first = replay(sequence["frames"], interval, min_observations)
            cell = " ".join(f"{CODES.get(alert, alert)}{index}"
                            for alert, index in sorted(first.items(), key=lambda item: item[1]))
            cells.append((cell or "-").rjust(12))
        lines.append(name.ljust(24) + "".join(cells))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gravacoes", nargs="*", type=Path, help="sequencias gravadas em JSON")
    parser.add_argument("--intervalos", nargs="+", type=float, default=INTERVALS)
    args = parser.parse_args()

    sequences = synthetic_sequences()
    recorded = {}
    for path in args.gravacoes:
        data = json.loads(path.read_text(encoding="utf-8"))
        recorded[path.stem] = data
        sequences[path.stem] = {"frames": data["frames"], "esperados": set(data.get("esperados", []))}

    print("F Mao Fechada, A Mao Fechada + Braco Estendido, R Rendicao, O Mao Oculta,")
    print("B Braco Estendido. R3 = Rendicao na terceira observacao, ou seja,")
    print("2 intervalos depois da primeira.\n")
    for title, minimum in (("Antes: so a duracao", PREVIOUS_MIN_OBSERVATIONS),
                           (f"Depois: duracao e observacoes seguidas {MIN_OBSERVATIONS}", None)):
        print(title)
        print(table(sequences, args.intervalos, minimum))
        print()

    if recorded:
        print("Gravacoes: parte das observacoes com a condicao ativa e a maior sequencia")
        for name, data in recorded.items():
            with_person = sum(1 for frame in data["frames"] if frame.get("keypoints") is not None)
            parts = [f"{CODES[alert]} {share:.0%} (seq. {longest})"
                     for alert, (share, longest) in activity(data["frames"]).items() if share]
            print(f"{name}: {len(data['frames'])} frames, pessoa em {with_person}; "
                  + (", ".join(parts) or "nenhuma condicao ativa"))


if __name__ == "__main__":
    main()
