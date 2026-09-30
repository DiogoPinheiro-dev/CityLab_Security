# Gravacoes reais de gestos

Seis videos gravados no celular em 29/09/2026, com uma pessoa em pe, convertidos
com `tools/record_gesture_sequence.py --video`. Cada arquivo guarda, por frame,
so o que o `GestureAnalyzer` recebe: 17 keypoints, a caixa da pessoa e o estado
das maos (visivel, fechada, no tronco). Nao ha imagem, rosto nem nome; os
videos ficaram com o responsavel.

Em cada video a pessoa aparece de frente, de um lado, do outro e de costas,
mantendo o gesto. O video neutro foi gravado na vertical, e os cinco de gestos,
na horizontal. Os frames foram reduzidos para 640 px no lado maior, como os que
o cliente envia ao stream, e amostrados a cerca de 10 por segundo.

| Arquivo | Gesto | Frames | Duracao |
|---|---|---|---|
| `neutro.json` | nenhum | 516 | 51 s |
| `mao_fechada.json` | mao fechada, braco solto | 337 | 34 s |
| `rendicao.json` | rendicao | 216 | 22 s |
| `mao_oculta.json` | mao oculta, cotovelo dobrado | 478 | 48 s |
| `braco_estendido.json` | braco estendido, mao aberta | 561 | 56 s |
| `ameaca.json` | braco estendido com punho fechado | 578 | 58 s |

No video neutro, a pessoa esta de frente de 2 a 11 s, de lado de 12 a 25 s, do
outro lado, meio virada, de 27 a 37 s, e de costas de 38 a 48 s.

Desde 30/09/2026 o punho so conta com o braco levantado. `mao_fechada.json`,
gravado com o braco solto, deixou de disparar o alerta de mao fechada: no
arquivo, o gesto passou a `punho_braco_solto`, sem alerta esperado. Ver
`docs/PLANO_GESTOS.md`.

A pose rodou sem o rastreador do Ultralytics: no PC, a politica de Controle de
Aplicativo do Windows bloqueou a DLL do pacote `lap`, que ele usa. Sem o
rastreador, as caixas vem direto do detector; os keypoints vem do mesmo modelo.

Para ver quando cada alerta dispara:

```bash
python tools/gesture_replay.py resultados/gestos-reais/*.json
```
