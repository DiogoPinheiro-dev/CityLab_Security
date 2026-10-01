# Uma pessoa com a pose em 416, commit c1f1ac0

Feita em 01/10/2026, com o mesmo codigo e o mesmo protocolo de
`../pi3-c1f1ac0-pose640/`, trocando so o `.env` do Pi: `POSE_IMGSZ=416`.
Resultados e leitura em `docs/PLANO_OTIMIZACAO.md`, "Estado verificado em
01/10/2026". Depois desta serie, 416 virou o padrao do perfil rpi3.

- `config-api.json`: saida de `tools/run_rpi.py --show-config` com o 416, igual
  a da serie de 640 exceto pelo `POSE_IMGSZ`. O `tls` aparece falso porque o
  comando rodou sem os certificados; a API subiu com eles.
- `uma-pessoa-r1.json` a `r3.json`: `tools/benchmark_stream.py` com a webcam do
  PC, 5 frames de aquecimento e 30 medidos, uma pessoa sentada de frente.

Horario do PC, inicio e fim de cada rodada: r1 12:56:25 a 12:59:31, r2 13:05:22
a 13:08:29, r3 13:12:40 a 13:15:51. O resumo do log de `vcgencmd`, que cobre
esta serie, esta em `../pi3-c1f1ac0-pose640/LEIAME.md`.
