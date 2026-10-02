# Reuso do rosto desligado, video fixo de uma pessoa, commit 419ca7f

Feita em 01/10/2026, das 20:59 as 21:29 pelo relogio do PC, com o commit
`419ca7f` implantado e conferido por hash no Pi: perfil rpi3 com a pose em 416
e 2 threads no rosto, e `FACE_REUSE_SECONDS` em 0, conferido com
`--show-config`. E a base da serie `../pi3-419ca7f-video-reuso15/`. Resultados e
leitura em `docs/PLANO_OTIMIZACAO.md`, "Estado verificado em 01/10/2026".

- `uma-pessoa-r1.json` a `r3.json`: `tools/benchmark_stream.py` no modo de
  video, com a mesma carga de `../pi3-89c93b8-video-onnx1/`, 5 frames de
  aquecimento e 30 medidos, API reiniciada antes de cada rodada. Este codigo ja
  publica o tempo do rosto por partes: deteccao, embedding e comparacao.
- `evidencia/vcgencmd-reuso0.txt`: log de `vcgencmd` a cada 5 s, das 20:58:18
  as 21:30:17 pelo relogio do Pi, cerca de 39 s a frente do PC. As tres
  rodadas aparecem como os trechos a 1,4 GHz; os outros sao repouso e reinicio
  da API. `throttled=0x0` nas 378 leituras, maxima de 60,7 C no fim da r3.
