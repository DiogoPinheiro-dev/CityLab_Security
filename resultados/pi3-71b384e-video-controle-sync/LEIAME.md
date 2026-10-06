# Controle sincrono do P5 no Pi, commit 71b384e

Feito em 06/10/2026 com o commit `71b384e` implantado e conferido por hash em
`App/FaceRecon/service.py` e `App/settings.py`. O controle usou o video de
carga, pose NCNN com 2 threads, detector facial `det_500m`, aprendizado
desligado e `FACE_ASYNC_RECOGNITION=0`. O video tem SHA-256
`b0ef593552469c93881cfa6bb1b0a6f7715f3f4452b0f34e90bd787a887b0431`.

- `uma-pessoa-r1.json`: 5 frames de aquecimento e 30 medidos. Media de
  1729,282 ms, mediana de 2308,346 ms, p95 de 2495,782 ms e 0,577 FPS.
- Pessoa, rosto e gesto em 30/30 frames. O nome ficou confirmado em 26/30,
  pendente em 4/30 e desconhecido em nenhum; houve 16 embeddings.
- Oito alertas: braco estendido no frame 17, mao fechada nos frames 21 a 23 e
  rendicao nos frames 26 a 29. Esta e a referencia da regra de gesto vigente
  em 06/10, com 4 s para mao fechada.
- O log acompanhado durante a rodada valida teve 39 leituras em
  `throttled=0x0`. O arquivo bruto dessa rodada nao foi copiado para esta
  pasta.

A primeira tentativa ficou em `descartadas/`: apesar da media de 1783,201 ms,
o log `vcgencmd-tentativa1.txt` registrou `0x50005` as 15:20:29 e o historico
`0x50000` depois. Essa tentativa nao entra na comparacao de desempenho.

Este controle serve de base para as rodadas assincronas em
`resultados/pi3-71b384e-video-async/`. Ele nao aprova o P5 por si so.
