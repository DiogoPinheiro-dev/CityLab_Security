# Rosto com 1 thread, video fixo de uma pessoa, commit 89c93b8

Feita em 01/10/2026, das 18:55 as 19:17 pelo relogio do PC, com o commit
`89c93b8` no Pi: perfil rpi3, pose em 416 e `ONNX_INTRA_OP_THREADS` tirado do
`.env`, ou seja, 1 thread no rosto. O tempo do rosto, igual ao da serie de 416
com webcam, confirma a thread unica. E a base da
serie `../pi3-89c93b8-video-onnx2/`. Resultados e leitura em
`docs/PLANO_OTIMIZACAO.md`, "Estado verificado em 01/10/2026".

- `uma-pessoa-r1.json` a `r3.json`: `tools/benchmark_stream.py` no modo de
  video, 5 frames de aquecimento e 30 medidos, API reiniciada antes de cada
  rodada.

A carga e um video de 36 frames de 640x480, montado com 6 frames de cada video
r1 das seis situacoes de validacao de `docs/PLANO_GESTOS.md`, um a cada 5 s
(SHA-256 `b0ef593552469c93881cfa6bb1b0a6f7715f3f4452b0f34e90bd787a887b0431`,
registrado em cada JSON). O video tem o rosto de quem gravou e fica fora do
repositorio. Os JSONs guardam contagens, confiancas, a semelhanca de cada rosto
com o cadastro e quantos foram reconhecidos, sem nome nem imagem.

O log de `vcgencmd` rodou junto no Pi, mas o resumo nao foi coletado; a
temperatura de cada frame esta no `temperature_c` da API.
