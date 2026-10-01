# Rosto com 2 threads, video fixo de uma pessoa, commit 89c93b8

Feita em 01/10/2026, das 19:22 as 19:43 pelo relogio do PC, com o mesmo codigo,
video e protocolo de `../pi3-89c93b8-video-onnx1/`, trocando so o `.env` do Pi:
`ONNX_INTRA_OP_THREADS=2`, conferido com `--show-config`. Resultados e leitura
em `docs/PLANO_OTIMIZACAO.md`, "Estado verificado em 01/10/2026". Depois desta
serie, 2 threads viraram o padrao do perfil rpi3.

- `uma-pessoa-r1.json` a `r3.json`: `tools/benchmark_stream.py` no modo de
  video, 5 frames de aquecimento e 30 medidos, API reiniciada antes de cada
  rodada. A rodada so comecou depois de o PC ver a API cair e voltar.

O log de `vcgencmd` rodou junto no Pi, mas o resumo nao foi coletado; a
temperatura de cada frame esta no `temperature_c` da API.
