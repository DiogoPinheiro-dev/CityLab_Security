# Pose em NCNN, video de carga, commit d974a3f

Feita em 04/10/2026, das 16:10 as 16:41 pelo relogio do PC, com o mesmo codigo
e configuracao da base `pi3-d974a3f-video-pt/` mais
`POSE_MODEL_PATH=App/GestureRecon/yolov8n-pose_ncnn_model` no `.env`. A pasta
e o export de 03/10/2026 em 320x416, copiada do PC e conferida por hash
(`model.ncnn.param` `aaa7f61e...`, `model.ncnn.bin` `d74d660a...`), com o
pacote `ncnn==1.0.20260526` no ambiente da API. Na r1, o log da API registrou
"Loading ... yolov8n-pose_ncnn_model for NCNN inference". Resultados e leitura
em `docs/PLANO_OTIMIZACAO.md`, "Estado verificado em 04/10/2026".

- `uma-pessoa-r1.json` a `r3.json`: mesmo video e protocolo da base, API
  reiniciada antes de cada rodada. Media de 2876,4, 2923,1 e 2872,1 ms, contra
  3737,6 ms da base, com os mesmos alertas frame a frame.
- As rodadas sairam com quedas de tensao: os tempos incluem os trechos com o
  processador em 600 MHz, abaixo.

Log de `vcgencmd` a cada 5 s em `~/vcgencmd-ncnn.txt` no Pi, colado do Pi, de
15:10:34 as 17:04:16 pelo relogio do Pi, cerca de 40 s a frente do PC. Ate a
primeira rodada do NCNN, `throttled=0x0` em todas as leituras. Trechos das
rodadas:

- 16:11:27 a 16:13:19: r1, 14 de 23 leituras com `0x50005`, ate 52,1 C.
- 16:31:02 a 16:32:48: r2, 9 de 22 leituras com `0x50005`, ate 52,1 C.
- 16:39:39 a 16:41:25: r3, 9 de 22 leituras com `0x50005`, ate 52,1 C.

`0x50005` e tensao baixa naquele instante e processador limitado; em todas
essas leituras ele estava em 600 MHz. Fora delas, `0x50000`: o registro de que
ja houve tensao baixa, que so zera quando o Pi reinicia. A subida da API, entre
as rodadas, nao derrubou a tensao.
