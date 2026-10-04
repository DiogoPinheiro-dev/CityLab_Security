# Pose em NCNN com 2 threads, video de carga, commit 50c0ddc

Feita em 04/10/2026, das 17:48 as 18:05 pelo relogio do PC, no codigo
`50c0ddc` conferido por hash (`App/GestureRecon/service.py` e
`App/settings.py`), com `FACE_LEARN_FROM_STREAM=0`,
`POSE_MODEL_PATH=App/GestureRecon/yolov8n-pose_ncnn_model` e
`NCNN_NUM_THREADS=2` no `.env`, conferidos com `--show-config`. Mesma pasta
exportada e mesmo pacote `ncnn` da serie `pi3-d974a3f-video-ncnn/`, que rodou
com as 4 threads do padrao. Base: `pi3-d974a3f-video-pt/`. Resultados e
leitura em `docs/PLANO_OTIMIZACAO.md`, "Estado verificado em 04/10/2026".

- `uma-pessoa-r1.json` a `r3.json`: video de carga, 5 frames de aquecimento e
  30 medidos, API reiniciada antes de cada rodada. Media de 2238,9, 2237,0 e
  2258,0 ms, contra 3737,6 ms da base, com os mesmos alertas frame a frame. A
  metrica `ncnn_threads` deu 2 em todos os frames.
- Fonte original do Pi, reiniciado as 17:18 pelo relogio do Pi, depois do
  teste com um carregador de celular. `vcgencmd get_throttled` deu `0x0` antes
  da serie e depois de cada rodada.

Log de `vcgencmd` a cada 5 s em `~/vcgencmd-ncnn2.txt` no Pi, resumido la, de
17:46:04 as 18:07:54 pelo relogio do Pi, cerca de 40 s a frente do PC:
`throttled=0x0` nas 257 leituras, nenhuma em 1,2 GHz, maxima de 55,3 C.
Trechos a 1,4 GHz:

- 17:49:15 a 17:50:41: r1, ate 54,8 C.
- 17:56:23 a 17:57:39: r2, ate 55,3 C.
- 18:04:38 a 18:06:04: r3, ate 54,8 C.
- Os trechos curtos de 17:46 a 17:48, 17:51 a 17:54 e 17:59 a 18:01 sao a API
  subindo.
