# Pose no .pt, video de carga, commit d974a3f

Feita em 04/10/2026 as 15:25 pelo relogio do PC, no codigo `d974a3f`
conferido por hash, com o perfil rpi3 e `FACE_LEARN_FROM_STREAM=0` no `.env`,
conferido com `--show-config`: o aprendizado fica desligado para as rodadas
nao dependerem umas das outras. Base da serie `pi3-d974a3f-video-ncnn/`.
Resultados e leitura em `docs/PLANO_OTIMIZACAO.md`, "Estado verificado em
04/10/2026".

- `uma-pessoa-r1.json`: video de carga, 5 frames de aquecimento e 30 medidos,
  API reiniciada antes. Uma rodada so: confere que o ambiente da API voltou ao
  normal depois da instalacao do `ncnn`, que tinha trocado o numpy e o OpenCV.
  Media de 3737,6 ms, os mesmos 11 alertas pelo nome e as mesmas semelhancas do
  rosto da serie `pi3-06a3040-video-fullframe/`.

Log de `vcgencmd` a cada 5 s em `~/vcgencmd-ncnn.txt` no Pi, colado do Pi, o
mesmo da serie NCNN. A rodada foi o trecho a 1,4 GHz de 15:26:08 a 15:28:24
pelo relogio do Pi, cerca de 40 s a frente do PC, ate 57,5 C, com
`throttled=0x0` e o processador em 1,4 GHz em todas as leituras.
