# Detector de rosto det_500m, video de carga, commit d0cb616

Feita em 04/10/2026, das 19:30 as 19:42 pelo relogio do PC, no codigo
`d0cb616` conferido por hash (`App/FaceRecon/service.py`), com a configuracao
da serie `pi3-50c0ddc-video-ncnn2/` (pose em NCNN com 2 threads, aprendizado
desligado) mais `FACE_DETECTOR_PATH=~/.insightface/models/det_500m.onnx` no
`.env`, conferido com `--show-config`. O arquivo e o detector do `buffalo_sc`
dos releases v0.7 do InsightFace, copiado do PC (sha256 `5e4447f5...`); o
reconhecedor segue o `w600k_r50` do `buffalo_l`. Base: a serie
`pi3-50c0ddc-video-ncnn2/`. Resultados e leitura em
`docs/PLANO_OTIMIZACAO.md`, "Estado verificado em 04/10/2026".

- `uma-pessoa-r1.json` a `r3.json`: video de carga, 5 frames de aquecimento e
  30 medidos, API reiniciada antes de cada rodada. Media de 1625,0, 1627,3 e
  1636,5 ms, contra 2237,0 a 2258,0 ms da base, com os mesmos alertas frame a
  frame. A deteccao do rosto caiu de 1356 para uns 385 ms.
- Rosto reconhecido em 24 de 30 frames, contra 26 na base, nas tres rodadas:
  os frames 30 e 31 herdaram pelo reuso o "NAO ALUNO" do rosto virado do
  frame 28, em vez de reconhecer de novo. A regra que reconhece de novo o
  desconhecido quase reconhecido entrou depois desta serie.
- Primeiro deploy com `tools/atualizar_dependencias_rpi.sh`: instalou o psutil
  7.0.0 no `.venv` da API e gravou a marca `7ed62d5b...`.

Log de `vcgencmd` a cada 5 s em `~/vcgencmd-det500m.txt` no Pi, colado do Pi,
de 19:28:29 as 19:56:09 pelo relogio do Pi, cerca de 40 s a frente do PC:
`throttled=0x0` em todas as leituras e no `get_throttled` depois da serie,
nenhuma leitura em 1,2 GHz, maxima de 52,6 C. Trechos a 1,4 GHz:

- 19:31:13 a 19:32:19: r1, ate 52,1 C.
- 19:37:27 a 19:38:27: r2, ate 52,1 C.
- 19:41:53 a 19:42:54: r3, ate 52,6 C.
- Os trechos de 19:28 a 19:30, 19:34 a 19:36 e 19:39 a 19:40 sao a API
  subindo.
