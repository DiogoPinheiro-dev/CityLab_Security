# Detector det_500m com a rechecagem do desconhecido, video de carga, commit 9ce1b8c

Feita em 04/10/2026, das 20:07 as 20:22 pelo relogio do PC, no codigo
`9ce1b8c` conferido por hash (`App/FaceRecon/service.py`), com a mesma
configuracao da serie `pi3-d0cb616-video-det500m/`: pose em NCNN com 2
threads, aprendizado desligado e `FACE_DETECTOR_PATH` com o `det_500m`. A
diferenca e a regra nova no reuso: um "NAO ALUNO" com semelhanca de 0,30 ate o
limite e reconhecido de novo no frame seguinte, em vez de herdar o nome. Base:
a serie `pi3-50c0ddc-video-ncnn2/`. Resultados e leitura em
`docs/PLANO_OTIMIZACAO.md`, "Estado verificado em 04/10/2026".

- `uma-pessoa-r1.json` a `r3.json`: video de carga, 5 frames de aquecimento e
  30 medidos. A r1 rodou logo depois do deploy, que reiniciou a API; antes da
  r2 e da r3 a API foi reiniciada a mao. Media de 1739,3, 1732,2 e 1733,7 ms,
  contra 2237,0 a 2258,0 ms da base, com os mesmos alertas frame a frame e o
  rosto reconhecido nos mesmos 26 frames da base.

Log de `vcgencmd` a cada 5 s em `~/vcgencmd-rechecagem.txt` no Pi, resumido
la, de 20:07:27 as 20:25:15 pelo relogio do Pi, cerca de 40 s a frente do PC:
`throttled=0x0` nas 211 leituras, nenhuma em 1,2 GHz, maxima de 52,6 C. Trechos
a 1,4 GHz:

- 20:07:58 a 20:09:09: r1, ate 51,5 C.
- 20:14:39 a 20:15:44: r2, ate 52,6 C.
- 20:21:38 a 20:22:34: r3, ate 52,6 C.
- Os trechos curtos de 20:12 a 20:14 e de 20:17 a 20:19 sao a API subindo.
