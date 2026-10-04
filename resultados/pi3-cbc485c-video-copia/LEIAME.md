# Filtro de copia nas referencias aprendidas, video de carga, commit cbc485c

Feita em 03/10/2026, das 20:06 as 20:36 pelo relogio do PC, no codigo
`cbc485c` conferido por hash (`App/FaceRecon/service.py`), com a mesma
configuracao da serie `pi3-cb972f7-video-aprende/`: `FACE_LEARN_FROM_STREAM=1`
e `FACE_LEARN_INTERVAL_SECONDS=0` no `.env`, conferido com `--show-config`.
Resultados e leitura em `docs/PLANO_OTIMIZACAO.md`, "Estado verificado em
03/10/2026".

- `uma-pessoa-r1.json` a `r3.json`: video de carga, 5 frames de aquecimento e
  30 medidos, API reiniciada antes de cada rodada. O banco foi limpo so antes
  da r1 (`tools/limpar_aprendidos.py --tudo` apagou as 2 referencias que
  sobraram da serie anterior, e a API subiu com 0). Antes da r2 e da r3 havia
  5 referencias no banco, e a API carregou as 5: nenhuma copia entrou.
- Na r2 e na r3, dois frames batem 1,000 com a propria referencia, guardada na
  r1, e nao entram de novo; nenhum frame dessas rodadas gravou referencia.

Log de `vcgencmd` a cada 5 s em `~/vcgencmd-copia.txt` no Pi, resumido la, de
19:58:53 as 21:03:52 pelo relogio do Pi, cerca de 40 s a frente do PC:
`throttled=0x0` nas 599 leituras, nenhuma em 1,2 GHz, maxima de 60,7 C.
Trechos a 1,4 GHz:

- 20:07:39 a 20:09:55: r1, ate 59,6 C.
- 20:13:23 a 20:15:39: r2, ate 60,7 C.
- 20:34:30 a 20:36:46: r3, ate 59,6 C.
- Os trechos curtos de 19:59 a 20:00, 20:04 a 20:06, 20:10 a 20:12 e 20:30 a
  20:33 sao a API subindo, e o de 20:03 e a limpeza do banco.
