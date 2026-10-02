# Cena vazia com o perfil atual, webcam, commit af2872d

Feita em 02/10/2026, das 17:55 as 18:12 pelo relogio do PC, com o commit
`af2872d` no Pi e so o perfil rpi3 no `.env`: pose em 416, 2 threads no rosto,
3 no PyTorch, reuso de 15 s e gate de movimento. A API roda como servico; a r1
pegou o processo novo depois de o Pi ser ligado, e antes da r2 e da r3 o
servico foi reiniciado. Resultados e leitura em `docs/PLANO_OTIMIZACAO.md`,
"Estado verificado em 02/10/2026".

- `vazia-r1.json` a `r3.json`: `tools/benchmark_stream.py` com a webcam do PC
  apontada para um canto sem ninguem, 5 frames de aquecimento e 30 medidos.
  Antes de cada rodada, uma contagem de 20 s para sair do quadro e uma
  sondagem de 3 frames sem pessoa, rosto nem gesto.

Log de `vcgencmd` a cada 5 s, colado do Pi em dois trechos, das 17:52:02 as
18:09:03 e das 18:10:33 as 18:12:39 pelo relogio do Pi, cerca de 39 s a frente
do PC. `throttled=0x0` em todas as leituras. Trechos a 1,4 GHz:

- 17:52 a 17:54: o Pi ligando e a API carregando os modelos.
- 17:55:58 a 17:56:03 e 17:56:28 a 17:57:03: sondagem e r1, ate 49,4 C.
- 18:01:45 a 18:03:16: o servico reiniciando.
- 18:04:31 a 18:04:36 e 18:05:01 a 18:05:42: sondagem e r2, ate 51,5 C.
- 18:06:42 a 18:08:23: o servico reiniciando.
- 18:10:53 a 18:10:58 e 18:11:24 a 18:12:04: sondagem e r3, ate 51,0 C.
