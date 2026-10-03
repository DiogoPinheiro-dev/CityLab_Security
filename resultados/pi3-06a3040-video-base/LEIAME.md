# Video de carga com o perfil atual, commit 06a3040

Feita em 03/10/2026, das 12:46 as 13:04 pelo relogio do PC, com o commit
`06a3040` no Pi, conferido por hash em cinco arquivos, e so o perfil rpi3 no
`.env`: pose em 416, 2 threads no rosto, 3 no PyTorch, reuso de 15 s, gate de
movimento e `FACE_EMBED_FULL_FRAME` desligado, conferido com `--show-config`.
E a base da comparacao com o embedding no frame inteiro, na pasta
`pi3-06a3040-video-fullframe/`, e a primeira serie com os modelos numa thread
de inferencia. Resultados e leitura em `docs/PLANO_OTIMIZACAO.md`, "Estado
verificado em 03/10/2026".

- `uma-pessoa-r1.json` a `r3.json`: `tools/benchmark_stream.py` com o video de
  carga, os 36 frames de 640x480 das seis situacoes de validacao, um a cada
  5 s, o mesmo das series de 01/10. 5 frames de aquecimento e 30 medidos, com
  a API reiniciada antes de cada rodada. Os JSONs guardam so o SHA-256 do video.

Log de `vcgencmd` a cada 5 s, colado do Pi, das 12:26:06 as 13:05:49 pelo
relogio do Pi, cerca de 40 s a frente do PC. `throttled=0x0` em todas as
leituras e nenhuma em 1,2 GHz. Trechos a 1,4 GHz:

- 12:42:32 a 12:44:26, 12:53:09 a 12:54:49 e 13:00:32 a 13:02:13: a API subindo.
- 12:46:57 a 12:49:07: r1, ate 58,0 C.
- 12:57:51 a 13:00:07: r2, ate 58,0 C.
- 13:02:53 a 13:05:04: r3, ate 59,6 C.
