# Embedding no frame inteiro, video de carga, commit 06a3040

Feita em 03/10/2026, das 13:07 as 13:53 pelo relogio do PC, com o mesmo codigo
e configuracao da serie `pi3-06a3040-video-base/` mais `FACE_EMBED_FULL_FRAME=1`
no `.env`, conferido com `--show-config`: a unica diferenca entre as duas
series. Resultados e leitura em `docs/PLANO_OTIMIZACAO.md`, "Estado verificado
em 03/10/2026".

- `uma-pessoa-r1.json` a `r3.json`: mesmo video de carga e mesmo protocolo da
  base, 5 frames de aquecimento e 30 medidos, API reiniciada antes de cada
  rodada. Entre a r2 e a r3 a API ficou uns 30 min no ar sem stream, e foi
  reiniciada antes da r3 como nas outras.

Log de `vcgencmd` a cada 5 s, colado do Pi, das 13:06:31 as 14:00:29 pelo
relogio do Pi, cerca de 40 s a frente do PC. `throttled=0x0` em todas as
leituras e nenhuma em 1,2 GHz. Trechos a 1,4 GHz:

- 13:06:31 a 13:08:02, 13:11:19 a 13:13:08 e 13:42:19 a 13:44:07: a API subindo.
- 13:08:17 a 13:10:29: r1, ate 60,1 C.
- 13:13:18 a 13:15:33: r2, ate 60,7 C.
- 13:51:40 a 13:53:56: r3, ate 58,5 C.
