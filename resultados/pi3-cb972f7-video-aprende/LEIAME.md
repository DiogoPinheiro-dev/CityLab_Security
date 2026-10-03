# Aprendizado pelo stream, video de carga, commit cb972f7

Feita em 03/10/2026, das 18:19 as 19:18 pelo relogio do PC, com o mesmo codigo
e configuracao da base `pi3-cb972f7-video-aprende-base/` mais
`FACE_LEARN_FROM_STREAM=1` e `FACE_LEARN_INTERVAL_SECONDS=0` no `.env`,
conferido com `--show-config`. Resultados e leitura em
`docs/PLANO_OTIMIZACAO.md`, "Estado verificado em 03/10/2026".

- `uma-pessoa-r1.json` a `r3.json`: mesmo video e protocolo da base, API
  reiniciada antes de cada rodada. O banco estava vazio antes da r1 e nao foi
  limpo entre as rodadas, de proposito: `tools/limpar_aprendidos.py` mostrou 5
  referencias antes da r2 e da r3, e o log da API registrou "5 referencias
  aprendidas carregadas" nas duas subidas. As rodadas nao sao independentes.
- Na r2 e na r3, um frame bate 1,000 com a propria copia, guardada na rodada
  anterior, porque o video e o mesmo. Nao e ganho de reconhecimento: e o
  defeito que o filtro de copia corrige.

Log de `vcgencmd`, o mesmo da base: a r1 foi o trecho a 1,4 GHz de 18:20:42 a
18:22:53 pelo relogio do Pi, ate 59,1 C, com `throttled=0x0`. O log parou as
18:27:45, antes da r2 e da r3; nelas, a temperatura que o proprio benchmark
registra ficou entre 51,5 e 59,1 C.
