# Teste de uma noite, uma pessoa em video fixo, commit 3ab0547

Feito de 01 para 02/10/2026, das 23:54 as 10:23 pelo relogio do PC, 10,48 h
numa conexao so, com o commit `3ab0547` no Pi e so o perfil rpi3 no `.env`:
pose em 416, 2 threads no rosto, 3 no PyTorch e reuso de 15 s. API iniciada
com `nohup`, fora da sessao SSH. Resultados e leitura em
`docs/PLANO_OTIMIZACAO.md`, "Estado verificado em 02/10/2026".

- `amostras.jsonl.gz`: as 10.000 amostras, uma por linha, com o horario UTC de
  cada uma (`at_utc`). `tools/benchmark_stream.py` com `--loop` sobre o video de
  carga de 36 frames de `../pi3-89c93b8-video-onnx1/` e `--samples-jsonl`, 5
  frames de aquecimento e 10.000 medidos.
- `uma-pessoa-noite-resumo.json`: o relatorio do benchmark sem a lista de
  amostras, que esta no arquivo acima. O relatorio completo tinha 13 MB.

Resumo do registro do Pi, `~/noite.txt`, uma leitura a cada 30 s com
`vcgencmd`, `free`, `vmstat` e a memoria da API em `/proc/<pid>/status`, das
23:52 as 10:48 pelo relogio do Pi. O arquivo ficou no Pi; este e o resumo
feito la com `awk`, com uma linha por hora:

```text
hora 00:45:52 throttled=0x0 temp=61.2'C disp=344 swap=298 si=0 so=0 api_rss=501 api_swap=210
hora 01:46:10 throttled=0x0 temp=62.3'C disp=327 swap=287 si=0 so=0 api_rss=503 api_swap=209
hora 02:46:27 throttled=0x0 temp=58.0'C disp=328 swap=286 si=0 so=0 api_rss=504 api_swap=209
hora 03:46:44 throttled=0x0 temp=59.1'C disp=322 swap=282 si=0 so=0 api_rss=507 api_swap=209
hora 04:47:00 throttled=0x0 temp=60.7'C disp=318 swap=282 si=0 so=0 api_rss=508 api_swap=208
hora 05:47:17 throttled=0x0 temp=62.3'C disp=322 swap=281 si=0 so=0 api_rss=507 api_swap=208
hora 06:47:35 throttled=0x0 temp=61.2'C disp=322 swap=281 si=0 so=0 api_rss=506 api_swap=209
hora 07:47:52 throttled=0x0 temp=61.2'C disp=317 swap=281 si=0 so=0 api_rss=506 api_swap=209
hora 08:48:08 throttled=0x0 temp=62.3'C disp=319 swap=281 si=0 so=0 api_rss=506 api_swap=209
hora 09:48:25 throttled=0x0 temp=62.3'C disp=316 swap=281 si=0 so=0 api_rss=508 api_swap=209
throttled 0x0 1318
1320 leituras | temp max 63.4 as 00:14:44 | disp min 189 as 23:54:38 | swap max 361 as 00:17:14
si>0 em 18 max 1940 | so>0 em 4 max 6512
api_rss min 500 max 661 | api_swap max 268
ultima 10:48:11 throttled=0x0 temp=39.2'C disp=315 swap=280 si=0 so=0 api_rss=508 api_swap=209
```

Memoria em MB; `si` e `so` em KiB/s. A primeira linha do arquivo era o aviso
do `nohup`, sem leitura. Um segundo registro rodou por engano nos primeiros
minutos e foi parado; as linhas duplicadas dele nao mudam o resumo.
