# Uma pessoa com a pose em 640, commit c1f1ac0

Feita em 01/10/2026, com o commit `c1f1ac0` implantado e conferido por hash no
Pi, e `POSE_IMGSZ` fora do `.env`: a pose no padrao de 640 do Ultralytics. E a
serie de comparacao de `../pi3-c1f1ac0-pose416/`. Resultados e leitura em
`docs/PLANO_OTIMIZACAO.md`, "Estado verificado em 01/10/2026".

- `config-api.json`: saida de `tools/run_rpi.py --show-config` antes da serie.
  O `tls` aparece falso porque o comando rodou sem os certificados; a API subiu
  com eles.
- `uma-pessoa-r1.json` a `r3.json`: `tools/benchmark_stream.py` com a webcam do
  PC, 5 frames de aquecimento e 30 medidos, uma pessoa sentada de frente.

Protocolo: API reiniciada antes de cada rodada, sondagem de 3 frames antes da
rodada cheia, runner do GitHub Actions parado. Antes da r1, duas sondagens
pararam por cena fora do padrao, sem ninguem no quadro e com a caixa entre 0,45
e 0,66 sem rosto, e mandaram 3 frames cada a API.

Horario do PC, inicio e fim de cada rodada: r1 12:24:34 a 12:27:53, r2 12:34:47
a 12:38:06, r3 12:42:19 a 12:45:39. O relogio do Pi estava 39 s adiantado.

Log de `vcgencmd` a cada 5 s, das 12:14:21 as 17:47:46 pelo relogio do Pi,
cobrindo esta serie, a de 416 e a rodada com gesto. O log ficou no Pi; o resumo
dele, feito la com `awk`:

```text
throttled 0x0 3967
clock 1000 MHz 1
clock 600 MHz 3171
clock 700 MHz 323
clock 900 MHz 5
clock 1400 MHz 460
clock 800 MHz 7
de 12:14:21 ate 17:47:46 - 3967 leituras - maxima 63.9 C as 12:46:11 - limite ativo em 0
```

Nenhuma leitura com limite de temperatura ou subtensao, e o clock nunca em
1,2 GHz. A maxima veio no fim da r3 desta serie.
