# Aprendizado desligado, video de carga, commit cb972f7

Feita em 03/10/2026 as 17:48 pelo relogio do PC, no codigo `cb972f7`
conferido por hash, com o perfil rpi3 e o aprendizado pelo stream desligado,
conferido com `--show-config`; `tools/limpar_aprendidos.py` mostrou o banco sem
referencias. Base da serie `pi3-cb972f7-video-aprende/`. Resultados e leitura
em `docs/PLANO_OTIMIZACAO.md`, "Estado verificado em 03/10/2026".

- `uma-pessoa-r1.json`: video de carga, 5 frames de aquecimento e 30 medidos,
  API reiniciada antes. Uma rodada so: com o aprendizado desligado, o rosto
  roda como na serie `pi3-06a3040-video-fullframe/`, e as semelhancas repetem
  as dela.

Log de `vcgencmd` a cada 5 s em `~/vcgencmd-aprende.txt` no Pi, resumido la,
de 17:32:04 as 18:27:45 pelo relogio do Pi, cerca de 40 s a frente do PC:
`throttled=0x0` nas 662 leituras, nenhuma em 1,2 GHz, maxima de 59,6 C. A
rodada foi o trecho a 1,4 GHz de 17:49:04 a 17:51:20, ate 59,6 C; os trechos
curtos de 17:32 a 17:34 e de 18:08 a 18:11 devem ser a API subindo antes de
cada serie.
