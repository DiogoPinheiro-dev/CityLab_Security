# Reuso do rosto em 15 s, video fixo de uma pessoa, commit 419ca7f

Feita em 01/10/2026, das 21:32 as 22:03 pelo relogio do PC, com o mesmo codigo,
video e protocolo de `../pi3-419ca7f-video-reuso0/`, trocando so o `.env` do Pi:
`FACE_REUSE_SECONDS=15`. A r1 confirmou o reuso pela metrica `face_reused`.
Resultados e leitura em `docs/PLANO_OTIMIZACAO.md`, "Estado verificado em
01/10/2026". Depois desta serie, 15 s viraram o padrao do perfil rpi3.

- `uma-pessoa-r1.json` a `r3.json`: 5 frames de aquecimento e 30 medidos, API
  reiniciada antes de cada rodada. O nome foi reaproveitado nos mesmos 16 dos
  30 frames nas tres rodadas.

O log de `vcgencmd` continuou rodando no Pi, mas desta serie nao foi coletado;
a temperatura de cada frame esta no `temperature_c` da API.
