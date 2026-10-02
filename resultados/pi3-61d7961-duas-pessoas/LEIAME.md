# Duas pessoas com o perfil atual, webcam, commit 61d7961

Feita em 02/10/2026, das 18:33 as 19:03 pelo relogio do PC, com o commit
`61d7961` no Pi, levado pelo deploy automatico, e so o perfil rpi3 no `.env`:
pose em 416, 2 threads no rosto, 3 no PyTorch, reuso de 15 s e gate de
movimento. A API roda como servico; a r1 pegou o processo subido pelo deploy,
pronto as 18:31, e antes da r2 e da r3 o servico foi reiniciado. Resultados e
leitura em `docs/PLANO_OTIMIZACAO.md`, "Estado verificado em 02/10/2026".

- `duas-pessoas-r1.json` a `r3.json`: `tools/benchmark_stream.py` com a webcam
  do PC e duas pessoas reais, a cena de setembro: uma de frente, com cadastro,
  e outra de lado, sem cadastro. 5 frames de aquecimento e 30 medidos. Antes de
  cada rodada, uma contagem de 10 s e uma sondagem de 3 frames que exige duas
  pessoas e um rosto.
- Os JSONs guardam contagens, confiancas e a semelhanca do rosto com o
  cadastro, sem nome nem imagem.

Sem log de `vcgencmd` nesta serie: o registro da cena vazia parou as 18:12:49
pelo relogio do Pi. No lugar dele, `vcgencmd get_throttled` devolveu
`throttled=0x0` as 19:37:08, com o Pi ligado desde 17:50:46 sem reiniciar
(`uptime -s`). Os bits 16 a 19 ficam marcados desde o boot quando algo
acontece, entao nao houve subtensao, corte de clock nem limite de temperatura
nas tres rodadas. A temperatura de cada frame esta em `temperature_c`, medida
pela API: de 50,5 a 56,9 C.
