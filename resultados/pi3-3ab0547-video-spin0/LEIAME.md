# Spinning do ONNX Runtime desligado, video fixo de uma pessoa, commit 3ab0547

Feita em 01/10/2026 as 22:34 pelo relogio do PC, com o commit `3ab0547`
implantado e conferido por hash no Pi: perfil rpi3 com a pose em 416, 2 threads
no rosto, 3 no PyTorch e reuso de 15 s vindo do perfil, mais
`ONNX_ALLOW_SPINNING=0` no `.env`, conferido com `--show-config`. ONNX Runtime
1.23.2 no Pi. A base e a serie `../pi3-419ca7f-video-reuso15/`, com a mesma
configuracao e o spinning ligado. Resultados e leitura em
`docs/PLANO_OTIMIZACAO.md`, "Estado verificado em 01/10/2026".

- `uma-pessoa-r1.json`: mesma carga em video, 5 frames de aquecimento e 30
  medidos, API reiniciada antes. A media ficou igual a da base, entao as
  rodadas 2 e 3 nao foram feitas.

O log de `vcgencmd` rodou no Pi, mas nao foi coletado; a temperatura de cada
frame esta no `temperature_c` da API.
