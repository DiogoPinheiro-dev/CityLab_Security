# Pose com 2 threads do PyTorch, video fixo de uma pessoa, commit 419ca7f

Feita em 01/10/2026 as 22:17 pelo relogio do PC, com o mesmo codigo, video e
protocolo da serie `../pi3-419ca7f-video-reuso15/`, que e a base, trocando so o
`.env` do Pi: `TORCH_NUM_THREADS=2`. A pose rodou com 2 threads em todos os
frames, pela metrica `gesture_torch_threads`. Resultados e leitura em
`docs/PLANO_OTIMIZACAO.md`, "Estado verificado em 01/10/2026".

- `uma-pessoa-r1.json`: 5 frames de aquecimento e 30 medidos, API reiniciada
  antes. A media ficou 2,4% a 3,2% abaixo da base, menos que os 5% do criterio,
  entao as rodadas 2 e 3 nao foram feitas: nao mudariam a conclusao.
- `descartadas/`: uma tentativa anterior em que a pose ainda rodou com 3
  threads.
