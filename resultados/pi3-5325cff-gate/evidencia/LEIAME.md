# Evidencia: gate de movimento cega pessoa parada

Rodada coletada em 21/09/2026 com o rotulo de cena vazia, mas havia uma pessoa
no enquadramento. Nao serve como medicao de cena vazia; serve como a prova do
defeito do gate na sua primeira versao.

- Rosto detectado em 30 de 30 frames: a pessoa estava presente o tempo todo.
- A pose rodou em 8 frames e devolveu caixa de pessoa entre 0,867 e 0,897, ou
  seja, deteccao real e forte, sem fantasma.
- Nos outros 22 frames o gate pulou a pose, porque pessoa sentada parada gera
  `motion_ratio` entre 0,00000 e 0,0017, abaixo do limiar de 0,002.
- Resultado: em 73% dos frames a pessoa nao existia para o pipeline de pose.
  Nenhuma caixa, nenhum track, nenhuma analise de gesto.
- O teto de tempo funcionou: o frame 31 rodou a pose com `motion_ratio` de
  0,00021, forcado por `GESTURE_MOTION_MAX_SKIP_SECONDS`.

Conclusao: decidir so por movimento e insuficiente. O gate precisa lembrar se a
cena ja estava ocupada, senao alguem parado fica invisivel para gesto por ate
30 segundos.
