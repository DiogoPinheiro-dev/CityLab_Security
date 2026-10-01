# Rodada com gesto e a pose em 416, commit c1f1ac0

Feita em 01/10/2026, das 16:55 as 17:47 pelo relogio do PC, com a API da serie
`../pi3-c1f1ac0-pose416/`: commit `c1f1ac0` e `POSE_IMGSZ=416` no `.env`. Mesmo
coletor e mesmas nove situacoes de `../pi3-338ac06-gestos/`, a rodada de 30/09
com a pose em 640: 6 frames por situacao e uma conexao por situacao. Nas duas
poses de braco, o braco foi esticado para o lado, como na r2 daquela pasta.
Resultados e leitura em `docs/PLANO_OTIMIZACAO.md`, "Estado verificado em
01/10/2026".

- `rodada-gestos-r1.json`: sondagem e as nove situacoes.

O JSON guarda, por frame, os alertas de cada pessoa pelo nome, a contagem de
pessoas e de rostos, caixas, tempos e metricas, e os eventos novos de `/logs`
so com tipo e horario. Nao guarda imagem nem nome de pessoa.

O relogio do Pi estava 39 s adiantado em relacao ao PC, pelos eventos e pelo
cabecalho `Date`.
