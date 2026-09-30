# Rodada com gesto no Pi, commit 338ac06

Feita em 30/09/2026, com o commit `338ac06` implantado e conferido por hash no
Pi. Uma pessoa na frente da webcam do PC; um script local, rascunho fora do
Git, enviou os frames ao stream `wss://192.168.15.200:8000/stream`, um por vez,
6 por situacao e uma conexao por situacao. Resultados e leitura em
`docs/PLANO_GESTOS.md`, "Rodada com gesto no Pi".

- `config-api.json`: configuracao que a API imprimiu ao subir.
- `rodada-gestos-r1.json`: sondagem e as nove situacoes. Em ameaca e em braco
  estendido, o braco foi apontado para a camera.
- `rodada-gestos-r2.json`: ameaca e braco estendido de novo, com o braco
  esticado para o lado.
- `eventos-alerta-gesto-r1.jsonl`: os 8 eventos `ALERTA_GESTO` da r1, lidos do
  MongoDB no Pi sem a imagem, com `alertas`, `alertas_novos` e `evidencia`.

Os JSONs das rodadas guardam, por frame, os alertas de cada pessoa pelo nome,
a contagem de pessoas e de rostos, caixas, tempos e metricas, e os eventos
novos de `/logs` so com tipo e horario. Nao guardam imagem nem nome de pessoa.

O relogio do Pi estava de 36 a 38 s adiantado em relacao ao PC. O cabecalho
`Date` das respostas da API atrasa enquanto o stream roda, porque o uvicorn so
o atualiza com o loop livre; por isso os eventos foram casados com as
situacoes pelo deslocamento que poe todos dentro delas.
