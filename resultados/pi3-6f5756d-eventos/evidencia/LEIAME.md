# Evidencia: limite de temperatura e eventos na rodada sem ar-condicionado

`vcgencmd-duas-pessoas-r1.txt` foi coletado no Pi em 27/09/2026, a cada 5 s,
durante a sondagem e a rodada `../duas-pessoas-r1.json` (commit `6f5756d`, duas
pessoas, ar-condicionado da sala desligado). Horarios do relogio do Pi.

- 17:04:17 a 17:04:37: sondagem de 3 frames.
- 17:05:53: a rodada comeca, 1,4 GHz e 49,9 C.
- 17:07:03, 70 s depois: primeiro `throttled=0x80008`, limite ativo e 1,2 GHz.
- Dali ate 17:09:24, fim da rodada: limite ativo em 22 de 29 leituras (76%),
  entre 58,0 e 60,7 C.

Os frames que terminaram antes do limite tiveram mediana de 5795,2 ms; os que
rodaram inteiros depois dele, sem o frame com 2 rostos, 6169,3 ms (+6,5%).

Eventos gravados no MongoDB durante a rodada, contados pela diferenca de
`/logs` antes e depois dela: 34 `ALUNO`, 2 `NAO_ALUNO` e 5 `ALERTA_GESTO`, todos
do mesmo track. O primeiro saiu no segundo frame de aquecimento; os outros, nos
frames 23, 24, 28 e 29, quando o track passou de 1 para 2 alertas e voltou.
