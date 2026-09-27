# Evidencia: limite de temperatura com ar-condicionado na sala

`vcgencmd-duas-pessoas-r1.txt` foi coletado no Pi em 27/09/2026, a cada 5 s,
durante a sondagem e a rodada `../duas-pessoas-r1.json` (commit `6f5756d`, duas
pessoas). O ar da sala estava em 21 C, com ventilacao maxima e sem vento direto
no Pi, ligado cerca de 20 min antes. Horarios do relogio do Pi.

- 17:48:31 a 17:49:27: Pi parado entre 39,2 e 39,7 C. Sem ar, ficava em 42 a
  43 C.
- 17:49:32 a 17:49:52: sondagem de 3 frames.
- 17:50:32: a rodada comeca, 1,4 GHz e 42,9 C.
- 17:52:58, 146 s depois: primeiro `throttled=0x80008`. Sem ar foram 70 s.
- Dali ate 17:53:58, fim da rodada: limite ativo em 6 de 13 leituras (46%),
  entre 58,0 e 60,1 C. Sem ar foram 76%.

Frames que terminaram antes do limite: mediana de 5806,0 ms, igual a rodada sem
ar. Frames inteiros no periodo limitado: 5983,7 ms (+3,1%).

Eventos gravados no MongoDB durante a rodada: 33 `ALUNO`, 2 `NAO_ALUNO` e 2
`ALERTA_GESTO`, a abertura do episodio de cada track.
