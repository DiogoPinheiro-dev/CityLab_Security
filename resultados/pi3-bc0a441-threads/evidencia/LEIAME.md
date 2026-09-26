# Evidencia: o Pi baixa o clock por temperatura no meio da rodada

`vcgencmd-duas-pessoas-r2.txt` foi coletado no Pi em 26/09/2026, a cada 5 s,
durante a rodada `duas-pessoas-r2.json` desta pasta (commit `bc0a441`, pose com 3
threads). Os horarios sao do relogio do Pi, cerca de 35 s a frente do PC: o
JSON registra o inicio da rodada as 17:49:02 e o fim as 17:52:43, horario do PC.

- 17:47 a 17:49: API reiniciando e carregando modelos, 43 a 46 C.
- 17:49:39: a rodada comeca, 1,4 GHz e 45,1 C.
- 17:50:54, 75 s depois e a 59,1 C: primeiro `throttled=0x80008`. O bit `0x8`
  quer dizer limite de temperatura ativo naquele instante, e o clock cai para
  1,2 GHz.
- De 17:50:54 a 17:53:10: o firmware alterna entre 1,2 e 1,4 GHz para segurar a
  temperatura entre 58,5 e 60,7 C. O limite aparece ativo em 15 de 28 leituras,
  com 1,2 GHz em 13 delas.
- 17:53:15: a rodada termina e o clock desce para 600 MHz, o repouso.

Leitura: no Pi 3 B+ o `temp_soft_limit` padrao e 60 C, e ao atingi-lo o clock
cai de 1,4 para 1,2 GHz. Com o processador metade do tempo a 1,2 GHz, a media
fica perto de 1,3 GHz, uns 7% mais lenta. E isso que aparece nas rodadas desta
pasta: os frames medidos a 59 C ou mais sao de 3% a 7% mais lentos que os
anteriores ao patamar de temperatura.

Uma leitura feita depois da rodada nao serve para isso: o bit `0x80000` fica
gravado desde o boot, mas o clock ja esta em repouso e a temperatura ja caiu.
Por isso o log precisa rodar durante a medicao.
