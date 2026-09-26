# Memoria do sistema e o runner do GitHub Actions

Coleta de 26/09/2026, commit `bc0a441`, para saber quanto do swap visto nas
rodadas de duas pessoas era do runner self-hosted.

- `duas-pessoas-runner-ligado.json`: rodada de duas pessoas com o runner
  ligado, protocolo normal. Mediana de 6117,0 ms, dentro da serie de
  `resultados/pi3-bc0a441-threads/` (6159,8 a 6182,2 ms). `process_rss_mb` da
  API entre 695,2 e 701,5 MB.
- `vcgencmd-runner-ligado.txt`: log de clock e temperatura durante essa rodada.
  Segunda confirmacao do limite de temperatura: primeiro `0x80008` com 60,1 C e
  1,2 GHz em 13 das 25 leituras dali ate o fim.
- `memoria-runner.txt`: memoria do sistema com a API desligada, antes e 5 s
  depois de parar o runner.

| | memoria disponivel | swap usado | runner |
|---|---|---|---|
| runner ligado | 787 MB | 104 de 511 MB | 7 MB em RAM e 40 MB em swap |
| runner parado | 785 MB | 32 de 511 MB | - |

Leitura: o runner ocioso fica quase todo no swap e ocupa 7 MB de RAM. Parar o
runner liberou 72 MB de swap e nada de memoria disponivel. O swap visto durante
as rodadas vem da propria API, com cerca de 700 MB residentes numa placa de
906 MB; o runner nao muda isso. A rodada com a API reiniciada sem o runner nao
foi feita porque nao ha o que ela pudesse mostrar.
