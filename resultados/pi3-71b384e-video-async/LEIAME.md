# Reconhecimento facial assincrono no Pi, duas rodadas validas

Feito em 06/10/2026 com o commit `71b384e` implantado e conferido por hash em
`App/FaceRecon/service.py` e `App/settings.py`. Mesmos video, configuracao e
protocolo do controle em `resultados/pi3-71b384e-video-controle-sync/`, com
`FACE_ASYNC_RECOGNITION=1` e `FACE_LEARN_FROM_STREAM=0`. O video tem SHA-256
`b0ef593552469c93881cfa6bb1b0a6f7715f3f4452b0f34e90bd787a887b0431`.

| Rodada | Media | Contra o controle | Mediana | p95 | FPS |
|---|---:|---:|---:|---:|---:|
| r1 | 1177,602 ms | -31,9% | 1224,585 ms | 1362,822 ms | 0,848 |
| r2 | 1175,393 ms | -32,0% | 1147,676 ms | 1462,026 ms | 0,849 |

- Pessoa, rosto e gesto em 30/30 frames nas duas rodadas. Em ambas, o nome
  ficou confirmado em 20/30, pendente em 10/30 e desconhecido em nenhum. Os
  frames pendentes foram 16 a 19, 24 a 26 e 31 a 33.
- Houve 11 embeddings por rodada. A mediana do embedding em segundo plano foi
  2653,6 ms na r1 e 2672,0 ms na r2, equivalente a 2 ou 3 frames.
- Sete alertas nas duas rodadas: braco estendido no frame 17, mao fechada nos
  frames 22 e 23 e rendicao nos frames 26 a 29. Contra o controle, so a mao
  fechada comecou um frame depois; com a duracao minima de 4 s, o frame mais
  rapido adia o alerta em numero de frames, nao em tempo.
- Na r1, o log acompanhado durante a serie teve 87 leituras em
  `throttled=0x0`. Na r2, o responsavel viu o log todo em `0x0`, mas o arquivo
  foi perdido. Os logs validos nao foram copiados para esta pasta.

As duas tentativas de r3 estao em `descartadas/`. As duas repetiram 20 frames
confirmados, 10 pendentes e nenhum desconhecido, mas nao valem para desempenho:

- `uma-pessoa-r3-subtensao.json`, media de 1880,447 ms: durante a carga,
  `vcgencmd-r3.txt` registrou 10 leituras em `0x50005` e clock de 600 MHz.
- `uma-pessoa-r3-subtensao-2.json`, media de 1599,239 ms:
  `vcgencmd-r3-2.txt` registrou `0x50005` e clock de 600 MHz depois de a
  rodada comecar.

Conclusao: o ganho passou de 5% nas duas rodadas validas, mas a terceira foi
impedida pela alimentacao. O P5 segue sem aprovacao. Alem da r3 valida, falta a
decisao do responsavel sobre os alertas e sobre 20 nomes confirmados em 30,
contra 26 no controle. O coletor remove nomes por privacidade: ele confirma
que havia uma identidade cadastrada, mas nao prova sozinho qual nome saiu.
