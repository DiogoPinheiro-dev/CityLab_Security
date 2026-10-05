# Reconhecimento facial assincrono - primeira rodada controlada

Rodada feita em 05/10/2026 com o commit `0ac0bf0` implantado no Raspberry Pi,
o video de carga e a mesma configuracao do controle sincrono
`resultados/pi3-ef3c0b5-video-controle-sync-outlet2/`, exceto por
`FACE_ASYNC_RECOGNITION=1`. `FACE_LEARN_FROM_STREAM=0`. O video tem SHA-256
`b0ef593552469c93881cfa6bb1b0a6f7715f3f4452b0f34e90bd787a887b0431`.

- `uma-pessoa-r1.json`: 5 frames de aquecimento e 30 medidos, com a API
  reiniciada pelo deploy antes da rodada.
- Media de 1212,636 ms, mediana de 1222,592 ms, p95 de 1495,908 ms e 0,823
  FPS. Contra os 1738,865 ms do controle sincrono, a media caiu 30,263%.
- Pessoa, rosto e gesto em 30/30. Os mesmos 11 alertas do controle, nas mesmas
  regras e nos mesmos frames; contagens, larguras do rosto e confiancas de
  pessoa e gesto tambem ficaram identicas quadro a quadro.
- Identidade conhecida em 2/30, contra 26/30 no controle; 27 frames ficaram
  pendentes e 1, desconhecido. Houve 12 embeddings concluidos. O resultado
  mostra que a confirmacao conservadora nao acompanhou o rosto em movimento.
- Temperatura de 46,7 a 52,1 C na resposta da API e RSS de ate 672 MB.
- O log `~/vcgencmd-async-0ac0bf0.log`, mantido no Pi, cobriu tambem a rodada:
  253 de 253 leituras em `throttled=0x0`. O nome inicial incorreto do arquivo,
  com o hash anterior, foi corrigido depois da medicao; isso nao muda o
  conteudo.

O coletor remove nomes por privacidade. Ele confirma apenas que dois frames
tiveram alguma identidade cadastrada; nao permite afirmar que era o nome
correto.

Conclusao: candidato reprovado por perda de recall facial. O ganho de tempo nao
e promocao valida, e as rodadas 2 e 3 nao foram feitas porque a falha funcional
ja estava demonstrada. O resultado deve ser usado como evidencia negativa do
experimento, nao como desempenho aprovado.
