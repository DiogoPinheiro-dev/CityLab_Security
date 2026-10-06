# Controle sincrono depois da troca de tomada

Rodada feita em 05/10/2026 com o mesmo Raspberry Pi, a mesma fonte, o mesmo
video e a mesma configuracao da serie
`resultados/pi3-9ce1b8c-video-rechecagem/`, com
`FACE_ASYNC_RECOGNITION=0` e `FACE_LEARN_FROM_STREAM=0`. A unica mudanca
eletrica foi ligar a fonte em outra tomada. O video tem SHA-256
`b0ef593552469c93881cfa6bb1b0a6f7715f3f4452b0f34e90bd787a887b0431`.

- `uma-pessoa-r1.json`: 5 frames de aquecimento e 30 medidos, com a API
  reiniciada antes da rodada.
- Media de 1738,865 ms, contra 1739,290 ms na r1 da base: -0,024%. Mediana de
  2324,197 ms contra 2355,568 ms; p95 de 2511,778 ms contra 2440,469 ms. As
  diferencas ficam dentro da variacao normal da serie.
- Pessoa, rosto e gesto em 30/30; identidade conhecida em 26/30; os mesmos 11
  alertas. Contagens, alertas, largura do rosto e confiancas de pessoa, gesto e
  rosto ficaram identicos quadro a quadro.
- O resumo do log `~/vcgencmd-tomada2.log`, mantido no Pi, teve 86 de 86
  leituras em `throttled=0x0`. Temperatura de 45,1 a 50,5 C na resposta da API.

Conclusao limitada ao que o controle demonstra: com a opcao assincrona
desligada e mudando so a tomada, o mesmo caminho sincrono voltou exatamente ao
desempenho e ao resultado da base, sem queda de tensao. Uma rodada nao compara
candidatos de desempenho; ela registra a condicao eletrica daquele momento.

Revisao em 06/10/2026: a queda voltou nessa mesma tomada, com a mesma fonte e o
mesmo cabo, nos dois modos (`resultados/pi3-71b384e-video-controle-sync/` e
`resultados/pi3-71b384e-video-async/`, pastas `descartadas`). A troca de
tomada nao isolou a causa entre tomada, fonte, cabo e contato.
