# Plano de correcao das regras de gesto

Documento separado de `docs/PLANO_OTIMIZACAO.md` de proposito. Aquele trata de
latencia e vazao; este trata de **comportamento do produto**: quando um alerta
dispara e o que ele significa. Sao problemas diferentes, e o segundo nao se
resolve com o primeiro.

Nada aqui esta autorizado a ser implementado. Cada acao e combinada com o
responsavel antes, uma de cada vez, como no plano de otimizacao.

## Estado verificado em 07/10/2026

### Triagem offline dos dois falsos positivos restantes

O responsavel autorizou medir hipoteses no replay, sem mudar o produto. A
versao `7ba3b97` reproduziu, a 1,7 s, os dois problemas registrados em 06/10:

- rendicao falsa em `36/51` fases dos videos de ameaca;
- mao fechada e ameaca falsas em `37/51` e `34/51` fases dos videos de braco
  estendido com a mao aberta.

Duas familias simples foram reprovadas nos conjuntos da webcam e do celular,
nos intervalos de 1,0, 1,7, 2,4, 3,4 e 6,2 s:

1. Exigir postura de rendicao nos dois lados reduziu o falso da webcam de
   `36/51` para `11/51`, sem perder a rendicao da webcam, mas tirou toda a
   rendicao do celular a 1,7 s (`17/17 -> 0/17`). Ignorar a mao levantada lida
   como fechada repetiu a perda ja conhecida: no celular, `17/17 -> 16/17` a
   1,7 s, `24/24 -> 19/24` a 2,4 s e `32/34 -> 21/34` a 3,4 s.
2. Uma grade de 72 combinacoes apertou quantidade de pontas compactas, media
   da distancia das pontas e abertura das pontas na regra lateral de punho.
   A regra atual foi a unica sem perda de acerto. A alteracao mais proxima,
   abertura `< 1,20` no lugar de `< 1,35`, reduziu os falsos agregados de 561
   para 525, mas perdeu punho verdadeiro no celular a 3,4 s (`28 -> 27`).

Nenhuma variante foi promovida. Os mesmos videos ja serviram para escolher
regras anteriores e nao podem ser o unico aceite de outra mudanca.

### Segundo conjunto independente

O responsavel gravou outros 18 videos em 07/10: tres de cada situacao, com 20 s
cada. A extracao usou o caminho do Pi, com escala 0,5, pose NCNN em 320x416 e
duas threads. Foram 3.600 frames, com a pessoa detectada nos 3.600. Um processo
paralelo do MediaPipe falhou em `braco_aberto_r3`; so esse video foi repetido
sequencialmente, com os 200 frames extraidos. Os videos continuam locais e os
JSONs guardam apenas keypoints e maos.

No replay a 1,7 s:

| Situacao | Esperado | Acerto | Falso |
|---|---|---:|---:|
| Ameaca | Mao fechada, ameaca e braco estendido | `51/51` nos tres | nenhum |
| Braco estendido, mao aberta | Braco estendido | `51/51` | mao fechada `1/51`; ameaca `16/51` |
| Rendicao | Rendicao | `51/51` | mao fechada `10/51` |
| Mao oculta | Mao oculta | `51/51` | nenhum |
| Neutro | nenhum | - | nenhum |
| Punho, braco solto | nenhum | - | nenhum |

A rendicao falsa da ameaca, `36/51` no conjunto anterior, nao apareceu no
novo (`0/51`). Endurecer a rendicao resolveria um resultado que nao se repetiu
e perderia o conjunto do celular, portanto continua reprovado.

A grade da regra lateral foi repetida com os tres conjuntos. A configuracao
atual voltou a ser a unica das 72 sem perda de acerto. Exigir quatro pontas
compactas reduziu os falsos agregados de 736 para 664, mas perdeu mao fechada
verdadeira na webcam anterior a 6,2 s (`185 -> 183`). Limitar a abertura a
`1,20` reduziu para 670, mas perdeu no celular a 3,4 s (`28 -> 27`).

Por fim, 40 combinacoes variaram as observacoes e a duracao da ameaca. Aumentar
a duracao de 0,22 para 1 s nao mudou nenhum resultado. Com 2 s, os falsos
agregados cairam de 308 para 291, mas a ameaca verdadeira do celular a 1,7 s
caiu de `17` para `6`. Exigir tres observacoes perdeu ainda mais acertos.

### Encerramento em 07/10/2026

O responsavel decidiu encerrar este plano sem mudar o produto. Os sinais
atuais nao separam a mao aberta do punho em todos os conjuntos sem perder
recall. Ficam registradas como limitacoes conhecidas a ameaca falsa em `16/51`
fases de um dos tres videos novos de braco estendido com a mao aberta, a mao
fechada falsa em `10/51` fases da rendicao e em `1/51` do braco estendido. A
rendicao falsa da ameaca, vista em `36/51` fases no conjunto anterior, nao se
repetiu no conjunto independente (`0/51`), portanto nao foi usada sozinha para
aprovar uma mudanca.

As grades avaliadas nao encontraram ajuste seguro de limiar, duracao ou numero
de observacoes. Corrigir as limitacoes exige outra entrada ou outro
classificador de mao, com dados rotulados proprios. Esse trabalho maior nao
faz parte deste encerramento e so reabre o plano se for combinado com o
responsavel.

## Estado verificado em 06/10/2026

O responsavel reabriu o plano como o ultimo item da ordem combinada no fim de
04/10/2026 (`docs/PLANO_OTIMIZACAO.md`, "Pedido do responsavel no fim de
04/10"). Antes de mudar qualquer regra, o conjunto de validacao de 30/09 foi
medido de novo, no ritmo atual do Pi.

### O frame do Pi ficou mais de 3 vezes mais rapido

A acao 4 deste plano aconteceu pelo plano de desempenho: com a pose em 416 px
e em NCNN e o detector de rosto `det_500m`. O frame com uma pessoa levou de 5,3
a 6,3 s na rodada com gesto de 30/09; no video de carga, desde 04/10, leva 1,7
s. As regras contam observacoes seguidas, e a duracao minima de cada uma, de
0,20 a 0,40 s, foi escolhida para 30 FPS. A 1,7 s, so as observacoes decidem:
cada alerta dispara mais de 3 vezes mais cedo, e condicoes que duram poucos
segundos passam a disparar tambem.

Medicao no PC, sem mudar o produto: os 18 videos do conjunto de validacao,
extraidos com a pose do Pi (NCNN em 320x416), com a regra de 30/09, em todas as
fases de amostragem a 6,2 s e a 1,7 s: 62 e 17 fases por video, 186 e 51 por
situacao (`.tmp/analisa_validacao_gestos.py --intervalo`, fora do Git). A 6,2
s, os JSONs extraidos em 30/09 reproduzem a tabela daquele dia. Parte das fases
em que o alerta dispara ao menos uma vez no video:

| Situacao | Alerta | 6,2 s | 1,7 s |
|---|---|---:|---:|
| Ameaca | Mao fechada (esperado) | 99% | 100% |
| Ameaca | Ameaca (esperado) | 26% | 33% |
| Ameaca | Braco estendido (esperado) | 0% | 33% |
| Braco estendido, mao aberta | Braco estendido (esperado) | 0% | 67% |
| Rendicao | Rendicao (esperado) | 90% | 100% |
| Mao oculta | Mao oculta (esperado) | 98% | 100% |
| Ameaca | Rendicao | 6% | 71% |
| Ameaca | Mao oculta | 2% | 65% |
| Braco estendido, mao aberta | Mao fechada | 46% | 94% |
| Braco estendido, mao aberta | Ameaca | 45% | 67% |
| Rendicao | Braco estendido | 0% | 67% |
| Rendicao | Mao oculta | 2% | 49% |
| Rendicao | Mao fechada | 3% | 41% |
| Punho, braco solto | Mao oculta | 0% | 31% |
| Mao oculta | Mao fechada | 0% | 18% |
| Neutro | Mao oculta | 0% | 12% |

O braco estendido, que pedia 18,6 s a 6,2 s e nunca disparava, passou a
disparar. Os alarmes falsos subiram em todas as situacoes.

### Duracao minima no ritmo do Pi

Variando a duracao minima de cada regra, com as observacoes seguidas de hoje,
nos intervalos de 1,0, 1,7, 2,4 e 3,4 s (`.tmp/varre_duracao_gestos.py`, fora
do Git):

- **Mao oculta:** com 8 s, as fases falsas a 1,7 s caem de 80 para 15 em 255, e
  o alerta esperado continua em todas as fases, nos quatro intervalos. Com 10 s,
  7 falsas, ainda sem perder o esperado.
- **Mao fechada:** com 4 s, as falsas caem de 78 para 38 em 255, e o punho da
  ameaca continua em todas as fases. O que sobra e a mao aberta lida como
  fechada na ponta do braco estendido; tirar isso pede 8 s ou mais e perde
  punho verdadeiro a 3,4 s.
- **Rendicao, braco estendido e ameaca:** nenhuma duracao tira os falsos sem
  perder alerta esperado. Eles vem da forma, nao do tempo: os punhos levantados
  da ameaca cumprem a regra da rendicao, os bracos erguidos da rendicao cumprem
  a do braco estendido, e a mao aberta na ponta do braco estendido e lida
  fechada.

### Decisao: mao oculta 8 s, mao fechada 4 s

Decisao do responsavel em 06/10/2026: a mao oculta passa a pedir 8 s, e a mao
fechada, 4 s, alem das observacoes seguidas de antes. As outras regras ficam
como estavam. A mudanca esta em `App/GestureRecon/detector.py`. O banco de
replay ganhou o intervalo de 1,7 s, e os gestos sinteticos passaram a 300
observacoes, que a 30 FPS duram 10 s.

Com a regra nova, no mesmo conjunto, fases com o alerta a 1,7 s:

| Situacao | Alerta | Antes | Depois |
|---|---|---:|---:|
| Mao oculta | Mao oculta (esperado) | 51/51 | 51/51 |
| Ameaca | Mao fechada (esperado) | 51/51 | 51/51 |
| Neutro | Mao oculta | 6/51 | 0/51 |
| Punho, braco solto | Mao oculta | 16/51 | 0/51 |
| Rendicao | Mao oculta | 25/51 | 5/51 |
| Ameaca | Mao oculta | 33/51 | 10/51 |
| Rendicao | Mao fechada | 21/51 | 0/51 |
| Mao oculta | Mao fechada | 9/51 | 1/51 |
| Braco estendido, mao aberta | Mao fechada | 48/51 | 37/51 |

- Nenhum alerta esperado perdeu fase a 1,0, 1,7, 2,4 ou 3,4 s. A 6,2 s o
  resultado ficou identico ao de antes, nos JSONs de 30/09 e nos da pose do Pi:
  nessa taxa, as observacoes seguidas ja pediam mais tempo que a duracao nova.
- Custo: a 1,7 s por frame, a mao oculta dispara na sexta observacao seguida,
  8,5 s depois da primeira, em vez de na terceira, 3,4 s; a mao fechada, na
  quarta, 5,1 s, em vez de na segunda, 1,7 s. Em 30/09, com o frame de uns 6 s,
  eram 12,4 s e 6,2 s. Um gesto breve, de duas observacoes, so confirma a
  ameaca.
- A 30 FPS, num computador rapido, a duracao passa a decidir as duas regras: 8 e
  4 s, em vez de 0,35 e 0,20 s.
- Falta conferir os tempos novos numa rodada com gesto no Pi, a criterio do
  responsavel.
- Commit `2aca583`, implantado no Pi e conferido por hash do `detector.py`,
  com `throttled=0x0`.

### Braco estendido so ate 135 graus

A regra do braco estendido valia do braco a 45 graus da vertical ate o braco
reto para cima, a 180. Com os bracos esticados para cima, a rendicao disparava
o braco estendido, e o detector nem avaliava a rendicao naquele frame: ela so e
conferida quando nao ha braco estendido. Medido com o braco estendido limitado
a 135 graus, ou seja, a ate 45 graus acima da horizontal, no conjunto da webcam
e nas gravacoes do celular de 29/09, uma por situacao
(`.tmp/varre_geometria_gestos.py`, fora do Git):

| Conjunto | Intervalo | Braco estendido na rendicao (falso) | Rendicao (esperado) |
|---|---|---:|---:|
| Webcam | 1,0 s | 20/30 -> 0/30 | 30/30 -> 30/30 |
| Webcam | 1,7 s | 34/51 -> 0/51 | 51/51 -> 51/51 |
| Webcam | 2,4 s | 39/72 -> 0/72 | 72/72 -> 72/72 |
| Webcam | 3,4 s | 16/102 -> 0/102 | 100/102 -> 102/102 |
| Webcam | 6,2 s | 0/186 -> 0/186 | 168/186 -> 179/186 |
| Celular | 1,0 a 1,7 s | 0 -> 0 | todas as fases, antes e depois |
| Celular | 2,4 s | 0/24 -> 0/24 | 21/24 -> 24/24 |
| Celular | 3,4 s | 0/34 -> 0/34 | 6/34 -> 32/34 |
| Celular | 6,2 s | 0/62 -> 0/62 | 10/62 -> 36/62 |

- O braco estendido esperado, nos videos de braco estendido e de ameaca, nao
  perdeu nenhuma fase em nenhum intervalo, nos dois conjuntos. Com 120 graus o
  resultado e o mesmo, salvo a rendicao do celular a 3,4 s, em 34/34.
- No celular, a 2,4 s, a ameaca falsa no video de rendicao caiu de 2 para 0
  fases em 24. A 3,4 s, apareceu 1 fase de rendicao falsa em 34 no video de
  ameaca.
- A rendicao falsa nos videos de ameaca, com os punhos levantados, nao muda
  com o limite: 36 de 51 fases a 1,7 s. O teste de 30/09, rendicao ignorando a
  mao levantada lida como fechada, foi repetido no ritmo de hoje, junto com o
  limite. Na webcam, a rendicao falsa cai de 36 para 13 em 51 fases; no
  celular, a rendicao verdadeira cai de 17 para 16 em 17 a 1,7 s, de 24 para
  19 em 24 a 2,4 s e de 32 para 21 em 34 a 3,4 s. Ficou de fora.

Decisao do responsavel em 06/10/2026: o braco estendido vale de 45 a 135 graus
da vertical (`MAX_AIMING_RAISE_DEGREES` em `App/GestureRecon/detector.py`). O
banco de replay ganhou a rendicao com os bracos retos, que dispara rendicao, e
nao braco estendido.

- Commit `4935202`, implantado no Pi.

### Rosto de perfil no episodio do aluno

Na rodada com gesto de 30/09, o rosto de perfil saia `NAO ALUNO`, e cada troca
abria um episodio: 11 `ALUNO` e 7 `NAO_ALUNO` da mesma pessoa na primeira
tentativa. A rechecagem de
04/10 ja reconhece de novo o desconhecido quase reconhecido, com semelhanca de
0,30 ate o limite, mas o frame em que ele sai `NAO ALUNO` ainda gravava o
evento e fechava o episodio do aluno.

Medido no PC com o servico de rosto e o registro de eventos reais, no video de
carga: reconhecimento normal com reuso de 15 s, `det_500m`, embedding no frame
inteiro, o frame 0 do video como cadastro e o relogio simulado no ritmo do Pi
(roteiro no scratchpad, fora do Git). Os rostos virados dos frames 22 e 23 (0,44
e 0,36) e 28 e 29 (0,47 e 0,41, as semelhancas medidas no Pi em 04/10) saiam
`NAO ALUNO`, e a mesma pessoa gravava 3 eventos `ALUNO` e 2 `NAO_ALUNO` nos 36
frames, a 1,7, 3,4 e 6,2 s.

O responsavel pediu a correcao em 06/10/2026. A regra, em
`App/FaceRecon/service.py`: o rosto que sai `NAO ALUNO` com semelhanca de 0,30
ate o limite fica com o nome do aluno, sem confirmar, quando:

- a pessoa mais parecida com ele, mesmo abaixo do limite, e esse aluno;
- o aluno estava a ate 120 px dele no frame anterior, a distancia que o
  registro de eventos usa para seguir um desconhecido, e nao aparece
  confirmado em outro rosto do mesmo frame;
- o aluno foi visto confirmado, pelo reconhecimento ou pelo reuso, ha menos de
  15 s, a validade do reuso.

O rosto aparece em laranja, com o nome e "verificando", nao grava evento e nao
fecha o episodio do aluno; o frame seguinte reconhece de novo. No mesmo video,
com a regra, ficou 1 evento `ALUNO` nos tres ritmos, e os outros frames, com os
mesmos nomes, semelhancas e embeddings de antes.

- So vale com o reuso ligado, como no perfil rpi3. Com a regra nova de
  confirmacao do P5, no mesmo dia, vale tambem no reconhecimento em segundo
  plano, desligado nos perfis (`docs/PLANO_OTIMIZACAO.md`, 06/10).
- Rosto de perfil com semelhanca abaixo de 0,30 continua `NAO ALUNO`. Na
  rodada de 30/09 as semelhancas nao foram gravadas; o efeito no Pi falta
  conferir pelos eventos de uma rodada.
- Estranho no lugar do aluno: so fica com o nome dele se for mais parecido com
  ele do que com qualquer outro cadastro e passar de 0,30, e no maximo por 15 s
  depois do aluno visto confirmado. Rostos de outras pessoas nao passaram de
  0,16 nos testes de 03 e 04/10.

### O que fica aberto

1. Rendicao com os punhos levantados: a 1,7 s, dispara em 71% das fases dos
   videos de ameaca.
2. Mao aberta lida como fechada com o braco levantado: mao fechada e ameaca
   falsas em 73% e 67% das fases dos videos de braco estendido.
3. Rodada com gesto no Pi para conferir os tempos, o limite e os eventos do
   rosto de perfil.

## Estado verificado em 30/09/2026

O responsavel combinou gravar um conjunto de validacao para a mao fechada, e o
resultado levou a mais uma decisao dele: o punho so conta com o braco
levantado. No mesmo dia, uma rodada com gesto no Pi conferiu essa regra e o
que estava pendente desde 29/09. Depois dos testes offline abaixo, ele decidiu
encerrar o plano de novo: a mao fechada com o braco levantado e a ameaca ficam
como limitacoes conhecidas, e o que fica aberto nao tem acao combinada.

### Conjunto de validacao

Os cortes e o veto de 29/09 foram escolhidos nos mesmos seis videos em que
foram medidos. O conjunto novo e separado e foi gravado pela webcam do projeto
em 640x480, o frame que o cliente envia ao stream: seis situacoes, tres videos
de 30 s cada, 5.400 frames amostrados, com a pessoa detectada em todos. Com
menos de 25 s, as quatro observacoes seguidas do braco estendido nao cabem em
todas as fases de amostragem a cada 6,2 s.

As situacoes sao neutro, punho com o braco solto, braco estendido com a mao
aberta, ameaca, rendicao e mao oculta atras das costas. A pessoa nao ficou
parada de frente: em cada video variou a pose e virou de frente, de lado e de
costas. Na ameaca as duas maos estao em punho, em guarda ou com o braco
estendido; no braco estendido, o braco aponta ora para o lado, ora para a
camera. Os videos mostram a pessoa e ficam fora do repositorio, com os JSONs
extraidos deles.

Parte dos frames com alguma mao fechada e, das 186 fases de amostragem a cada
6,2 s, 62 por video, em quantas o alerta de mao fechada disparou:

| Situacao | Punho real | Frames com mao fechada | Alerta de mao fechada |
|---|---|---:|---:|
| Neutro | nao | 56% | 129/186 |
| Punho, braco solto | sim | 99% | 186/186 |
| Braco estendido, mao aberta | nao | 86% | 181/186 |
| Ameaca | sim | 96% | 186/186 |
| Rendicao | nao | 11% | 6/186 |
| Mao oculta | nao | 55% | 147/186 |

O veto `Open_Palm >= 0,55` so mudou a rendicao, de 11% para 8% dos frames, com
as mesmas 6 fases, e nao perdeu nenhum punho. Com as maos no frame cheio, a pose
ficou identica nos 5.400 frames e o resultado quase nao mudou: neutro em 55%,
mao oculta em 56%, rendicao em 7%. As duas alternativas seguem descartadas.

Separando por mao, pelo que se ve em cada situacao:

| Mao | Marcada como fechada |
|---|---:|
| Punho, braco solto | 99% |
| Punho, braco levantado | 97% |
| Solta ao lado do corpo, relaxada | 51%, de 7% a 99% conforme o video |
| Solta ou na cintura, com a outra mao oculta | 87% |
| Aberta, braco estendido | 44% |
| Aberta, bracos levantados | 11% |

Com a mao solta ao lado do corpo, o classificador responde `None` para quase
todas as maos, em punho ou relaxadas, e a regra lateral marca as duas. O
detector so separa o punho da mao aberta quando a mao esta levantada.

Outros alertas nos mesmos videos: rendicao em 167/186 fases dos videos de
rendicao e mao oculta em 184/186 dos de mao oculta. O braco estendido nao
disparou em nenhuma fase: a condicao ficou ativa em ate 63% dos frames de um
video, mas por no maximo 11 s seguidos, e a regra pede quatro observacoes, ou
18,6 s. A ameaca disparou em 48/186 fases dos videos de ameaca e, sem punho, em
104/186 dos videos de braco estendido com a mao aberta. Nos videos de ameaca, a
rendicao disparou em 13/186 fases, com os punhos levantados acima dos ombros.

### Punho so com o braco levantado

Decisao do responsavel em 30/09/2026: a mao fechada so conta, no alerta de mao
fechada e no de ameaca, quando o braco esta a 45 graus ou mais da vertical, a
mesma medida do braco estendido. O punho com o braco solto deixa de alertar:
ele alertava em todas as fases, mas a pessoa parada com a mao relaxada tambem
alertava em 129 de 186.

Fases com o alerta de mao fechada, antes e depois, nos dois conjuntos:

| Conjunto | Situacao | Antes | Depois |
|---|---|---:|---:|
| Webcam | Neutro | 129/186 | 0/186 |
| Webcam | Punho, braco solto | 186/186 | 0/186 |
| Webcam | Braco estendido, mao aberta | 181/186 | 75/186 |
| Webcam | Ameaca | 186/186 | 186/186 |
| Webcam | Rendicao | 6/186 | 5/186 |
| Webcam | Mao oculta | 147/186 | 0/186 |
| Celular | Neutro | 0/62 | 0/62 |
| Celular | Mao fechada, braco solto | 61/62 | 0/62 |
| Celular | Braco estendido, mao aberta | 16/62 | 2/62 |
| Celular | Ameaca | 48/62 | 48/62 |
| Celular | Rendicao | 27/62 | 23/62 |
| Celular | Mao oculta | 62/62 | 0/62 |

O alerta de ameaca sem punho, nos videos de braco estendido com a mao aberta,
caiu de 104 para 75 fases na webcam e de 10 para 1 no celular; nos videos de
ameaca ficou igual, em 48/186 e 2/62. O resultado quase nao muda com o limite
entre 30 e 60 graus. O que sobra de falso e a mao aberta ou relaxada lida como
fechada com o braco levantado: o braco estendido com a mao aberta e, no
celular, a rendicao.

Pelo criterio de aceite, o item 3 deixa de valer para o punho com o braco
solto: a regra perde essa deteccao de proposito. No banco sintetico, a mao
fechada passou a ser o punho levantado com o cotovelo dobrado, que dispara nas
mesmas observacoes de antes (F7 a 30 FPS, F2 no Pi), e o punho com o braco
solto entrou como postura que nao dispara. A gravacao
`resultados/gestos-reais/mao_fechada.json`, de punho com o braco solto, deixou
de disparar.

### Rodada com gesto no Pi

Commit `338ac06`, implantado e conferido por hash no Pi em 30/09/2026, com o
perfil `rpi3`. Uma pessoa na frente da webcam do projeto, enviando um frame por
vez ao stream. Cada situacao teve 6 frames seguidos numa conexao nova, entao o
servidor zerou o historico de gestos e os episodios de evento no primeiro
frame. O frame levou de 5,3 a 6,3 s, com a placa entre 47 e 61 C. Arquivos em
`resultados/pi3-338ac06-gestos/`.

Alertas por frame (F mao fechada, A ameaca, R rendicao, O mao oculta, B braco
estendido):

| Situacao | Alertas nos 6 frames | Esperado |
|---|---|---|
| Neutro, de frente | nenhum | nenhum |
| Punho, braco solto | nenhum | nenhum |
| Punho levantado, cotovelo dobrado | F do 2 ao 4 | F no 2 |
| Rendicao | R do 3 em diante; F no 2, no 5 e no 6 | R no 3 |
| Mao oculta, de frente | O do 3 em diante | O no 3 |
| De lado | O do 4 em diante | O no 3 |
| De costas | nenhum | nenhum |
| Braco para o lado, mao aberta | B do 4 em diante; F e A no 3 e no 4 | B no 4 |
| Braco para o lado, punho fechado | B do 4 em diante | F e A no 2, B no 4 |

- O criterio duplo se confirmou: mao fechada e ameaca na segunda observacao
  seguida, rendicao e mao oculta na terceira, braco estendido na quarta. De
  lado, a condicao de mao oculta so apareceu no segundo frame, e o alerta veio
  no quarto.
- O punho com o braco solto nao alertou; levantado, alertou no segundo frame.
- Com os bracos soltos, nem braco estendido nem mao oculta, de frente ou de
  costas; de lado, mao oculta, como decidido em 29/09.
- Os 32 eventos das duas tentativas cairam no frame previsto pelo episodio:
  cada alerta gravou quando comecou e nao de novo enquanto seguiu, e
  `alertas_novos` trouxe so o que abriu episodio. O campo `evidencia` saiu em
  todos os 8 alertas lidos do banco: 2 observacoes em 5,7 a 5,9 s, 3 em 11,1 a
  11,7 s e 5 em 22,9 s.
- Os rostos gravaram um evento por episodio, mas o rosto de perfil nao foi
  reconhecido e virou `NAO_ALUNO`. Cada troca entre reconhecido e nao
  reconhecido abriu um episodio novo: 11 `ALUNO` e 7 `NAO_ALUNO` da mesma
  pessoa na primeira tentativa.

A primeira tentativa de ameaca e de braco estendido saiu com o braco apontado
para a camera: a caixa da pessoa nao alargou, e o braco estendido nao disparou
em nenhum dos 12 frames, como ja se via nas gravacoes de 29/09. Com o punho
acima do ombro, disparou a rendicao do terceiro frame em diante. A tabela traz
a repeticao, com o braco para o lado.

Problemas vistos na rodada:

- Na ponta do braco estendido, a mao nao foi lida direito nos dois sentidos: o
  punho fechado nao contou em nenhum dos 6 frames, e a ameaca nao disparou; a
  mao aberta contou como fechada em 2 frames seguidos e disparou a ameaca.
- Uma segunda pessoa, de confianca 0,27 a 0,60, apareceu em 8 frames da
  primeira tentativa, no canto de baixo a direita da imagem, provavelmente a
  cadeira com uma camisa que aparece nos videos. Nao teve alerta.

### Testes offline depois da rodada

Sem mudar o produto, nos dois conjuntos de gravacoes, contando as fases de
amostragem a cada 6,2 s:

- Rendicao ignorando a mao levantada lida como fechada: nos videos de ameaca
  da webcam, a rendicao falsa caiu de 13 para 0 fases, mas a rendicao de
  verdade caiu de 167 para 164 na webcam e de 10 para 3 no celular, onde a mao
  pequena e lida fechada com frequencia. Vetar so quando o classificador diz
  `Closed_Fist` nao mudou nada: ele nao se pronuncia nessas maos. Nenhuma das
  duas foi promovida.
- Mao cortada pela caixa da pessoa: o detector de maos roda so dentro dela,
  sem margem. Na webcam, a mao aberta na ponta do braco estendido foi lida
  fechada em 89% das vezes com o centro estimado fora da caixa, contra 40%
  dentro. Com a regiao das maos 15% maior de cada lado, a pose ficou identica.
  Na webcam, a ameaca falsa caiu de 75 para 61 fases e a mao aberta levantada
  lida fechada, de 11% para 6%, sem perder punho. No celular piorou: a mao
  fechada falsa na rendicao subiu de 23 para 29 fases, a ameaca falsa de 1
  para 3, e a mao fechada da ameaca de verdade caiu de 48 para 42. Nao foi
  promovida.

### O que fica aberto

1. Mao lida errado com o braco levantado: a mao aberta ou relaxada conta como
   fechada, nos videos e no Pi, na rendicao e no braco estendido; e o punho na
   ponta do braco estendido nao foi lido no Pi, entao a ameaca nao disparou.
   A margem na regiao das maos, acima, nao resolveu. Sem acao combinada.
2. Rendicao com os punhos levantados acima dos ombros, vista nos videos de
   ameaca e no Pi: a regra nao olha se as maos estao abertas, e olhar custa
   rendicao de verdade enquanto a leitura da mao for a de hoje. Sem acao
   combinada.
3. Rosto de perfil vira `NAO_ALUNO` e reabre o episodio do aluno. Fora das
   acoes deste plano, sem acao combinada.
4. Acao 4 deste plano.

## Estado verificado em 29/09/2026

O responsavel reabriu o plano para as acoes 1, 2 e 5. As tres foram feitas, e o
video de pose neutra levou a duas mudancas de geometria, tambem decididas por
ele. Falta verificar tudo no Pi.

### Acao 1 - banco de replay

`tools/gesture_replay.py` entrega a mesma sequencia de observacoes ao
`GestureAnalyzer` com o intervalo escolhido e mostra em qual observacao cada
alerta dispara, antes e depois do criterio duplo. Traz poses sinteticas para as
cinco regras e para as posturas neutras abaixo, e aceita gravacoes reais no
mesmo formato. `tools/record_gesture_sequence.py` grava essas sequencias da
webcam ou de um video, guardando so keypoints, caixa e estado das maos, sem
imagem. Os testes estao em `tests/test_gesture_replay.py`.

As gravacoes reais estao em `resultados/gestos-reais/`: seis videos de celular
de uma pessoa em pe, em 29/09/2026, com a pose neutra e os cinco gestos, cada
um de frente, de um lado, do outro e de costas. Detalhes no `LEIAME.md` da
pasta.

### Acao 2 - criterio duplo

Cada regra passa a exigir, alem da duracao, um minimo de observacoes seguidas:
mao fechada e ameaca 2, rendicao e mao oculta 3, braco estendido 4. Sao os
valores de partida deste plano, mantidos pelo responsavel. Uma observacao em
que o gesto falha zera a contagem.

Observacao em que cada alerta dispara com o gesto mantido, no banco sintetico
(F mao fechada, A ameaca, R rendicao, O mao oculta, B braco estendido):

| Intervalo entre observacoes | Antes | Agora |
|---|---|---|
| 1/30 s | F7, A8, R10, O12, B13 | igual |
| 1 s ou mais | todas na segunda | F2, A2, R3, O3, B4 |

Com o frame de 6,2 s do Pi, isso da cerca de 6 s para mao fechada e ameaca,
12 s para rendicao e mao oculta e 18 s para braco estendido, contados da
primeira observacao do gesto. Um gesto visto em so duas observacoes seguidas
dispara apenas mao fechada e ameaca. Pelo criterio de aceite, o item 1 fica
atendido no replay.

### Mudancas de geometria

O video neutro mostrou alertas sem gesto nenhum. Com as regras anteriores:

| Pessoa vista | Condicao ativa (parte dos frames) |
|---|---|
| De frente | braco estendido 20%, mao oculta 7% |
| De lado | mao oculta 93%, braco estendido 21%, mao fechada 7% |
| Do outro lado, meio virada | mao oculta 100%, mao fechada 5% |
| De costas | mao oculta 100% |

Braco caido ao lado do corpo, a no maximo 23 graus da vertical, passava por
braco estendido. Braco solto perto do corpo, com o cotovelo entre 142 e 161
graus e a mao perdida pelo detector de maos, passava por mao oculta. De costas,
a camera nunca ve as maos. Decisoes do responsavel:

- Braco estendido so com o braco levantado, a 45 graus ou mais da vertical.
- Mao oculta conforme a orientacao da pessoa. De frente, so com o cotovelo
  dobrado, abaixo de 130 graus, ou com o punho fora de vista e o cotovelo
  cruzando o corpo, como antes. De lado, o braco que a camera nao ve conta
  como oculto. De costas, nao conta. A pessoa esta de costas quando o rosto
  nao aparece e as orelhas sim, e de lado quando a largura dos ombros fica
  abaixo de 45% da altura do tronco ou so um ombro aparece.
- Foram descartadas avaliar a mao oculta so de frente e redefinir a ameaca como
  so braco estendido com punho fechado.

### Validacao com as gravacoes reais

- Video neutro: braco estendido zerou nas quatro situacoes, e mao oculta zerou
  de frente e de costas e continua de lado.
- Os gestos continuam disparando: rendicao de frente, de lado e de costas; mao
  oculta de frente com o cotovelo dobrado (82% do trecho, contra 89% antes) e
  de lado; braco estendido de frente e de lado com o braco na altura do ombro
  (47% a 100% dos trechos); mao fechada; e ameaca de lado. De frente a ameaca
  quase nao dispara, antes ou agora, porque o braco apontado para a camera
  aparece curto na imagem.
- O que as regras novas deixaram de acusar nos videos de gestos eram bracos
  caidos ou dobrados junto ao corpo. No video de mao fechada visto de frente,
  braco estendido ficava ativo em 64% a 96% dos frames so pelo braco caido, e
  foi a 0%.

### Acao 5 - evidencia em cada alerta

O evento `ALERTA_GESTO` ganhou o campo `evidencia`: para cada alerta, quantas
observacoes seguidas o sustentaram e quanto tempo real elas cobriram, por
exemplo `{"alerta": "Rendicao", "observacoes": 3, "duracao_s": 12.4}`. No Pi,
um alerta fica com 2 a 4 observacoes em 6 a 19 s; a 30 FPS, com 7 a 13
observacoes em 0,2 a 0,4 s. Quando o alerta dispara nao muda, e o stream
enviado ao navegador tambem nao.

### Eventos de rosto por episodio

Decidido pelo responsavel em 29/09/2026, fora das acoes deste plano. Os eventos
`ALUNO` e `NAO_ALUNO` seguem a logica dos alertas: o aluno grava quando aparece
e nao grava de novo enquanto aparece nos frames seguintes; se some de um frame,
a proxima aparicao grava outra vez. Rosto desconhecido nao tem nome: o episodio
continua quando ele aparece a ate 120 px de um desconhecido ja gravado no frame
anterior, a mesma distancia do casamento de reserva do rastreador de gestos. O
cooldown de 5 s continua valendo por cima. Pela sequencia de eventos das duas
rodadas de 27/09, isso daria 4 e 5 eventos de rosto por rodada, contra 36 e 35.

### Achado: mao fechada acende demais

A condicao de mao fechada ficou ativa em videos sem punho fechado: 14% a 80% dos
frames na rendicao, 42% a 100% na mao oculta e 10% a 59% no braco estendido
com a mao aberta. No video neutro, gravado na vertical, ficou entre 0% e 8%. A
primeira suspeita era que, com a mao pequena na imagem, o reconhecedor de gestos
da mao confundisse mao aberta ou relaxada com punho.

O responsavel autorizou o diagnostico em 29/09/2026. O gravador passou a guardar,
para cada mao associada a pessoa, tamanho, gesto e confianca do classificador e
se a marcou como fechada o classificador, a regra frontal ou a regra lateral.
Nos seis videos, quem marcou quase todos os falsos positivos foi a regra lateral;
o classificador respondeu `None` para a maioria das maos. No video de mao
fechada com o braco solto, porem, so a regra lateral reconheceu o punho.

Foi testada a deteccao de maos no frame de 640 px, mantendo a pose no mesmo
frame reduzido de 320 px e remapeando somente as maos. Tempo, keypoints e caixa
da pose ficaram identicos em todos os 2.686 frames das duas variantes. O tamanho
mediano entregue ao detector de maos praticamente dobrou, de 13-34 px para
25-68 px conforme o video.

| Video | Punho real | Mao fechada no frame reduzido | Mao fechada no frame cheio |
|---|---:|---:|---:|
| Neutro | nao | 5% | 4% |
| Rendicao | nao | 47% | 29% |
| Mao oculta | nao | 79% | 81% |
| Braco estendido | nao | 28% | 26% |
| Mao fechada | sim | 82% | 78% |
| Ameaca | sim | 47% | 49% |

A resolucao cheia melhorou apenas a rendicao neste conjunto, nao separou os
outros falsos positivos e nao aumentou o reconhecimento dos dois punhos reais.
Por isso, nao foi promovida para o servico nem levada ao Pi.

O responsavel autorizou em seguida medir os indices internos da regra lateral.
Foram registradas 3.092 maos associadas nos mesmos videos: pontas compactas,
distancia media ate a palma, abertura entre pontas e as duas distancias do
polegar, todas normalizadas pelo tamanho da palma. Os indices reproduziram a
decisao atual em todas as maos, sem divergencia.

Uma grade de 896 combinacoes mais estritas e uma varredura exata de cada valor
nao acharam um corte geometrico util que preservasse todos os frames positivos.
Por exemplo, limitar a abertura a 1,25 removeu 25 frames falsos, mas tambem um
frame da ameaca; limitar a media das pontas a 1,04 removeu 21 falsos, mas perdeu
um frame da ameaca e um da mao fechada. As distribuicoes dos punhos reais e das
maos abertas que a regra lateral confunde se sobrepoem.

O unico sinal mais discriminante foi a classificacao negativa `Open_Palm`. Um
veto com confianca a partir de 0,55 preservou os frames positivos deste conjunto
e removeu 50 frames falsos da rendicao. A margem, porem, e pequena: a unica mao
`Open_Palm` no video de ameaca teve 0,546. Simulando as 62 fases possiveis de
amostragem a cada 6,2 s, o veto mudou assim os alertas de mao fechada:

| Video | Regra atual | Veto `Open_Palm >= 0,55` |
|---|---:|---:|
| Neutro | 0/62 | 0/62 |
| Rendicao | 27/62 | 8/62 |
| Mao oculta | 62/62 | 62/62 |
| Braco estendido | 16/62 | 16/62 |
| Mao fechada | 61/62 | 61/62 |
| Ameaca | 48/62 | 48/62 |

O veto ajuda somente a rendicao e nao resolve os dois falsos positivos mais
graves. Por isso, nenhum limiar foi promovido. Tirar a regra lateral tambem nao
foi feito: perderia o punho real com o braco solto.

O responsavel autorizou entao testar a mao num recorte menor ao redor de cada
punho. Foram comparados tres quadrados do frame original, com meio lado igual a
0,65, 0,85 e 1,05 vezes o comprimento do antebraco visto pela pose. A pose
continuou no frame reduzido e ficou identica, inclusive tempos, caixas e
keypoints, nos 2.686 frames das quatro variantes.

| Video | Corpo reduzido | Recorte 0,65 | Recorte 0,85 | Recorte 1,05 |
|---|---:|---:|---:|---:|
| Neutro | 25/512 | 119/512 | 72/512 | 110/512 |
| Rendicao | 101/216 | 71/216 | 52/216 | 50/216 |
| Mao oculta | 378/478 | 307/478 | 377/478 | 384/478 |
| Braco estendido | 158/561 | 135/561 | 140/561 | 150/561 |
| Mao fechada | 276/337 | 253/337 | 257/337 | 259/337 |
| Ameaca | 273/578 | 293/578 | 317/578 | 322/578 |

O recorte ajudou a rendicao e o punho da ameaca, mas introduziu muito mais
punhos no video neutro. O menor reduziu o falso positivo da mao oculta, mas
deixou de ver alguma mao em 40 frames adicionais. Os tres tambem reduziram o
punho real com o braco solto.

Nas 62 fases de amostragem a cada 6,2 s, o video neutro passou de zero alertas
de mao fechada para 35, 13 e 21 fases. O punho real com o braco solto caiu de
61 para 48, 51 e 50 fases. Na mao oculta, o melhor resultado ainda alertou em
56/62 fases. No braco estendido aberto, os dois recortes maiores aumentaram o
alerta composto falso de 10 para 15 e 14 fases. Combinar os recortes com o veto
`Open_Palm >= 0,55` so melhorou novamente a rendicao.

Nenhum recorte foi promovido. O custo no Pi nao foi medido porque a qualidade
ja reprovou no conjunto local; essa variante chamaria o detector uma vez por
punho, em vez de uma vez por pessoa.

### O que fica aberto

1. Verificar no Pi o criterio duplo, as mudancas de geometria, o episodio por
   alerta (`8801350`), o campo `evidencia` e os eventos de rosto por episodio,
   numa rodada com gesto.
2. Falsos positivos de mao fechada, acima: foram descartados a resolucao cheia,
   os novos cortes geometricos e os recortes por punho. A proxima acao ainda
   precisa ser combinada.
3. Acao 4 deste plano.

## Encerramento em 27/09/2026

O plano foi encerrado junto com o de desempenho. Feita: a acao 3, verificada no
Pi com a chave por conjunto de alertas e trocada depois, por decisao do
responsavel, por um episodio por alerta (`8801350`, implantado e conferido por
hash no Pi, sem rodada medida com ele). Ficaram para depois as acoes 1, 2, 4
e 5.

Pelo criterio de aceite do fim deste documento, o item 2 foi atendido e o item
1 nao: no Pi, as cinco regras continuam exigindo o mesmo, o gesto presente em
duas observacoes seguidas, cerca de 12 s com o frame em 6 s. Os limiares de
0,20 a 0,40 s de cada regra nao tem efeito nessa velocidade. E a principal
limitacao conhecida da versao final.

## Problema medido

Em 20 e 21/09/2026, no Raspberry Pi 3 B+, com o pipeline completo:

- O intervalo real entre analises de gesto e de **7,4 s** por frame no perfil
  otimizado e era de **13,2 s** antes dele.
- Os limiares das regras, apos a conversao para tempo decorrido, sao de
  **0,20 a 0,40 segundos**.
- O intervalo entre observacoes e portanto **18 vezes maior** que o maior
  limiar existente.

Em `App/GestureRecon/detector.py`, `_update_counter` acumula assim:

```python
if was_active:
    self.history[track_id][key] = min(limit, self.history[track_id][key] + elapsed)
```

Com `elapsed` de 7,4 s contra um `limit` de 0,40 s, o contador **satura no
primeiro incremento**. O `if was_active` existe por um bom motivo: nao creditar
o intervalo anterior a um gesto que acabou de aparecer. O efeito colateral e que
a confirmacao passa a exigir exatamente **duas observacoes consecutivas**,
qualquer que seja o limiar.

### As cinco regras colapsaram em uma

O analisador e construido com `GESTURE_ANALYZER_FPS`, cujo padrao e **12**, nao
30. Antes da conversao, os limiares equivaliam a:

| Regra | Limiar antigo em observacoes | Limiar atual efetivo |
|---|---|---|
| Maos escondidas | 4 | 2 |
| Rendicao | 3 | 2 |
| Braco estendido | 4 | 2 |
| Mao fechada | 2 | 2 |
| Mao fechada + braco | 3 | 2 |

A distincao entre punho, rendicao, mira e ameaca **deixou de existir neste
hardware**. Todas exigem o mesmo: gesto presente em dois frames seguidos.

### Consequencias observadas

- Na rodada `uma-pessoa-r3.json` do perfil default, **7 alertas em 7 frames
  consecutivos**, confirmados pelo responsavel como gesto real. Nao foi falso
  positivo: foi um gesto unico gravado sete vezes.
- `COOLDOWN_ALERTA_GESTO_SECONDS` e 5 s e o frame custa 7,4 s. O cooldown
  **nunca deduplica**: todo frame com o gesto ativo grava um evento novo no
  MongoDB, com recorte de imagem.
- Nas tres rodadas de duas pessoas de 26/09/2026, com o frame em cerca de
  7,6 s, foram **56, 52 e 42 alertas**, presentes em 85 de 90 frames, quase
  sempre 1 ou 2 por frame. Nao houve confirmacao de que os gestos eram reais.
- A contagem de alertas continua **sem servir para comparar versoes**, agora por
  excesso de sensibilidade em vez de falta. O plano de otimizacao tratava a
  conversao para tempo como pre-requisito cumprido para essa comparacao; nao e.

## Por que ajustar o limiar nao resolve

Nao da para resolver um evento de 0,35 s amostrando a cada 7,4 s. E limite de
amostragem, nao de calibracao. Subir os limiares para 15 s ou 30 s faria as
regras voltarem a discriminar, mas ao custo de exigir que alguem mantenha um
gesto ameacador por meio minuto antes do alerta.

Vale separar duas coisas que o limiar de hoje mistura:

1. **Persistencia real do gesto.** Quanto tempo de mundo real o gesto precisa
   durar para valer alerta. Isso e decisao de produto.
2. **Confianca estatistica.** Quantas observacoes concordantes sao necessarias
   para nao disparar por um unico erro de estimativa de pose. Isso e decisao de
   engenharia, e depende de quantas amostras existem.

A 30 FPS, 0,35 s equivalem a 11 amostras: persistencia curta e confianca alta.
A 0,13 FPS, duas amostras cobrem 15 s: persistencia longa e confianca baixa.
**O mesmo numero hoje governa as duas coisas, e por isso nao serve para nenhuma.**

## Acoes propostas, em ordem

### 1. Banco de replay para medir sem o Pi

Pre-requisito de todo o resto. `GestureAnalyzer.analyze` ja aceita `observed_at`
explicito, entao da para reproduzir uma sequencia de keypoints com qualquer
intervalo simulado e observar quais alertas disparam, sem modelo, sem camera e
sem Raspberry.

- Gravar sequencias rotuladas de keypoints: rendicao, braco estendido, punho,
  maos escondidas e cena neutra. Podem vir do proprio coletor, com
  `include_keypoints`, ou de gravacao dedicada.
- Reproduzir cada sequencia com intervalos de 0,033 s, 1 s, 3 s, 7,4 s e 13,2 s
  e registrar o que dispara em cada um.
- Isso transforma "ajustar limiar" de chute em medicao, e roda no CI.

Risco: nenhum. Nao altera o comportamento do produto.

Feita em 29/09/2026; ver "Estado verificado em 29/09/2026".

### 2. Criterio duplo: observacoes e tempo, o que for mais dificil

Confirmar um gesto exigindo **ao mesmo tempo** um numero minimo de observacoes
concordantes e uma duracao minima:

```
confirmado = observacoes >= N  E  tempo_decorrido >= T
```

A 30 FPS, T domina e o comportamento fica igual ao pretendido hoje. A 0,13 FPS,
N domina e as regras voltam a discriminar entre si. O criterio degrada de forma
previsivel em qualquer taxa, em vez de colapsar numa delas.

Valores de partida, a validar com a acao 1, nao escolhidos por medicao ainda:

| Regra | N | T |
|---|---|---|
| Mao fechada | 2 | 0,20 s |
| Mao fechada + braco | 2 | 0,22 s |
| Rendicao | 3 | 0,30 s |
| Maos escondidas | 3 | 0,35 s |
| Braco estendido | 4 | 0,40 s |

Contrapartida honesta: a 7,4 s por frame, N=4 significa cerca de 30 s de braco
estendido antes do alerta. **Nessa taxa nao existe almoco gratis**: ou dispara
rapido com pouca evidencia, ou devagar com muita. A escolha e do responsavel, e
a acao 4 e o que muda o dilema de lugar.

Risco: muda quando cada alerta dispara. Exige a acao 1 antes.

Feita em 29/09/2026, com os valores de partida acima; ver "Estado verificado em
29/09/2026".

### 3. Deduplicar alerta enquanto o gesto continua ativo

O cooldown em segundos nao funciona quando o frame custa mais que ele. Enquanto
um gesto permanece continuamente confirmado no mesmo track, deve gerar **um**
evento, nao um por frame. Um novo evento so apos o gesto cair e voltar, ou apos
um intervalo bem maior que o frame.

Risco: baixo. Independe das acoes 1 e 2 e pode ser feita antes delas.

**Implementada em 26/09/2026 e verificada no Pi em 27/09/2026.** O
`EventLogger` guarda os alertas ja gravados no episodio atual, com a mesma
chave do cooldown: o `track_id` mais o conjunto de alertas. Enquanto a chave
aparece em frames seguidos, nao grava de novo; quando ela some de um frame, o
episodio termina e a proxima aparicao grava outra vez. A chave so entra no
episodio depois do insert com sucesso, entao uma falha de banco tenta de novo
no frame seguinte. O servidor reinicia os episodios junto com o historico de
gestos, no primeiro frame da conexao e depois de espera longa. O cooldown de
5 s continua valendo por cima. Ficou de fora a repeticao periodica de um
alerta que dura muito; se for desejada, e um intervalo configuravel a mais.

O payload do stream nao muda: o cliente segue recebendo `alerts` em todo frame e
o `alerts_count` do benchmark continua igual. Muda so o que vai para o MongoDB.
Verificacao prevista: com alertas em todos os frames, `logs_ms` deve cair para o
custo do evento de rosto fora do primeiro frame de cada episodio, cerca de 30 ms
como nas rodadas de uma pessoa, contra cerca de 60 ms nas rodadas de duas
pessoas de 26/09.

Verificacao de 27/09/2026, commit `6f5756d`, duas pessoas, 35 frames por
rodada contando o aquecimento. Os eventos foram contados pela diferenca de
`/logs` antes e depois de cada rodada, sem guardar nomes nem imagens. A segunda
rodada foi feita com o ar-condicionado da sala ligado; ver
`docs/PLANO_OTIMIZACAO.md`.

| Rodada | Frames medidos com alerta | `ALERTA_GESTO` gravados | `logs_ms` mediano |
|---|---|---|---|
| Sala sem ar-condicionado | 30/30 | 5 | 30,7 ms |
| Sala com ar-condicionado | 30/30 | 2 | 28,4 ms |

- Nas rodadas de 26/09, sem a acao 3, todo frame com alerta gravava evento e
  `logs_ms` passava de 45 ms em 29 ou 30 de 30 frames. Agora passa em 5 e 3 de
  30: 4 e 1 deles gravaram alerta, os outros so gravaram eventos de rosto.
- Na rodada com ar, os 2 eventos sao a abertura do episodio de cada pessoa. Na
  rodada sem ar, 1 evento abriu o episodio e os outros 4 vieram da mesma pessoa
  passando de 1 para 2 alertas e voltando, duas vezes: como a chave inclui o
  conjunto de alertas, cada mudanca abre um episodio novo. O responsavel
  decidiu que a volta para um alerta que ja estava ativo nao deve gravar de
  novo; ver o paragrafo seguinte.
- Os eventos de rosto seguem um por frame: 34 e 33 `ALUNO`, e 2 `NAO_ALUNO` em
  cada rodada. Ver "O que este plano nao cobre".

**Episodio por alerta, decidido em 27/09/2026; falta verificar no Pi.**
Substitui a chave por conjunto de alertas descrita acima. Cada alerta de cada
track tem o proprio episodio: grava quando comeca e nao grava de novo enquanto
continuar, mesmo que outro alerta do mesmo track entre ou saia. Alertas que
comecam juntos geram um evento so. O evento ganhou o campo `alertas_novos`, com
os alertas que abriram episodio; `alertas` segue com todos os alertas ativos do
track. Nenhum consumidor le esses campos hoje, e a rota `/logs` nao os expoe.
Na rodada sem ar de 27/09, a regra nova teria gravado 3 eventos em vez de 5.
Quem sai de cena e volta grava de novo: o historico de gestos do track e
apagado no primeiro frame sem a pessoa, e o alerta precisa ser confirmado
outra vez.

### 4. Aumentar a taxa de observacao do caminho de gesto

E o unico caminho que desfaz o dilema da acao 2. Candidatos, um por vez:

- **`imgsz` explicito na pose.** Hoje a chamada nao fixa tamanho e o Ultralytics
  usa 640. O frame ja chega reduzido por `PROCESS_SCALE=0.5`. Fixar 320 corta a
  computacao em cerca de 4x. Se a pose cair de 6,9 s para perto de 2 s, o
  intervalo vai a cerca de 3 s e N=3 passa a significar 9 s em vez de 22 s.
  Medido em 01/10/2026 em `docs/PLANO_OTIMIZACAO.md`: 320 perdeu deteccao, e
  416 virou padrao do rpi3. A pose caiu de 5,0 para 2,2 s, mas o frame com uma
  pessoa so de 5,7 para 5,3 s, porque o rosto passou a decidir o frame: o
  intervalo entre observacoes quase nao mudou.
- **NCNN no modelo de pose.** Ja preparado em `tools/export_pose_ncnn.py` e
  `POSE_MODEL_PATH`. Exportar no PC, nunca no Pi: sao 906 MB de RAM. Falta
  confirmar o pacote `ncnn` instalado no dispositivo.
- **Pose sobre o recorte da pessoa** em vez do frame inteiro, reaproveitando a
  caixa do frame anterior.

Risco alto em todos: mexem na qualidade de keypoints, que e a entrada das
regras. Nenhum vale sem a acao 1 medindo o recall antes e depois.

Feita pelo plano de desempenho: em 04/10/2026, com a pose em NCNN e o detector
de rosto `det_500m`, o frame com uma pessoa chegou a 1,7 s. O efeito nas regras
e a duracao minima nova estao em "Estado verificado em 06/10/2026".

### 5. Registrar a resolucao temporal em cada alerta

Gravar no evento o intervalo entre observacoes e quantas observacoes o
sustentaram. Um alerta apoiado em 2 amostras ao longo de 15 s e um apoiado em 11
amostras ao longo de 0,4 s nao sao a mesma coisa, e hoje o banco nao distingue.

Risco: nenhum no comportamento. Muda o formato do evento.

Feita em 29/09/2026; ver "Estado verificado em 29/09/2026".

## Criterio de aceite

1. As cinco regras voltam a ter comportamento distinto entre si na taxa real do
   dispositivo, demonstrado pelo banco de replay da acao 1.
2. Um gesto unico e continuo gera **um** alerta, nao um por frame.
3. Nenhuma regra perde deteccao que hoje acerta, verificado no replay antes de
   ir ao hardware.
4. O que mudar de comportamento fica registrado aqui, com o numero medido.

## O que este plano nao cobre

Latencia e vazao seguem em `docs/PLANO_OTIMIZACAO.md`. Se a acao 4 for
executada, o ganho de desempenho dela e registrado la, e o efeito sobre as
regras, aqui.

Os eventos de rosto (`ALUNO` e `NAO_ALUNO`) sao registro de presenca, nao
alerta. Ate 29/09/2026 seguiam so o cooldown de 5 s e, com o frame em cerca de
6 s, gravavam um evento por frame; desde entao gravam por episodio, por decisao
a parte do responsavel. Ver "Estado verificado em 29/09/2026".
