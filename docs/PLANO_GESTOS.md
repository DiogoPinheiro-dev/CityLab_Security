# Plano de correcao das regras de gesto

Documento separado de `docs/PLANO_OTIMIZACAO.md` de proposito. Aquele trata de
latencia e vazao; este trata de **comportamento do produto**: quando um alerta
dispara e o que ele significa. Sao problemas diferentes, e o segundo nao se
resolve com o primeiro.

Nada aqui esta autorizado a ser implementado. Cada acao e combinada com o
responsavel antes, uma de cada vez, como no plano de otimizacao.

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
suspeita e que, com a mao pequena na imagem, o reconhecedor de gestos da mao
confunda mao aberta ou relaxada com punho. Para confirmar, a gravacao teria de
guardar o tamanho e a confianca da mao, o que ainda nao faz. Fica como item
novo, sem acao combinada.

### O que fica aberto

1. Verificar no Pi o criterio duplo, as mudancas de geometria, o episodio por
   alerta (`8801350`), o campo `evidencia` e os eventos de rosto por episodio,
   numa rodada com gesto.
2. Falsos positivos de mao fechada, acima.
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
- **NCNN no modelo de pose.** Ja preparado em `tools/export_pose_ncnn.py` e
  `POSE_MODEL_PATH`. Exportar no PC, nunca no Pi: sao 906 MB de RAM. Falta
  confirmar o pacote `ncnn` instalado no dispositivo.
- **Pose sobre o recorte da pessoa** em vez do frame inteiro, reaproveitando a
  caixa do frame anterior.

Risco alto em todos: mexem na qualidade de keypoints, que e a entrada das
regras. Nenhum vale sem a acao 1 medindo o recall antes e depois.

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
