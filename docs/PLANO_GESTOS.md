# Plano de correcao das regras de gesto

Documento separado de `docs/PLANO_OTIMIZACAO.md` de proposito. Aquele trata de
latencia e vazao; este trata de **comportamento do produto**: quando um alerta
dispara e o que ele significa. Sao problemas diferentes, e o segundo nao se
resolve com o primeiro.

Nada aqui esta autorizado a ser implementado. Cada acao e combinada com o
responsavel antes, uma de cada vez, como no plano de otimizacao.

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

Os eventos de rosto (`ALUNO` e `NAO_ALUNO`) seguem so o cooldown de 5 s: com o
frame em cerca de 6 s, cada rosto gera um evento por frame. Sao registro de
presenca, nao alerta, e mudar isso e uma decisao a parte.
