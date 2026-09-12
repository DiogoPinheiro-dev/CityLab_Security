# Benchmark do stream

O utilitario `tools/benchmark_stream.py` reproduz um video fixo contra a API real.
Ele envia todos os frames selecionados, em ordem, com uma pendencia por vez e sem
pausa artificial. Mede a vazao desse perfil; nao simula o navegador com duas
pendencias ou captura a 10 FPS. Nao mede CPU/renderizacao do navegador.

## Preparacao

Use tres videos locais: cena vazia, uma pessoa e varias pessoas. Cada um precisa
ter pelo menos 330 frames para os valores padrao. Mantenha os mesmos arquivos em
todas as comparacoes; o relatorio registra SHA-256. Os videos nao estao incluidos
no repositorio. Use uma instancia e banco de teste: o stream grava eventos reais.

No servidor, habilite `ENABLE_PERFORMANCE_METRICS=1` e
`ENABLE_SYSTEM_MONITOR=1` antes de iniciar. Instale o perfil Bookworm, incluindo
psutil, e as dependencias opcionais dos modelos que serao avaliados. Para a
referencia completa, use `CITYLAB_ALLOW_PARTIAL_PIPELINE=0` e ambos os servicos
habilitados. Confirme no startup que os dois foram carregados. Instale previamente
os pesos e mantenha os mesmos pesos e cadastros em todas as execucoes.

Guarde junto aos resultados, no Raspberry:

```bash
git rev-parse HEAD > benchmark-commit.txt
python --version > benchmark-python.txt
python -m pip freeze > benchmark-packages.txt
uname -a > benchmark-host.txt
```

Registre tambem resolucao, escala, workers, paralelismo, limites de threads,
servicos ativos, hashes dos pesos, quantidade de cadastros, alimentacao e
refrigeracao. Nao inclua credenciais do banco nos artefatos. `--run-label`
identifica esse registro; nao detecta automaticamente o ambiente remoto.

## Execucao

Na raiz do projeto, com a API iniciada e sem outras cameras conectadas:

```bash
python tools/benchmark_stream.py videos/vazia.mp4 --scenario empty --run-label pi3-baseline-01 --output resultados/vazia-01.json
python tools/benchmark_stream.py videos/uma-pessoa.mp4 --scenario one-person --run-label pi3-baseline-01 --output resultados/uma-01.json
python tools/benchmark_stream.py videos/varias-pessoas.mp4 --scenario many-persons --run-label pi3-baseline-01 --output resultados/varias-01.json
```

Reinicie a API antes de cada execucao para zerar rastreador e caches de eventos.
Repita cada cenario tres vezes com o mesmo aquecimento e estado inicial do banco.
Espere temperatura comparavel antes das repeticoes. O cliente pode rodar em outra
maquina usando `--url ws://ENDERECO:8000/stream`; registre a topologia de rede.
Nao misture resultados do cliente local e remoto.

## Significado das medidas

- `rtt_ms`: envio do JPEG ate recebimento da resposta, incluindo espera no
  servidor e transporte. No navegador, comeca depois da codificacao JPEG.
- `completed_fps`: respostas medidas / tempo real da carga, excluindo aquecimento
  e incluindo preparacao dos frames posteriores ao primeiro frame medido.
- `receive_wait_ms`: tempo aguardando `receive_bytes`; inclui ociosidade e nao
  deve ser interpretado como latencia de rede.
- `response_ready_ms`: decodificacao, pipeline, eventos e montagem da resposta,
  antes da coleta de recursos, serializacao e envio.
- `send_ms` e `total_ms`: medidos no servidor apos envio; total exclui espera
  pelo proximo frame e inclui envio ao ASGI, sem garantir recebimento no cliente.
  Nao aparecem na resposta do proprio frame; alimentam o monitor do servidor.
- `persons_ms`, `faces_ms`, `pose_ms`, `hands_ms`, `gestures_ms`: duracoes dos
  estagios existentes, nao tempo exclusivo de cada rede. Estagios paralelos e
  tempos de gestos/pose/maos se sobrepoem; nao some esses valores.
- `process_rss_mb`: RAM residente do processo servidor; `temperature_c`: leitura
  disponivel no host. Ausencia aparece como null e contagem zero no resumo.
- Trafego: bytes dos payloads JPEG e JSON, sem cabecalhos WebSocket/TCP/TLS.

Os resumos usam mediana e p95 pelo metodo nearest-rank; as amostras individuais
tambem ficam no JSON. A coleta de recursos tem custo e deve ficar igualmente
habilitada nas comparacoes. O utilitario falha em video curto, timeout, erro da
API ou metricas desativadas, em vez de publicar uma execucao incompleta.

## Validacao disponivel

```bash
node --test tests/client_stream.test.cjs
python -m unittest discover -s tests -p 'test_*.py'
```

Esses testes validam correlacao FIFO, callbacks de conexoes antigas, falha de
codificacao, limites dos tempos do handler com dependencias simuladas e resumo
estatistico. Nao executam modelos, MongoDB ou Raspberry. A fase de medicao so
fecha com as tres cargas repetidas no hardware e os resultados registrados.
