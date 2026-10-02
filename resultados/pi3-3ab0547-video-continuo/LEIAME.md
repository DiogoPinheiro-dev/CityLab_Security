# Teste continuo de uma pessoa, 33,6 min em video fixo, commit 3ab0547

Feito em 01/10/2026, das 22:54 as 23:28 pelo relogio do PC, com o commit
`3ab0547` no Pi e so o perfil rpi3 no `.env`: pose em 416, 2 threads no rosto,
3 no PyTorch, reuso de 15 s e spinning no padrao. API reiniciada antes. E o
teste continuo de P0 em `docs/PLANO_OTIMIZACAO.md`, feito sem camera.
Resultados e leitura em "Estado verificado em 01/10/2026".

- `uma-pessoa-continuo-r1.json`: `tools/benchmark_stream.py` numa conexao so,
  5 frames de aquecimento e 535 medidos. A carga e um video de 540 frames com
  os 36 frames do video de carga repetidos 15 vezes (SHA-256
  `084478cc074dce0fe9373fe1d905cd026a59feb78070ae337761ba23632fc2e6`, fora do
  repositorio), entao cada volta de 36 frames se compara com as outras e com as
  rodadas curtas.

Resumo do registro do Pi, `~/continuo.txt`, uma leitura a cada 5 a 6 s com
`vcgencmd`, `free -m` e `vmstat`, das 22:54:10 as 23:39:02 pelo relogio do Pi,
que inclui uns 10 min depois do fim da rodada. O arquivo ficou no Pi; este e o
resumo feito la com `awk`:

```text
throttled 0x0 532
de 22:54:10 ate 23:39:02 - 532 leituras
temp max 62.3 C as 23:19:56
memoria disponivel min 150 MB as 22:55:31
swap usado: inicio 255 max 302 as 23:10:33 fim 267 MB
si>0 em 34 leituras, max 8156 | so>0 em 4 leituras, max 10568
```

`si` e `so` sao KiB/s trocados com o swap no segundo medido pelo `vmstat`.
