# Instrucoes para agentes (Codex e Claude)

Vale para qualquer agente que abrir este repositorio.

## Trabalho em andamento

O trabalho acontece na `main`, que tambem e a branch de deploy. Os dois planos
abaixo sao a fonte unica de cada assunto; o estado atual e o que fica aberto
estao na secao "Estado verificado" mais recente de cada um.

- **Desempenho do stream no Raspberry Pi 3 B+:** leia `docs/PLANO_OTIMIZACAO.md`
  antes de mexer em performance e atualize o "Estado verificado" ao concluir
  uma etapa.
- **Regras de gesto e quando um alerta dispara:** leia `docs/PLANO_GESTOS.md`.
  Sao problemas diferentes: desempenho nao conserta comportamento.

Regras que valem agora:

1. Cada acao dos planos e combinada com o responsavel antes de ser
   implementada, uma por vez.
2. Medicao antes de otimizacao: com a webcam escolhida pelo responsavel,
   manter camera, cenario e configuracao tao estaveis quanto possivel e
   registrar a variacao entre rodadas, conforme `docs/BENCHMARK.md`. So tratar
   como ganho real uma variacao acima de 5% repetida nas tres rodadas do mesmo
   cenario. No Pi, deixar o log de `vcgencmd` rodando durante as rodadas: o
   limite de temperatura baixa o clock no meio da medicao.
3. Preservar recall de rostos, gestos e alertas. Ganho de tempo que perde
   deteccao nao e ganho. A contagem de alertas nao serve para comparar versoes;
   ver `docs/PLANO_GESTOS.md`.
4. O deploy so dispara em push na `main` e roda no runner self-hosted do
   proprio Pi. Parar a API antes do push, e confirmar a conclusao do workflow e
   o processo ativo no Raspberry antes de afirmar que ele roda o novo codigo.
5. O deploy sincroniza com `rsync` sem `--delete`: arquivo renomeado ou apagado
   no repositorio continua no Pi ate ser removido la.

## Convencoes do repositorio

- Documentacao e comentarios em portugues, **sem acentos** (padrao dos arquivos
  em `docs/` e `README.md`).
- Configuracao por variavel de ambiente fica em `App/settings.py`.
- Perfil de dependencias do Pi: `requirements-rpi-bookworm.txt`
  (Bookworm 64-bit, Python 3.11). Ver `docs/RASPBERRY_PI.md`.

## Validacao local

Nao exige modelos, MongoDB nem o Raspberry:

```bash
python -m unittest discover -s tests -p 'test_*.py'
node --test tests/client_stream.test.cjs
```
