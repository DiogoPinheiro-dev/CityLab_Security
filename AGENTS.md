# Instrucoes para agentes (Codex e Claude)

Vale para qualquer agente que abrir este repositorio.

## Trabalho em andamento

A branch `otimizations-tests` esta no meio de um plano faseado de otimizacao do
stream para Raspberry Pi 3 B+. **Leia `docs/PLANO_OTIMIZACAO.md` antes de mexer
em performance** e atualize a secao "Estado verificado" de la ao concluir uma
etapa. Esse arquivo e a fonte unica do plano.

Regras que valem agora:

1. A fase 1 esta **fechada desde 19/09/2026**: 9 rodadas de webcam nos tres
   cenarios, resultados em `resultados/pi3-b05058f-webcam/`. O que vem agora
   esta no "Backlog de otimizacao" do plano, ordenado por retorno medido. Cada
   acao e combinada com o responsavel antes de ser implementada, uma por vez.
2. Medicao antes de otimizacao: com a webcam escolhida pelo responsavel,
   manter camera, cenario e configuracao tao estaveis quanto possivel e
   registrar a variacao entre rodadas, conforme `docs/BENCHMARK.md`. So tratar
   como ganho real uma variacao acima de 5% repetida nas tres rodadas do mesmo
   cenario.
3. Preservar recall de rostos, gestos e alertas. Ganho de tempo que perde
   deteccao nao e ganho. As regras de gesto agora usam tempo decorrido; a
   contagem de alertas da linha de base antiga nao serve como comparacao direta.
4. O deploy so dispara em push na `main`. Confirmar a conclusao do workflow e
   o processo ativo no Raspberry antes de afirmar que ele roda o novo codigo.

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
