# Fluxo de Branches

Este repositorio usa a `main` como branch estavel do app.

## Papel de cada branch

- `main`: versao estavel, pronta para demonstracao, validacao externa e deploy.
- `server`: branch principal de desenvolvimento da integracao do backend, pipeline e cliente.
- outras branches de desenvolvimento: usadas para features, refactors, correcao de bugs e experimentos antes da promocao para a `main`.

## Regras do fluxo

- Nao desenvolver direto na `main`.
- Toda alteracao deve nascer em uma branch de desenvolvimento.
- A `main` so recebe mudancas promovidas depois de validacao.
- A promocao para a `main` deve preservar contexto, por isso o padrao recomendado e `merge commit`.

## Fluxo recomendado

1. Atualize a branch de desenvolvimento a partir da base mais recente.
2. Desenvolva e teste localmente.
3. Confirme que o working tree esta limpo.
4. Promova a branch para a `main`.
5. Valide a `main` como versao estavel.
6. Publique a `main` no remoto.

## Convencao de nomes

- `feature/<tema>`
- `fix/<tema>`
- `refactor/<tema>`
- `experiment/<tema>`

Exemplos:

- `feature/mobile-client`
- `fix/websocket-reconnect`
- `refactor/pipeline-split`

## Promocao para a main

Use o script:

```powershell
.\scripts\promote-to-main.ps1 -SourceBranch server
```

Se quiser publicar no remoto no mesmo fluxo:

```powershell
.\scripts\promote-to-main.ps1 -SourceBranch server -Push
```

## Protecao local da main

Este repositorio inclui um hook local que bloqueia commits diretos na `main`.

Se voce realmente precisar criar um commit intencional na `main`, use:

```powershell
$env:ALLOW_MAIN_COMMIT='1'
git commit -m "mensagem"
Remove-Item Env:ALLOW_MAIN_COMMIT
```

O script de promocao ja faz isso automaticamente quando necessario.

## Recomendacoes para o GitHub

Se quiser refletir esse fluxo no remoto, o ideal e configurar:

- branch protection na `main`
- bloquear push direto na `main`
- exigir pull request ou merge controlado para a `main`
- usar tags para marcar snapshots estaveis

## Tags estaveis

Quando quiser marcar uma versao estavel validada, use algo como:

```powershell
git tag -a stable-2026-04-18 -m "Stable main snapshot"
git push CityLab_Security stable-2026-04-18
```
